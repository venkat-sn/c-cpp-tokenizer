import os
import re
from datetime import datetime
from pathlib import Path
import tiktoken
import traceback
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from dataclasses import dataclass, asdict
import hashlib
from typing import List, Optional, Set, Any, Dict, Tuple
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time
from src.utils.file_utils import FileScanner, detect_language
from config.default_config import get_default_config, update_config

@dataclass
class CodeSnippet:
    """Represents a processed code snippet"""
    content: str
    file_path: str
    language: str
    n_tokens: int
    hash: str
    size_bytes: int
    
    def to_dict(self):
        return asdict(self)

def read_file_with_encoding(file_path: Path) -> Tuple[str, str]:
    """
    Try to read file content with different encodings
    
    Returns:
        Tuple of (content, encoding_used)
    Raises:
        ValueError if file cannot be read with any encoding
    """
    # List of encodings to try, in order of preference
    encodings = [
        'utf-8', 
        'utf-8-sig',  # UTF-8 with BOM
        'ascii',
        'iso-8859-1',
        'cp1252',     # Windows-1252
        'latin1',
        'utf-16',
        'utf-32'
    ]
    
    # First, try to detect if it's a binary file
    try:
        with open(file_path, 'rb') as f:
            is_binary = False
            block = f.read(8192)  # Read first 8KB
            
            # Check for null bytes and other binary indicators
            textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
            is_binary = bool(block.translate(None, textchars))
            
            if is_binary:
                raise ValueError(f"File appears to be binary")
    except Exception as e:
        raise ValueError(f"Error checking file type: {str(e)}")

    # Try each encoding
    errors = []
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return content, encoding
        except UnicodeDecodeError as e:
            errors.append(f"{encoding}: {str(e)}")
        except Exception as e:
            errors.append(f"{encoding}: Unexpected error: {str(e)}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(traceback.format_exc())
    
    # If we get here, none of the encodings worked
    raise ValueError(f"Failed to read with any encoding. Errors:\n" + "\n".join(errors))

class CodePreprocessor:
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 additional_extensions: Set[str] = None,
                 exclude_dirs: Set[str] = None,
                 max_tokens: int = None,
                 min_tokens: int = None,
                 tokenizer: str = None,
                 n_workers: int = None,
                 appconfig: Dict[str, Any] = None):
        """
        Initialize the code preprocessor
        """
        self.config = appconfig or get_default_config()
        
        # Setup paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Set number of workers
        self.n_workers = n_workers or self.config['processing']['n_workers'] or os.cpu_count()
        
        # Log initialization parameters
        logging.debug(f"Initializing CodePreprocessor with parameters:")
        logging.debug(f"  Input directory: {self.input_dir}")
        logging.debug(f"  Output directory: {self.output_dir}")
        logging.debug(f"  Additional extensions: {additional_extensions}")
        logging.debug(f"  Exclude dirs: {exclude_dirs}")
        logging.debug(f"  Number of workers: {self.n_workers}")
        logging.debug(f"  Console Log Level: {self.config['logging']['console_level']}")
        logging.debug(f"  File Log Level: {self.config['logging']['file_level']}")
        
        # Override specific settings from arguments
        if max_tokens:
            self.config['token_settings']['max_tokens'] = max_tokens
        if min_tokens:
            self.config['token_settings']['min_tokens'] = min_tokens
        if tokenizer:
            self.config['token_settings']['tokenizer'] = tokenizer
        
        # Update file extensions and excluded dirs
        if additional_extensions:
            self.config['file_settings']['default_extensions'].update(additional_extensions)
        if exclude_dirs:
            self.config['file_settings']['exclude_dirs'].update(exclude_dirs)
        
        # Create file scanner with updated settings
        self.file_scanner = FileScanner(
            self.input_dir,
            self.config['file_settings']['default_extensions'],
            self.config['file_settings']['exclude_dirs'],
            self.config['file_settings']['include_filenames']
        )
        
        # Setup tokenizer
        self.tokenizer = tiktoken.get_encoding(
            self.config['token_settings']['tokenizer']
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_tokens': 0,
            'skipped_files': 0,
            'examples_generated': 0,
            'total_bytes': 0,
            'processing_time': 0,
            'files_per_language': {}
        }
        
        logging.info(f"CodePreprocessor initialized with {self.n_workers} workers")

    def clean_code(self, content: str) -> str:
        """Clean and normalize code content"""
        # Remove consecutive empty lines
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Normalize line endings
        content = content.replace('\r\n', '\n')
        
        # Remove trailing whitespace
        content = '\n'.join(line.rstrip() for line in content.split('\n'))
        
        return content.strip()

    def process_file(self, file_path: Path) -> Optional[Tuple[CodeSnippet, str]]:
        """Process a single code file"""
        try:
            logging.debug(f"Processing file: {file_path}")
            
            try:
                content, encoding = read_file_with_encoding(file_path)
                logging.debug(f"Successfully read file with {encoding} encoding")
            except ValueError as e:
                return None, f"Encoding error: {str(e)}"
            
            # Get file size
            size_bytes = file_path.stat().st_size
            logging.debug(f"File size: {size_bytes} bytes")
            
            # Clean content
            content = self.clean_code(content)
            
            # Skip if too short
            if not content or len(content.split()) < 1:
                return None, "Content too short (less than 1 words)"
            
            # Skip if content seems to be binary or non-text
            if '\0' in content or any(ord(c) > 127 for c in content[:1024]):
                return None, "Content appears to be binary or contains non-ASCII characters"
            
            # Tokenize
            try:
                tokens = self.tokenizer.encode(content)
                n_tokens = len(tokens)
                logging.debug(f"Tokens generated: {n_tokens}")
            except Exception as e:
                return None, f"Tokenization failed: {str(e)}"
            
            # Skip if too short or too long
            if n_tokens < self.config['token_settings']['min_tokens']:
                return None, f"Too few tokens ({n_tokens} < {self.config['token_settings']['min_tokens']})"
            if n_tokens > self.config['token_settings']['max_tokens']:
                return None, f"Too many tokens ({n_tokens} > {self.config['token_settings']['max_tokens']})"
            
            # Create content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Determine language
            language = detect_language(file_path)
            
            return CodeSnippet(
                content=content,
                file_path=str(file_path.relative_to(self.input_dir)),
                language=language,
                n_tokens=n_tokens,
                hash=content_hash,
                size_bytes=size_bytes
            ), None
                
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def process_files(self):
        """Process all files in the input directory"""
        start_time = time.time()
        
        # Scan for files
        files = self.file_scanner.scan_files()
        self.stats['total_files'] = len(files)
        
        # Track encoding statistics
        self.stats['encoding_stats'] = {}
        self.stats['skipped_reasons'] = {}
        
        # Process files in parallel
        processed_files = []
        skipped_files = []
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task(
                "Processing files...", 
                total=len(files)
            )
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for i, result in enumerate(executor.map(self.process_file, files)):
                    if isinstance(result, tuple):
                        snippet, skip_reason = result
                    else:
                        snippet, skip_reason = result, "Unknown reason"
                        
                    if snippet:
                        processed_files.append(snippet)
                        self.stats['processed_files'] += 1
                        self.stats['total_tokens'] += snippet.n_tokens
                        self.stats['total_bytes'] += snippet.size_bytes
                        
                        # Update language statistics
                        self.stats['files_per_language'][snippet.language] = \
                            self.stats['files_per_language'].get(snippet.language, 0) + 1
                    else:
                        skipped_files.append((str(files[i]), skip_reason))
                        self.stats['skipped_files'] += 1
                        # Track skip reasons
                        self.stats['skipped_reasons'][skip_reason] = \
                            self.stats['skipped_reasons'].get(skip_reason, 0) + 1
                    
                    progress.update(task, advance=1)
                    
                    # Log progress periodically
                    if (i + 1) % 100 == 0:
                        logging.info(
                            f"Processed {i + 1}/{len(files)} files. "
                            f"Success: {len(processed_files)}, "
                            f"Skipped: {len(skipped_files)}"
                        )
        
        self.stats['processing_time'] = time.time() - start_time
        
        # Log summary of skipped files
        if skipped_files:
            skipped_log = Path(self.config['logging']['logs_folder']) / f'skipped_files_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(skipped_log, 'w', encoding='utf-8') as f:
                for file_path, reason in skipped_files:
                    f.write(f"{file_path}\t{reason}\n")
            logging.info(f"List of skipped files saved to: {skipped_log}")
        
        # Log skip reasons summary
        if self.stats['skipped_reasons']:
            logging.info("\nSkip reasons summary:")
            for reason, count in self.stats['skipped_reasons'].items():
                logging.info(f"  {reason}: {count:,} files")
        
        logging.info(f"Processing complete. {len(processed_files)} files processed successfully")
        
        # Save results
        self.save_results(processed_files)

    def save_results(self, processed_files: List[CodeSnippet]):
        """Save processed results to disk"""
        # Convert to pandas DataFrame
        df = pd.DataFrame([f.to_dict() for f in processed_files])
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as parquet
        parquet_file = self.output_dir / f'processed_data_{timestamp}.parquet'
        df.to_parquet(parquet_file)
        
        # Save sample as JSON
        sample_file = self.output_dir / f'sample_{timestamp}.json'
        sample = df.head(10).to_dict(orient='records')
        with open(sample_file, 'w') as f:
            json.dump(sample, f, indent=2)
        
        # Save statistics
        stats_file = self.output_dir / f'stats_{timestamp}.json'
        self.stats['examples_generated'] = len(processed_files)
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Log summary
        logging.info("\nProcessing Summary:")
        logging.info(f"Total files scanned: {self.stats['total_files']:,}")
        logging.info(f"Files processed: {self.stats['processed_files']:,}")
        logging.info(f"Files skipped: {self.stats['skipped_files']:,}")
        logging.info(f"Total tokens: {self.stats['total_tokens']:,}")
        logging.info(f"Total size: {self.stats['total_bytes']/1024/1024:.2f} MB")
        logging.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        # Log language distribution
        logging.info("\nFiles per language:")
        for lang, count in self.stats['files_per_language'].items():
            logging.info(f"  {lang}: {count:,}")
        
        logging.info(f"\nResults saved to {self.output_dir}")
        logging.info(f"  - Processed data: {parquet_file.name}")
        logging.info(f"  - Sample data: {sample_file.name}")
        logging.info(f"  - Statistics: {stats_file.name}")