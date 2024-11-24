import os
import re
import json
import logging
import time
import hashlib
import tiktoken
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import fnmatch as fnmatch
import traceback

import pandas as pd
from config.default_config import get_default_config, update_config
from src.utils.file_utils import FileScanner, detect_language

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

@dataclass
class ScanPattern:
    """Represents a file/directory scanning pattern"""
    pattern: str
    is_file: bool = False
    is_dir: bool = False
    is_wildcard: bool = False
    
    @classmethod
    def from_path(cls, path: str) -> 'ScanPattern':
        """Create a ScanPattern from a path string"""
        path = path.strip()
        
        # Check if it's a wildcard pattern
        if any(char in path for char in '*?[]'):
            return cls(path, is_wildcard=True)
            
        path_obj = Path(path)
        if path_obj.exists():
            return cls(path, is_file=path_obj.is_file(), is_dir=path_obj.is_dir())
        else:
            # Assume it's a wildcard if it doesn't exist
            return cls(path, is_wildcard=True)

class EnhancedFileScanner:
    """Enhanced file scanner supporting explicit include/exclude patterns"""
    
    def __init__(
        self,
        include_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
        allowed_extensions: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        use_default_includes: bool = True,
        use_default_excludes: bool = True
    ):
        self.config = config or get_default_config()
        
        # Get effective patterns by merging CLI patterns with defaults if requested
        self.include_patterns = [
            ScanPattern.from_path(p) for p in self._merge_include_patterns(
                include_patterns, 
                use_defaults=use_default_includes
            )
        ]
        
        self.exclude_patterns = [
            ScanPattern.from_path(p) for p in self._merge_exclude_patterns(
                exclude_patterns,
                use_defaults=use_default_excludes
            )
        ]
        
        # Get allowed extensions
        self.allowed_extensions = allowed_extensions or set()
        
        # Log the final scanning configuration
        logging.info("\nScanner Configuration:")
        logging.info("Include patterns:")
        for pattern in self.include_patterns:
            logging.info(f"  {pattern.pattern}")
        logging.info("\nExclude patterns:")
        for pattern in self.exclude_patterns:
            logging.info(f"  {pattern.pattern}")

    def _merge_include_patterns(self, cli_patterns: List[str], use_defaults: bool) -> List[str]:
        """Get final include patterns after merging with defaults"""
        patterns = set(cli_patterns)
        
        if use_defaults:
            # Add patterns for file extensions from config
            for ext in self.config['file_settings']['default_extensions']:
                patterns.add(f"**/*{ext}")
            
            # Add patterns for specific filenames
            for filename in self.config['file_settings']['include_filenames']:
                patterns.add(f"**/{filename}")
                
            logging.debug("Using default include patterns")
            
        return list(patterns)

    def _merge_exclude_patterns(self, cli_patterns: Optional[List[str]], use_defaults: bool) -> List[str]:
        """Get final exclude patterns after merging with defaults"""
        patterns = set(cli_patterns or [])
        
        if use_defaults:
            # Add exclude patterns from config's exclude_dirs
            for exclude_dir in self.config['file_settings']['exclude_dirs']:
                patterns.add(f"**/{exclude_dir}/**")
                
            logging.debug("Using default exclude patterns")
            
        return list(patterns)

    def should_include_path(self, path: Path) -> bool:
        """Determine if a path should be included"""
        path_str = str(path)
        
        # First check exclusions
        for pattern in self.exclude_patterns:
            if pattern.is_wildcard:
                if fnmatch.fnmatch(path_str, pattern.pattern):
                    return False
            elif pattern.is_file and path_str == pattern.pattern:
                return False
            elif pattern.is_dir and path_str.startswith(pattern.pattern):
                return False
        
        # Then check inclusions
        for pattern in self.include_patterns:
            if pattern.is_wildcard:
                if fnmatch.fnmatch(path_str, pattern.pattern):
                    return self._check_extension(path)
            elif pattern.is_file and path_str == pattern.pattern:
                return self._check_extension(path)
            elif pattern.is_dir and path_str.startswith(pattern.pattern):
                return self._check_extension(path)
                
        return False

    def _check_extension(self, path: Path) -> bool:
        """Check if file has allowed extension"""
        if not self.allowed_extensions:
            return True
        return path.suffix.lower() in self.allowed_extensions

    def scan_files(self) -> List[Path]:
        """Scan for files based on patterns"""
        included_files = set()
        
        # Process each include pattern
        for pattern in self.include_patterns:
            if pattern.is_file:
                path = Path(pattern.pattern)
                if path.exists() and path.is_file() and self.should_include_path(path):
                    included_files.add(path)
            else:
                # For directories and wildcards, walk the tree
                base_dir = Path(pattern.pattern).parent
                if str(base_dir) == '.':
                    base_dir = Path.cwd()
                    
                for dirpath, _, filenames in os.walk(str(base_dir)):
                    dir_path = Path(dirpath)
                    for filename in filenames:
                        file_path = dir_path / filename
                        if self.should_include_path(file_path):
                            included_files.add(file_path)
        
        files_list = sorted(list(included_files))
        logging.info(f"\nFound {len(files_list)} matching files")
        return files_list

class CodePreprocessor:
    def __init__(
        self,
        include_patterns: List[str],
        output_dir: str,
        exclude_patterns: Optional[List[str]] = None,
        additional_extensions: Optional[Set[str]] = None,
        exclude_dirs: Optional[Set[str]] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        tokenizer: Optional[str] = None,
        n_workers: Optional[int] = None,
        appconfig: Optional[Dict[str, Any]] = None,
        use_default_includes: bool = True,
        use_default_excludes: bool = True
    ):
        """
        Initialize the code preprocessor
        
        Args:
            include_patterns: List of files/directories/patterns to include
            output_dir: Directory where processed files will be saved
            exclude_patterns: List of files/directories/patterns to exclude
            additional_extensions: Additional file extensions to process
            exclude_dirs: Additional directories to exclude (maintained for backwards compatibility)
            max_tokens: Maximum tokens per example
            min_tokens: Minimum tokens per example
            tokenizer: Name of the tokenizer to use
            n_workers: Number of worker processes
            appconfig: Application configuration dictionary
        """
        self.config = appconfig or get_default_config()
        
        # Setup paths
        self.output_dir = Path(output_dir)
        
        # Set number of workers
        self.n_workers = n_workers or self.config['processing']['n_workers'] or os.cpu_count()
        
        # Log initialization parameters
        logging.debug(f"Initializing CodePreprocessor with parameters:")
        logging.debug(f"  Output directory: {self.output_dir}")
        logging.debug(f"  Include patterns: {include_patterns}")
        logging.debug(f"  Exclude patterns: {exclude_patterns}")
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
        
        # Update file extensions and excluded dirs in config
        if additional_extensions:
            self.config['file_settings']['default_extensions'].update(additional_extensions)
            
        # Convert exclude_dirs to exclude_patterns if provided (for backwards compatibility)
        if exclude_dirs:
            exclude_patterns = (exclude_patterns or []) + [
                f"**/{d}/**" for d in exclude_dirs  # Convert to glob pattern
            ]
            
        # Create enhanced file scanner
        self.file_scanner = EnhancedFileScanner(
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            allowed_extensions=self.config['file_settings']['default_extensions'],
            config=self.config,
            use_default_includes=use_default_includes,
            use_default_excludes=use_default_excludes
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
                file_path=str(file_path),  # Changed from relative_to(self.input_dir)
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