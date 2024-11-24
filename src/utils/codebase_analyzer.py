import subprocess
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

@dataclass
class TagEntry:
    """Represents a ctags entry"""
    name: str
    path: str
    pattern: str  # Search pattern
    line: int
    kind: str    # f:function, c:class, m:member, etc.
    access: Optional[str]  # public/private/protected
    signature: Optional[str]
    scope: Optional[str]  # Class or namespace scope

@dataclass
class CrossReference:
    """Represents a cscope cross-reference"""
    symbol: str
    refs: List[Dict[str, str]]  # List of {file, line, context}
    ref_type: str  # One of: called_by, calls, text, assign, global

class CodebaseAnalyzer:
    def __init__(self, parquet_path: str):
        """Initialize analyzer with just the parquet data"""
        self.parquet_path = Path(parquet_path).resolve()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Verify parquet exists
        if not self.parquet_path.exists():
            raise RuntimeError(f"Parquet file does not exist: {self.parquet_path}")
        
        # Create tools directory next to parquet file
        self.tools_dir = self.parquet_path.parent / '.code_tools'
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for tag and cscope files
        self.tags_file = self.tools_dir / "tags"
        self.cscope_files = {
            'out': self.tools_dir / "cscope.out",
            'in': self.tools_dir / "cscope.files",
            'po': self.tools_dir / "cscope.po.out"
        }
        
        # Load the parquet data
        self.df = pd.read_parquet(self.parquet_path)
        
        # Caches
        self.tags_cache: Dict[str, TagEntry] = {}
        self.cscope_cache: Dict[str, CrossReference] = {}
        
        # Create temporary directory for file reconstruction
        self.temp_dir = self.tools_dir / "temp_files"
        self.temp_dir.mkdir(exist_ok=True)

    def _reconstruct_files(self):
        """Reconstruct files from parquet content for analysis"""
        self.logger.info("Reconstructing files from parquet data...")
        
        reconstructed_files = []
        for _, row in self.df.iterrows():
            # Create file path in temp directory maintaining relative structure
            temp_file = self.temp_dir / row['file_path']
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to temp file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(row['content'])
            
            reconstructed_files.append(str(temp_file))
            
        return reconstructed_files

    def initialize(self):
        """Initialize the analysis environment"""
        try:
            # Try to generate tags first
            try:
                self._generate_tag_files()
            except Exception as e:
                self.logger.error(f"Failed to generate tags: {str(e)}")
                raise
                
            # Only try cscope if tags worked
            try:
                self._generate_cscope_files()
            except Exception as e:
                self.logger.error(f"Failed to generate cscope database, continuing with tags only: {str(e)}")
                
            self._load_tags()
            self.logger.info("Initialization complete")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    def _generate_tag_files(self):
        """Generate ctags file using reconstructed files"""
        self.logger.info("Generating ctags data...")
        
        try:
            # Get list of reconstructed files
            valid_files = self._reconstruct_files()
            
            if not valid_files:
                raise RuntimeError("No valid files found in parquet dataset")
            
            # Write file list
            file_list_path = self.tools_dir / "files_to_process.txt"
            with open(file_list_path, 'w') as f:
                for file_path in valid_files:
                    f.write(f"{file_path}\n")
            
            # Create tags
            cmd = [
                "ctags",
                "--sort=no",
                "--fields=+ailmnKzS",
                "--extras=+q",
                "--c++-kinds=+p",
                "--language-force=C++",
                "-L", str(file_list_path),
                "-f", str(self.tags_file)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Clean up
            file_list_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Error generating tags: {str(e)}")
            raise

    def _generate_cscope_files(self):
        """Generate cscope database using reconstructed files"""
        self.logger.info("Generating cscope cross-reference...")
        
        try:
            # Use already reconstructed files
            valid_files = [str(f) for f in self.temp_dir.rglob('*') if f.is_file()]
            
            if not valid_files:
                raise RuntimeError("No valid files found in temp directory")
            
            # Write file list for cscope
            with open(self.cscope_files['in'], 'w') as f:
                for file_path in valid_files:
                    f.write(f"{file_path}\n")
            
            # Build cscope database
            cmd = ["cscope", "-b", "-q", "-k", "-i", str(self.cscope_files['in'])]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
        except Exception as e:
            self.logger.error(f"Error generating cscope files: {str(e)}")
            raise
            
    def _load_tags(self):
        """Load and parse ctags data into memory"""
        self.logger.info("Loading ctags data...")
        
        try:
            with open(self.tags_file) as f:
                for line in f:
                    if line.startswith('!'):  # Skip meta information
                        continue
                        
                    # Parse tag line
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        name = parts[0]
                        path = parts[1]
                        pattern = parts[2]
                        
                        # Extract additional fields
                        fields = {}
                        for part in parts[3:]:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                fields[key] = value
                        
                        entry = TagEntry(
                            name=name,
                            path=path,
                            pattern=pattern,
                            line=int(fields.get('line', 0)),
                            kind=fields.get('kind', ''),
                            access=fields.get('access'),
                            signature=fields.get('signature'),
                            scope=fields.get('class') or fields.get('namespace')
                        )
                        
                        self.tags_cache[name] = entry
                        
        except Exception as e:
            self.logger.error(f"Error loading tags: {str(e)}")
            raise
            
    def query_symbol(self, symbol: str) -> Dict:
        """
        Get comprehensive information about a symbol
        
        Args:
            symbol: Symbol name to query
            
        Returns:
            Dictionary containing symbol information and references
        """
        result = {
            'symbol': symbol,
            'tag_info': None,
            'definitions': [],
            'references': [],
            'callers': [],
            'callees': []
        }
        
        # Get tag information
        if symbol in self.tags_cache:
            result['tag_info'] = self.tags_cache[symbol]
        
        # Query cscope for references
        try:
            # Find definition
            cmd_def = f"cscope -d -L1{symbol}"
            def_output = subprocess.run(cmd_def, shell=True, capture_output=True, text=True)
            
            for line in def_output.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    result['definitions'].append({
                        'file': parts[0],
                        'line': int(parts[2]),
                        'context': ' '.join(parts[3:])
                    })
            
            # Find references
            cmd_ref = f"cscope -d -L0{symbol}"
            ref_output = subprocess.run(cmd_ref, shell=True, capture_output=True, text=True)
            
            for line in ref_output.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    result['references'].append({
                        'file': parts[0],
                        'line': int(parts[2]),
                        'context': ' '.join(parts[3:])
                    })
            
            # Find callers
            cmd_callers = f"cscope -d -L2{symbol}"
            caller_output = subprocess.run(cmd_callers, shell=True, capture_output=True, text=True)
            
            for line in caller_output.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    result['callers'].append({
                        'file': parts[0],
                        'line': int(parts[2]),
                        'context': ' '.join(parts[3:])
                    })
            
            # Find callees
            cmd_callees = f"cscope -d -L3{symbol}"
            callee_output = subprocess.run(cmd_callees, shell=True, capture_output=True, text=True)
            
            for line in callee_output.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    result['callees'].append({
                        'file': parts[0],
                        'line': int(parts[2]),
                        'context': ' '.join(parts[3:])
                    })
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error querying cscope: {str(e)}")
            
        return result
    
    def generate_training_chunk(self, symbol: str) -> Dict:
        """
        Generate a training chunk for a symbol with rich context
        
        Args:
            symbol: Symbol to generate training data for
            
        Returns:
            Dictionary containing training data and metadata
        """
        # Get symbol information
        info = self.query_symbol(symbol)
        
        if not info['definitions']:
            return None
            
        # Build context
        context_parts = []
        
        # Add symbol information
        if info['tag_info']:
            tag = info['tag_info']
            context_parts.append(f"Type: {tag.kind}")
            if tag.scope:
                context_parts.append(f"Scope: {tag.scope}")
            if tag.signature:
                context_parts.append(f"Signature: {tag.signature}")
            if tag.access:
                context_parts.append(f"Access: {tag.access}")
                
        # Add relationship information
        if info['callers']:
            caller_contexts = [f"{c['file']}:{c['line']} - {c['context']}" 
                             for c in info['callers'][:5]]  # Limit to 5 examples
            context_parts.append("Called by:\n" + "\n".join(caller_contexts))
            
        if info['callees']:
            callee_contexts = [f"{c['file']}:{c['line']} - {c['context']}" 
                             for c in info['callees'][:5]]  # Limit to 5 examples
            context_parts.append("Calls:\n" + "\n".join(callee_contexts))
            
        # Get the actual implementation
        impl_file = info['definitions'][0]['file']
        impl_line = info['definitions'][0]['line']
        
        with open(impl_file) as f:
            lines = f.readlines()
            
        # Extract the implementation (simple approach - could be enhanced)
        implementation = []
        i = impl_line - 1
        brace_count = 0
        started = False
        
        while i < len(lines):
            line = lines[i]
            if not started:
                if '{' in line:
                    started = True
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                implementation.append(line)
            else:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                implementation.append(line)
                if brace_count == 0:
                    break
            i += 1
            
        return {
            'symbol': symbol,
            'context': '\n'.join(context_parts),
            'implementation': ''.join(implementation),
            'metadata': {
                'file': impl_file,
                'line': impl_line,
                'num_callers': len(info['callers']),
                'num_callees': len(info['callees'])
            }
        }
    
    def batch_process_symbols(self, batch_size: int = 100) -> List[Dict]:
        """
        Process symbols in batches to generate training data
        
        Args:
            batch_size: Number of symbols to process in parallel
            
        Returns:
            List of training chunks
        """
        chunks = []
        
        # Get all function and method symbols
        function_symbols = [
            name for name, tag in self.tags_cache.items()
            if tag.kind in ('function', 'method')
        ]
        
        def process_symbol(symbol):
            try:
                return self.generate_training_chunk(symbol)
            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {str(e)}")
                return None
                
        # Process in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for chunk in executor.map(process_symbol, function_symbols):
                if chunk:
                    chunks.append(chunk)
                
                if len(chunks) >= batch_size:
                    yield chunks
                    chunks = []
                    
        if chunks:
            yield chunks