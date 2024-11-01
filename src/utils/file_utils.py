import os
from pathlib import Path
from typing import Set, List
import logging
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)

class FileScanner:
    """Utility class for scanning and filtering source code files"""
    
    def __init__(self, 
                 base_dir: str, 
                 additional_extensions: Set[str] = None,
                 exclude_dirs: Set[str] = None,
                 include_filenames: Set[str] = None):
        """
        Initialize file scanner
        
        Args:
            base_dir: Base directory to scan
            additional_extensions: Additional file extensions to include
            exclude_dirs: Directories to exclude from scanning
            include_filenames: Set of exact filenames to include (e.g. {'Makefile', 'CMakeLists.txt'})
        """
        self.base_dir = Path(base_dir)
        self.extensions = additional_extensions or set()    
        self.exclude_dirs = exclude_dirs or set()
        self.include_filenames = include_filenames or set()

        logger.debug(f"Initialized scanner with extensions: {self.extensions}")
        logger.debug(f"Excluded directories: {self.exclude_dirs}")
        logger.debug(f"Include Filenames: {self.include_filenames}")
    
    def is_makefile(self, path: Path) -> bool:
        """Check if file is a makefile (case insensitive)"""
        return path.name.lower() == 'makefile'
    
    def should_process_file(self, path: Path) -> bool:
        """Determine if a file should be processed"""
        # Check if file is in excluded directory
        for exclude_dir in self.exclude_dirs:
            if exclude_dir in path.parts:
                return False
        
        # Process makefiles
        if self.is_makefile(path):
            return True
            
        # Check extension
        return ((path.suffix.lower() in self.extensions) or (path.name in self.include_filenames))
    
    def scan_files(self, show_progress: bool = True) -> List[Path]:
        """
        Scan directory for processable files
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            List of file paths to process
        """
        files = []
        total_scanned = 0
        
        logger.info(f"Scanning directory: {self.base_dir}")
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)
            
            for root, dirs, filenames in os.walk(self.base_dir):
                # Remove excluded dirs
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
                
                for filename in filenames:
                    total_scanned += 1
                    path = Path(root) / filename
                    
                    if self.should_process_file(path):
                        files.append(path)
                    #else:
                    #    logger.debug(f"Skipping file: {path}")

                    if show_progress and total_scanned % 100 == 0:
                        progress.update(task, description=f"Found {len(files)} files...")
        
        logger.info(f"Scan complete. Found {len(files)} files to process out of {total_scanned} total files")
        return files

def detect_language(file_path: Path) -> str:
    """Detect programming language from file extension"""
    if file_path.suffix.lower() in {'.cpp', '.hpp', '.cc', '.xx'}:
        return 'cpp'
    elif file_path.suffix.lower() in {'.c', '.h', '.x'}:
        return 'c'
    elif file_path.name.lower() == 'makefile' or file_path.suffix.lower() in {'.mak', '.make'}:
        return 'makefile'
    else:
        return 'other'