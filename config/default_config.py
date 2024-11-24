import json
import sys
import logging
from typing import Dict, Any
from pathlib import Path

LOGS_DIR = 'logs'

def get_default_config() -> Dict[str, Any]:
    """
    Returns default configuration settings for the preprocessor
    """
    return {
        # File processing settings
        'file_settings': {
            # Default file extensions to process
            'default_extensions': {
                # C/C++ files
                '.x', '.c', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.C',
                # Headers
                '.hxx', '.H', '.hh',
                # Make files
                'makefile', '.mak', '.make', 'Makefile', 'GNUmakefile',
                # Build files
                '.build', '.cmake', 'CMakeLists.txt',
                # Scripts
                '.sh', '.bash', '.ksh',
                # Configuration
                '.ini', '.conf', '.cfg', '.xml', '.json', '.yaml', '.yml',
                # Documentation
                '.txt', '.md', '.rst'
            },
            
            # Default include/exclude patterns
            'default_include_patterns': [
                # C/C++
                '**/*.c', '**/*.cpp', '**/*.h', '**/*.x', '**/*.hpp', '**/*.cc',
                '**/*.cxx', '**/*.C', '**/*.hxx', '**/*.H', '**/*.hh',
                # Make/Build
                '**/[Mm]akefile', '**/*.mak', '**/*.make', '**/GNUmakefile',
                '**/CMakeLists.txt', '**/*.build', '**/*.cmake',
                # Scripts
                '**/*.sh', '**/*.bash', '**/*.ksh',
                # Config
                '**/*.ini', '**/*.conf', '**/*.cfg', '**/*.xml', '**/*.json', 
                '**/*.yaml', '**/*.yml',
                # Docs
                '**/*.txt', '**/*.md', '**/*.rst'
            ],
            
            'default_exclude_patterns': [
                # Version control
                '**/.git/**', '**/.svn/**', '**/.hg/**', '**/.bzr/**',
                '**/_darcs/**', '**/CVS/**',
                # Build artifacts and dependencies
                '**/node_modules/**', '**/bower_components/**',
                '**/vendor/**', '**/third_party/**', '**/3rdparty/**',
                '**/build/**', '**/dist/**', '**/out/**',
                '**/target/**', '**/Debug/**', '**/Release/**',
                '**/x64/**', '**/x86/**',
                '**/venv/**', '**/env/**', '**/.env/**',
                '**/__pycache__/**', '**/*.pyc', '**/*.pyo', '**/*.pyd',
                # IDE and editor files
                '**/.idea/**', '**/.vscode/**', '**/.vs/**',
                '**/*.swp', '**/*~', '**/.DS_Store',
                # Test directories
                '**/tests/**', '**/test/**', '**/testing/**',
                '**/ut/**', '**/unit_test/**', '**/unittest/**',
                '**/acceptance/**', '**/integration/**',
                # Project-specific excludes
                '**/cu_emulator/**', '**/asn1/**',
                '**/3rdParty/**', '**/defs/**',
                '**/CU_autoscript/**', '**/mnxt_oam/**',
                '**/transport/**', '**/FlexRAN/**',
                '**/xRANc/**', '**/lteclpal/**',
                # Generated files
                '**/generated/**', '**/gen/**', '**/autogen/**',
                # Object files and libraries
                '**/*.o', '**/*.obj', '**/*.a', '**/*.lib',
                '**/*.so', '**/*.so.*', '**/*.dylib', '**/*.dll',
                # Temporary and backup files
                '**/tmp/**', '**/temp/**', '**/backup/**',
                '**/*.bak', '**/*.backup', '**/*.tmp'
            ],

            # Legacy exclude_dirs (maintaining your existing ones plus some additions)
            'exclude_dirs': {
                'node_modules', 'bower_components',
                'vendor', 'third_party', '3rdparty',
                'build', 'dist', 'out', 'target',
                'Debug', 'Release', 'x64', 'x86',
                '.git', '.svn', '.hg', 'CVS',
                'venv', 'env', '.env',
                '__pycache__',
                '.idea', '.vscode', '.vs',
                'tests', 'test', 'testing',
                'ut', 'unit_test', 'unittest',
                'acceptance', 'integration',
                'cu_emulator', 'asn1',
                '3rdParty', 'defs',
                'CU_autoscript', 'mnxt_oam',
                'transport', 'FlexRAN',
                'xRANc', 'lteclpal',
                'generated', 'gen', 'autogen',
                'tmp', 'temp', 'backup'
            },

            'include_filenames': {
                'BUILD_DU',
                'CMakeLists.txt',
                'configure',
                'Makefile',
                'GNUmakefile',
                'README',
                'README.md',
                'LICENSE',
                'COPYING'
            },
            
            # Pattern matching settings
            'pattern_matching': {
                'case_sensitive': False,
                'follow_symlinks': False,
                'recursive': True,
                'ignore_hidden_files': True,
                'ignore_hidden_dirs': True
            },
            
            # File size limits (in bytes)
            'max_file_size': 10 * 1024 * 1024,  # 10 MB
            'min_file_size': 1  # 10 bytes
        },
        
        # Added some more token settings
        'token_settings': {
            'tokenizer': 'cl100k_base',
            'max_tokens': 10000000,
            'min_tokens': 1,
            'batch_size': 1024,
            'skip_empty_lines': True,
            'normalize_whitespace': True,
            'remove_comments': False,  # Keep comments by default
            'preserve_indentation': True
        },
        
        # Added some more processing settings
        'processing': {
            'n_workers': None,
            'chunk_size': 1024 * 1024,
            'show_progress': True,
            'progress_update_interval': 100,
            'fail_on_error': False,  # Continue processing if some files fail
            'max_retries': 3,  # Number of times to retry failed files
            'retry_delay': 1,  # Seconds between retries
            'skip_binary_check': False,  # Check for binary files
            'max_file_count': None  # Optional limit on total files to process
        },
        
        # Output settings
        'output': {
            # Output file names
            'data_file': 'processed_data_{timestamp}.parquet',
            'sample_file': 'sample_{timestamp}.json',
            'stats_file': 'stats_{timestamp}.json',
            
            # Number of samples to save
            'n_samples': 10,
            
            # Compression settings for parquet
            'parquet_compression': 'snappy',
            
            # Whether to save intermediate results
            'save_intermediate': False,
            'intermediate_interval': 1000  # files
        },
        
        # Logging settings
        'logging': {
            # Log levels
            'console_level': 'DEBUG',
            'file_level': 'DEBUG',
            
            # Log format
            'console_format': '%(message)s',
            'file_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            
            # Rich console settings
            'use_rich_console': True,
            'show_time': True,
            'show_path': True,
            'rich_tracebacks': True,
            
            # Log file settings
            'log_filename': 'preprocessing_{timestamp}.log',
            'max_log_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,

            # Log Folder
            'logs_folder': LOGS_DIR
        },
        
        # Language detection settings
        'language_detection': {
            'default_language': 'other',
            'extensions_map': {
                'cpp': {'.cpp', '.hpp', '.cc'},
                'c': {'.c', '.h'},
                'makefile': {'makefile', '.mak', '.make'},
                'cmake': {'.cmake'},
                'build': {'.build'}
            }
        },
        
        # Scanner settings (new)
        'scanner': {
            # Maximum depth for recursive directory traversal
            'max_depth': None,
            
            # Whether to expand environment variables in paths (e.g., $HOME, %USERPROFILE%)
            'expand_vars': True,
            
            # Whether to resolve and follow symbolic links
            'follow_links': False,
            
            # Whether to use gitignore-style pattern matching
            'use_gitignore_style': True,
            
            # Additional pattern matching flags
            'pattern_flags': {
                'case_sensitive': False,
                'match_hidden': False,
                'require_literal_separator': True,
                'require_literal_leading_dot': True
            }
        }
    }

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration file: {str(e)}")
        sys.exit(1)

def update_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update base configuration with override values
    
    Args:
        base_config: Base configuration dictionary
        overrides: Dictionary of override values
    
    Returns:
        Updated configuration dictionary
    """
    result = base_config.copy()
    
    def update_recursive(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                update_recursive(d1[k], v)
            else:
                d1[k] = v
    
    update_recursive(result, overrides)
    return result

# Make sure your ConfigEncoder is defined (which you already have):
class ConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle sets and other special types"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def save_default_config(output_path: str):
    """Save default configuration to JSON file"""
    try:
        config = get_default_config()
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, cls=ConfigEncoder)  # Added cls=ConfigEncoder
        print(f"Default configuration saved to: {output_path}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error saving default configuration: {str(e)}")
        sys.exit(1)