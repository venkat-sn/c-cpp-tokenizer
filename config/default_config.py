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
                '.x', '.c', '.cpp', '.h', '.hpp', '.cc',
                # Make files
                'makefile', '.mak', '.make',
                # Build files
                '.build', '.cmake'
            },
            
            # Directories to exclude by default
            'exclude_dirs': {
                'node_modules',
                '.git',
                '.svn',
                'dist',
                'venv',
                '__pycache__',
                '.idea',
                '.vscode',
                'ut',
                'unit_test',
                'acceptance',
                'cu_emulator',
                'asn1',
                '3rdParty',
                'defs',
                'CU_autoscript',
                'mnxt_oam',
                "transport",
                "FlexRAN",
                "xRANc",
                "lteclpal"
            },

            'include_filenames': {
                'BUILD_DU'
            },
            
            # File size limits (in bytes)
            'max_file_size': 10 * 1024 * 1024,  # 10 MB
            'min_file_size': 1  # 10 bytes
        },
        
        # Token processing settings
        'token_settings': {
            # Default tokenizer
            'tokenizer': 'cl100k_base',
            
            # Token limits
            'max_tokens': 10000000, # 10 million tokens
            'min_tokens': 1,
            
            # Batch size for processings
            'batch_size': 1024
        },
        
        # Processing settings
        'processing': {
            # Number of worker processes (None = CPU count)
            'n_workers': None,
            
            # Chunk size for reading large files
            'chunk_size': 1024 * 1024,  # 1MB
            
            # Progress bar settings
            'show_progress': True,
            'progress_update_interval': 100
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

def save_default_config(output_path: str):
    """Save default configuration to JSON file"""
    try:
        config = get_default_config()
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration saved to: {output_path}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error saving default configuration: {str(e)}")
        sys.exit(1)