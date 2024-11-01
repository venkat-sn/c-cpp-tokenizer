import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from datetime import datetime
import os
from typing import Dict, Any

def setup_logging(appconfig: Dict[str, Any]) -> Path:
    # Create logs directory
    log_dir = Path( appconfig['logging']['logs_folder'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'code_processor_{timestamp}.log'
    
    # Set base log level
    root_logger = logging.getLogger()
    console_log_level_str = appconfig['logging']['console_level'].upper()
    file_log_level_str = appconfig['logging']['file_level'].upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    file_log_level = getattr(logging, file_log_level_str, logging.DEBUG)
    root_logger.setLevel(file_log_level)
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Create console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        show_time=True,
        markup=True,
        tracebacks_show_locals=True,
        log_time_format='[%Y-%m-%d %H:%M:%S]',
        console=Console(theme=Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "red"
        }))
    )
    console_handler.setLevel(console_log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_log_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]'
    )
    
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial information
    logging.info("Code Processing Tool Started")
    logging.debug(f"Python version: {sys.version}")
    logging.debug(f"Operating System: {os.name}")
    logging.debug(f"Working Directory: {os.getcwd()}")
    logging.debug(f"Log file: {log_file}")
    
    return log_file