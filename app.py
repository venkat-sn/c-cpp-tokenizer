import argparse
import sys
import json
import traceback
from pathlib import Path
from rich.console import Console
from typing import Dict, Any
import logging
from datetime import datetime
import traceback
from src.preprocessor import CodePreprocessor
from src.utils.data_explorer import DataExplorer
from src.utils.logging_utils import setup_logging
from config.default_config import *

class ConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle sets and other special types"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

class CodeProcessingCLI:
    def __init__(self):
        self.parser = self.create_parser()
        self.config = get_default_config()
    
    def create_parser(self):
        """Create main parser and subparsers for different commands"""
        parser = argparse.ArgumentParser(
            description='Code Processing and Analysis Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=self.get_usage_examples()
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Preprocess command
        preprocess_parser = subparsers.add_parser(
            'preprocess', 
            help='Preprocess source code files',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_preprocess_arguments(preprocess_parser)
        
        # Explore command
        explore_parser = subparsers.add_parser(
            'explore', 
            help='Explore processed data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_explore_arguments(explore_parser)
        
        # Config command
        config_parser = subparsers.add_parser(
            'config', 
            help='Manage configuration',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_config_arguments(config_parser)
        
        return parser
    
    def _add_preprocess_arguments(self, parser):
        """Add preprocessing-related arguments"""
        # Configuration saving argument
        parser.add_argument(
            '--save-config',
            type=str,
            help='Save default configuration to specified JSON file and exit'
        )
        
        # Processing arguments group
        processing_args = parser.add_argument_group('Processing arguments')
        
        processing_args.add_argument(
            '--input-dir',
            help='Input directory containing code'
        )
        
        processing_args.add_argument(
            '--output-dir',
            help='Output directory for processed data'
        )
        
        processing_args.add_argument(
            '--max-tokens',
            type=int,
            help='Maximum tokens per example'
        )
        
        processing_args.add_argument(
            '--min-tokens',
            type=int,
            help='Minimum tokens per example'
        )
        
        processing_args.add_argument(
            '--workers',
            type=int,
            help='Number of worker processes'
        )
        
        processing_args.add_argument(
            '--tokenizer',
            type=str,
            help='Tokenizer to use (default: cl100k_base)'
        )
        
        processing_args.add_argument(
            '--additional-extensions',
            type=str,
            nargs='+',
            help='Additional file extensions to process (e.g., .gradle .conf)'
        )
        
        processing_args.add_argument(
            '--exclude-dirs',
            type=str,
            nargs='+',
            help='Additional directories to exclude'
        )
        
        processing_args.add_argument(
            '--config',
            type=str,
            help='Path to JSON configuration file'
        )

        processing_args.add_argument(
            '--explore-after',
            action='store_true',
            help='Launch explorer after preprocessing'
        )
    
    def _add_explore_arguments(self, parser):
        """Add exploration-related arguments"""
        base_explore = parser.add_argument_group('Basic Options')
        base_explore.add_argument('file', help='Path to parquet file to explore')
        
        # Create subcommands for different exploration modes
        subparsers = parser.add_subparsers(
            dest='explore_command',
            title='Exploration Commands',
            description='Available exploration commands'
        )
        
        # Search command
        search_parser = subparsers.add_parser(
            'search',
            help='Search code content',
            description='Search through processed code content'
        )
        search_parser.add_argument('pattern', help='Pattern to search for')
        search_parser.add_argument(
            '--context',
            type=int,
            default=2,
            help='Number of context lines (default: 2)'
        )
        search_parser.add_argument(
            '--export',
            help='Export results to file (.json, .csv, or .parquet)'
        )
        
        # Structure command
        structure_parser = subparsers.add_parser(
            'structure',
            help='Analyze folder structure',
            description='Analyze and display folder structure of processed code'
        )
        structure_parser.add_argument(
            '--max-depth',
            type=int,
            help='Maximum depth to display in folder tree'
        )
        structure_parser.add_argument(
            '--min-files',
            type=int,
            default=1,
            help='Minimum files in folder to display (default: 1)'
        )
        structure_parser.add_argument(
            '--export',
            help='Export structure analysis to file (.json, .csv, or .parquet)'
        )
        
        # Validate command
        validate_parser = subparsers.add_parser(
            'validate',
            help='Validate processed files',
            description='Validate random samples of processed files'
        )
        validate_parser.add_argument(
            '--sample-size',
            type=int,
            default=5,
            help='Number of random files to validate (default: 5)'
        )
    
    def _add_config_arguments(self, parser):
        """Add configuration-related arguments"""
        config_subparsers = parser.add_subparsers(dest='config_command', help='Configuration commands')
        
        # Save default config
        save_parser = config_subparsers.add_parser('save', help='Save default configuration')
        save_parser.add_argument('output', help='Output path for configuration file')
        
        # Show current config
        show_parser = config_subparsers.add_parser('show', help='Show current configuration')
        show_parser.add_argument('--file', help='Configuration file to show')
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Convert lists back to sets where needed
            if 'file_settings' in config:
                if 'default_extensions' in config['file_settings']:
                    config['file_settings']['default_extensions'] = set(
                        config['file_settings']['default_extensions']
                    )
                if 'exclude_dirs' in config['file_settings']:
                    config['file_settings']['exclude_dirs'] = set(
                        config['file_settings']['exclude_dirs']
                    )
            
            if 'language_detection' in config:
                if 'extensions_map' in config['language_detection']:
                    for lang in config['language_detection']['extensions_map']:
                        config['language_detection']['extensions_map'][lang] = set(
                            config['language_detection']['extensions_map'][lang]
                        )
            
            return config
            
        except Exception as e:
            logging.exception(f"[red]Error loading configuration file: {str(e)}[/red]")
            logging.exception(traceback.format_exc())
            sys.exit(1)
    
    def save_config(self, output_path: str):
        """Save configuration to JSON file"""
        try:
            # Create directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config = get_default_config()
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, cls=ConfigEncoder)
            logging.debug(f"[green]Default configuration saved to: {output_path}[/green]")
            return True
        except Exception as e:
            logging.exception(f"[red]Error saving configuration: {str(e)}[/red]")
            logging.exception(traceback.format_exc())
            return False
    
    def validate_preprocess_args(self, args):
        """Validate preprocessing arguments"""
        if not args.save_config:
            if not args.input_dir:
                self.parser.error('--input-dir is required for processing')
            if not args.output_dir:
                self.parser.error('--output-dir is required for processing')
    
    def handle_preprocess(self, args):
        logging.info("\n[yellow]Starting code preprocessing...[/yellow]")

        # Validate arguments
        self.validate_preprocess_args(args)

        logging.debug("\n[yellow]Loading configuration...[/yellow]")
        
        # Handle configuration saving request
        if args.save_config:
            if self.save_config(args.save_config):
                return
            sys.exit(1)
        
        try:
            logging.debug("\n[yellow]Loaded configuration:[/yellow]")

            # Load configuration file if provided
            if args.config:
                self.config = self.load_config_file(args.config)
            
            # Convert additional extensions to set with dots
            additional_extensions = None
            if args.additional_extensions:
                additional_extensions = {
                    ext if ext.startswith('.') else f'.{ext}'
                    for ext in args.additional_extensions
                }

            logging.debug("\n[yellow]Initializing preprocessor...[/yellow]")

            # Initialize preprocessor
            processor = CodePreprocessor(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                additional_extensions=additional_extensions,
                exclude_dirs=set(args.exclude_dirs) if args.exclude_dirs else None,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                tokenizer=args.tokenizer,
                n_workers=args.workers,
                appconfig=self.config
            )

            logging.debug("\n[yellow]Starting preprocessing...[/yellow]")
            
            # Process files
            output_file = processor.process_files()
            
            # Launch explorer if requested
            if args.explore_after and output_file:
                logging.debug("\n[yellow]Launching data explorer...[/yellow]")
                explorer = DataExplorer(output_file)
                explorer.show_basic_stats()
                explorer.validate_random_files()
                
        except KeyboardInterrupt:
            logging.exception("\n[red]Processing interrupted by user[/red]")
            sys.exit(1)
        except Exception as e:
            logging.exception(f"[red]Error during processing: {str(e)}[/red]")
            logging.exception(traceback.format_exc())
            sys.exit(1)
    
    def handle_explore(self, args):
        """Handle explore command"""
        try:
            explorer = DataExplorer(args.file)
            
            # Always show basic stats first
            explorer.show_basic_stats()
            
            # Handle different exploration subcommands
            if not hasattr(args, 'explore_command') or not args.explore_command:
                # If no subcommand, just show stats
                return

            if args.explore_command == 'search':
                explorer.search_content(
                    pattern=args.pattern,
                    context_lines=args.context,
                    export_file=args.export if hasattr(args, 'export') else None
                )
                
            elif args.explore_command == 'structure':
                explorer.analyze_folder_structure(
                    max_depth=args.max_depth if hasattr(args, 'max_depth') else None,
                    min_files=args.min_files if hasattr(args, 'min_files') else 1
                )
                if hasattr(args, 'export') and args.export:
                    explorer.export_folder_structure(args.export)
                    
            elif args.explore_command == 'validate':
                explorer.validate_random_files(
                    n=args.sample_size if hasattr(args, 'sample_size') else 5
                )
                
        except Exception as e:
            logging.exception(f"[red]Error during exploration: {str(e)}[/red]")
            logging.exception(traceback.format_exc())
            sys.exit(1)
        
    def handle_config(self, args):
        """Handle config command"""
        try:
            if args.config_command == 'save':
                save_config(args.output)
                logging.debug(f"[green]Configuration saved to: {args.output}[/green]")
            
            elif args.config_command == 'show':
                if args.file:
                    # Load and display specified config
                    config = load_config(args.file)
                else:
                    # Show default config
                    config = get_default_config()
                
                # Pretty print config
                logging.debug_json(data=config)
                
        except Exception as e:
            logging.exception(f"[red]Error handling configuration: {str(e)}[/red]")
            sys.exit(1)
        
    def run(self):
        """Main entry point"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Setup logging
        log_file = setup_logging(appconfig=self.config)
        
        try:
            logging.info(f"Executing command: {args.command}")
            logging.debug(f"Command arguments: {args}")
            
            # Dispatch to appropriate handler
            if args.command == 'preprocess':
                logging.info("Starting preprocessing operation")
                self.handle_preprocess(args)
                
            elif args.command == 'explore':
                logging.info("Starting exploration operation")
                self.handle_explore(args)
                
            elif args.command == 'config':
                logging.info("Starting configuration operation")
                self.handle_config(args)
            
            logging.info("Operation completed successfully")
            
        except KeyboardInterrupt:
            logging.warning("Operation interrupted by user")
            sys.exit(1)
            
        except Exception as e:
            logging.exception(f"Error during operation: {str(e)}", exc_info=True)
            logging.exception(traceback.format_exc())
            
            sys.exit(1)
            
        finally:
            logging.info(f"Log file saved to: {log_file}")

    def get_usage_examples(self):
        """Return usage examples for help text"""
        return """
Examples:
    # Preprocess code:
    python app.py preprocess --input-dir ./code --output-dir ./processed_data
    
    # Save configuration:
    python app.py preprocess --save-config config/my_config.json
    
    # Preprocess with configuration:
    python app.py preprocess --input-dir ./code --output-dir ./processed_data --config my_config.json
    
    # Preprocess with additional options:
    python app.py preprocess --input-dir ./code --output-dir ./processed_data \\
        --max-tokens 4096 --workers 4 --additional-extensions .gradle .conf
    
    # Explore processed data:
    python app.py explore processed_data/processed_data_latest.parquet --search "main("
    
    # Manage configuration:
    python app.py config save my_config.json
    python app.py config show --file my_config.json
    """

def main():
    cli = CodeProcessingCLI()
    cli.run()

if __name__ == '__main__':
    main()