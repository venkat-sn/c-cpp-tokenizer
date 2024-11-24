import argparse
import sys
import json
import traceback
from pathlib import Path
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from typing import Dict, Any
import logging
from datetime import datetime
from collections import defaultdict
import traceback
from src.preprocessor import CodePreprocessor
from src.utils.data_explorer import DataExplorer
from src.utils.logging_utils import setup_logging
from rich.table import Table
import pandas as pd
from src.utils.codebase_analyzer import CodebaseAnalyzer
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

        # Analyze command
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='Analyze code structure and relationships',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_analyze_arguments(analyze_parser)
        
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
            '--include',
            nargs='+',
            required=True,
            help='Files/directories/patterns to include (e.g., src/**/*.py lib/utils/helpers.py tests/unit/)'
        )
        
        processing_args.add_argument(
            '--exclude',
            nargs='+',
            help='Files/directories/patterns to exclude (e.g., **/__pycache__/** tests/fixtures/)'
        )
        
        processing_args.add_argument(
            '--output-dir',
            required=True,
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
        
        # Keep exclude-dirs for backward compatibility
        processing_args.add_argument(
            '--exclude-dirs',
            type=str,
            nargs='+',
            help='Additional directories to exclude (deprecated, use --exclude instead)'
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

        processing_args.add_argument(
            '--no-default-includes',
            action='store_true',
            help='Do not use default include patterns from config'
        )
        
        processing_args.add_argument(
            '--no-default-excludes',
            action='store_true',
            help='Do not use default exclude patterns from config'
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

    def _add_analyze_arguments(self, parser):
        """Add analysis-related arguments"""
        analyze_parser = parser.add_argument_group('Analysis Options')
        analyze_parser.add_argument('parquet_file', help='Path to parquet file containing processed code')
        analyze_parser.add_argument('source_dir', help='Path to original source code directory')
        
        subparsers = parser.add_subparsers(
            dest='analyze_command',
            title='Analysis Commands',
            description='Available analysis commands'
        )
        
        # Create training data command
        training_parser = subparsers.add_parser(
            'create-training',
            help='Create complete training dataset',
            description='Generate comprehensive training chunks from codebase'
        )
        
        # Chunking strategy
        training_parser.add_argument(
            '--strategy',
            choices=['function', 'class', 'file', 'smart'],
            default='smart',
            help='Chunking strategy (default: smart)'
        )
        
        # Output format and location
        training_parser.add_argument(
            '--output-dir',
            required=True,
            help='Directory to store generated training chunks'
        )
        
        training_parser.add_argument(
            '--format',
            choices=['jsonl', 'parquet', 'csv'],
            default='jsonl',
            help='Output format (default: jsonl)'
        )
        
        # Content controls
        training_parser.add_argument(
            '--min-chunk-size',
            type=int,
            default=50,
            help='Minimum tokens per chunk (default: 50)'
        )
        
        training_parser.add_argument(
            '--max-chunk-size',
            type=int,
            default=2000,
            help='Maximum tokens per chunk (default: 2000)'
        )
        
        # Context controls
        training_parser.add_argument(
            '--include-context',
            action='store_true',
            help='Include rich context (imports, dependencies, etc.)'
        )
        
        training_parser.add_argument(
            '--context-depth',
            type=int,
            default=1,
            help='Depth of context relationships to include (default: 1)'
        )

    def _handle_training_data_creation(self, args, analyzer):
        """Handle creation of complete training dataset using only parquet data"""
        try:
            output_dir = Path(args.output_dir)
            if output_dir.exists():
                logging.info(f"[yellow]Cleaning existing training data in {output_dir}[/yellow]")
                for file in output_dir.glob("*"):
                    file.unlink()
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                
            logging.info("[yellow]Creating fresh training dataset...[/yellow]")
            
            # Create progress display
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TextColumn("[yellow]{task.fields[current]}"),
                console=Console()
            )
            
            # Read tags generated from parquet content
            file_symbols = defaultdict(list)
            logging.info("Reading tags file...")
            
            with open(analyzer.tags_file, 'r') as f:
                for line in f:
                    if line.startswith('!'):  # Skip headers
                        continue
                        
                    try:
                        # Parse tag line
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                            
                        symbol = parts[0]
                        file_path = parts[1]
                        
                        # Convert temp file path back to original path
                        rel_path = Path(file_path).relative_to(analyzer.temp_dir)
                        
                        # Extract kind from extended fields
                        kind = None
                        line_number = None
                        for part in parts[3:]:
                            if part.startswith('kind:'):
                                kind = part.split(':')[1]
                            elif part.startswith('line:'):
                                line_number = int(part.split(':')[1])
                        
                        if kind in ('function', 'method', 'class') and line_number:
                            file_symbols[str(rel_path)].append({
                                'symbol': symbol,
                                'kind': kind,
                                'line': line_number
                            })
                            
                    except Exception as e:
                        logging.warning(f"Error parsing tag line: {str(e)}")
                        continue
            
            logging.info(f"Found {sum(len(symbols) for symbols in file_symbols.values())} symbols in {len(file_symbols)} files")
            
            def save_chunk(chunk, stats):
                """Save single chunk immediately to disk"""
                chunk_file = output_dir / f"chunk_{stats['batch_num']:06d}.jsonl"
                with open(chunk_file, 'w') as f:
                    json.dump(chunk, f)
                    f.write('\n')
                stats['processed_chunks'] += 1
                stats['batch_num'] += 1
            
            # Create file content lookup from parquet data
            file_content_map = {
                row['file_path']: row 
                for _, row in analyzer.df.iterrows()
            }
            
            # Process files with progress bar
            with progress:
                main_task = progress.add_task(
                    "[green]Processing files...", 
                    total=len(file_symbols),
                    current="Starting..."
                )
                
                stats = {
                    'processed_files': 0,
                    'processed_chunks': 0,
                    'batch_num': 0
                }
                
                # Process each file from symbols
                for file_path, symbols in file_symbols.items():
                    progress.update(main_task, current=f"File: {file_path}")
                    
                    try:
                        # Only process if file exists in parquet data
                        if file_path not in file_content_map:
                            continue
                            
                        # Get content from parquet data
                        row = file_content_map[file_path]
                        lines = row['content'].splitlines(keepends=True)
                        
                        # Process each symbol in file
                        for symbol_info in symbols:
                            symbol = symbol_info['symbol']
                            kind = symbol_info['kind']
                            line_num = symbol_info['line'] - 1  # 0-based index
                            
                            # Extract content
                            content = []
                            brace_count = 0
                            started = False
                            
                            # Look for content
                            for i in range(line_num, len(lines)):
                                line = lines[i]
                                if not started:
                                    if '{' in line:
                                        started = True
                                        content.append(line)
                                        brace_count = line.count('{') - line.count('}')
                                    else:
                                        content.append(line)
                                else:
                                    content.append(line)
                                    brace_count += line.count('{') - line.count('}')
                                    if brace_count == 0:
                                        break
                            
                            if content:
                                content_str = ''.join(content)
                                tokens = analyzer.tokenizer.encode(content_str)
                                
                                if len(tokens) >= args.min_chunk_size and len(tokens) <= args.max_chunk_size:
                                    chunk = {
                                        'content': content_str,
                                        'n_tokens': len(tokens),
                                        'type': kind,
                                        'metadata': {
                                            'symbol': symbol,
                                            'file': file_path,
                                            'line': line_num + 1,
                                            'language': row['language'],
                                            'size_bytes': row['size_bytes']
                                        }
                                    }
                                    save_chunk(chunk, stats)
                        
                        stats['processed_files'] += 1
                        progress.update(main_task, advance=1)
                        
                    except Exception as e:
                        logging.warning(f"Error processing file {file_path}: {str(e)}")
                        continue
                
                # Process any files that don't have symbols but meet size requirements
                remaining_task = progress.add_task(
                    "[blue]Processing remaining files...", 
                    total=len(analyzer.df),
                    current="Starting..."
                )
                
                for _, row in analyzer.df.iterrows():
                    file_path = row['file_path']
                    progress.update(remaining_task, current=f"File: {file_path}")
                    
                    # Only process if not already processed through symbols
                    if file_path not in file_symbols:
                        content = row['content']
                        tokens = analyzer.tokenizer.encode(content)
                        
                        if len(tokens) >= args.min_chunk_size and len(tokens) <= args.max_chunk_size:
                            chunk = {
                                'content': content,
                                'type': 'file',
                                'n_tokens': len(tokens),
                                'metadata': {
                                    'file_path': file_path,
                                    'language': row['language'],
                                    'size_bytes': row['size_bytes']
                                }
                            }
                            save_chunk(chunk, stats)
                    
                    progress.update(remaining_task, advance=1)

            # Create summary
            summary = {
                'creation_time': datetime.now().isoformat(),
                'stats': {
                    'total_chunks': stats['processed_chunks'],
                    'total_files': stats['processed_files']
                },
                'configuration': {
                    'strategy': args.strategy,
                    'min_chunk_size': args.min_chunk_size,
                    'max_chunk_size': args.max_chunk_size,
                    'include_context': args.include_context,
                    'context_depth': args.context_depth
                }
            }
            
            with open(output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logging.info(f"\n[bold green]Training Data Creation Complete![/bold green]")
            logging.info(f"Processed {stats['processed_files']} files")
            logging.info(f"Created {stats['processed_chunks']} chunks")
            logging.info(f"\nOutput directory: {output_dir}")
            
        except Exception as e:
            logging.exception("Error creating training data")
            raise

    def handle_analyze(self, args):
        """Handle analyze command"""
        try:
            logging.info("[yellow]Initializing code analysis...[/yellow]")
            
            # Initialize DataExplorer first (but don't pass to training data creation)
            explorer = DataExplorer(args.parquet_file)
            
            # Initialize analyzer
            analyzer = CodebaseAnalyzer(args.parquet_file)  # Remove source_dir
            
            # Add tokenizer directly
            analyzer.tokenizer = explorer.tokenizer
            
            # Initialize cscope/ctags databases
            logging.info("[yellow]Building code cross-reference database...[/yellow]")
            analyzer.initialize()
            
            # Dispatch to appropriate handler based on analyze_command
            if args.analyze_command == 'create-training':
                # Don't pass explorer here
                self._handle_training_data_creation(args, analyzer)
            elif args.analyze_command == 'symbol':
                self._handle_symbol_analysis(args, analyzer)
            elif args.analyze_command == 'batch':
                self._handle_batch_analysis(args, analyzer)
            elif args.analyze_command == 'deps':
                self._handle_dependency_analysis(args, analyzer)
                
        except Exception as e:
            logging.exception(f"[red]Error during analysis: {str(e)}[/red]")
            logging.exception(traceback.format_exc())
            sys.exit(1)


    def _handle_symbol_analysis(self, args, analyzer, explorer):
        """Handle single symbol analysis"""
        try:
            # Get symbol information
            symbol_info = analyzer.query_symbol(args.symbol)
            
            # Create rich table for symbol info
            table = Table(title=f"Analysis of symbol: {args.symbol}")
            table.add_column("Category", style="cyan")
            table.add_column("Information")
            
            # Basic symbol info
            if symbol_info['tag_info']:
                tag = symbol_info['tag_info']
                table.add_row("Type", tag.kind)
                if tag.scope:
                    table.add_row("Scope", tag.scope)
                if tag.signature:
                    table.add_row("Signature", tag.signature)
                    
            # References
            table.add_row("Definitions", str(len(symbol_info['definitions'])))
            table.add_row("References", str(len(symbol_info['references'])))
            table.add_row("Callers", str(len(symbol_info['callers'])))
            table.add_row("Callees", str(len(symbol_info['callees'])))
            
            logging.info(table)
            
            # Show detailed references if requested
            if args.with_context:
                self._show_detailed_references(symbol_info)
                
            # Export if requested
            if args.export:
                df = pd.DataFrame({
                    'symbol': [args.symbol],
                    'info': [symbol_info]
                })
                self._export_data(df, args.export)
                
        except Exception as e:
            logging.exception(f"Error analyzing symbol {args.symbol}: {str(e)}")
            raise

    def _handle_batch_analysis(self, args, analyzer, explorer):
        """Handle batch analysis of symbols"""
        try:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info("[yellow]Starting batch analysis...[/yellow]")
            
            # Process symbols in batches
            batch_num = 0
            total_chunks = 0
            
            for chunks in analyzer.batch_process_symbols(args.batch_size):
                # Filter chunks based on minimum references
                filtered_chunks = [
                    chunk for chunk in chunks
                    if chunk['metadata']['num_callers'] + chunk['metadata']['num_callees'] >= args.min_refs
                ]
                
                if filtered_chunks:
                    # Save batch to parquet
                    batch_file = output_dir / f"training_chunks_batch_{batch_num}.parquet"
                    df = pd.DataFrame(filtered_chunks)
                    df.to_parquet(batch_file)
                    
                    total_chunks += len(filtered_chunks)
                    batch_num += 1
                    
                    logging.info(f"Processed batch {batch_num}: {len(filtered_chunks)} chunks")
                    
            logging.info(f"[green]Analysis complete. Generated {total_chunks} training chunks in {batch_num} batches[/green]")
            
        except Exception as e:
            logging.exception("Error during batch analysis")
            raise

    def _handle_dependency_analysis(self, args, analyzer, explorer):
        """Handle dependency analysis"""
        try:
            logging.info("[yellow]Analyzing dependencies...[/yellow]")
            
            # Get all symbols
            symbols = list(analyzer.tags_cache.keys())
            
            # Build dependency graph
            deps_graph = {}
            for symbol in track(symbols, description="Building dependency graph..."):
                info = analyzer.query_symbol(symbol)
                deps_graph[symbol] = {
                    'callers': set(c['context'].split()[0] for c in info['callers']),
                    'callees': set(c['context'].split()[0] for c in info['callees'])
                }
                
            # Create summary table
            table = Table(title="Dependency Analysis")
            table.add_column("Metric", style="cyan")
            table.add_column("Value")
            
            # Calculate metrics
            total_deps = sum(len(d['callers']) + len(d['callees']) for d in deps_graph.values())
            max_deps = max(len(d['callers']) + len(d['callees']) for d in deps_graph.values())
            avg_deps = total_deps / len(deps_graph) if deps_graph else 0
            
            table.add_row("Total Symbols", str(len(deps_graph)))
            table.add_row("Total Dependencies", str(total_deps))
            table.add_row("Average Dependencies", f"{avg_deps:.2f}")
            table.add_row("Max Dependencies", str(max_deps))
            
            logging.info(table)
            
            # Export if requested
            if args.export:
                df = pd.DataFrame([
                    {
                        'symbol': symbol,
                        'callers': list(data['callers']),
                        'callees': list(data['callees'])
                    }
                    for symbol, data in deps_graph.items()
                ])
                self._export_data(df, args.export)
                
        except Exception as e:
            logging.exception("Error during dependency analysis")
            raise

    def _show_detailed_references(self, symbol_info):
        """Show detailed reference information"""
        # Show definitions
        if symbol_info['definitions']:
            tree = Tree("[bold]Definitions[/bold]")
            for def_info in symbol_info['definitions']:
                tree.add(f"{def_info['file']}:{def_info['line']} - {def_info['context']}")
            logging.info(tree)
        
        # Show callers
        if symbol_info['callers']:
            tree = Tree("[bold]Callers[/bold]")
            for caller in symbol_info['callers']:
                tree.add(f"{caller['file']}:{caller['line']} - {caller['context']}")
            logging.info(tree)
        
        # Show callees
        if symbol_info['callees']:
            tree = Tree("[bold]Callees[/bold]")
            for callee in symbol_info['callees']:
                tree.add(f"{callee['file']}:{callee['line']} - {callee['context']}")
            logging.info(tree)

    def _export_data(self, df, output_path):
        """Export data to specified format"""
        path = Path(output_path)
        if path.suffix == '.csv':
            df.to_csv(path, index=False)
        elif path.suffix == '.json':
            df.to_json(path, orient='records', indent=2)
        else:
            df.to_parquet(path)
        logging.info(f"[green]Data exported to: {output_path}[/green]")
        
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
            if not args.include:
                self.parser.error('--include patterns are required for processing')
            if not args.output_dir:
                self.parser.error('--output-dir is required for processing')

    def handle_preprocess(self, args):
        """Handle preprocess command"""
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

            # Initialize preprocessor with config
            processor = CodePreprocessor(
                include_patterns=args.include,
                output_dir=args.output_dir,
                exclude_patterns=args.exclude,
                additional_extensions=additional_extensions,
                exclude_dirs=set(args.exclude_dirs) if args.exclude_dirs else None,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                tokenizer=args.tokenizer,
                n_workers=args.workers,
                appconfig=self.config,
                use_default_includes=not args.no_default_includes,
                use_default_excludes=not args.no_default_excludes
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
                save_default_config(args.output)
                logging.debug(f"[green]Configuration saved to: {args.output}[/green]")
            
            elif args.config_command == 'show':
                if args.file:
                    # Load and display specified config
                    config = load_config_file(args.file)
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

            elif args.command == 'analyze':
                logging.info("Starting analysis operation")
                self.handle_analyze(args)
            
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