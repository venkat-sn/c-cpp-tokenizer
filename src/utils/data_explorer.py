import os
import pandas as pd
import re
import sys
import logging
import tiktoken
from collections import defaultdict
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree
from rich import box
from typing import Optional, Union, Any

class DataExplorer:
    def __init__(self, parquet_path: str):
        """
        Initialize the data explorer
        
        Args:
            parquet_path: Path to the parquet file
        """
        self.parquet_path = Path(parquet_path)
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        try:
            self.df = pd.read_parquet(self.parquet_path)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.debug(f"Successfully loaded {len(self.df)} records from {self.parquet_path}")
        except Exception as e:
            self.logger.error(f"Failed to load parquet file: {str(e)}")
            raise
    
    def _clean_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _log_rich(self, message: Union[str, Table, Tree, Any], level: str = "info", file_message: str = None, line_number: int = None):
        """
        Log messages with rich formatting for console and plain text for file
        """
        try:
            # Get frame information for line numbers
            frame = sys._getframe(1)  # Get the caller's frame
            line_no = line_number or frame.f_lineno

            # Create console for rich rendering
            console = Console()
            
            # Handle different types of messages
            if isinstance(message, str):
                console_message = message
                if file_message is None:
                    file_message = re.sub(r'\[.*?\]', '', message)
                    file_message = re.sub(r'📁', 'DIR', file_message)
            elif isinstance(message, (Tree, Table)):
                # For trees and tables, pass directly to console handler
                console_message = message
                if file_message is None:
                    # Capture plain text for file
                    plain_console = Console(color_system=None, width=100)
                    with plain_console.capture() as capture:
                        plain_console.print(message)
                    file_message = self._clean_ansi(capture.get())
            else:
                # For other Rich objects
                console_message = message
                if file_message is None:
                    plain_console = Console(color_system=None, width=100)
                    with plain_console.capture() as capture:
                        plain_console.print(message)
                    file_message = self._clean_ansi(capture.get())

            # Create record with proper line number
            record_dict = {
                'name': self.logger.name,
                'level': getattr(logging, level.upper()),
                'pathname': __file__,
                'lineno': line_no,
                'msg': None,
                'args': (),
                'exc_info': None,
                'func': frame.f_code.co_name,
            }

            # Log with different messages for console and file handlers
            for handler in self.logger.root.handlers:
                if isinstance(handler, RichHandler):
                    if isinstance(message, (Tree, Table)):
                        # For trees and tables, use the handler's console directly
                        handler.console.print(message)
                    else:
                        # For other messages, use the record
                        record = logging.LogRecord(**{**record_dict, 'msg': console_message})
                        handler.handle(record)
                else:
                    # Use plain text for file
                    record = logging.LogRecord(**{**record_dict, 'msg': file_message})
                    handler.handle(record)
                    
        except Exception as e:
            self.logger.error(f"Error in _log_rich: {str(e)}")
            # Fallback to basic logging
            self.logger.log(getattr(logging, level.upper()), str(message))

    def show_basic_stats(self):
        """Show basic statistics about the dataset"""
        try:
            # Create rich table
            table = Table(title="Dataset Statistics")
            table.add_column("Metric")
            table.add_column("Value")
            
            # Add rows
            table.add_row("Total Files", str(len(self.df)))
            table.add_row("Total Tokens", f"{self.df['n_tokens'].sum():,}")
            table.add_row("Average Tokens per File", f"{self.df['n_tokens'].mean():.1f}")
            table.add_row("Total Size (Bytes)", f"{self.df['size_bytes'].sum():,}")
            table.add_row("Compressed Parquet Size", f"{self.parquet_path.stat().st_size:,}")
            table.add_row(
                "Compression Ratio",
                f"{self.df['size_bytes'].sum() / self.parquet_path.stat().st_size:.1f}x"
            )
            
            # Log the table
            self._log_rich(table)
            
            # Language distribution
            self._log_rich("\n[bold]Language Distribution:[/bold]")
            for lang in sorted(self.df['language'].unique()):
                count = len(self.df[self.df['language'] == lang])
                self._log_rich(f"[cyan]Files ({lang}):[/cyan] {count}")
                
        except Exception as e:
            self.logger.error(f"Error showing basic stats: {str(e)}")
            raise
    
    def search_content(self, pattern: str, context_lines: int = 2, 
                      export_file: Optional[str] = None):
        """
        Search for pattern in code content
        
        Args:
            pattern: Regular expression pattern to search for
            context_lines: Number of lines of context to show
            export_file: Optional file to export results to
        """
        try:
            matches = []
            total_files = len(self.df)
            
            self._log_rich(f"\n[yellow]Searching for pattern: '{pattern}' across {total_files} files[/yellow]")
            
            for idx, row in track(self.df.iterrows(), 
                                total=total_files, 
                                description="Searching files..."):
                try:
                    if re.search(pattern, row['content'], re.IGNORECASE):
                        matches.append({
                            'file': row['file_path'],
                            'language': row['language'],
                            'content': row['content'],
                            'size': row['size_bytes'],
                            'n_tokens': row['n_tokens']
                        })
                except Exception as e:
                    self.logger.warning(f"Error searching file {row['file_path']}: {str(e)}")
                    continue
            
            if not matches:
                self._log_rich("[yellow]No matches found[/yellow]")
                return
            
            self._log_rich(f"\n[green]Found {len(matches)} matching files:[/green]")
            
            # Prepare export content if needed
            if export_file:
                export_content = []
            
            for match in matches:
                self._log_rich(f"\n[blue]File:[/blue] {match['file']}")
                self._log_rich(f"[blue]Language:[/blue] {match['language']}")
                self._log_rich(f"[blue]Size:[/blue] {match['size']:,} bytes")
                self._log_rich(f"[blue]Tokens:[/blue] {match['n_tokens']:,}")
                
                # Split content into lines
                lines = match['content'].split('\n')
                
                # Find matching lines
                match_sections = []
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        # Console output
                        self._log_rich("\n[cyan]Match context:[/cyan]")
                        for j in range(start, i):
                            self._log_rich(f"{j+1:4d} | {lines[j]}")
                        self._log_rich(f"[yellow]{i+1:4d} | {line}[/yellow]")
                        for j in range(i + 1, end):
                            self._log_rich(f"{j+1:4d} | {lines[j]}")
                        
                        # Collect for export
                        if export_file:
                            section = {
                                'file': match['file'],
                                'language': match['language'],
                                'line_number': i + 1,
                                'matched_line': line,
                                'context': '\n'.join(lines[start:end])
                            }
                            match_sections.append(section)
                
                if export_file and match_sections:
                    export_content.extend(match_sections)
            
            # Export if requested
            if export_file and export_content:
                export_path = Path(export_file)
                df_export = pd.DataFrame(export_content)
                
                if export_path.suffix == '.csv':
                    df_export.to_csv(export_path, index=False)
                elif export_path.suffix == '.json':
                    df_export.to_json(export_path, orient='records', indent=2)
                else:
                    df_export.to_parquet(export_path)
                    
                self._log_rich(f"\n[green]Search results exported to: {export_file}[/green]")
                
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise
    
    def show_file_content(self, pattern: str):
        """Show complete content of files matching pattern"""
        try:
            matches = self.df[self.df['file_path'].str.contains(pattern, case=False)]
            
            if len(matches) == 0:
                self._log_rich(f"[yellow]No files found matching: {pattern}[/yellow]")
                return
            
            for _, row in matches.iterrows():
                self._log_rich(f"\n[blue]File:[/blue] {row['file_path']}")
                self._log_rich(f"[blue]Language:[/blue] {row['language']}")
                self._log_rich(f"[blue]Size:[/blue] {row['size_bytes']:,} bytes")
                self._log_rich(f"[blue]Tokens:[/blue] {row['n_tokens']:,}")
                self._log_rich("\n[green]Content:[/green]")
                self._log_rich(row['content'])
                self._log_rich("\n" + "-"*80)
            
        except Exception as e:
            self.logger.error(f"Error showing file content: {str(e)}")
            raise
    
    def validate_random_files(self, n: int = 5):
        """Show n random files from the dataset"""
        try:
            sample = self.df.sample(n=min(n, len(self.df)))
            for _, row in sample.iterrows():
                self._log_rich(f"\n[blue]File:[/blue] {row['file_path']}")
                self._log_rich(f"[blue]First 200 characters:[/blue]")
                self._log_rich(row['content'][:200] + "...")
                
                # Validate token count
                tokens = self.tokenizer.encode(row['content'])
                stored_count = row['n_tokens']
                calculated_count = len(tokens)
                
                if stored_count == calculated_count:
                    self._log_rich(f"[green]Token count matches: {stored_count}[/green]")
                else:
                    self._log_rich(
                        f"[red]Token count mismatch: stored={stored_count}, "
                        f"calculated={calculated_count}[/red]"
                    )
                
                self._log_rich("-"*80)
                
        except Exception as e:
            self.logger.error(f"Error validating files: {str(e)}")
            raise

    def analyze_folder_structure(self, max_depth: int = None, min_files: int = 1):
        """
        Analyze and display the folder structure of processed files
        """
        try:
            log_messages = []  # Store messages for logging
            
            # Build folder structure - using full paths as keys
            folder_structure = defaultdict(list)
            folder_stats = defaultdict(lambda: {'files': 0, 'size': 0, 'tokens': 0})
            
            # First pass: gather all folder paths and files
            for _, row in self.df.iterrows():
                # Normalize path separators
                clean_path = str(row['file_path']).replace('\\', '/')
                
                # Split path into parts
                parts = [p for p in clean_path.split('/') if p]
                
                # Process each folder in the path
                current_path = ''
                for part in parts[:-1]:  # Process folders
                    if current_path:
                        current_path = f"{current_path}/{part}"
                    else:
                        current_path = part
                        
                # Add file to its parent folder
                if parts:
                    parent_path = '/'.join(parts[:-1]) if len(parts) > 1 else ''
                    folder_structure[parent_path].append({
                        'name': parts[-1],
                        'path': clean_path,
                        'language': row['language'],
                        'size': row['size_bytes'],
                        'tokens': row['n_tokens']
                    })
                    
                    # Update stats for all parent folders
                    current_path = ''
                    for part in parts[:-1]:
                        if current_path:
                            current_path = f"{current_path}/{part}"
                        else:
                            current_path = part
                        folder_stats[current_path]['files'] += 1
                        folder_stats[current_path]['size'] += row['size_bytes']
                        folder_stats[current_path]['tokens'] += row['n_tokens']
            
            # Create summary table
            summary_table = Table(title="Folder Statistics", box=box.ROUNDED)
            summary_table.add_column("Folder", style="cyan")
            summary_table.add_column("Files", justify="right")
            summary_table.add_column("Total Size", justify="right")
            summary_table.add_column("Total Tokens", justify="right")
            
            # Sort folders by number of files
            sorted_folders = sorted(
                folder_stats.items(),
                key=lambda x: x[1]['files'],
                reverse=True
            )
            
            # Add folder statistics to summary
            for folder, stats in sorted_folders:
                if stats['files'] >= min_files:
                    summary_table.add_row(
                        folder or ".",
                        f"{stats['files']:,}",
                        f"{self._format_size(stats['size'])}",
                        f"{stats['tokens']:,}"
                    )
            
            # Build tree
            tree = Tree("📁 Root", guide_style="blue")
            
            def get_immediate_subfolders(parent_path):
                """Get immediate subfolders of a given path"""
                prefix = f"{parent_path}/" if parent_path else ""
                subfolders = set()
                for path in folder_structure.keys():
                    if path.startswith(prefix) and path != parent_path:
                        parts = path[len(prefix):].split('/')
                        if parts[0]:  # Avoid empty strings
                            subfolders.add(parts[0])
                return sorted(subfolders)
            
            def add_to_tree(parent_path: str, parent_tree: Tree, current_depth: int = 0):
                if max_depth is not None and current_depth >= max_depth:
                    return
                
                # Add files in current folder
                items = folder_structure[parent_path]
                sorted_files = sorted(items, key=lambda x: x['name'])
                
                # Add files with their stats
                for file in sorted_files:
                    file_label = (
                        f"[green]{file['name']}[/green] "
                        f"({self._format_size(file['size'])}, "
                        f"{file['tokens']:,} tokens, "
                        f"lang: {file['language']})"
                    )
                    parent_tree.add(file_label)
                
                # Get immediate subfolders
                subfolders = get_immediate_subfolders(parent_path)
                
                # Process each subfolder
                for subfolder in subfolders:
                    full_subfolder_path = f"{parent_path}/{subfolder}" if parent_path else subfolder
                    stats = folder_stats[full_subfolder_path]
                    
                    # Skip empty folders if min_files is set
                    if stats['files'] < min_files:
                        continue
                    
                    folder_label = (
                        f"📁 [blue]{subfolder}[/blue] "
                        f"({stats['files']} files, "
                        f"{self._format_size(stats['size'])}, "
                        f"{stats['tokens']:,} tokens)"
                    )
                    subtree = parent_tree.add(folder_label)
                    add_to_tree(full_subfolder_path, subtree, current_depth + 1)
            
            # Build the complete tree
            add_to_tree('', tree)
            
            # Display results
            self._log_rich("\n[bold]Folder Structure Analysis[/bold]")
            self._log_rich(summary_table)
            self._log_rich("\n[bold]Folder Tree[/bold]")
            self._log_rich(tree)
            
            # Additional statistics
            total_size = sum(stat['size'] for stat in folder_stats.values())
            total_tokens = sum(stat['tokens'] for stat in folder_stats.values())
            
            stats_table = Table(title="Overall Statistics", box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", justify="right")
            
            stats_table.add_row("Total Folders", f"{len(folder_stats):,}")
            stats_table.add_row("Total Files", f"{len(self.df):,}")
            stats_table.add_row("Total Size", self._format_size(total_size))
            stats_table.add_row("Total Tokens", f"{total_tokens:,}")
            stats_table.add_row(
                "Average Files per Folder",
                f"{len(self.df) / max(1, len(folder_stats)):.1f}"
            )
            
            self._log_rich("\n")
            self._log_rich(stats_table)
            
            # Log all output
            log_messages.append("Folder Structure Analysis Complete")
            log_messages.append(f"Total Files: {len(self.df)}")
            log_messages.append(f"Total Folders: {len(folder_stats)}")
            log_messages.append(f"Total Size: {self._format_size(total_size)}")
            log_messages.append(f"Total Tokens: {total_tokens:,}")
            self._log_rich("\n".join(log_messages))
            
        except Exception as e:
            self.logger.error(f"Error analyzing folder structure: {str(e)}")
            raise
    
    def export_folder_structure(self, output_file: str):
        """
        Export folder structure analysis to a file
        
        Args:
            output_file: Path to output file (.json, .csv, or .parquet)
        """
        try:
            # Build folder structure with statistics
            folder_data = []
            
            for _, row in self.df.iterrows():
                path = Path(row['file_path'])
                current_path = ''
                
                # Add entry for each folder level
                for part in path.parts[:-1]:
                    current_path = os.path.join(current_path, part)
                    folder_data.append({
                        'folder_path': current_path,
                        'depth': len(Path(current_path).parts),
                        'file_path': row['file_path'],
                        'file_name': path.name,
                        'language': row['language'],
                        'size_bytes': row['size_bytes'],
                        'n_tokens': row['n_tokens']
                    })
            
            # Convert to DataFrame
            df_structure = pd.DataFrame(folder_data)
            
            # Save to file
            output_path = Path(output_file)
            if output_path.suffix == '.csv':
                df_structure.to_csv(output_path, index=False)
            elif output_path.suffix == '.json':
                df_structure.to_json(output_path, orient='records', indent=2)
            else:
                df_structure.to_parquet(output_path)
            
            self._log_rich(f"[green]Folder structure exported to: {output_file}[/green]")
            
        except Exception as e:
            self.logger.error(f"Error exporting folder structure: {str(e)}")
            raise
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"