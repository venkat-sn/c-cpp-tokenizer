# Code Processing and Analysis Tool

A powerful command-line tool for preprocessing, analyzing, and exploring source code across multiple programming languages. This tool helps developers process large codebases, analyze their structure, and explore code content efficiently.

## Features

- **Code Preprocessing**: Process source code files with customizable tokenization
- **Data Exploration**: Analyze processed code with search and structure visualization
- **Configuration Management**: Flexible configuration system with JSON support
- **Multi-threaded Processing**: Parallel processing support for better performance
- **Extensible Design**: Easy to add support for additional file types and languages

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-processing-tool.git
cd code-processing-tool

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

1. Preprocess your code:
```bash
python app.py preprocess --input-dir ./code --output-dir ./processed_data
```

2. Explore processed data:
```bash
python app.py explore processed_data/processed_data_latest.parquet search "main("
```

3. Manage configuration:
```bash
python app.py config save my_config.json
```

### Command Reference

#### Preprocess Command

Process source code files with various options:

```bash
python app.py preprocess [options]

Options:
  --input-dir DIR           Input directory containing code
  --output-dir DIR         Output directory for processed data
  --max-tokens N           Maximum tokens per example
  --min-tokens N           Minimum tokens per example
  --workers N              Number of worker processes
  --tokenizer NAME         Tokenizer to use (default: cl100k_base)
  --additional-extensions  Additional file extensions to process
  --exclude-dirs          Additional directories to exclude
  --config FILE           Path to JSON configuration file
  --explore-after         Launch explorer after preprocessing
```

#### Explore Command

Analyze processed code data:

```bash
python app.py explore FILE [command] [options]

Commands:
  search                   Search code content
    --context N           Number of context lines (default: 2)
    --export FILE         Export results to file

  structure               Analyze folder structure
    --max-depth N        Maximum depth to display
    --min-files N        Minimum files in folder (default: 1)
    --export FILE        Export structure analysis

  validate                Validate processed files
    --sample-size N      Number of random files to validate
```

#### Config Command

Manage tool configuration:

```bash
python app.py config [command] [options]

Commands:
  save OUTPUT             Save default configuration
  show                    Show current configuration
    --file FILE          Configuration file to show
```

## Configuration

The tool uses a JSON configuration file system. You can:
1. Generate a default configuration:
```bash
python app.py config save my_config.json
```

2. Modify the configuration file as needed
3. Use the modified configuration:
```bash
python app.py preprocess --config my_config.json --input-dir ./code --output-dir ./processed
```

## Requirements

- Python 3.7+
- Required packages:
  - rich
  - pandas
  - pyarrow
  - tiktoken (for tokenization)

## Development

### Project Structure

```
code-processing-tool/
├── app.py              # Main entry point
├── src/
│   ├── preprocessor.py # Code preprocessing logic
│   └── utils/
│       ├── data_explorer.py    # Data exploration utilities
│       └── logging_utils.py    # Logging configuration
└── config/
    └── default_config.py       # Default configuration
```

### Adding New Features

1. **New File Types**: Add extensions to the configuration file
2. **Custom Processing**: Extend the `CodePreprocessor` class
3. **New Analysis Tools**: Add commands to the explore subparser

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the terms of the LICENSE file included in the repository. See LICENSE for more information.

## Support

For issues and feature requests, please use the GitHub issue tracker.