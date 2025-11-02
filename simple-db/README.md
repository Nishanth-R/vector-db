## Introduction 

This is an implementation of a vector database with advanced features. 
It stores vector embeddings for efficient similarity search and supports B-tree indexing for fast primary key lookups.

## Features

- **B-tree Indexing**: Fast query by ID using B-tree data structure
- **Vector Embeddings**: Uses Google's embedding models via sentence-transformers (fallback to all-MiniLM-L6-v2)
- **UMAP-based Similarity Search**: Uses UMAP for dimensionality reduction and cosine similarity for finding closest vectors
- **LIFO File Management**: Tracks insert order for Last-In-First-Out operations
- **Primary Key Queries**: Fast lookup by ID using indexed B-tree structure

## Installation

This project uses `uv` as the package manager. Choose one of the installation methods below:

### Option 1: Install as CLI (Recommended for regular use)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Navigate to the project directory
cd simple-db

# Install the package in editable mode (allows CLI usage + development)
uv pip install -e .

# Pre-install required models and dependencies (recommended for first-time setup)
vdb install

# The `vdb` command should now be available
vdb --help
```

### Option 2: Use with virtual environment

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project directory
cd simple-db

# Create and sync virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .

# Pre-install required models and dependencies
vdb install

# Use the CLI
vdb --help
```

### Option 3: Install globally

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project directory
cd simple-db

# Install globally (requires appropriate permissions)
uv pip install .

# The `vdb` command should now be available globally
vdb --help
```

**Note:** After installation, the `vdb` command will be available in your terminal. If it's not found, make sure the Python scripts directory is in your PATH, or use the virtual environment method (Option 2).

## Data Structure

- Tables are stored as JSON files
- Each row contains an `encoded_data` field with vector embeddings
- B-tree indices are stored as pickle files for fast primary key lookups
- LIFO stack tracks insert order in a separate pickle file

## Encoding

The database uses **sentence-transformers** for text encoding:
- Primary model: Attempts to load `google/gemma-2-2b-it` 
- Fallback model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Embeddings are stored as lists in the `encoded_data` column

## Application 

The application provides a command-line interface for interacting with the vector database.

## Usage 

Once installed, you can use the `vdb` command from anywhere in your terminal:

```bash
# Get help
vdb --help

# Get help for a specific command
vdb insert-url --help
```

### `insert-url <url>`
Insert text contents from a URL into the database.

**Usage:**
```bash
vdb insert-url <url>
```

**Example:**
```bash
vdb insert-url https://example.com/article
vdb insert-url https://en.wikipedia.org/wiki/Machine_learning
```

### `insert-file <filename>`
Insert text contents from a file (.txt, .pkl, .pickle) into the database.

**Usage:**
```bash
vdb insert-file <filename>
```

**Example:**
```bash
vdb insert-file document.txt
vdb insert-file data.pkl
```

### `closest <text> [--num-results=1]`
Find the closest document(s) to the entered text using UMAP + cosine similarity.

**Usage:**
```bash
vdb closest "<text>" [--num-results <number>]
```

**Example:**
```bash
vdb closest "machine learning algorithms"
vdb closest "neural networks" --num-results 5
```

### `search <text> [--num-results=5]`
Search for articles similar to the given text and return top N matches.

**Usage:**
```bash
vdb search "<text>" [--num-results <number>]
```

**Example:**
```bash
vdb search "deep learning"
vdb search "artificial intelligence" --num-results 10
```

### `query-id <table_name> <id_value>`
Query a row by its primary key (ID) using B-tree index.

**Usage:**
```bash
vdb query-id <table_name> <id_value>
```

**Example:**
```bash
vdb query-id articles 1
vdb query-id articles 42
```

### `install`
Pre-install and download required models and dependencies. Run this after installing the package to download:
- Sentence transformer models (all-MiniLM-L6-v2 and optionally google/gemma-2-2b-it)
- NLTK stopwords corpus
- Verify other dependencies

**Usage:**
```bash
vdb install
```

**Example:**
```bash
# After installing the package
uv pip install -e .
vdb install  # Download models and dependencies
```

## Quick Start Example

```bash
# 1. Install the CLI (see Installation section above)
cd simple-db
uv pip install -e .

# 2. Pre-install models and dependencies (recommended for first-time setup)
vdb install

# 3. Insert some content
vdb insert-url https://en.wikipedia.org/wiki/Vector_database

# 4. Search for similar content
vdb search "database storage" --num-results 3

# 5. Query by ID (assuming an article was inserted with ID 1)
vdb query-id articles 1
```

## Troubleshooting

**Command not found:**
- Make sure you've activated the virtual environment if using Option 2
- Check that the installation completed successfully: `uv pip list | grep svdb`
- Ensure Python scripts directory is in your PATH

**Import errors:**
- Run `uv sync` to ensure all dependencies are installed
- Activate the virtual environment if using one
- Check Python version: `python --version` (requires Python 3.10+) 