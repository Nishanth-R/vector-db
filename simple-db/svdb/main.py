import typer
import asyncio
from typing import Optional
# Lazy import AppFlow - only needed when commands are actually run
# from .app import AppFlow
import pickle
# Lazy import Rich - only needed when commands run
# from rich.console import Console
# from rich.table import Table
import os

app = typer.Typer()

# Lazy console creation
_console = None
_table_class = None

def get_console():
    """Lazily get or create console."""
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console()
    return _console

def get_table():
    """Lazily get Table class."""
    global _table_class
    if _table_class is None:
        from rich.table import Table
        _table_class = Table
    return _table_class

async def get_app_flow():
    """Get or create an AppFlow instance."""
    if not hasattr(get_app_flow, 'instance'):
        from .app import AppFlow
        get_app_flow.instance = AppFlow()
    return get_app_flow.instance

@app.command()
def insert_url(url: str):
    """
    Insert text contents of the entered URL into the database.
    
    Args:
        url (str): URL which should be inserted into the database
    """
    async def _insert_url():
        try:
            from .app import AppFlow
            app_flow = await get_app_flow()
            async with app_flow:
                success = await app_flow.write_to_db_from_url(url)
                if success:
                    timing_stats = app_flow.get_timing_stats()
                    console = get_console()
                    console.print(f"[green]Successfully inserted content from {url}[/green]")
                    console.print("\n[bold]Timing Statistics:[/bold]")
                    console.print(f"Fetch time: {timing_stats['fetch_time']:.2f} μs")
                    console.print(f"Insert time: {timing_stats['insert_time']:.2f} μs")
                    console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
                else:
                    get_console().print(f"[red]Failed to insert content from {url}[/red]")
        except Exception as e:
            get_console().print(f"[red]Error inserting URL: {str(e)}[/red]")

    asyncio.run(_insert_url())

@app.command()
def insert_file(filename: str):
    """
    Insert text contents of the entered file into the database.
    
    Args:
        filename (str): Filename to write to database (supports .txt, .pkl, .pickle)
    """
    async def _insert_file():
        try:
            app_flow = await get_app_flow()
            async with app_flow:
                # Read file content
                if filename.endswith('.txt'):
                    with open(filename, 'r', encoding='utf-8') as file:
                        content = file.read()
                elif filename.endswith(('.pkl', '.pickle')):
                    with open(filename, 'rb') as file:
                        content = pickle.load(file)
                else:
                    get_console().print("[red]Unsupported file format. Please use .txt, .pkl, or .pickle files.[/red]")
                    return

                # Create a table name using only the file name without extension
                file_name = os.path.basename(filename)
                table_name = f"file_{os.path.splitext(file_name)[0]}"
                
                # Insert content into database
                success = await app_flow.write_to_db_from_url(
                    f"file://{filename}",
                    table_name,
                    content=content
                )
                
                if success:
                    timing_stats = app_flow.get_timing_stats()
                    console = get_console()
                    console.print(f"[green]Successfully inserted content from {filename}[/green]")
                    console.print("\n[bold]Timing Statistics:[/bold]")
                    console.print(f"Insert time: {timing_stats['insert_time']:.2f} μs")
                    console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
                else:
                    get_console().print(f"[red]Failed to insert content from {filename}[/red]")
        except Exception as e:
            get_console().print(f"[red]Error inserting file: {str(e)}[/red]")

    asyncio.run(_insert_file())

@app.command()
def closest(text: str, num_results: int = 1):
    """
    Find the closest documents to the given text.
    
    Args:
        text (str): Text to compare the documents against
        num_results (int): Number of results to return (default: 1)
    """
    async def _closest():
        try:
            app_flow = await get_app_flow()
            async with app_flow:
                articles = await app_flow.find_closest_articles_by_text(text, num_results=num_results)
                
                if not articles:
                    get_console().print("[yellow]No matching articles found.[/yellow]")
                    return
                
                # Handle single result vs list
                if not isinstance(articles, list):
                    articles = [articles]

                # Create a rich table for better output formatting
                console = get_console()
                Table = get_table()
                table = Table(title=f"Top {num_results} Matching Articles")
                table.add_column("Source", style="magenta")
                table.add_column("Title", style="cyan")
                table.add_column("Content", style="green")
                table.add_column("URL", style="blue")

                for article in articles:
                    table.add_row(
                        article.get('table_name', 'N/A'),
                        article.get('title', 'N/A'),
                        article.get('content', 'N/A')[:200] + '...' if article.get('content') else 'N/A',
                        article.get('url', 'N/A')
                    )

                console.print(table)
                
                # Display timing statistics
                timing_stats = app_flow.get_timing_stats()
                console.print("\n[bold]Timing Statistics:[/bold]")
                console.print(f"Encoding time: {timing_stats.get('encode_time', 0):.2f} μs")
                console.print(f"Loading time: {timing_stats.get('load_time', 0):.2f} μs")
                console.print(f"Similarity calculation time: {timing_stats.get('similarity_time', 0):.2f} μs")
                console.print(f"Total time: {timing_stats.get('total_time', 0):.2f} μs")
        except Exception as e:
            get_console().print(f"[red]Error finding closest articles: {str(e)}[/red]")

    asyncio.run(_closest())

@app.command()
def search(text: str, num_results: int = 5):
    """
    Search for articles similar to the given text and return top N matches.
    
    Args:
        text (str): Text to search for
        num_results (int): Number of results to return (default: 5)
    """
    async def _search():
        try:
            app_flow = await get_app_flow()
            async with app_flow:
                articles = await app_flow.find_closest_articles_by_text(text, num_results=num_results)
                
                if not articles:
                    get_console().print("[yellow]No matching articles found.[/yellow]")
                    return
                
                # Handle single result vs list
                if not isinstance(articles, list):
                    articles = [articles]

                # Create a rich table for better output formatting
                console = get_console()
                Table = get_table()
                table = Table(title=f"Search Results for: {text}")
                table.add_column("Rank", style="bold")
                table.add_column("Source", style="magenta")
                table.add_column("Title", style="cyan")
                table.add_column("Content Preview", style="green")
                table.add_column("URL", style="blue")

                for idx, article in enumerate(articles, 1):
                    table.add_row(
                        str(idx),
                        article.get('table_name', 'N/A'),
                        article.get('title', 'N/A'),
                        article.get('content', 'N/A')[:150] + '...' if article.get('content') else 'N/A',
                        article.get('url', 'N/A')
                    )

                console.print(table)
                
                # Display timing statistics
                timing_stats = app_flow.get_timing_stats()
                console.print("\n[bold]Timing Statistics:[/bold]")
                console.print(f"Encoding time: {timing_stats.get('encode_time', 0):.2f} μs")
                console.print(f"Loading time: {timing_stats.get('load_time', 0):.2f} μs")
                console.print(f"Similarity calculation time: {timing_stats.get('similarity_time', 0):.2f} μs")
                console.print(f"Total time: {timing_stats.get('total_time', 0):.2f} μs")
        except Exception as e:
            get_console().print(f"[red]Error searching articles: {str(e)}[/red]")

    asyncio.run(_search())

@app.command()
def query_id(table_name: str, id_value: int):
    """
    Query a row by its primary key (ID) using B-tree index.
    
    Args:
        table_name (str): Name of the table to query
        id_value (int): The ID value to search for
    """
    async def _query_id():
        try:
            app_flow = await get_app_flow()
            async with app_flow:
                row = app_flow.database.query_by_id(table_name, id_value)
                
                if not row:
                    get_console().print(f"[yellow]No row found with ID {id_value} in table '{table_name}'[/yellow]")
                    return
                
                # Create a rich table for output
                console = get_console()
                Table = get_table()
                table = Table(title=f"Row with ID {id_value} from '{table_name}'")
                table.add_column("Column", style="bold")
                table.add_column("Value", style="green")
                
                for key, value in row.items():
                    if key == 'encoded_data':
                        # Truncate embedding for display
                        if isinstance(value, list):
                            value_str = f"[{len(value)} dims] {str(value[:5])}..."
                        else:
                            value_str = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                    else:
                        value_str = str(value)[:200] + '...' if len(str(value)) > 200 else str(value)
                    table.add_row(key, value_str)
                
                console.print(table)
        except Exception as e:
            get_console().print(f"[red]Error querying by ID: {str(e)}[/red]")

    asyncio.run(_query_id())

@app.command()
def install():
    """
    Pre-install and download required models and dependencies.
    This will download:
    - Sentence transformer models (google/embeddinggemma-300m with fallback to all-MiniLM-L6-v2)
    - NLTK stopwords corpus
    """
    console = get_console()
    console.print("[bold cyan]Installing required models and dependencies...[/bold cyan]\n")
    
    # Install NLTK stopwords
    console.print("[yellow]Downloading NLTK stopwords...[/yellow]")
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            stopwords.words('english')
            console.print("[green]✓ NLTK stopwords already downloaded[/green]")
        except LookupError:
            console.print("[yellow]  Downloading stopwords corpus...[/yellow]")
            nltk.download('stopwords', quiet=True)
            stopwords.words('english')  # Verify it works
            console.print("[green]✓ NLTK stopwords downloaded successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to download NLTK stopwords: {e}[/red]")
    
    console.print()
    
    # Install sentence-transformers models
    console.print("[yellow]Downloading sentence-transformer models...[/yellow]")
    
    # Primary model: all-MiniLM-L6-v2
    console.print("[yellow]  Loading all-MiniLM-L6-v2 model...[/yellow]")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Trigger a test encode to ensure model is fully loaded
        _ = model.encode(["test"], convert_to_numpy=True)
        console.print("[green]✓ all-MiniLM-L6-v2 model loaded successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load all-MiniLM-L6-v2 model: {e}[/red]")
    
    # Optional model: google/embeddinggemma-300m
    console.print("[yellow]  Attempting to load google/embeddinggemma-300m model (optional)...[/yellow]")
    try:
        from sentence_transformers import SentenceTransformer
        model_gemma = SentenceTransformer('google/embeddinggemma-300m', trust_remote_code=True)
        # Trigger a test encode
        _ = model_gemma.encode(["test"], convert_to_numpy=True)
        console.print("[green]✓ google/embeddinggemma-300m model loaded successfully[/green]")
    except Exception as e:
        console.print(f"[yellow]  ⚠ google/embeddinggemma-300m model not available (using fallback): {str(e)[:100]}[/yellow]")
    
    console.print()
    
    # Verify other dependencies
    console.print("[yellow]Verifying other dependencies...[/yellow]")
    dependencies = {
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'umap-learn': 'umap',
        'hdbscan': 'hdbscan',
    }
    
    all_ok = True
    for package_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            console.print(f"[green]✓ {package_name} is installed[/green]")
        except ImportError:
            console.print(f"[red]✗ {package_name} is not installed[/red]")
            all_ok = False
    
    console.print()
    if all_ok:
        console.print("[bold green]✓ Installation complete! All dependencies are ready.[/bold green]")
    else:
        console.print("[bold yellow]⚠ Installation complete, but some optional dependencies are missing.[/bold yellow]")
        console.print("[yellow]  Run: uv pip install -e .[/yellow]")

def main():
    """Main entry point for the application."""
    app()

if __name__ == '__main__':
    main()
