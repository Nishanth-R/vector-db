import typer
import asyncio
from typing import Optional
from app import AppFlow
import pickle
from rich.console import Console
from rich.table import Table
import os

app = typer.Typer()
console = Console()

async def get_app_flow() -> AppFlow:
    """Get or create an AppFlow instance."""
    if not hasattr(get_app_flow, 'instance'):
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
            app_flow = await get_app_flow()
            async with app_flow:
                success = await app_flow.write_to_db_from_url(url)
                if success:
                    timing_stats = app_flow.get_timing_stats()
                    console.print(f"[green]Successfully inserted content from {url}[/green]")
                    console.print("\n[bold]Timing Statistics:[/bold]")
                    console.print(f"Fetch time: {timing_stats['fetch_time']:.2f} μs")
                    console.print(f"Insert time: {timing_stats['insert_time']:.2f} μs")
                    console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
                else:
                    console.print(f"[red]Failed to insert content from {url}[/red]")
        except Exception as e:
            console.print(f"[red]Error inserting URL: {str(e)}[/red]")

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
                    console.print("[red]Unsupported file format. Please use .txt, .pkl, or .pickle files.[/red]")
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
                    console.print(f"[green]Successfully inserted content from {filename}[/green]")
                    console.print("\n[bold]Timing Statistics:[/bold]")
                    console.print(f"Insert time: {timing_stats['insert_time']:.2f} μs")
                    console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
                else:
                    console.print(f"[red]Failed to insert content from {filename}[/red]")
        except Exception as e:
            console.print(f"[red]Error inserting file: {str(e)}[/red]")

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
                    console.print("[yellow]No matching articles found.[/yellow]")
                    return

                # Create a rich table for better output formatting
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
                console.print(f"Encoding time: {timing_stats['encode_time']:.2f} μs")
                console.print(f"Loading time: {timing_stats['load_time']:.2f} μs")
                console.print(f"Similarity calculation time: {timing_stats['similarity_time']:.2f} μs")
                console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
        except Exception as e:
            console.print(f"[red]Error finding closest articles: {str(e)}[/red]")

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
                    console.print("[yellow]No matching articles found.[/yellow]")
                    return

                # Create a rich table for better output formatting
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
                console.print(f"Encoding time: {timing_stats['encode_time']:.2f} μs")
                console.print(f"Loading time: {timing_stats['load_time']:.2f} μs")
                console.print(f"Similarity calculation time: {timing_stats['similarity_time']:.2f} μs")
                console.print(f"Total time: {timing_stats['total_time']:.2f} μs")
        except Exception as e:
            console.print(f"[red]Error searching articles: {str(e)}[/red]")

    asyncio.run(_search())

def main():
    """Main entry point for the application."""
    app()

if __name__ == '__main__':
    main()
