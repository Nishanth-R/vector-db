import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import ssl
import socket
from aiohttp_socks import ProxyConnector
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from database import Database
from errors import FetchFailure, InsertIntoException, LoadingException, InvalidRowError, EncodingError
from helper import filter_stopwords_in_text

console = Console()

class AppFlow:
    def __init__(self, db_dir: str = None):
        """
        Initialize the application flow with a database instance.
        
        Args:
            db_dir (str, optional): Directory for database files. Defaults to None.

        Raises:
            LoadingException: If there's an error initializing the database.
        """
        try:
            self.database = Database(db_dir)
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._session = None
            self._timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE
            self._timing_stats = {}
        except Exception as err:
            raise LoadingException(f"Failed to initialize AppFlow: {str(err)}")

    async def _init_session(self):
        """Initialize aiohttp session if not already initialized."""
        if self._session is None:
            # Configure connector with optimized settings for Windows
            connector = aiohttp.TCPConnector(
                ssl=self._ssl_context,
                force_close=True,
                enable_cleanup_closed=True,
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                family=socket.AF_INET
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )

    async def _close_session(self):
        """Close aiohttp session if it exists."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL asynchronously with optimized settings for Windows.
        
        Args:
            url (str): URL to fetch content from
            
        Returns:
            Optional[str]: Fetched content or None if failed

        Raises:
            FetchFailure: If there's an error fetching the URL.
        """
        await self._init_session()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Fetching content from {url}...", total=None)
                
                async with self._session.get(url, ssl=False) as response:
                    if response.status != 200:
                        raise FetchFailure(f"HTTP {response.status} error fetching URL: {url}")
                    
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    article_selectors = [
                        ("article", None),
                        ("section", {"class": "post-content"}),
                        ("div", {"class": "article-content"}),
                        ("div", {"itemprop": "articleBody"})
                    ]
                    
                    for tag, attrs in article_selectors:
                        elements = soup.find_all(tag, attrs=attrs)
                        if elements:
                            return "\n".join(
                                p.get_text().strip()
                                for element in elements
                                for p in element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
                            ).strip()
                    
                    paragraphs = soup.find_all("p")
                    if paragraphs:
                        return "\n".join(p.get_text().strip() for p in paragraphs).strip()
                    
                    raise FetchFailure(f"Could not find article elements on the page: {url}")
                    
        except aiohttp.ClientError as e:
            raise FetchFailure(f"Network error fetching URL {url}: {str(e)}")
        except Exception as e:
            raise FetchFailure(f"Error fetching URL {url}: {str(e)}")

    @staticmethod
    def _pad_arrays(vector1: np.ndarray, vector2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad arrays to the same length for similarity calculation.
        
        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Padded vectors
        """
        len1 = vector1.shape[1]
        len2 = vector2.shape[1]
        if len1 < len2:
            padding = np.zeros((1, len2 - len1))
            vector1 = np.concatenate([vector1, padding], axis=1)
        elif len2 < len1:
            padding = np.zeros((1, len1 - len2))
            vector2 = np.concatenate([vector2, padding], axis=1)
        return vector1, vector2

    async def _calculate_similarity(self, text_vector: np.ndarray, other_vector: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            text_vector (np.ndarray): First vector
            other_vector (np.ndarray): Second vector
            
        Returns:
            float: Similarity score
        """
        try:
            if text_vector.size == 0 or other_vector.size == 0:
                return 0.0
                
            text_vector, other_vector = self._pad_arrays(text_vector, other_vector)
            return cosine_similarity(text_vector, other_vector)[0][0]
        except Exception as e:
            console.print(f"[red]Error calculating similarity: {e}[/red]")
            return 0.0

    async def find_closest_articles_by_text(self, text: str, table_name: str = "articles", num_results: int = 1) -> Optional[List[Dict[str, Any]]]:
        """
        Find the most similar articles to the given text.
        
        Args:
            text (str): Text to find similar articles for
            table_name (str): Name of the table to search in (if None, searches all tables)
            num_results (int): Number of results to return
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of similar articles or None if not found

        Raises:
            LoadingException: If there's an error loading the table or calculating similarities.
            EncodingError: If there's an error encoding the text.
        """
        try:
            start_time = time.perf_counter_ns()  # Use nanoseconds for more precision
            
            # Filter and encode the input text
            text_filtered = filter_stopwords_in_text(text)
            if not text_filtered:
                console.print("[yellow]No valid text content after filtering stopwords[/yellow]")
                return None

            # Encode the input text
            encode_start = time.perf_counter_ns()
            text_vector = np.array(self.database.encode_text(text_filtered)).reshape(1, -1)
            encode_time = (time.perf_counter_ns() - encode_start) / 1000  # Convert to microseconds
            
            # Get all articles from all tables
            load_start = time.perf_counter_ns()
            all_articles = []
            for table_name in self.database.tables:
                table = self.database.get_table(table_name)
                if table:
                    articles = table.get_rows()
                    for article in articles:
                        article['table_name'] = table_name  # Add table name to article
                        all_articles.append(article)
            load_time = (time.perf_counter_ns() - load_start) / 1000  # Convert to microseconds

            if not all_articles:
                console.print("[yellow]No articles found in any table[/yellow]")
                return None

            # Calculate similarities
            similarity_start = time.perf_counter_ns()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Calculating similarities...", total=len(all_articles))
                
                similarities = await asyncio.gather(*[
                    self._calculate_similarity(
                        text_vector,
                        np.array(article.get('encoded_data', [])).reshape(1, -1)
                    )
                    for article in all_articles
                ])
            similarity_time = (time.perf_counter_ns() - similarity_start) / 1000  # Convert to microseconds

            if not similarities:
                return None
            
            # Store timing information
            self._timing_stats = {
                'encode_time': encode_time,
                'load_time': load_time,
                'similarity_time': similarity_time,
                'total_time': (time.perf_counter_ns() - start_time) / 1000  # Convert to microseconds
            }
            
            if num_results == 1:
                closest_index = np.argmax(similarities)
                return all_articles[closest_index]
            else:
                sorted_articles = [article for _, article in sorted(zip(similarities, all_articles), key=lambda pair: pair[0], reverse=True)]
                return sorted_articles[:num_results]
        except (LoadingException, EncodingError):
            raise
        except Exception as err:
            raise LoadingException(f"Failed to find similar articles: {str(err)}")

    async def write_to_db_from_url(self, url: str, table_name: str = "articles", content: str = None) -> bool:
        """
        Fetch content from URL and write it to the database.
        
        Args:
            url (str): URL to fetch content from
            table_name (str): Name of the table to write to
            content (str, optional): Pre-fetched content to insert
            
        Returns:
            bool: True if successful, False otherwise

        Raises:
            FetchFailure: If there's an error fetching the URL.
            InsertIntoException: If there's an error inserting into the database.
            InvalidRowError: If the table doesn't exist or row is invalid.
            LoadingException: If there's an error saving to disk.
        """
        try:
            start_time = time.perf_counter_ns()  # Use nanoseconds for more precision
            
            # Create table if it doesn't exist
            if table_name not in self.database.tables:
                self.database.create_table(table_name, ["title", "content", "url", "encoded_data"])

            # Fetch content if not provided
            fetch_time = 0
            if content is None:
                fetch_start = time.perf_counter_ns()
                content = await self._fetch_url_content(url)
                fetch_time = (time.perf_counter_ns() - fetch_start) / 1000  # Convert to microseconds
                if not content:
                    return False

            # Insert into database
            insert_start = time.perf_counter_ns()
            self.database.insert_into(table_name, {
                "title": url.split('/')[-1],  # Use last part of URL as title
                "content": content,
                "url": url
            })
            insert_time = (time.perf_counter_ns() - insert_start) / 1000  # Convert to microseconds
            
            # Store timing information
            self._timing_stats = {
                'fetch_time': fetch_time,
                'insert_time': insert_time,
                'total_time': (time.perf_counter_ns() - start_time) / 1000  # Convert to microseconds
            }
            
            return True
        except (FetchFailure, InsertIntoException, InvalidRowError, LoadingException):
            raise
        except Exception as err:
            raise InsertIntoException(f"Failed to write content from URL to database: {str(err)}")

    def get_timing_stats(self) -> Dict[str, float]:
        """Get the timing statistics for the last operation."""
        return self._timing_stats

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()

