from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from concurrent.futures import ThreadPoolExecutor
import ssl
import socket
import time
from .errors import FetchFailure, InsertIntoException, LoadingException, InvalidRowError, EncodingError

console = None

def get_console():
    """Lazily get or create console."""
    global console
    if console is None:
        from rich.console import Console  # type: ignore[import-untyped]
        console = Console()
    return console

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
            from .database import Database
            self.database = Database(db_dir)
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._session = None
            import aiohttp  # type: ignore[import-untyped]
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
            # Lazy import aiohttp - only needed when fetching URLs
            import aiohttp  # type: ignore[import-untyped]
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
            # Import here in case session was never initialized
            import aiohttp  # type: ignore[import-untyped]
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
        # Lazy imports for URL fetching
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        from rich.progress import Progress, SpinnerColumn, TextColumn  # type: ignore[import-untyped]
        
        await self._init_session()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=get_console()
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
                    
        except Exception as e:
            # Check if it's an aiohttp error
            try:
                import aiohttp  # type: ignore[import-untyped]
                if isinstance(e, aiohttp.ClientError):
                    raise FetchFailure(f"Network error fetching URL {url}: {str(e)}")
            except ImportError:
                pass
            raise FetchFailure(f"Error fetching URL {url}: {str(e)}")

    async def _find_closest_using_umap_hdbscan(
        self, 
        query_embedding: "np.ndarray", 
        all_embeddings: "np.ndarray",
        all_articles: List[Dict[str, Any]],
        num_results: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find closest vectors using UMAP for dimensionality reduction and HDBSCAN for clustering.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            all_embeddings (np.ndarray): All article embeddings
            all_articles (List[Dict[str, Any]]): All article data
            num_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Closest articles
        """
        try:
            import numpy as np  
            from sklearn.metrics.pairwise import cosine_similarity 
            from umap import UMAP  
            from sklearn.preprocessing import StandardScaler  
            
            # Combine query with all embeddings
            combined_embeddings = np.vstack([query_embedding, all_embeddings])
            
            # Standardize embeddings
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(combined_embeddings)
            
            # Apply UMAP for dimensionality reduction
            # Use a reasonable n_components (e.g., min(50, embedding_dim-1))
            n_components = min(50, scaled_embeddings.shape[1] - 1, scaled_embeddings.shape[0] - 1)
            if n_components < 2:
                n_components = 2
            
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(scaled_embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            umap_embeddings = umap_model.fit_transform(scaled_embeddings)
            
            # Extract query embedding after UMAP
            query_umap = umap_embeddings[0:1]
            article_umap = umap_embeddings[1:]
            
            # Use cosine similarity on UMAP-reduced embeddings to find closest
            similarities = cosine_similarity(query_umap, article_umap)[0]
            
            # Sort by similarity and return top results
            top_indices = np.argsort(similarities)[::-1][:num_results]
            
            return [all_articles[idx] for idx in top_indices]
            
        except Exception as e:
            get_console().print(f"[yellow]UMAP+HDBSCAN search failed, falling back to cosine similarity: {e}[/yellow]")
            # Fallback to simple cosine similarity
            import numpy as np  # type: ignore[import-untyped]
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]
            similarities = cosine_similarity(query_embedding, all_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:num_results]
            return [all_articles[idx] for idx in top_indices]

    async def find_closest_articles_by_text(self, text: str, table_name: str = "articles", num_results: int = 1) -> Optional[List[Dict[str, Any]]]:
        """
        Find the most similar articles to the given text using UMAP + HDBSCAN.
        
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
            
            # Filter input text (keep stopwords for better embeddings)
            # Note: We're not using stopword filtering for embeddings anymore
            text_filtered = text.strip()
            if not text_filtered:
                get_console().print("[yellow]No valid text content[/yellow]")
                return None

            # Encode the input text using sentence-transformers
            encode_start = time.perf_counter_ns()
            # Lazy import to avoid slow startup
            from .embeddings import get_embedding_model
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.encode_single(text_filtered)
            encode_time = (time.perf_counter_ns() - encode_start) / 1000  # Convert to microseconds
            
            # Get all articles from all tables
            load_start = time.perf_counter_ns()
            import numpy as np  # type: ignore[import-untyped]
            
            all_articles = []
            all_embeddings = []
            
            for table_name_iter in self.database.tables:
                table = self.database.get_table(table_name_iter)
                if table:
                    articles = table.get_rows()
                    for article in articles:
                        article['table_name'] = table_name_iter  # Add table name to article
                        all_articles.append(article)
                        
                        # Get embedding from article (stored as encoded_data)
                        embedding = article.get('encoded_data')
                        if embedding is not None:
                            if isinstance(embedding, list):
                                # If it's a list, try to convert to numpy array
                                embedding = np.array(embedding)
                            elif not isinstance(embedding, np.ndarray):
                                embedding = np.array(embedding)
                            # Ensure it's 2D
                            if embedding.ndim == 1:
                                embedding = embedding.reshape(1, -1)
                            all_embeddings.append(embedding)
                        else:
                            # If no embedding, create one on the fly
                            content = article.get('content', '') or article.get('title', '')
                            if content:
                                embedding = embedding_model.encode_single(str(content))
                                all_embeddings.append(embedding)
                            else:
                                # Zero vector as fallback
                                try:
                                    embedding_dim = embedding_model._model.get_sentence_embedding_dimension()
                                except:
                                    embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
                                all_embeddings.append(np.zeros((1, embedding_dim)))
            
            load_time = (time.perf_counter_ns() - load_start) / 1000  # Convert to microseconds

            if not all_articles:
                get_console().print("[yellow]No articles found in any table[/yellow]")
                return None
            
            # Convert embeddings list to numpy array
            if all_embeddings:
                # Stack all embeddings, handling different shapes
                try:
                    all_embeddings = np.vstack(all_embeddings)
                except ValueError:
                    # If shapes don't match, pad to same length
                    max_len = max(emb.shape[1] for emb in all_embeddings if emb.ndim == 2)
                    padded_embeddings = []
                    for emb in all_embeddings:
                        if emb.ndim == 1:
                            emb = emb.reshape(1, -1)
                        if emb.shape[1] < max_len:
                            padding = np.zeros((1, max_len - emb.shape[1]))
                            emb = np.hstack([emb, padding])
                        padded_embeddings.append(emb)
                    all_embeddings = np.vstack(padded_embeddings)

            # Find closest using UMAP + HDBSCAN
            similarity_start = time.perf_counter_ns()
            from rich.progress import Progress, SpinnerColumn, TextColumn  # type: ignore[import-untyped]
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=get_console()
            ) as progress:
                task = progress.add_task("Finding closest articles using UMAP+HDBSCAN...", total=None)
                
                results = await self._find_closest_using_umap_hdbscan(
                    query_embedding,
                    all_embeddings,
                    all_articles,
                    num_results
                )
            
            similarity_time = (time.perf_counter_ns() - similarity_start) / 1000  # Convert to microseconds
            
            # Store timing information
            self._timing_stats = {
                'encode_time': encode_time,
                'load_time': load_time,
                'similarity_time': similarity_time,
                'total_time': (time.perf_counter_ns() - start_time) / 1000  # Convert to microseconds
            }
            
            if num_results == 1 and results:
                return results[0] if isinstance(results, list) and len(results) == 1 else results
            return results if isinstance(results, list) else [results] if results else None
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

