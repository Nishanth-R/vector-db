"""
Pure python implementation of a vector database, will include next to no libraries.
Simple and easy to understand code.
Class definitions -
Database - Implementation of a simple database that will load data to and from disk

Assumptions -
    id - is the only primary key
"""
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import pickle
from typing import Dict, List, Any, Optional
from collections import deque

from .errors import InsertIntoException, LoadingException, InvalidRowError, EncodingError
from .btree import BTree

class Table:
    def __init__(self, table_name: str, columns: List[str], primary_key: str = 'id', db_dir: str = None):
        """
        Initializes a new table object.

        Args:
            table_name (str): The name of the table.
            columns (list): A list of column names for the table.
            primary_key (str, optional): The name of the primary key column. Defaults to 'id'.
            db_dir (str, optional): Directory for storing index files.
        """
        self.table_name = table_name
        self.columns = columns
        self.primary_key = primary_key
        if primary_key not in columns:
            self.columns.insert(0, primary_key)
        self.data = []
        self._next_id = 1 if primary_key == 'id' else None
        self._lock = threading.Lock()
        self.db_dir = db_dir or os.getcwd()
        
        # Initialize B-tree index for primary key
        index_filename = os.path.join(self.db_dir, f"{table_name}_index.pickle")
        self._index = BTree(max_keys=3, filename=index_filename)

    def insert_row(self, row_values: Dict[str, Any]) -> None:
        """
        Inserts a new row into the table.

        Args:
            row_values (dict): A dictionary where keys are column names and values are the row data.

        Raises:
            ValueError: If the provided keys do not match the table columns (excluding primary key if auto-generated).
            InvalidRowError: If the row ID already exists.
        """
        with self._lock:
            try:
                if self.primary_key == 'id':
                    if self.primary_key not in row_values:
                        row_values[self.primary_key] = self._next_id
                        self._next_id += 1
                    elif row_values[self.primary_key] >= self._next_id:
                        self._next_id = row_values[self.primary_key] + 1

                # Check for duplicate IDs
                if self.primary_key in row_values:
                    existing_ids = {row[0] for row in self.data}
                    if row_values[self.primary_key] in existing_ids:
                        raise InvalidRowError(f"ID {row_values[self.primary_key]} already exists in table '{self.table_name}'")

                if set(row_values.keys()) != set(self.columns):
                    raise ValueError(f"Provided keys {set(row_values.keys())} do not match table columns {set(self.columns)}.")

                ordered_values = tuple(row_values[col] for col in self.columns)
                self.data.append(ordered_values)
                
                # Update B-tree index
                primary_key_value = row_values[self.primary_key]
                row_dict = dict(zip(self.columns, ordered_values))
                self._index.insert(primary_key_value, len(self.data) - 1)
            except Exception as err:
                raise InsertIntoException(f"Failed to insert row into table '{self.table_name}': {str(err)}")

    def get_rows(self) -> List[Dict[str, Any]]:
        """
        Returns all rows in the table as a list of dictionaries.

        Raises:
            LoadingException: If there's an error retrieving the rows.
        """
        try:
            return [dict(zip(self.columns, row)) for row in self.data]
        except Exception as err:
            raise LoadingException(f"Failed to get rows from table '{self.table_name}': {str(err)}")
    
    def get_by_id(self, primary_key_value: Any) -> Optional[Dict[str, Any]]:
        """
        Get a row by its primary key value using B-tree index.
        
        Args:
            primary_key_value (Any): The primary key value to search for
            
        Returns:
            Optional[Dict[str, Any]]: The row if found, None otherwise
            
        Raises:
            LoadingException: If there's an error retrieving the row.
        """
        try:
            row_index = self._index.search(primary_key_value)
            if row_index is not None and 0 <= row_index < len(self.data):
                return dict(zip(self.columns, self.data[row_index]))
            return None
        except Exception as err:
            raise LoadingException(f"Failed to get row by ID from table '{self.table_name}': {str(err)}")

    def save_to_disk(self, filename: str) -> None:
        """
        Saves the table data to a JSON file.

        Args:
            filename (str): The name of the file to save to.

        Raises:
            LoadingException: If there's an error saving to disk.
        """
        try:
            table_data = {
                "table_name": self.table_name,
                "columns": self.columns,
                "primary_key": self.primary_key,
                "data": [dict(zip(self.columns, row)) for row in self.data]
            }
            with open(filename, 'w') as jsonfile:
                json.dump(table_data, jsonfile, indent=4)
            print(f"Table '{self.table_name}' saved to '{filename}'")
        except Exception as err:
            raise LoadingException(f"Failed to save table '{self.table_name}' to disk: {str(err)}")

    @classmethod
    def load_from_disk(cls, filename: str) -> 'Table':
        """
        Loads a table from a JSON file.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            Table: A new Table object loaded from the file.

        Raises:
            LoadingException: If there's an error loading from disk.
        """
        try:
            db_dir = os.path.dirname(filename) or os.getcwd()
            with open(filename, 'r') as jsonfile:
                table_data = json.load(jsonfile)
                table_name = table_data['table_name']
                columns = table_data['columns']
                primary_key = table_data['primary_key']
                new_table = cls(table_name, columns, primary_key, db_dir=db_dir)
                
                # Rebuild index while loading data
                for row_dict in table_data['data']:
                    # Insert row without triggering index update in insert_row
                    ordered_values = tuple(row_dict[col] for col in new_table.columns)
                    new_table.data.append(ordered_values)
                    
                    # Manually update index
                    primary_key_value = row_dict[primary_key]
                    new_table._index.insert(primary_key_value, len(new_table.data) - 1)
                
                # Update next_id if using auto-increment
                if primary_key == 'id' and new_table.data:
                    max_id = max(row[0] for row in new_table.data if isinstance(row[0], int))
                    new_table._next_id = max_id + 1
                
                return new_table
        except Exception as err:
            raise LoadingException(f"Failed to load table from '{filename}': {str(err)}")


class Database:
    def __init__(self, db_dir: str = None):
        """
        Initialize the database with a directory for storing files.
        
        Args:
            db_dir (str, optional): Directory to store database files. Defaults to current directory.

        Raises:
            LoadingException: If there's an error initializing the database.
        """
        try:
            self.db_dir = db_dir or os.getcwd()
            self.tables: Dict[str, Table] = {}
            self.bow_filename = os.path.join(self.db_dir, 'bow.pickle')
            self._bow_cache = None
            self._bow_lock = threading.Lock()
            self._executor = ThreadPoolExecutor(max_workers=4)
            
            # LIFO file management - track insert order
            self._lifo_stack = deque()
            self._lifo_filename = os.path.join(self.db_dir, 'lifo_stack.pickle')
            self._load_lifo_stack()
            
            self._load_tables()
        except Exception as err:
            raise LoadingException(f"Failed to initialize database: {str(err)}")
    
    def _load_lifo_stack(self) -> None:
        """Load LIFO stack from disk."""
        try:
            if os.path.exists(self._lifo_filename):
                with open(self._lifo_filename, 'rb') as file:
                    self._lifo_stack = pickle.load(file)
        except Exception:
            self._lifo_stack = deque()
    
    def _save_lifo_stack(self) -> None:
        """Save LIFO stack to disk."""
        try:
            with open(self._lifo_filename, 'wb') as file:
                pickle.dump(self._lifo_stack, file)
        except Exception as err:
            raise LoadingException(f"Failed to save LIFO stack: {str(err)}")

    def _load_tables(self) -> None:
        """
        Load all tables from disk (lazy - only load when database is accessed).

        Raises:
            LoadingException: If there's an error loading tables.
        """
        try:
            # Only load tables if the directory exists and has files
            # Skip loading if just checking help
            if not os.path.exists(self.db_dir):
                return
            try:
                files = os.listdir(self.db_dir)
            except (OSError, PermissionError):
                # Can't read directory, skip loading
                return
            for filename in files:
                if filename.endswith('.json'):
                    table_name = filename[:-5]  # Remove .json extension
                    table_path = os.path.join(self.db_dir, filename)
                    self.tables[table_name] = Table.load_from_disk(table_path)
        except Exception as err:
            # Don't raise exception during initialization - just log it
            # raise LoadingException(f"Failed to load tables from directory '{self.db_dir}': {str(err)}")
            pass

    def create_table(self, table_name: str, columns: List[str], primary_key: str = 'id') -> Table:
        """
        Create a new table in the database.
        
        Args:
            table_name (str): Name of the table
            columns (List[str]): List of column names
            primary_key (str, optional): Primary key column name. Defaults to 'id'.
            
        Returns:
            Table: The created table object

        Raises:
            InvalidRowError: If the table already exists.
            InsertIntoException: If there's an error creating the table.
        """
        try:
            if table_name in self.tables:
                raise InvalidRowError(f"Table '{table_name}' already exists")
            
            table = Table(table_name, columns, primary_key, db_dir=self.db_dir)
            self.tables[table_name] = table
            return table
        except InvalidRowError:
            raise
        except Exception as err:
            raise InsertIntoException(f"Failed to create table '{table_name}': {str(err)}")

    def get_table(self, table_name: str) -> Optional[Table]:
        """
        Get a table by name.
        
        Args:
            table_name (str): Name of the table to retrieve
            
        Returns:
            Optional[Table]: The table if it exists, None otherwise

        Raises:
            LoadingException: If there's an error retrieving the table.
        """
        try:
            return self.tables.get(table_name)
        except Exception as err:
            raise LoadingException(f"Failed to get table '{table_name}': {str(err)}")
    
    def query_by_id(self, table_name: str, primary_key_value: Any) -> Optional[Dict[str, Any]]:
        """
        Query a row by primary key value using B-tree index.
        
        Args:
            table_name (str): Name of the table to query
            primary_key_value (Any): The primary key value to search for
            
        Returns:
            Optional[Dict[str, Any]]: The row if found, None otherwise
            
        Raises:
            LoadingException: If there's an error querying the table.
            InvalidRowError: If the table doesn't exist.
        """
        try:
            if table_name not in self.tables:
                raise InvalidRowError(f"Table '{table_name}' does not exist")
            
            return self.tables[table_name].get_by_id(primary_key_value)
        except InvalidRowError:
            raise
        except Exception as err:
            raise LoadingException(f"Failed to query by ID in table '{table_name}': {str(err)}")

    def _get_bow(self) -> Dict[str, int]:
        """
        Get the bag of words dictionary, loading from disk if necessary.

        Raises:
            LoadingException: If there's an error loading the BOW dictionary.
        """
        with self._bow_lock:
            try:
                if self._bow_cache is None:
                    try:
                        with open(self.bow_filename, 'rb') as file:
                            self._bow_cache = pickle.load(file)
                    except (EOFError, FileNotFoundError):
                        self._bow_cache = {}
                        with open(self.bow_filename, 'wb') as file:
                            pickle.dump(self._bow_cache, file)
                return self._bow_cache
            except Exception as err:
                raise LoadingException(f"Failed to load BOW dictionary: {str(err)}")

    def _save_bow(self) -> None:
        """
        Save the bag of words dictionary to disk.

        Raises:
            LoadingException: If there's an error saving the BOW dictionary.
        """
        with self._bow_lock:
            try:
                if self._bow_cache is not None:
                    with open(self.bow_filename, 'wb') as file:
                        pickle.dump(self._bow_cache, file)
            except Exception as err:
                raise LoadingException(f"Failed to save BOW dictionary: {str(err)}")

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return text.lower().split()

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text into a list of word IDs using the bag of words dictionary.
        
        Args:
            text (str): Text to encode
            
        Returns:
            List[int]: List of word IDs

        Raises:
            EncodingError: If there's an error encoding the text.
        """
        try:
            words = self.tokenize_text(text)
            if not words:
                return []

            bow = self._get_bow()
            tokens = []
            bow_changed = False

            for word in words:
                word = word.strip()
                if not word:
                    continue
                    
                word_id = bow.get(word)
                if word_id is None:
                    word_id = len(bow) + 1
                    bow[word] = word_id
                    bow_changed = True
                tokens.append(word_id)

            if bow_changed:
                self._save_bow()

            return tokens

        except Exception as err:
            raise EncodingError(f"Error during encoding process: {str(err)}")

    def _encode_text_fields(self, table_name: str, row_values: Dict[str, Any]) -> None:
        """
        Encode text fields in a row using sentence-transformers and add encoded_data column.
        
        Args:
            table_name (str): Name of the table
            row_values (Dict[str, Any]): Row values to encode

        Raises:
            EncodingError: If there's an error encoding the text fields.
        """
        try:
            # Import numpy first before any usage
            import numpy as np  # type: ignore[import-untyped]
            from .embeddings import get_embedding_model
            
            table = self.tables[table_name]
            text_fields = [col for col in table.columns if col != 'id' and col != 'encoded_data']
            
            # Combine all text fields
            combined_text = ' '.join(str(row_values.get(field, '')) for field in text_fields)
            
            # Encode the combined text using sentence-transformers
            if combined_text.strip():
                embedding_model = get_embedding_model()
                embedding = embedding_model.encode_single(combined_text)
                # Store as list for JSON serialization
                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    row_values['encoded_data'] = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    row_values['encoded_data'] = embedding.tolist()
                elif isinstance(embedding, list):
                    row_values['encoded_data'] = embedding
                else:
                    # Fallback: convert to list
                    row_values['encoded_data'] = list(embedding)
            else:
                # Empty embedding vector
                embedding_model = get_embedding_model()
                try:
                    embedding_dim = embedding_model._model.get_sentence_embedding_dimension()
                except:
                    embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
                row_values['encoded_data'] = [0.0] * embedding_dim
        except Exception as err:
            raise EncodingError(f"Failed to encode text fields for table '{table_name}': {str(err)}")

    def insert_into(self, table_name: str, row_values: Dict[str, Any]) -> None:
        """
        Insert a row into a table and encode text fields in parallel.
        
        Args:
            table_name (str): Name of the table to insert into
            row_values (Dict[str, Any]): Row values to insert

        Raises:
            InsertIntoException: If there's an error inserting the row.
            InvalidRowError: If the row is invalid.
            LoadingException: If there's an error saving to disk.
        """
        try:
            if table_name not in self.tables:
                raise InvalidRowError(f"Table '{table_name}' does not exist")

            future = self._executor.submit(self._encode_text_fields, table_name, row_values)
            future.result()
            
            self.tables[table_name].insert_row(row_values)
            
            # Update LIFO stack - push (table_name, primary_key_value) to top
            primary_key_value = row_values.get(self.tables[table_name].primary_key)
            if primary_key_value is not None:
                self._lifo_stack.append((table_name, primary_key_value))
                self._save_lifo_stack()
            
            self.tables[table_name].save_to_disk(os.path.join(self.db_dir, f"{table_name}.json"))
        except (InvalidRowError, InsertIntoException, LoadingException):
            raise
        except Exception as err:
            raise InsertIntoException(f"Failed to insert row into table '{table_name}': {str(err)}")

    def convert_vector_to_text(self, vector: List[int]) -> str:
        """
        Convert a vector of word IDs back to text.
        
        Args:
            vector (List[int]): Vector of word IDs
            
        Returns:
            str: Decoded text

        Raises:
            LoadingException: If there's an error loading the BOW dictionary.
        """
        try:
            bow = self._get_bow()
            reverse_bow = {value: key for key, value in bow.items()}
            return ' '.join(reverse_bow.get(word_id, '') for word_id in vector)
        except Exception as err:
            raise LoadingException(f"Failed to convert vector to text: {str(err)}")
    
    def get_lifo_inserts(self, limit: int = None) -> List[tuple]:
        """
        Get the most recently inserted rows in LIFO order.
        
        Args:
            limit (int, optional): Maximum number of recent inserts to return
            
        Returns:
            List[tuple]: List of (table_name, primary_key_value) tuples in LIFO order
        """
        if limit is None:
            return list(reversed(self._lifo_stack))
        return list(reversed(list(self._lifo_stack)[-limit:]))
