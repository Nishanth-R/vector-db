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

from errors import InsertIntoException, LoadingException, InvalidRowError, EncodingError

class Table:
    def __init__(self, table_name: str, columns: List[str], primary_key: str = 'id'):
        """
        Initializes a new table object.

        Args:
            table_name (str): The name of the table.
            columns (list): A list of column names for the table.
            primary_key (str, optional): The name of the primary key column. Defaults to 'id'.
        """
        self.table_name = table_name
        self.columns = columns
        self.primary_key = primary_key
        if primary_key not in columns:
            self.columns.insert(0, primary_key)
        self.data = []
        self._next_id = 1 if primary_key == 'id' else None
        self._lock = threading.Lock()

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
            with open(filename, 'r') as jsonfile:
                table_data = json.load(jsonfile)
                table_name = table_data['table_name']
                columns = table_data['columns']
                primary_key = table_data['primary_key']
                new_table = cls(table_name, columns, primary_key)
                for row_dict in table_data['data']:
                    new_table.insert_row(row_dict)
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
            self._load_tables()
        except Exception as err:
            raise LoadingException(f"Failed to initialize database: {str(err)}")

    def _load_tables(self) -> None:
        """
        Load all tables from disk.

        Raises:
            LoadingException: If there's an error loading tables.
        """
        try:
            for filename in os.listdir(self.db_dir):
                if filename.endswith('.json'):
                    table_name = filename[:-5]  # Remove .json extension
                    table_path = os.path.join(self.db_dir, filename)
                    self.tables[table_name] = Table.load_from_disk(table_path)
        except Exception as err:
            raise LoadingException(f"Failed to load tables from directory '{self.db_dir}': {str(err)}")

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
            
            table = Table(table_name, columns, primary_key)
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
        Encode text fields in a row and add encoded_data column.
        
        Args:
            table_name (str): Name of the table
            row_values (Dict[str, Any]): Row values to encode

        Raises:
            EncodingError: If there's an error encoding the text fields.
        """
        try:
            table = self.tables[table_name]
            text_fields = [col for col in table.columns if col != 'id' and col != 'encoded_data']
            
            # Combine all text fields
            combined_text = ' '.join(str(row_values.get(field, '')) for field in text_fields)
            
            # Encode the combined text
            encoded_data = self.encode_text(combined_text)
            row_values['encoded_data'] = encoded_data
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
