"""
Embedding module using Google's embedding-gemma model via sentence-transformers.
"""
from __future__ import annotations
import threading
from typing import List
from .errors import EncodingError


class EmbeddingModel:    
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance
    
    def _ensure_model_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer  
                        try:
                            self._model = SentenceTransformer('all-MiniLM-L6-v2')
                        except Exception as e:
                            raise EncodingError(f"Failed to load any embedding model: {str(e)}")
                    except Exception as e:
                        raise EncodingError(f"Failed to load embedding model: {str(e)}")
    
    def __init__(self):
        # Don't load the model here - load it only when encode() is called
        pass
    
    def encode(self, texts: List[str]):
        """
        Encode a list of texts into embedding vectors.
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            numpy.ndarray: Array of embedding vectors (shape: [len(texts), embedding_dim])
            
        Raises:
            EncodingError: If encoding fails
        """
        try:
            import numpy as np
            self._ensure_model_loaded()
            
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts or all(not text or not text.strip() for text in texts):
                embedding_dim = self._model.get_sentence_embedding_dimension()
                return np.zeros((1, embedding_dim)) # return zero vector if empty
            
            filtered_texts = [text if text and text.strip() else " " for text in texts]
            embeddings = self._model.encode(filtered_texts, convert_to_numpy=True)
            return embeddings
        except Exception as err:
            raise EncodingError(f"Failed to encode texts: {str(err)}")
    
    def encode_single(self, text: str):
        """Encode a single text into an embedding vector."""
        return self.encode([text])


def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()

