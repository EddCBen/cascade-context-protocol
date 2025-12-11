"""
Dedicated embedding service using SentenceTransformer.
Handles all embedding generation separately from LLM text generation.
"""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embeddings via SentenceTransformer."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service.
        
        Args:
            model_name: HuggingFace model identifier for SentenceTransformer
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"[EmbeddingService] Initialized with model={model_name}")
    
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"[EmbeddingService] Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"[EmbeddingService] Model loaded successfully")
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("[EmbeddingService] Empty text provided, returning zero vector")
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"[EmbeddingService] Failed to generate embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("[EmbeddingService] Empty text list provided")
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"[EmbeddingService] Failed to generate batch embeddings: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        # Load model to get dimension
        test_embedding = self.embed("test")
        return len(test_embedding)
