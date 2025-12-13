import logging
import requests
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Client for the separated Embedding Service running in Docker.
    Connects to http://local-models:8082
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service client.
        
        Args:
            model_name: Kept for compatibility, but model is fixed on server side.
        """
        from src.ccp.core.settings import settings
        import os
        # Prioritize ENV, then Settings, then Fallback
        self.base_url = os.getenv("EMBEDDING_SERVICE_URL", settings.embedding_service_url)
        self.model_name = model_name
        logger.info(f"[EmbeddingService] Initialized client for {self.base_url}")
        
        # Test connection? No, let lazy load or first call handle it to avoid init crashes
    
    @property
    def model(self):
        """Mock property for compatibility if anything accesses it directly."""
        return None
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text via API.
        """
        if not text or not text.strip():
            logger.warning("[EmbeddingService] Empty text provided, returning zero vector")
            return [0.0] * 384
        
        try:
            return self.embed_batch([text])[0]
        except Exception as e:
            logger.error(f"[EmbeddingService] Failed to generate embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts via API.
        """
        if not texts:
            logger.warning("[EmbeddingService] Empty text list provided")
            return []
        
        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
                timeout=30  # Adjust timeout as needed
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            logger.error(f"[EmbeddingService] API request failed: {e}")
            # Fallback or raise? Raise is safer to know something is wrong.
            raise
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        # Hardcoded for now based on MiniLM, or could query /health endpoint
        return 384
