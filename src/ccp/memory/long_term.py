from typing import List, Dict, Any, Optional
import logging
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.ccp.neural.models import VectorNormalizer
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)

class QdrantMemory:
    """
    Manages long-term memory using Qdrant.
    Handles 'knowledge' and 'functions' collections.
    """
    def __init__(self, domain_id: str = "default", llm_service=None):
        self.domain_id = domain_id
        # Assuming VectorNormalizer can be initialized without input_dim or it's handled internally
        # If not, a default or settings-based input_dim would be needed here.
        # For now, keeping the original input_dim value as a placeholder if not provided by settings.
        # The original code had input_dim: int = 768, so we'll use that as a default if not specified elsewhere.
        self.normalizer = VectorNormalizer(input_dim=settings.embedding_dim) # Assuming settings provides embedding_dim
        # Attempt to load weights, gracefully handle failure/random init
        self.normalizer.load_weights(domain_id)
        self.normalizer.eval() # Set to evaluation mode
        
        self.llm_service = llm_service
        
        # Initialize Qdrant client
        # We assume Qdrant is running locally on default port or use env vars in real prod
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding from LLM Service.
        """
        if self.llm_service:
            emb_list = self.llm_service.get_embedding(text)
            return torch.tensor(emb_list)
        else:
            # Fallback for testing if no service provided, though discouraged
            return torch.randn(self.normalizer.input_dim)

    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches long-term knowledge base.
        """
        # Step A: Generate raw embedding
        raw_embedding = self._get_embedding(query)
        
        # Step B: Normalize vector (Hippocampal Replay)
        with torch.no_grad():
            normalized_embedding = self.normalizer(raw_embedding)
        
        # Step C: Perform Qdrant Search
        # We assume a collection named 'knowledge' exists. 
        # In a real setup, we should ensure it exists.
        try:
            search_result = self.client.search(
                collection_name="knowledge",
                query_vector=normalized_embedding.tolist(),
                limit=top_k
            )
            return [
                {"id": hit.id, "score": hit.score, "payload": hit.payload}
                for hit in search_result
            ]
        except Exception as e:
            print(f"Qdrant search error (knowledge): {e}")
            return []

    def search_functions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches available functions/tools.
        """
        # Step A: Generate raw embedding
        raw_embedding = self._get_embedding(query)
        
        # Step B: Normalize vector (Pattern Separation)
        with torch.no_grad():
            normalized_embedding = self.normalizer(raw_embedding)
        
        # Step C: Perform Qdrant Search
        try:
            search_result = self.client.search(
                collection_name="functions",
                query_vector=normalized_embedding.tolist(),
                limit=top_k
            )
            return [
                {"id": hit.id, "score": hit.score, "payload": hit.payload}
                for hit in search_result
            ]
        except Exception as e:
            print(f"Qdrant search error (functions): {e}")
            return []
