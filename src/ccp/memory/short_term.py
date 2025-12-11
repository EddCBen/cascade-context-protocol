from typing import Any, Optional
import torch
import redis
import logging
from redis.commands.search.query import Query
from src.ccp.neural.models import VectorNormalizer
from src.ccp.core.settings import settings

logger = logging.getLogger(__name__)

class RedisMemory:
    """
    Manages short-term memory (context window) using Redis.
    """
    def __init__(self, session_id: str, llm_service=None, input_dim: int = 768):
        self.session_id = session_id
        self.normalizer = VectorNormalizer(input_dim=input_dim)
        self.normalizer.load_weights(session_id) # Assuming domain_id is now session_id for loading weights
        self.normalizer.eval()
        
        self.llm_service = llm_service
        
        # Initialize Redis client
        # decode_responses=True checks that we get strings back not bytes
        self.client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding from LLM Service.
        """
        if self.llm_service:
            emb_list = self.llm_service.get_embedding(text)
            return torch.tensor(emb_list)
        else:
             return torch.randn(self.normalizer.input_dim)

    def get(self, key: str) -> Optional[Any]:
        """
        Standard key-value lookup.
        """
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    def vector_search(self, query: str) -> Optional[Any]:
        """
        Performs vector similarity search in Redis (RediSearch).
        Assumption: An index named 'embeddings_idx' exists.
        """
        # Step A: Generate raw embedding
        raw_embedding = self._get_embedding(query)
        
        # Step B: Normalize vector
        with torch.no_grad():
            normalized_embedding = self.normalizer(raw_embedding)
            
        # Step C: Search
        # This requires RediSearch module (+ RediSearch-py features if used directly, 
        # but standardized redis-py has .ft())
        try:
            # Construct query
            # KNersh neighbor search: "*=>[KNN 1 @vector $vec AS score]"
            q = Query(f"*=>[KNN 1 @vector $vec AS score]").return_fields("id", "score").dialect(2)
            
            params = {"vec": normalized_embedding.numpy().tobytes()}
            
            res = self.client.ft("embeddings_idx").search(q, query_params=params)
            
            if res.docs:
                return res.docs[0]
            return None
            
        except Exception as e:
            # If index doesn't exist or other error
            print(f"Redis vector search error: {e}")
            return None
