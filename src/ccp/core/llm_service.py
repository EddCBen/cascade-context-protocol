"""
LLM Service for CCP - Pure text generation interface.
Uses custom CCPLLMClient for direct HTTP communication with local LLM server.
Embeddings handled separately via EmbeddingService.
"""
import logging
from typing import List, Iterator, Any, Optional
from src.ccp.core.settings import settings
from src.ccp.core.ccp_llm_client import CCPLLMClient
from src.ccp.core.embedding_service import EmbeddingService
from src.ccp.functions.registry import FunctionRegistry
from pymongo import MongoClient
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class LLMService:
    """
    Pure Local LLM Service for text generation and embeddings.
    Uses custom clients without external dependencies (no OpenAI library).
    """
    
    def __init__(self, context_window_size: int = 131_072):
        """
        Initialize LLM Service.
        
        Args:
            context_window_size: Maximum context window (DeepSeek-R1-Distill: 131K tokens)
        """
        self._context_window_size = context_window_size
        self.current_usage = 0
        
        # Initialize Function Registry for tool discovery
        logger.info("[LLMService] Initializing Function Registry...")
        self.registry = FunctionRegistry()
        
        # Provider metadata
        self.provider_type = "local"
        self.model_name = settings.local_llm_model
        
        # Initialize CCP LLM Client (pure text generation)
        logger.info(f"[LLMService] Initializing CCPLLMClient for {self.model_name}")
        self.llm_client = CCPLLMClient(
            base_url=settings.local_llm_base_url,
            model_name=self.model_name
        )
        
        # Initialize Embedding Service (separate from LLM)
        logger.info("[LLMService] Initializing EmbeddingService...")
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        
        # Preload embedding model for stability
        try:
            logger.info("[LLMService] Preloading SentenceTransformer...")
            self.embedding_service.embed("warmup")
            logger.info("[LLMService] Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"[LLMService] FATAL: Embedding model failed to load: {e}")
            raise
        
        # Synchronize Function Registry with persistence stores
        self._synchronize_registry()
    
    def _synchronize_registry(self):
        """Synchronize function registry with MongoDB and Qdrant."""
        try:
            mongo_client = MongoClient(settings.mongo_uri)
            qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            
            logger.info("[LLMService] Synchronizing Function Registry with vector store...")
            self.registry.synchronize_stores(
                mongo_client=mongo_client,
                qdrant_client=qdrant_client,
                list_embedding_func=self.get_embedding
            )
            logger.info("[LLMService] Registry synchronization complete")
        except Exception as e:
            logger.error(f"[LLMService] Registry synchronization failed: {e}")
    
    # ==================== Context Management ====================
    
    @property
    def context_window_size(self) -> int:
        """Get maximum context window size."""
        return self._context_window_size
    
    def get_remaining_context(self) -> int:
        """Get remaining context window space."""
        return max(0, self._context_window_size - self.current_usage)
    
    def update_usage(self, tokens: int):
        """Update context usage counter."""
        self.current_usage += tokens
    
    # ==================== Embedding Methods ====================
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using SentenceTransformer.
        
        Args:
            text: Input text to embed
        
        Returns:
            384-dimensional embedding vector
        """
        return self.embedding_service.embed(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        """
        return self.embedding_service.embed_batch(texts)
    
    # ==================== Text Generation Methods ====================
    
    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion (non-streaming).
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        
        text = self.llm_client.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        )
        
        if text:
            # Rough token count approximation
            self.update_usage(len(prompt.split()) + len(text.split()))
        
        return text
    
    def generate_content_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text completion with streaming.
        
        Args:
            Same as generate_content()
        
        Yields:
            Text chunks as they are generated
        """
        messages = [{"role": "user", "content": prompt}]
        
        logger.info(f"[LLMService] Streaming generation (temp={temperature}, max_tokens={max_tokens})")
        
        for chunk in self.llm_client.generate_stream(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        ):
            if chunk:
                self.update_usage(len(chunk.split()))
            yield chunk
