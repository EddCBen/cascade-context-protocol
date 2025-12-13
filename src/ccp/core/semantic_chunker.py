"""
Semantic chunking utilities for creating meaningful ContextBlocks.
Buffers LLM tokens into semantic units (sentences, paragraphs, reasoning steps).
"""
import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Buffers LLM tokens into semantic blocks for meaningful ContextBlock creation.
    
    Supports multiple chunking modes:
    - sentence: Buffer until sentence boundary (. ! ? \\n\\n)
    - paragraph: Buffer until paragraph break (\\n\\n)
    - semantic: Detect reasoning steps, numbered lists, headers
    """
    
    def __init__(
        self,
        chunk_mode: str = "semantic",
        max_chunk_size: int = 512,
        min_chunk_size: int = 50
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_mode: Chunking strategy ("sentence", "paragraph", "semantic")
            max_chunk_size: Maximum tokens per block (hard limit)
            min_chunk_size: Minimum tokens before checking boundaries
        """
        self.buffer = ""
        self.chunk_mode = chunk_mode
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.step_counter = 0
        
        logger.debug(f"[SemanticChunker] Initialized: mode={chunk_mode}, max={max_chunk_size}, min={min_chunk_size}")
    
    def add_token(self, token: str) -> Optional[Dict]:
        """
        Add token to buffer, return block data if boundary reached.
        
        Args:
            token: Single token from LLM stream
        
        Returns:
            Block data dict if boundary reached, None otherwise
        """
        self.buffer += token
        
        # Check if we should flush the buffer
        if self._should_flush():
            block_data = self._create_block()
            self.buffer = ""
            return block_data
        
        return None
    
    def _should_flush(self) -> bool:
        """Determine if buffer should be flushed based on chunking mode."""
        buffer_size = len(self.buffer.split())
        
        # Hard limit: always flush if max size exceeded
        if buffer_size >= self.max_chunk_size:
            logger.debug(f"[SemanticChunker] Flushing: max_chunk_size exceeded ({buffer_size} tokens)")
            return True
        
        # Don't flush if below minimum size (unless at hard limit)
        if buffer_size < self.min_chunk_size:
            return False
        
        # Mode-specific boundary detection
        if self.chunk_mode == "sentence":
            return self._is_sentence_boundary()
        elif self.chunk_mode == "paragraph":
            return self._is_paragraph_boundary()
        elif self.chunk_mode == "semantic":
            return self._is_semantic_boundary()
        
        return False
    
    def _is_sentence_boundary(self) -> bool:
        """Check for sentence-ending punctuation."""
        # Look for sentence terminators followed by space or newline
        return bool(re.search(r'[.!?]\s+$', self.buffer))
    
    def _is_paragraph_boundary(self) -> bool:
        """Check for paragraph break (double newline)."""
        return '\n\n' in self.buffer
    
    def _is_semantic_boundary(self) -> bool:
        """
        Detect semantic reasoning step boundaries.
        
        Looks for:
        - Numbered lists (1., 2., etc.)
        - Markdown headers (##, ###)
        - Bullet points (-, *)
        - Reasoning markers (Therefore, In conclusion, Step N:)
        - Code blocks (```)
        """
        patterns = [
            r'\n\d+\.\s',           # Numbered list: "1. "
            r'\n#+\s',              # Markdown header: "## "
            r'\n[-*]\s',            # Bullet point: "- " or "* "
            r'Therefore[,:]',       # Conclusion marker
            r'In conclusion[,:]',   # Conclusion marker
            r'Step \d+:',           # Explicit step marker
            r'```\n',               # Code block end
            r'\n\n',                # Paragraph break
        ]
        
        for pattern in patterns:
            if re.search(pattern, self.buffer):
                logger.debug(f"[SemanticChunker] Semantic boundary detected: {pattern}")
                return True
        
        # Also check for sentence boundary in semantic mode
        return self._is_sentence_boundary()
    
    def _create_block(self) -> Dict:
        """
        Create semantic block from buffer.
        
        Returns:
            Dict with content, type, and metadata
        """
        self.step_counter += 1
        
        # Infer block type from content
        block_type = self._infer_block_type()
        
        content = self.buffer.strip()
        token_count = len(content.split())
        
        logger.info(f"[SemanticChunker] Created block #{self.step_counter}: type={block_type}, tokens={token_count}")
        
        return {
            "content": content,
            "type": block_type,
            "metadata": {
                "step": self.step_counter,
                "token_count": token_count,
                "chunk_mode": self.chunk_mode
            }
        }
    
    def _infer_block_type(self) -> str:
        """
        Infer semantic type of block from content.
        
        Returns:
            Block type string
        """
        content_lower = self.buffer.lower()
        
        # Calculation/math
        if re.search(r'\d+\s*[+\-*/=]\s*\d+', self.buffer):
            return "calculation"
        
        # Conclusion
        if re.search(r'(therefore|thus|hence|in conclusion|finally)', content_lower):
            return "conclusion"
        
        # Reasoning step
        if re.search(r'(step|first|second|next|then|now)', content_lower):
            return "reasoning_step"
        
        # Question
        if self.buffer.strip().endswith('?'):
            return "question"
        
        # Code block
        if '```' in self.buffer:
            return "code"
        
        # List item
        if re.match(r'^\s*[\d\-*]+\.?\s', self.buffer):
            return "list_item"
        
        # Default
        return "text"
    
    def flush(self) -> Optional[Dict]:
        """
        Flush remaining buffer content.
        
        Returns:
            Block data dict if buffer has content, None otherwise
        """
        if self.buffer.strip():
            block_data = self._create_block()
            self.buffer = ""
            return block_data
        return None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk static text into semantic blocks.
        Useful for large user inputs.
        
        Args:
            text: Full input text
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        # Split by semantic boundaries (paragraphs mostly for inputs)
        # We can also use sentence splitting for finer granularity
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If paragraph is too large, split by sentences
            if len(para.split()) > self.max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len((current_chunk + "\n" + sentence).split()) > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += "\n" + sentence if current_chunk else sentence
            else:
                if len((current_chunk + "\n\n" + para).split()) > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def reset(self):
        """Reset chunker state."""
        self.buffer = ""
        self.step_counter = 0
        logger.debug("[SemanticChunker] Reset")
