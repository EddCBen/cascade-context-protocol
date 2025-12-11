"""
Custom LLM client for direct communication with local llama.cpp server.
Provides full control over text generation parameters without external dependencies.
"""
import requests
import json
import logging
from typing import List, Dict, Optional, Iterator, Union, Any

logger = logging.getLogger(__name__)


class CCPLLMClient:
    """Custom HTTP client for local LLM server communication."""
    
    def __init__(self, base_url: str, model_name: str):
        """
        Initialize CCP LLM Client.
        
        Args:
            base_url: Base URL of the local LLM server (e.g., "http://localhost:8081/v1")
            model_name: Model identifier (used for logging)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        logger.info(f"[CCPLLMClient] Initialized with base_url={base_url}, model={model_name}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text completion (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling limit
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            repeat_penalty: Repetition penalty (1.0 = no penalty)
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            **kwargs: Additional parameters passed to the API
        
        Returns:
            Generated text string
        """
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False,
            **kwargs
        )
        
        try:
            response = self._make_request(payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract content from response
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0].get('message', {}).get('content', '')
                return content
            else:
                logger.error(f"Unexpected response format: {data}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse response: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text completion with streaming.
        
        Args:
            Same as generate()
        
        Yields:
            Text chunks as they are generated
        """
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
            **kwargs
        )
        
        try:
            response = self._make_request(payload, stream=True)
            response.raise_for_status()
            
            # Parse SSE stream
            for chunk in self._parse_stream(response):
                yield chunk
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming request failed: {e}")
            raise
    
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        stop: Optional[List[str]],
        repeat_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for the API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if stop:
            payload["stop"] = stop
        
        # Add any additional parameters
        payload.update(kwargs)
        
        return payload
    
    def _make_request(
        self,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> requests.Response:
        """Make HTTP request to the LLM server."""
        url = f"{self.base_url}/chat/completions"
        
        logger.debug(f"[CCPLLMClient] POST {url}")
        logger.debug(f"[CCPLLMClient] Payload: {json.dumps(payload, indent=2)}")
        
        response = self.session.post(
            url,
            json=payload,
            stream=stream,
            timeout=300  # 5 minute timeout for long generations
        )
        
        return response
    
    def _parse_stream(self, response: requests.Response) -> Iterator[str]:
        """
        Parse Server-Sent Events (SSE) stream.
        
        Yields:
            Text content from each chunk
        """
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            
            # SSE format: "data: {json}"
            if line.startswith('data: '):
                data_str = line[6:]  # Remove "data: " prefix
                
                # Check for stream end
                if data_str.strip() == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    
                    # Extract delta content
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        
                        if content:
                            yield content
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse stream chunk: {data_str[:100]}... Error: {e}")
                    continue
