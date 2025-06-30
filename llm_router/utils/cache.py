# llm_router/utils/cache.py

"""Response caching for llm-router."""

import hashlib
import json
import time
from typing import Any, Dict, Optional
from cachetools import TTLCache

from ..types import LLMResponse, CacheConfig
from ..exceptions import CacheError


class ResponseCache:
    """In-memory cache for LLM responses."""
    
    def __init__(self, config: CacheConfig):
        """Initialize the cache with configuration."""
        self.config = config
        self.cache: Optional[TTLCache] = None
        
        if config.enabled:
            self.cache = TTLCache(
                maxsize=config.max_size,
                ttl=config.ttl
            )
    
    def _generate_key(self, prompt: str, model: str, provider: str, **kwargs) -> str:
        """Generate a cache key from request parameters."""
        # Create a deterministic string representation
        key_data = {
            "prompt": prompt,
            "model": model,
            "provider": provider,
            **kwargs
        }
        
        # Sort keys for deterministic ordering
        key_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        
        # Create hash for consistent key length
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"{self.config.key_prefix}:{key_hash}"
    
    def get(self, prompt: str, model: str, provider: str, **kwargs) -> Optional[LLMResponse]:
        """Get a cached response if available."""
        if not self.config.enabled or not self.cache:
            return None
        
        try:
            key = self._generate_key(prompt, model, provider, **kwargs)
            cached_data = self.cache.get(key)
            
            if cached_data:
                # Convert back to LLMResponse
                return LLMResponse(**cached_data)
            
            return None
        except Exception as e:
            raise CacheError(f"Error retrieving from cache: {e}")
    
    def set(self, response: LLMResponse, prompt: str, model: str, provider: str, **kwargs) -> None:
        """Cache a response."""
        if not self.config.enabled or not self.cache:
            return
        
        try:
            key = self._generate_key(prompt, model, provider, **kwargs)
            # Convert LLMResponse to dict for caching
            response_dict = response.dict()
            self.cache[key] = response_dict
        except Exception as e:
            raise CacheError(f"Error setting cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        if self.cache:
            self.cache.clear()
    
    def size(self) -> int:
        """Get the current cache size."""
        if self.cache:
            return len(self.cache)
        return 0
    
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.config.enabled and self.cache is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {
                "enabled": False,
                "size": 0,
                "max_size": 0,
                "ttl": 0
            }
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "ttl": self.config.ttl,
            "key_prefix": self.config.key_prefix
        } 