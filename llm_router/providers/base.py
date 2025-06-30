# llm_router/providers/base.py

"""Abstract base provider interface for llm-router."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Any
from datetime import datetime

from ..types import LLMResponse, ProviderConfig, ModelConfig, CompletionRequest
from ..exceptions import ProviderError, ModelNotAvailableError


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self.name = config.name
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_models()
    
    @abstractmethod
    def _initialize_models(self) -> None:
        """Initialize the available models for this provider."""
        pass
    
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> LLMResponse:
        """Complete a text generation request."""
        pass
    
    @abstractmethod
    async def stream_complete(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a text generation request."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and available."""
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate the cost of a request."""
        pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        return model in self.models and self.models[model].available
    
    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model)
    
    def validate_request(self, request: CompletionRequest) -> None:
        """Validate a completion request."""
        if not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if request.model and not self.is_model_available(request.model):
            raise ModelNotAvailableError(request.model, self.name)
        
        if request.max_tokens is not None and request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0 <= request.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= request.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
    
    async def _execute_with_timeout(self, coro, timeout: Optional[int] = None) -> Any:
        """Execute a coroutine with timeout."""
        timeout = timeout or self.config.timeout
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ProviderError(
                self.name,
                f"Request timed out after {timeout} seconds"
            )
    
    def _calculate_tokens(self, text: str) -> int:
        """Calculate approximate token count for text."""
        # Simple approximation: 1 token ≈ 4 characters
        # This is a rough estimate and should be replaced with proper tokenization
        return len(text) // 4
    
    def _create_response(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Create an LLMResponse object."""
        cost = self.estimate_cost("", content, model)  # Simplified cost calculation
        
        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            tokens_used=input_tokens + output_tokens,
            cost_estimate=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "available_models": self.get_available_models(),
            "config": self.config.dict(),
            "healthy": None  # Will be set by health monitoring
        } 