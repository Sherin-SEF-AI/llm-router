# llm_router/providers/anthropic.py

"""Anthropic provider implementation for llm-router."""

import asyncio
import time
from typing import AsyncGenerator, Dict, List, Optional, Any

from .base import BaseProvider
from ..types import LLMResponse, ProviderConfig, ModelConfig, CompletionRequest
from ..exceptions import ProviderError, RateLimitExceededError


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        self.client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup the Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.anthropic.com",
                timeout=self.config.timeout * 1000  # Anthropic uses milliseconds
            )
        except ImportError:
            raise ImportError("Anthropic client not available. Install with: pip install anthropic")
    
    def _initialize_models(self) -> None:
        """Initialize available Anthropic models."""
        # Claude 3 models
        self.models["claude-3-opus-20240229"] = ModelConfig(
            name="claude-3-opus-20240229",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            context_length=200000,
            available=True
        )
        
        self.models["claude-3-sonnet-20240229"] = ModelConfig(
            name="claude-3-sonnet-20240229",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            context_length=200000,
            available=True
        )
        
        self.models["claude-3-haiku-20240307"] = ModelConfig(
            name="claude-3-haiku-20240307",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125,
            context_length=200000,
            available=True
        )
        
        # Claude 2 models (legacy)
        self.models["claude-2.1"] = ModelConfig(
            name="claude-2.1",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.008,
            output_cost_per_1k=0.024,
            context_length=200000,
            available=True
        )
        
        self.models["claude-2.0"] = ModelConfig(
            name="claude-2.0",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.008,
            output_cost_per_1k=0.024,
            context_length=100000,
            available=True
        )
    
    async def complete(self, request: CompletionRequest) -> LLMResponse:
        """Complete a text generation request using Anthropic."""
        self.validate_request(request)
        
        start_time = time.time()
        
        try:
            # Use default model if none specified
            model = request.model or "claude-3-sonnet-20240229"
            
            # Prepare completion parameters
            completion_params = {
                "model": model,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "system": "You are a helpful AI assistant.",
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            if request.stop:
                completion_params["stop_sequences"] = request.stop
            
            # Make the API call
            response = await self._execute_with_timeout(
                self.client.messages.create(**completion_params)
            )
            
            # Calculate timing
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response content
            content = response.content[0].text if response.content else ""
            
            # Get token usage
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            
            # Create response object
            return self._create_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": usage.dict() if usage else None
                }
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitExceededError(f"Anthropic rate limit exceeded: {e}")
            else:
                raise ProviderError(self.name, f"Anthropic API error: {e}", e)
    
    async def stream_complete(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a text generation request using Anthropic."""
        self.validate_request(request)
        
        try:
            # Use default model if none specified
            model = request.model or "claude-3-sonnet-20240229"
            
            # Prepare completion parameters
            completion_params = {
                "model": model,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "system": "You are a helpful AI assistant.",
                "messages": [{"role": "user", "content": request.prompt}],
                "stream": True
            }
            
            if request.stop:
                completion_params["stop_sequences"] = request.stop
            
            # Make the streaming API call
            stream = await self._execute_with_timeout(
                self.client.messages.create(**completion_params)
            )
            
            # Yield chunks as they arrive
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text
                    
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitExceededError(f"Anthropic rate limit exceeded: {e}")
            else:
                raise ProviderError(self.name, f"Anthropic streaming API error: {e}", e)
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is healthy."""
        try:
            # Try to list models as a health check
            await self._execute_with_timeout(
                self.client.models.list(),
                timeout=10
            )
            return True
        except Exception:
            return False
    
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate the cost of a request."""
        if model not in self.models:
            return 0.0
        
        model_config = self.models[model]
        
        # Calculate token counts (simplified)
        input_tokens = self._calculate_tokens(prompt)
        output_tokens = self._calculate_tokens(response)
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_config.output_cost_per_1k
        
        return input_cost + output_cost
    
    async def close(self) -> None:
        """Close the Anthropic client."""
        if self.client:
            # Anthropic client doesn't have a close method, but we can clean up
            self.client = None 