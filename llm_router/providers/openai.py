# llm_router/providers/openai.py

"""OpenAI provider implementation for llm-router."""

import asyncio
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx

from .base import BaseProvider
from ..types import LLMResponse, ProviderConfig, ModelConfig, CompletionRequest
from ..exceptions import ProviderError, RateLimitExceededError


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup the OpenAI client."""
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.openai.com/v1",
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("OpenAI client not available. Install with: pip install openai")
    
    def _initialize_models(self) -> None:
        """Initialize available OpenAI models."""
        # GPT-4 models
        self.models["gpt-4"] = ModelConfig(
            name="gpt-4",
            provider=self.name,
            max_tokens=8192,
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
            context_length=8192,
            available=True
        )
        
        self.models["gpt-4-turbo"] = ModelConfig(
            name="gpt-4-turbo",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
            context_length=128000,
            available=True
        )
        
        self.models["gpt-4-turbo-preview"] = ModelConfig(
            name="gpt-4-turbo-preview",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
            context_length=128000,
            available=True
        )
        
        # GPT-3.5 models
        self.models["gpt-3.5-turbo"] = ModelConfig(
            name="gpt-3.5-turbo",
            provider=self.name,
            max_tokens=4096,
            input_cost_per_1k=0.0015,
            output_cost_per_1k=0.002,
            context_length=16385,
            available=True
        )
        
        self.models["gpt-3.5-turbo-16k"] = ModelConfig(
            name="gpt-3.5-turbo-16k",
            provider=self.name,
            max_tokens=16384,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.004,
            context_length=16384,
            available=True
        )
    
    async def complete(self, request: CompletionRequest) -> LLMResponse:
        """Complete a text generation request using OpenAI."""
        self.validate_request(request)
        
        start_time = time.time()
        
        try:
            # Use default model if none specified
            model = request.model or "gpt-3.5-turbo"
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": request.prompt}]
            
            # Prepare completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
            }
            
            if request.max_tokens:
                completion_params["max_tokens"] = request.max_tokens
            
            if request.stop:
                completion_params["stop"] = request.stop
            
            # Make the API call
            response = await self._execute_with_timeout(
                self.client.chat.completions.create(**completion_params)
            )
            
            # Calculate timing
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            # Get token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # Create response object
            return self._create_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": usage.dict() if usage else None
                }
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitExceededError(f"OpenAI rate limit exceeded: {e}")
            else:
                raise ProviderError(self.name, f"OpenAI API error: {e}", e)
    
    async def stream_complete(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a text generation request using OpenAI."""
        self.validate_request(request)
        
        try:
            # Use default model if none specified
            model = request.model or "gpt-3.5-turbo"
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": request.prompt}]
            
            # Prepare completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": True
            }
            
            if request.max_tokens:
                completion_params["max_tokens"] = request.max_tokens
            
            if request.stop:
                completion_params["stop"] = request.stop
            
            # Make the streaming API call
            stream = await self._execute_with_timeout(
                self.client.chat.completions.create(**completion_params)
            )
            
            # Yield chunks as they arrive
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitExceededError(f"OpenAI rate limit exceeded: {e}")
            else:
                raise ProviderError(self.name, f"OpenAI streaming API error: {e}", e)
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is healthy."""
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
        """Close the OpenAI client."""
        if self.client:
            # OpenAI client doesn't have a close method, but we can clean up
            self.client = None 