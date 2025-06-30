# llm_router/router.py

"""Main LLM router implementation."""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from .types import (
    LLMResponse, ProviderConfig, CompletionRequest, CacheConfig, 
    RetryConfig, RouterStats, CostSummary
)
from .exceptions import (
    LLMRouterError, NoProvidersAvailableError, ProviderError,
    InvalidConfigurationError
)
from .providers.base import BaseProvider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .strategies.priority import PriorityStrategy
from .strategies.cost_optimized import CostOptimizedStrategy
from .strategies.round_robin import RoundRobinStrategy
from .utils.cache import ResponseCache
from .utils.retry import RetryHandler
from .utils.monitor import HealthMonitor
from .utils.cost_tracker import CostTracker
from .utils.metrics import MetricsCollector


class LLMRouter:
    """Main LLM router that intelligently routes requests across providers."""
    
    def __init__(
        self,
        strategy: str = "priority",
        cache_ttl: int = 3600,
        cache_max_size: int = 1000,
        retry_attempts: int = 3,
        enable_health_monitoring: bool = True,
        health_check_interval: int = 300,
        enable_cost_tracking: bool = True,
        enable_metrics: bool = True
    ):
        """Initialize the LLM router."""
        self.providers: Dict[str, BaseProvider] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self.strategy_name = strategy
        self.strategy = None
        
        # Initialize components
        self.cache = ResponseCache(CacheConfig(
            enabled=True,
            ttl=cache_ttl,
            max_size=cache_max_size
        ))
        
        self.retry_handler = RetryHandler(RetryConfig(
            max_attempts=retry_attempts,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True
        ))
        
        self.health_monitor = HealthMonitor(health_check_interval)
        self.cost_tracker = CostTracker()
        self.metrics = MetricsCollector()
        
        # Start health monitoring if enabled
        if enable_health_monitoring:
            asyncio.create_task(self.health_monitor.start_monitoring())
    
    def add_provider(
        self,
        name: str,
        provider_type: str,
        api_key: str,
        priority: int = 1,
        **kwargs
    ) -> None:
        """Add a provider to the router."""
        # Create provider configuration
        config = ProviderConfig(
            name=name,
            api_key=api_key,
            priority=priority,
            **kwargs
        )
        
        # Create provider instance
        if provider_type.lower() == "openai":
            provider = OpenAIProvider(config)
        elif provider_type.lower() == "anthropic":
            provider = AnthropicProvider(config)
        else:
            raise InvalidConfigurationError(f"Unknown provider type: {provider_type}")
        
        # Store provider and configuration
        self.providers[name] = provider
        self.provider_configs[name] = config
        
        # Register with health monitoring
        self.health_monitor.register_provider(name)
        
        # Add model configurations to cost tracker
        for model_config in provider.models.values():
            self.cost_tracker.add_model_config(model_config)
        
        # Update strategy
        self._update_strategy()
    
    def remove_provider(self, name: str) -> None:
        """Remove a provider from the router."""
        if name in self.providers:
            # Close provider
            provider = self.providers[name]
            if hasattr(provider, 'close'):
                asyncio.create_task(provider.close())
            
            # Remove from collections
            del self.providers[name]
            del self.provider_configs[name]
            
            # Unregister from health monitoring
            self.health_monitor.unregister_provider(name)
            
            # Update strategy
            self._update_strategy()
    
    def _update_strategy(self) -> None:
        """Update the routing strategy based on current providers."""
        if not self.provider_configs:
            self.strategy = None
            return
        
        if self.strategy_name == "priority":
            self.strategy = PriorityStrategy(self.provider_configs)
        elif self.strategy_name == "cost_optimized":
            # Collect all model configurations
            model_configs = {}
            for provider in self.providers.values():
                model_configs.update(provider.models)
            self.strategy = CostOptimizedStrategy(self.provider_configs, model_configs)
        elif self.strategy_name == "round_robin":
            self.strategy = RoundRobinStrategy(self.provider_configs)
        else:
            raise InvalidConfigurationError(f"Unknown strategy: {self.strategy_name}")
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete a text generation request."""
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            **kwargs
        )
        
        return await self._process_request(request)
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a text generation request."""
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            stream=True,
            **kwargs
        )
        
        async for chunk in self._process_stream_request(request):
            yield chunk
    
    async def _process_request(self, request: CompletionRequest) -> LLMResponse:
        """Process a completion request with routing and fallback."""
        start_time = time.time()
        
        # Check cache first
        if not request.stream:
            cached_response = self._check_cache(request)
            if cached_response:
                self.metrics.record_cache_hit(cached_response.provider)
                return cached_response
        
        # Get healthy providers
        healthy_providers = self.health_monitor.get_healthy_providers()
        
        if not healthy_providers:
            raise NoProvidersAvailableError("No healthy providers available")
        
        # Select provider using strategy
        if not self.strategy:
            raise InvalidConfigurationError("No routing strategy configured")
        
        selected_provider_name = self.strategy.select_provider(request, healthy_providers)
        selected_provider = self.providers[selected_provider_name]
        
        # Try the selected provider with retry logic
        try:
            response = await self.retry_handler.execute_with_retry(
                self._execute_provider_request,
                selected_provider,
                request
            )
            
            # Record success
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_success(selected_provider_name, request, response, latency_ms)
            
            # Cache the response
            if not request.stream:
                self._cache_response(request, response)
            
            return response
            
        except Exception as e:
            # Record failure
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_failure(selected_provider_name, request, str(e), latency_ms)
            
            # Try fallback providers
            return await self._try_fallback_providers(request, healthy_providers, selected_provider_name)
    
    async def _process_stream_request(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Process a streaming request."""
        # Get healthy providers
        healthy_providers = self.health_monitor.get_healthy_providers()
        
        if not healthy_providers:
            raise NoProvidersAvailableError("No healthy providers available")
        
        # Select provider using strategy
        if not self.strategy:
            raise InvalidConfigurationError("No routing strategy configured")
        
        selected_provider_name = self.strategy.select_provider(request, healthy_providers)
        selected_provider = self.providers[selected_provider_name]
        
        try:
            async for chunk in selected_provider.stream_complete(request):
                yield chunk
        except Exception as e:
            # For streaming, we can't easily fallback, so just raise the error
            self._record_failure(selected_provider_name, request, str(e), 0)
            raise
    
    async def _execute_provider_request(
        self,
        provider: BaseProvider,
        request: CompletionRequest
    ) -> LLMResponse:
        """Execute a request on a specific provider."""
        return await provider.complete(request)
    
    async def _try_fallback_providers(
        self,
        request: CompletionRequest,
        healthy_providers: List[str],
        failed_provider: str
    ) -> LLMResponse:
        """Try fallback providers if the primary provider fails."""
        # Remove the failed provider from the list
        available_providers = [p for p in healthy_providers if p != failed_provider]
        
        if not available_providers:
            raise NoProvidersAvailableError("No fallback providers available")
        
        # Try each remaining provider
        for provider_name in available_providers:
            try:
                provider = self.providers[provider_name]
                response = await provider.complete(request)
                
                # Record success
                self._record_success(provider_name, request, response, 0)
                
                return response
                
            except Exception as e:
                # Record failure and continue to next provider
                self._record_failure(provider_name, request, str(e), 0)
                continue
        
        # All providers failed
        raise NoProvidersAvailableError("All providers failed")
    
    def _check_cache(self, request: CompletionRequest) -> Optional[LLMResponse]:
        """Check if response is cached."""
        if not request.model:
            return None
        
        # Try to find a cached response
        for provider_name in self.providers.keys():
            cached_response = self.cache.get(
                request.prompt,
                request.model,
                provider_name,
                **request.dict(exclude={'prompt', 'model', 'stream', 'metadata'})
            )
            if cached_response:
                return cached_response
        
        self.metrics.record_cache_miss("unknown")
        return None
    
    def _cache_response(self, request: CompletionRequest, response: LLMResponse) -> None:
        """Cache a response."""
        if not request.model:
            return
        
        self.cache.set(
            response,
            request.prompt,
            request.model,
            response.provider,
            **request.dict(exclude={'prompt', 'model', 'stream', 'metadata'})
        )
    
    def _record_success(
        self,
        provider_name: str,
        request: CompletionRequest,
        response: LLMResponse,
        latency_ms: int
    ) -> None:
        """Record a successful request."""
        # Update health monitor
        self.health_monitor.record_request_result(
            provider_name, True, latency_ms
        )
        
        # Update metrics
        self.metrics.record_request(
            provider_name,
            response.model,
            True,
            latency_ms,
            response.tokens_used,
            response.cost_estimate
        )
        
        # Update cost tracker
        self.cost_tracker.record_cost(
            provider_name,
            response.model,
            response.tokens_used - response.tokens_used // 2,  # Rough estimate
            response.tokens_used // 2,  # Rough estimate
            response.cost_estimate
        )
    
    def _record_failure(
        self,
        provider_name: str,
        request: CompletionRequest,
        error: str,
        latency_ms: int
    ) -> None:
        """Record a failed request."""
        # Update health monitor
        self.health_monitor.record_request_result(
            provider_name, False, latency_ms, error
        )
        
        # Update metrics
        self.metrics.record_request(
            provider_name,
            request.model or "unknown",
            False,
            latency_ms
        )
    
    async def get_stats(self) -> RouterStats:
        """Get router statistics."""
        return self.metrics.get_router_stats()
    
    async def get_cost_summary(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> CostSummary:
        """Get cost summary for a time period."""
        return self.cost_tracker.get_cost_summary(period_start, period_end)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return self.health_monitor.get_all_health_statuses()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers."""
        return {
            name: provider.get_provider_info()
            for name, provider in self.providers.items()
        }
    
    async def close(self) -> None:
        """Close the router and all providers."""
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Close all providers
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        
        # Clear all data
        self.providers.clear()
        self.provider_configs.clear()
        self.cache.clear()
        self.cost_tracker.reset_costs()
        self.metrics.reset_metrics() 