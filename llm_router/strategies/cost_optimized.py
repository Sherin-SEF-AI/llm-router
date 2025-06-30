# llm_router/strategies/cost_optimized.py

"""Cost-optimized routing strategy for llm-router."""

from typing import List, Dict, Any
from .base import BaseStrategy
from ..types import ProviderConfig, CompletionRequest, ModelConfig
from ..exceptions import NoProvidersAvailableError


class CostOptimizedStrategy(BaseStrategy):
    """Routes requests to the cheapest available provider for the given model."""
    
    def __init__(self, providers: Dict[str, ProviderConfig], model_configs: Dict[str, ModelConfig]):
        """Initialize with providers and model configurations."""
        super().__init__(providers)
        self.model_configs = model_configs
    
    def select_provider(
        self,
        request: CompletionRequest,
        healthy_providers: List[str]
    ) -> str:
        """Select the cheapest provider for the requested model."""
        self.validate_providers(healthy_providers)
        
        available_providers = self.get_available_providers(healthy_providers)
        
        # If no specific model requested, use priority strategy as fallback
        if not request.model:
            return self._select_by_priority(available_providers)
        
        # Find the cheapest provider for the requested model
        cheapest_provider = None
        lowest_cost = float('inf')
        
        for provider_name in available_providers:
            # Check if this provider has the requested model
            model_config = self._get_model_config(provider_name, request.model)
            if model_config and model_config.available:
                # Calculate estimated cost for this model
                estimated_cost = self._estimate_request_cost(request, model_config)
                
                if estimated_cost < lowest_cost:
                    lowest_cost = estimated_cost
                    cheapest_provider = provider_name
        
        if cheapest_provider is None:
            # Fallback to priority strategy if no provider has the requested model
            return self._select_by_priority(available_providers)
        
        return cheapest_provider
    
    def _get_model_config(self, provider_name: str, model_name: str) -> ModelConfig:
        """Get model configuration for a provider and model."""
        # This would typically come from the provider's model registry
        # For now, we'll use a simplified approach
        for config in self.model_configs.values():
            if config.provider == provider_name and config.name == model_name:
                return config
        return None
    
    def _estimate_request_cost(self, request: CompletionRequest, model_config: ModelConfig) -> float:
        """Estimate the cost of a request for a given model."""
        # Simple cost estimation based on prompt length
        estimated_input_tokens = len(request.prompt) // 4  # Rough token estimation
        estimated_output_tokens = request.max_tokens or 100  # Default output length
        
        input_cost = (estimated_input_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (estimated_output_tokens / 1000) * model_config.output_cost_per_1k
        
        return input_cost + output_cost
    
    def _select_by_priority(self, available_providers: List[str]) -> str:
        """Fallback to priority-based selection."""
        sorted_providers = sorted(
            available_providers,
            key=lambda p: self.providers[p].priority
        )
        return sorted_providers[0]
    
    def get_cost_estimates(self, request: CompletionRequest) -> Dict[str, float]:
        """Get cost estimates for all available providers."""
        estimates = {}
        
        for provider_name, provider_config in self.providers.items():
            if not provider_config.enabled:
                continue
            
            if not request.model:
                estimates[provider_name] = 0.0
                continue
            
            model_config = self._get_model_config(provider_name, request.model)
            if model_config and model_config.available:
                estimates[provider_name] = self._estimate_request_cost(request, model_config)
            else:
                estimates[provider_name] = float('inf')
        
        return estimates 