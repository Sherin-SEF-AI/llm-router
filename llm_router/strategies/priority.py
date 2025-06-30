# llm_router/strategies/priority.py

"""Priority-based routing strategy for llm-router."""

from typing import List, Dict, Any
from .base import BaseStrategy
from ..types import ProviderConfig, CompletionRequest
from ..exceptions import NoProvidersAvailableError


class PriorityStrategy(BaseStrategy):
    """Routes requests to the highest priority healthy provider."""
    
    def select_provider(
        self,
        request: CompletionRequest,
        healthy_providers: List[str]
    ) -> str:
        """Select the highest priority provider from healthy providers."""
        self.validate_providers(healthy_providers)
        
        available_providers = self.get_available_providers(healthy_providers)
        
        # Sort by priority (lower number = higher priority)
        sorted_providers = sorted(
            available_providers,
            key=lambda p: self.providers[p].priority
        )
        
        return sorted_providers[0]
    
    def get_provider_priorities(self) -> Dict[str, int]:
        """Get the priority of each provider."""
        return {
            name: config.priority
            for name, config in self.providers.items()
        } 