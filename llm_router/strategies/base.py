# llm_router/strategies/base.py

"""Abstract base strategy interface for llm-router."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..types import ProviderConfig, CompletionRequest
from ..exceptions import NoProvidersAvailableError


class BaseStrategy(ABC):
    """Abstract base class for routing strategies."""
    
    def __init__(self, providers: Dict[str, ProviderConfig]):
        """Initialize the strategy with available providers."""
        self.providers = providers
    
    @abstractmethod
    def select_provider(
        self,
        request: CompletionRequest,
        healthy_providers: List[str]
    ) -> str:
        """Select a provider for the given request."""
        pass
    
    def get_available_providers(self, healthy_providers: List[str]) -> List[str]:
        """Get list of available providers that are healthy."""
        available = []
        for provider_name in healthy_providers:
            if provider_name in self.providers:
                config = self.providers[provider_name]
                if config.enabled:
                    available.append(provider_name)
        return available
    
    def validate_providers(self, healthy_providers: List[str]) -> None:
        """Validate that there are available providers."""
        available = self.get_available_providers(healthy_providers)
        if not available:
            raise NoProvidersAvailableError("No healthy and enabled providers available")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "",
            "providers": list(self.providers.keys())
        } 