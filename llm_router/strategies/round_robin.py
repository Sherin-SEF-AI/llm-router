# llm_router/strategies/round_robin.py

"""Round-robin routing strategy for llm-router."""

from typing import List, Dict, Any
from .base import BaseStrategy
from ..types import ProviderConfig, CompletionRequest


class RoundRobinStrategy(BaseStrategy):
    """Routes requests in a round-robin fashion across healthy providers."""
    
    def __init__(self, providers: Dict[str, ProviderConfig]):
        """Initialize the round-robin strategy."""
        super().__init__(providers)
        self.current_index = 0
    
    def select_provider(
        self,
        request: CompletionRequest,
        healthy_providers: List[str]
    ) -> str:
        """Select the next provider in round-robin order."""
        self.validate_providers(healthy_providers)
        
        available_providers = self.get_available_providers(healthy_providers)
        
        if not available_providers:
            raise NoProvidersAvailableError("No healthy and enabled providers available")
        
        # Get the next provider in round-robin order
        selected_provider = available_providers[self.current_index % len(available_providers)]
        
        # Move to next provider
        self.current_index += 1
        
        return selected_provider
    
    def reset_counter(self) -> None:
        """Reset the round-robin counter."""
        self.current_index = 0
    
    def get_current_index(self) -> int:
        """Get the current round-robin index."""
        return self.current_index 