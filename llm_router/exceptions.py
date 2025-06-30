# llm_router/exceptions.py

"""Custom exceptions for llm-router."""

from typing import Optional


class LLMRouterError(Exception):
    """Base exception for all llm-router errors."""
    pass


class NoProvidersAvailableError(LLMRouterError):
    """Raised when no providers are available for routing."""
    pass


class ProviderTimeoutError(LLMRouterError):
    """Raised when a provider times out."""
    pass


class InvalidConfigurationError(LLMRouterError):
    """Raised when configuration is invalid."""
    pass


class RateLimitExceededError(LLMRouterError):
    """Raised when rate limit is exceeded."""
    pass


class ProviderError(LLMRouterError):
    """Raised when a provider encounters an error."""
    def __init__(self, provider: str, message: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider '{provider}' error: {message}")


class ModelNotAvailableError(LLMRouterError):
    """Raised when a requested model is not available."""
    def __init__(self, model: str, provider: Optional[str] = None):
        self.model = model
        self.provider = provider
        message = f"Model '{model}' is not available"
        if provider:
            message += f" on provider '{provider}'"
        super().__init__(message)


class CacheError(LLMRouterError):
    """Raised when cache operations fail."""
    pass


class RetryExhaustedError(LLMRouterError):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, provider: str, attempts: int, last_error: Optional[Exception] = None):
        self.provider = provider
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted for provider '{provider}' after {attempts} attempts")


class StrategyError(LLMRouterError):
    """Raised when routing strategy encounters an error."""
    pass 