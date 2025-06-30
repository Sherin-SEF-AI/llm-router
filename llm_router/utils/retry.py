# llm_router/utils/retry.py

"""Retry logic for llm-router."""

import asyncio
import random
import time
from typing import Any, Callable, Optional, TypeVar
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import structlog

from ..exceptions import RetryExhaustedError, ProviderError
from ..types import RetryConfig

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry handler with configuration."""
        self.config = config
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        if attempt <= 0:
            return 0
        
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(0, 0.1 * delay)
            delay += jitter
        
        # Cap at maximum delay
        return min(delay, self.config.max_delay)
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute a function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - return result
                if attempt > 1:
                    logger.info(
                        "Retry successful",
                        attempt=attempt,
                        max_attempts=self.config.max_attempts
                    )
                return result
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain exceptions
                if isinstance(e, (ValueError, TypeError)):
                    raise e
                
                # Log the error
                logger.warning(
                    "Retry attempt failed",
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                # If this is the last attempt, don't wait
                if attempt == self.config.max_attempts:
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                logger.info(
                    "Waiting before retry",
                    attempt=attempt + 1,
                    delay=delay
                )
                
                # Wait before next attempt
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise RetryExhaustedError(
            provider="unknown",
            attempts=self.config.max_attempts,
            last_error=last_exception
        )
    
    def create_retry_decorator(self, retryable_exceptions: tuple = None):
        """Create a retry decorator for functions."""
        if retryable_exceptions is None:
            retryable_exceptions = (Exception,)
        
        return retry(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait_exponential(
                multiplier=self.config.base_delay,
                max=self.config.max_delay,
                exp=self.config.exponential_base
            ),
            retry=retry_if_exception_type(retryable_exceptions),
            before_sleep=before_sleep_log(logger, structlog.stdlib.DEBUG),
            after=after_log(logger, structlog.stdlib.DEBUG)
        )


def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = None
):
    """Decorator for retrying functions on failure."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter
    )
    
    handler = RetryHandler(config)
    return handler.create_retry_decorator(retryable_exceptions) 