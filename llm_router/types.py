# llm_router/types.py

"""Type definitions and models for llm-router."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    PRIORITY = "priority"
    COST_OPTIMIZED = "cost_optimized"
    ROUND_ROBIN = "round_robin"


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str = Field(..., description="The generated text content")
    model: str = Field(..., description="The model used for generation")
    provider: str = Field(..., description="The provider that generated the response")
    tokens_used: int = Field(..., description="Number of tokens used")
    cost_estimate: float = Field(..., description="Estimated cost in USD")
    latency_ms: int = Field(..., description="Response time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    name: str = Field(..., description="Provider name")
    api_key: str = Field(..., description="API key for the provider")
    priority: int = Field(default=1, description="Routing priority (lower = higher priority)")
    weight: float = Field(default=1.0, description="Weight for weighted routing strategies")
    base_url: Optional[str] = Field(None, description="Custom base URL for the API")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    enabled: bool = Field(default=True, description="Whether the provider is enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider that hosts this model")
    max_tokens: int = Field(default=4096, description="Maximum tokens for this model")
    input_cost_per_1k: float = Field(..., description="Cost per 1k input tokens")
    output_cost_per_1k: float = Field(..., description="Cost per 1k output tokens")
    context_length: int = Field(..., description="Maximum context length")
    available: bool = Field(default=True, description="Whether the model is available")


class CacheConfig(BaseModel):
    """Configuration for response caching."""
    enabled: bool = Field(default=True, description="Whether caching is enabled")
    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum number of cached items")
    key_prefix: str = Field(default="llm_router", description="Cache key prefix")


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    max_delay: float = Field(default=60.0, description="Maximum delay between retries")
    exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Whether to add jitter to delays")


class HealthStatus(BaseModel):
    """Health status of a provider."""
    provider: str = Field(..., description="Provider name")
    healthy: bool = Field(..., description="Whether the provider is healthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: Optional[int] = Field(None, description="Last response time")
    error_count: int = Field(default=0, description="Number of consecutive errors")
    success_rate: float = Field(default=1.0, description="Success rate (0.0 to 1.0)")


class CostSummary(BaseModel):
    """Summary of costs across providers."""
    total_cost: float = Field(..., description="Total cost in USD")
    provider_costs: Dict[str, float] = Field(..., description="Costs per provider")
    model_costs: Dict[str, float] = Field(..., description="Costs per model")
    period_start: datetime = Field(..., description="Start of the cost period")
    period_end: datetime = Field(..., description="End of the cost period")
    request_count: int = Field(..., description="Total number of requests")


class RouterStats(BaseModel):
    """Statistics for the router."""
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")
    average_latency_ms: float = Field(default=0.0, description="Average response latency")
    provider_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Stats per provider")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Stats creation timestamp")


class CompletionRequest(BaseModel):
    """Request for LLM completion."""
    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")


# Type aliases for convenience
ProviderName = str
ModelName = str
CacheKey = str 