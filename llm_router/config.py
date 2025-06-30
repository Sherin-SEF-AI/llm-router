# llm_router/config.py

"""Configuration management for llm-router."""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field, validator


class RouterConfig(BaseSettings):
    """Main configuration for the LLM router."""
    
    # Routing configuration
    default_strategy: str = Field(default="priority", description="Default routing strategy")
    enable_health_monitoring: bool = Field(default=True, description="Enable health monitoring")
    health_check_interval: int = Field(default=300, description="Health check interval in seconds")
    
    # Caching configuration
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    cache_key_prefix: str = Field(default="llm_router", description="Cache key prefix")
    
    # Retry configuration
    retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_base_delay: float = Field(default=1.0, description="Base delay between retries")
    retry_max_delay: float = Field(default=60.0, description="Maximum delay between retries")
    retry_exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    retry_jitter: bool = Field(default=True, description="Add jitter to retry delays")
    
    # Timeout configuration
    default_timeout: int = Field(default=30, description="Default request timeout")
    health_check_timeout: int = Field(default=10, description="Health check timeout")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    
    # Provider configurations
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(None, description="OpenAI base URL")
    openai_priority: int = Field(default=1, description="OpenAI routing priority")
    openai_enabled: bool = Field(default=True, description="Enable OpenAI provider")
    
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    anthropic_base_url: Optional[str] = Field(None, description="Anthropic base URL")
    anthropic_priority: int = Field(default=2, description="Anthropic routing priority")
    anthropic_enabled: bool = Field(default=True, description="Enable Anthropic provider")
    
    # Cost tracking
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    cost_tracking_period: int = Field(default=86400, description="Cost tracking period in seconds")
    
    # Metrics and monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_export_interval: int = Field(default=300, description="Metrics export interval")
    
    class Config:
        env_prefix = "LLM_ROUTER_"
        case_sensitive = False
    
    @validator("default_strategy")
    def validate_strategy(cls, v):
        valid_strategies = ["priority", "cost_optimized", "round_robin"]
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get provider configurations as a dictionary."""
        configs = {}
        
        if self.openai_enabled and self.openai_api_key:
            configs["openai"] = {
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "priority": self.openai_priority,
                "timeout": self.default_timeout,
                "max_retries": self.retry_attempts,
            }
        
        if self.anthropic_enabled and self.anthropic_api_key:
            configs["anthropic"] = {
                "api_key": self.anthropic_api_key,
                "base_url": self.anthropic_base_url,
                "priority": self.anthropic_priority,
                "timeout": self.default_timeout,
                "max_retries": self.retry_attempts,
            }
        
        return configs


def load_config_from_env() -> RouterConfig:
    """Load configuration from environment variables."""
    return RouterConfig()


def load_config_from_file(file_path: str) -> RouterConfig:
    """Load configuration from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Load environment variables from file
    from dotenv import load_dotenv
    load_dotenv(file_path)
    
    return RouterConfig() 