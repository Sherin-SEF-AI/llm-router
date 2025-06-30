# tests/test_router.py

"""Tests for the main LLMRouter class."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from llm_router import LLMRouter
from llm_router.types import LLMResponse, CompletionRequest
from llm_router.exceptions import NoProvidersAvailableError, InvalidConfigurationError


class TestLLMRouter:
    """Test cases for LLMRouter."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance for testing."""
        return LLMRouter(
            strategy="priority",
            cache_ttl=3600,
            retry_attempts=2,
            enable_health_monitoring=False
        )
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.name = "test_provider"
        provider.models = {
            "test-model": Mock(
                name="test-model",
                provider="test_provider",
                available=True
            )
        }
        provider.complete = AsyncMock()
        provider.stream_complete = AsyncMock()
        provider.health_check = AsyncMock(return_value=True)
        provider.estimate_cost = Mock(return_value=0.001)
        provider.get_provider_info = Mock(return_value={"name": "test_provider"})
        return provider
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.strategy_name == "priority"
        assert router.providers == {}
        assert router.provider_configs == {}
        assert router.strategy is None
    
    def test_add_provider_invalid_type(self, router):
        """Test adding provider with invalid type."""
        with pytest.raises(InvalidConfigurationError):
            router.add_provider(
                name="test",
                provider_type="invalid",
                api_key="test-key"
            )
    
    @patch('llm_router.providers.openai.OpenAIProvider')
    def test_add_openai_provider(self, mock_openai_class, router):
        """Test adding OpenAI provider."""
        mock_provider = Mock()
        mock_openai_class.return_value = mock_provider
        
        router.add_provider(
            name="openai",
            provider_type="openai",
            api_key="test-key",
            priority=1
        )
        
        assert "openai" in router.providers
        assert "openai" in router.provider_configs
        mock_openai_class.assert_called_once()
    
    @patch('llm_router.providers.anthropic.AnthropicProvider')
    def test_add_anthropic_provider(self, mock_anthropic_class, router):
        """Test adding Anthropic provider."""
        mock_provider = Mock()
        mock_anthropic_class.return_value = mock_provider
        
        router.add_provider(
            name="anthropic",
            provider_type="anthropic",
            api_key="test-key",
            priority=2
        )
        
        assert "anthropic" in router.providers
        assert "anthropic" in router.provider_configs
        mock_anthropic_class.assert_called_once()
    
    def test_remove_provider(self, router, mock_provider):
        """Test removing a provider."""
        router.providers["test"] = mock_provider
        router.provider_configs["test"] = Mock()
        
        router.remove_provider("test")
        
        assert "test" not in router.providers
        assert "test" not in router.provider_configs
    
    @pytest.mark.asyncio
    async def test_complete_no_providers(self, router):
        """Test completion with no providers."""
        with pytest.raises(NoProvidersAvailableError):
            await router.complete("test prompt")
    
    @pytest.mark.asyncio
    async def test_complete_with_provider(self, router, mock_provider):
        """Test completion with a provider."""
        # Setup
        router.providers["test"] = mock_provider
        router.provider_configs["test"] = Mock()
        router.health_monitor.health_status["test"] = Mock(healthy=True)
        router._update_strategy()
        
        # Mock response
        mock_response = LLMResponse(
            content="test response",
            model="test-model",
            provider="test_provider",
            tokens_used=10,
            cost_estimate=0.001,
            latency_ms=100
        )
        mock_provider.complete.return_value = mock_response
        
        # Test
        response = await router.complete("test prompt", model="test-model")
        
        assert response.content == "test response"
        assert response.provider == "test_provider"
        mock_provider.complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stream_no_providers(self, router):
        """Test streaming with no providers."""
        with pytest.raises(NoProvidersAvailableError):
            async for _ in router.stream("test prompt"):
                pass
    
    @pytest.mark.asyncio
    async def test_stream_with_provider(self, router, mock_provider):
        """Test streaming with a provider."""
        # Setup
        router.providers["test"] = mock_provider
        router.provider_configs["test"] = Mock()
        router.health_monitor.health_status["test"] = Mock(healthy=True)
        router._update_strategy()
        
        # Mock streaming response
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        mock_provider.stream_complete.return_value = mock_stream()
        
        # Test
        chunks = []
        async for chunk in router.stream("test prompt", model="test-model"):
            chunks.append(chunk)
        
        assert chunks == ["chunk1", "chunk2", "chunk3"]
        mock_provider.stream_complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, router):
        """Test getting router statistics."""
        stats = await router.get_stats()
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
    
    @pytest.mark.asyncio
    async def test_get_cost_summary(self, router):
        """Test getting cost summary."""
        cost_summary = await router.get_cost_summary()
        
        assert cost_summary.total_cost == 0.0
        assert cost_summary.request_count == 0
        assert cost_summary.provider_costs == {}
        assert cost_summary.model_costs == {}
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, router):
        """Test getting health status."""
        health_status = await router.get_health_status()
        
        assert health_status == {}
    
    def test_get_provider_info(self, router):
        """Test getting provider information."""
        provider_info = router.get_provider_info()
        
        assert provider_info == {}
    
    @pytest.mark.asyncio
    async def test_close(self, router, mock_provider):
        """Test closing the router."""
        router.providers["test"] = mock_provider
        
        await router.close()
        
        # Verify providers are cleared
        assert router.providers == {}
        assert router.provider_configs == {}
    
    def test_update_strategy_priority(self, router):
        """Test updating strategy to priority."""
        router.provider_configs["test"] = Mock()
        router.strategy_name = "priority"
        
        router._update_strategy()
        
        assert router.strategy is not None
        assert router.strategy.__class__.__name__ == "PriorityStrategy"
    
    def test_update_strategy_cost_optimized(self, router):
        """Test updating strategy to cost optimized."""
        router.provider_configs["test"] = Mock()
        router.providers["test"] = Mock(models={})
        router.strategy_name = "cost_optimized"
        
        router._update_strategy()
        
        assert router.strategy is not None
        assert router.strategy.__class__.__name__ == "CostOptimizedStrategy"
    
    def test_update_strategy_round_robin(self, router):
        """Test updating strategy to round robin."""
        router.provider_configs["test"] = Mock()
        router.strategy_name = "round_robin"
        
        router._update_strategy()
        
        assert router.strategy is not None
        assert router.strategy.__class__.__name__ == "RoundRobinStrategy"
    
    def test_update_strategy_invalid(self, router):
        """Test updating strategy with invalid name."""
        router.provider_configs["test"] = Mock()
        router.strategy_name = "invalid"
        
        with pytest.raises(InvalidConfigurationError):
            router._update_strategy()
    
    def test_update_strategy_no_providers(self, router):
        """Test updating strategy with no providers."""
        router._update_strategy()
        
        assert router.strategy is None 