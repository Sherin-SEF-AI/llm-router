# LLM Router AI

Intelligent routing of LLM API calls across multiple providers with automatic fallbacks, cost optimization, caching, health monitoring, and retry logic.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, and extensible for other providers
- **Intelligent Routing**: Priority-based, cost-optimized, and round-robin strategies
- **Automatic Fallbacks**: Seamless failover between providers
- **Cost Optimization**: Track and optimize API usage costs
- **Caching**: Redis-based response caching for improved performance
- **Health Monitoring**: Real-time provider health checks
- **Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Async Support**: Full async/await support for high-performance applications
- **Streaming**: Support for streaming responses
- **CLI Interface**: Command-line tool for easy integration
- **Metrics**: Prometheus-compatible metrics collection

## Installation

```bash
pip install llm-router-ai
```

## Quick Start

```python
from llm_router import LLMRouter, RouterConfig
from llm_router.providers import OpenAIProvider, AnthropicProvider

# Configure providers
config = RouterConfig(
    providers=[
        OpenAIProvider(api_key="your-openai-key"),
        AnthropicProvider(api_key="your-anthropic-key")
    ],
    strategy="priority"
)

# Create router
router = LLMRouter(config)

# Make a request
response = await router.chat_completion(
    messages=[{"role": "user", "content": "Hello, world!"}],
    model="gpt-4"
)
```

## Documentation

For detailed documentation, examples, and API reference, visit our [GitHub repository](https://github.com/Sherin-SEF-AI/llm-router).

## Author

**Sherin Joseph** - [LinkedIn](https://www.linkedin.com/in/sherin-roy-deepmost/)

## License

MIT License - see [LICENSE](LICENSE) file for details. 