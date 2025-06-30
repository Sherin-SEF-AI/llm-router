# llm-router

Intelligent LLM API routing with automatic fallbacks, cost optimization, and monitoring.

## Features
- Multi-provider support (OpenAI, Anthropic, easily extensible)
- Intelligent routing (priority, cost, performance-based)
- Automatic fallbacks and retries
- Cost tracking and optimization
- Response caching (in-memory, TTL, optional Redis)
- Health monitoring and metrics
- Async and streaming support
- Easy configuration (env or code)

## Installation
```bash
pip install llm-router
```

## Quickstart
```python
from llm_router import LLMRouter

router = LLMRouter()
router.add_provider("openai", api_key="sk-...", priority=1)
router.add_provider("anthropic", api_key="sk-ant-...", priority=2)

response = await router.complete("Explain quantum computing")
print(response.content, response.cost_estimate, response.provider)
```

See full documentation at [https://llm-router.readthedocs.io](https://llm-router.readthedocs.io) 