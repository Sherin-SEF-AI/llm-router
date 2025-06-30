# examples/basic_usage.py

"""Basic usage example for llm-router."""

import asyncio
import os
from llm_router import LLMRouter, RouterConfig
from llm_router.providers import OpenAIProvider, AnthropicProvider


async def main():
    """Demonstrate basic usage of llm-router."""
    
    # Create router instance
    config = RouterConfig(
        providers=[
            OpenAIProvider(api_key="your-openai-key"),
            AnthropicProvider(api_key="your-anthropic-key")
        ],
        strategy="priority"
    )
    router = LLMRouter(config)
    
    # Simple completion
    print("=== Basic Completion ===")
    try:
        response = await router.chat_completion(
            messages=[{"role": "user", "content": "Tell me a joke."}],
            model="gpt-4"
        )
        print("Response:", response.content)
        print("Provider:", response.provider)
        print("Cost:", response.cost)
    except Exception as e:
        print(f"Error: {e}")
    
    # Streaming completion
    print("\n=== Streaming Completion ===")
    try:
        print("Streaming response:")
        async for chunk in router.stream(
            "Write a short story about AI",
            model="gpt-3.5-turbo"
        ):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error: {e}")
    
    # Get statistics
    print("\n=== Statistics ===")
    try:
        stats = await router.get_stats()
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        print(f"Cache hits: {stats.cache_hits}")
        print(f"Cache misses: {stats.cache_misses}")
        print(f"Average latency: {stats.average_latency_ms:.2f}ms")
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    # Get cost summary
    print("\n=== Cost Summary ===")
    try:
        cost_summary = await router.get_cost_summary()
        print(f"Total cost: ${cost_summary.total_cost:.6f}")
        print(f"Total requests: {cost_summary.request_count}")
        print("Provider costs:")
        for provider, cost in cost_summary.provider_costs.items():
            print(f"  {provider}: ${cost:.6f}")
    except Exception as e:
        print(f"Error getting cost summary: {e}")
    
    # Get health status
    print("\n=== Health Status ===")
    try:
        health_status = await router.get_health_status()
        for provider, status in health_status.items():
            print(f"{provider}: {'Healthy' if status.healthy else 'Unhealthy'}")
            print(f"  Success rate: {status.success_rate:.2%}")
            print(f"  Error count: {status.error_count}")
    except Exception as e:
        print(f"Error getting health status: {e}")
    
    # Clean up
    await router.close()


if __name__ == "__main__":
    asyncio.run(main()) 