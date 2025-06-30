# examples/advanced_config.py

"""Advanced configuration example for llm-router."""

import asyncio
import os
from datetime import datetime, timedelta
from llm_router import LLMRouter, RouterConfig
from llm_router.providers import OpenAIProvider
from llm_router.utils.cache import RedisCache
from llm_router.utils.retry import RetryConfig
from llm_router.utils.monitor import HealthMonitorConfig


async def main():
    """Demonstrate advanced configuration of llm-router."""
    
    # Create router with advanced configuration
    config = RouterConfig(
        providers=[OpenAIProvider(api_key="your-openai-key")],
        strategy="cost_optimized",
        cache=RedisCache(host="localhost", port=6379, db=0),
        retry=RetryConfig(max_attempts=3, backoff_factor=2),
        monitor=HealthMonitorConfig(interval_seconds=30)
    )
    router = LLMRouter(config)
    response = await router.chat_completion(
        messages=[{"role": "user", "content": "What's the weather today?"}],
        model="gpt-3.5-turbo"
    )
    print(response.content)
    
    # Test different routing strategies
    print("=== Testing Different Strategies ===")
    
    # Test cost-optimized routing
    print("\n--- Cost-Optimized Routing ---")
    try:
        response = await router.complete(
            "Explain machine learning",
            model="gpt-3.5-turbo"  # This should route to the cheapest provider
        )
        print(f"Provider: {response.provider}")
        print(f"Cost: ${response.cost_estimate:.6f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Switch to priority strategy
    router.strategy_name = "priority"
    router._update_strategy()
    
    print("\n--- Priority Routing ---")
    try:
        response = await router.complete(
            "Explain machine learning",
            model="gpt-4"  # This should route to highest priority provider
        )
        print(f"Provider: {response.provider}")
        print(f"Cost: ${response.cost_estimate:.6f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Switch to round-robin strategy
    router.strategy_name = "round_robin"
    router._update_strategy()
    
    print("\n--- Round-Robin Routing ---")
    for i in range(3):
        try:
            response = await router.complete(
                f"Request {i+1}: Explain AI",
                model="gpt-3.5-turbo"
            )
            print(f"Request {i+1} -> Provider: {response.provider}")
        except Exception as e:
            print(f"Request {i+1} -> Error: {e}")
    
    # Test caching
    print("\n=== Testing Caching ===")
    prompt = "What is the capital of France?"
    
    # First request (cache miss)
    print("First request (should be cache miss):")
    start_time = datetime.now()
    response1 = await router.complete(prompt, model="gpt-3.5-turbo")
    time1 = (datetime.now() - start_time).total_seconds()
    print(f"Time: {time1:.3f}s, Provider: {response1.provider}")
    
    # Second request (cache hit)
    print("Second request (should be cache hit):")
    start_time = datetime.now()
    response2 = await router.complete(prompt, model="gpt-3.5-turbo")
    time2 = (datetime.now() - start_time).total_seconds()
    print(f"Time: {time2:.3f}s, Provider: {response2.provider}")
    print(f"Cache hit: {time2 < time1}")
    
    # Test concurrent requests
    print("\n=== Testing Concurrent Requests ===")
    
    async def make_request(i: int):
        try:
            response = await router.complete(
                f"Concurrent request {i}: Explain quantum physics",
                model="gpt-3.5-turbo"
            )
            return f"Request {i} -> {response.provider}"
        except Exception as e:
            return f"Request {i} -> Error: {e}"
    
    # Make 5 concurrent requests
    tasks = [make_request(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)
    
    # Test cost tracking over time
    print("\n=== Cost Tracking Over Time ===")
    
    # Make several requests to accumulate costs
    for i in range(3):
        try:
            await router.complete(
                f"Cost tracking request {i+1}: Write a short story",
                model="gpt-3.5-turbo",
                max_tokens=100
            )
        except Exception as e:
            print(f"Error in cost tracking request {i+1}: {e}")
    
    # Get cost summary for different periods
    try:
        # Last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        cost_summary_hour = await router.get_cost_summary(hour_ago)
        print(f"Cost in last hour: ${cost_summary_hour.total_cost:.6f}")
        
        # Last day
        day_ago = datetime.now() - timedelta(days=1)
        cost_summary_day = await router.get_cost_summary(day_ago)
        print(f"Cost in last day: ${cost_summary_day.total_cost:.6f}")
        
        # All time
        cost_summary_all = await router.get_cost_summary()
        print(f"Total cost: ${cost_summary_all.total_cost:.6f}")
        
    except Exception as e:
        print(f"Error getting cost summary: {e}")
    
    # Test health monitoring
    print("\n=== Health Monitoring ===")
    try:
        health_status = await router.get_health_status()
        for provider, status in health_status.items():
            print(f"{provider}:")
            print(f"  Healthy: {status.healthy}")
            print(f"  Success rate: {status.success_rate:.2%}")
            print(f"  Error count: {status.error_count}")
            print(f"  Last check: {status.last_check}")
            if status.response_time_ms:
                print(f"  Response time: {status.response_time_ms}ms")
    except Exception as e:
        print(f"Error getting health status: {e}")
    
    # Test provider performance metrics
    print("\n=== Provider Performance Metrics ===")
    try:
        stats = await router.get_stats()
        print("Provider statistics:")
        for provider, provider_stats in stats.provider_stats.items():
            print(f"  {provider}:")
            print(f"    Total requests: {provider_stats['total_requests']}")
            print(f"    Success rate: {provider_stats['success_rate']:.2%}")
            print(f"    Average latency: {provider_stats['average_latency']:.2f}ms")
            print(f"    Cache hit rate: {provider_stats['cache_hit_rate']:.2%}")
    except Exception as e:
        print(f"Error getting provider stats: {e}")
    
    # Test error handling and fallback
    print("\n=== Error Handling and Fallback ===")
    
    # This will test fallback if one provider fails
    try:
        response = await router.complete(
            "Test fallback mechanism with a long prompt that might cause issues",
            model="gpt-3.5-turbo",
            max_tokens=500
        )
        print(f"Fallback test successful -> Provider: {response.provider}")
    except Exception as e:
        print(f"Fallback test failed: {e}")
    
    # Clean up
    await router.close()
    print("\n=== Router closed ===")


if __name__ == "__main__":
    asyncio.run(main()) 