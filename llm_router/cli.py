"""Command-line interface for llm-router."""

import asyncio
import argparse
import json
import os
import sys
from typing import Optional

from .router import LLMRouter


def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Router - Intelligent routing for LLM API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple completion
  llm-router complete "Explain quantum computing"
  
  # Completion with specific model
  llm-router complete "Write a story" --model gpt-4
  
  # Streaming completion
  llm-router stream "Write a poem" --model gpt-3.5-turbo
  
  # Get statistics
  llm-router stats
  
  # Get cost summary
  llm-router costs
  
  # Get health status
  llm-router health
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Complete a text generation request")
    complete_parser.add_argument("prompt", help="Input prompt")
    complete_parser.add_argument("--model", help="Model to use")
    complete_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    complete_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    complete_parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    complete_parser.add_argument("--strategy", default="priority", 
                               choices=["priority", "cost_optimized", "round_robin"],
                               help="Routing strategy")
    complete_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    complete_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream a text generation request")
    stream_parser.add_argument("prompt", help="Input prompt")
    stream_parser.add_argument("--model", help="Model to use")
    stream_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    stream_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    stream_parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    stream_parser.add_argument("--strategy", default="priority",
                             choices=["priority", "cost_optimized", "round_robin"],
                             help="Routing strategy")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get router statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Costs command
    costs_parser = subparsers.add_parser("costs", help="Get cost summary")
    costs_parser.add_argument("--period", choices=["hour", "day", "week", "month", "all"],
                            default="all", help="Time period for cost summary")
    costs_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Get health status")
    health_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Providers command
    providers_parser = subparsers.add_parser("providers", help="Get provider information")
    providers_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    return parser


async def run_complete(args, router: LLMRouter):
    """Run completion command."""
    try:
        response = await router.complete(
            args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        if args.json:
            print(json.dumps({
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "cost_estimate": response.cost_estimate,
                "latency_ms": response.latency_ms,
                "metadata": response.metadata
            }, indent=2))
        else:
            print(f"Provider: {response.provider}")
            print(f"Model: {response.model}")
            print(f"Cost: ${response.cost_estimate:.6f}")
            print(f"Latency: {response.latency_ms}ms")
            print(f"Tokens: {response.tokens_used}")
            print("-" * 50)
            print(response.content)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def run_stream(args, router: LLMRouter):
    """Run streaming command."""
    try:
        print(f"Streaming response from {args.strategy} strategy...")
        print("-" * 50)
        
        async for chunk in router.stream(
            args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "-" * 50)
        print("Streaming completed")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def run_stats(args, router: LLMRouter):
    """Run stats command."""
    try:
        stats = await router.get_stats()
        
        if args.json:
            print(json.dumps(stats.dict(), indent=2))
        else:
            print("Router Statistics:")
            print(f"  Total requests: {stats.total_requests}")
            print(f"  Successful requests: {stats.successful_requests}")
            print(f"  Failed requests: {stats.failed_requests}")
            print(f"  Cache hits: {stats.cache_hits}")
            print(f"  Cache misses: {stats.cache_misses}")
            print(f"  Average latency: {stats.average_latency_ms:.2f}ms")
            
            if stats.provider_stats:
                print("\nProvider Statistics:")
                for provider, provider_stats in stats.provider_stats.items():
                    print(f"  {provider}:")
                    print(f"    Total requests: {provider_stats['total_requests']}")
                    print(f"    Success rate: {provider_stats['success_rate']:.2%}")
                    print(f"    Average latency: {provider_stats['average_latency']:.2f}ms")
                    print(f"    Cache hit rate: {provider_stats['cache_hit_rate']:.2%}")
                    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def run_costs(args, router: LLMRouter):
    """Run costs command."""
    try:
        from datetime import datetime, timedelta
        
        # Calculate period
        now = datetime.utcnow()
        if args.period == "hour":
            period_start = now - timedelta(hours=1)
        elif args.period == "day":
            period_start = now - timedelta(days=1)
        elif args.period == "week":
            period_start = now - timedelta(weeks=1)
        elif args.period == "month":
            period_start = now - timedelta(days=30)
        else:  # all
            period_start = None
        
        cost_summary = await router.get_cost_summary(period_start)
        
        if args.json:
            print(json.dumps(cost_summary.dict(), indent=2))
        else:
            print(f"Cost Summary ({args.period}):")
            print(f"  Total cost: ${cost_summary.total_cost:.6f}")
            print(f"  Total requests: {cost_summary.request_count}")
            
            if cost_summary.provider_costs:
                print("\nProvider Costs:")
                for provider, cost in cost_summary.provider_costs.items():
                    print(f"  {provider}: ${cost:.6f}")
            
            if cost_summary.model_costs:
                print("\nModel Costs:")
                for model, cost in cost_summary.model_costs.items():
                    print(f"  {model}: ${cost:.6f}")
                    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def run_health(args, router: LLMRouter):
    """Run health command."""
    try:
        health_status = await router.get_health_status()
        
        if args.json:
            print(json.dumps({
                provider: status.dict() for provider, status in health_status.items()
            }, indent=2))
        else:
            print("Health Status:")
            for provider, status in health_status.items():
                status_icon = "✅" if status.healthy else "❌"
                print(f"  {status_icon} {provider}:")
                print(f"    Status: {'Healthy' if status.healthy else 'Unhealthy'}")
                print(f"    Success rate: {status.success_rate:.2%}")
                print(f"    Error count: {status.error_count}")
                print(f"    Last check: {status.last_check}")
                if status.response_time_ms:
                    print(f"    Response time: {status.response_time_ms}ms")
                print()
                    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def run_providers(args, router: LLMRouter):
    """Run providers command."""
    try:
        provider_info = router.get_provider_info()
        
        if args.json:
            print(json.dumps(provider_info, indent=2))
        else:
            print("Provider Information:")
            for name, info in provider_info.items():
                print(f"  {name}:")
                print(f"    Type: {info['type']}")
                print(f"    Available models: {', '.join(info['available_models'])}")
                print(f"    Priority: {info['config']['priority']}")
                print(f"    Enabled: {info['config']['enabled']}")
                print()
                    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def create_router(args) -> LLMRouter:
    """Create and configure router instance."""
    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("Warning: No API keys found in environment variables", file=sys.stderr)
        print("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to use providers", file=sys.stderr)
    
    # Create router
    router = LLMRouter(
        strategy=args.strategy,
        cache_ttl=3600,
        retry_attempts=3,
        enable_health_monitoring=True
    )
    
    # Add providers
    if openai_key:
        router.add_provider(
            name="openai",
            provider_type="openai",
            api_key=openai_key,
            priority=1
        )
    
    if anthropic_key:
        router.add_provider(
            name="anthropic",
            provider_type="anthropic",
            api_key=anthropic_key,
            priority=2
        )
    
    return router


async def main():
    """Main CLI function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create router
    router = create_router(args)
    
    try:
        # Execute command
        if args.command == "complete":
            await run_complete(args, router)
        elif args.command == "stream":
            await run_stream(args, router)
        elif args.command == "stats":
            await run_stats(args, router)
        elif args.command == "costs":
            await run_costs(args, router)
        elif args.command == "health":
            await run_health(args, router)
        elif args.command == "providers":
            await run_providers(args, router)
    finally:
        # Clean up
        await router.close()


if __name__ == "__main__":
    asyncio.run(main()) 