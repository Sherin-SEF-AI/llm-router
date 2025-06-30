# examples/streaming_example.py

"""Streaming example for llm-router."""

import asyncio
import os
import time
from llm_router import LLMRouter


async def streaming_demo():
    """Demonstrate streaming capabilities of llm-router."""
    
    # Create router
    router = LLMRouter(
        strategy="priority",
        cache_ttl=3600,
        retry_attempts=3
    )
    
    # Add providers
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
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
    
    print("=== Streaming Demo ===")
    
    # Basic streaming
    print("\n--- Basic Streaming ---")
    prompt = "Write a short story about a robot learning to paint"
    
    print(f"Prompt: {prompt}")
    print("Response:")
    
    try:
        start_time = time.time()
        full_response = ""
        
        async for chunk in router.stream(prompt, model="gpt-3.5-turbo"):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        end_time = time.time()
        print(f"\n\nStreaming completed in {end_time - start_time:.2f} seconds")
        print(f"Total characters: {len(full_response)}")
        
    except Exception as e:
        print(f"\nError during streaming: {e}")
    
    # Interactive streaming
    print("\n--- Interactive Streaming ---")
    
    async def interactive_stream():
        """Interactive streaming where user can see response being generated."""
        prompt = "Explain the concept of artificial intelligence step by step"
        
        print(f"Prompt: {prompt}")
        print("Generating response...")
        print("-" * 50)
        
        try:
            chunk_count = 0
            start_time = time.time()
            
            async for chunk in router.stream(prompt, model="gpt-3.5-turbo"):
                print(chunk, end="", flush=True)
                chunk_count += 1
                
                # Add a small delay to simulate real-time generation
                await asyncio.sleep(0.01)
            
            end_time = time.time()
            print(f"\n\nGeneration completed!")
            print(f"Chunks received: {chunk_count}")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nError: {e}")
    
    await interactive_stream()
    
    # Streaming with different models
    print("\n--- Streaming with Different Models ---")
    
    models_to_test = ["gpt-3.5-turbo", "gpt-4"]
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        prompt = f"Write a haiku about {model}"
        
        try:
            print(f"Prompt: {prompt}")
            print("Response:")
            
            start_time = time.time()
            async for chunk in router.stream(prompt, model=model):
                print(chunk, end="", flush=True)
            
            end_time = time.time()
            print(f"\nCompleted in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    # Streaming with custom parameters
    print("\n--- Streaming with Custom Parameters ---")
    
    try:
        prompt = "Create a creative story about time travel"
        
        print(f"Prompt: {prompt}")
        print("Response (with custom temperature and max_tokens):")
        
        start_time = time.time()
        async for chunk in router.stream(
            prompt,
            model="gpt-3.5-turbo",
            temperature=0.8,  # More creative
            max_tokens=200,   # Limit length
            top_p=0.9
        ):
            print(chunk, end="", flush=True)
        
        end_time = time.time()
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Concurrent streaming
    print("\n--- Concurrent Streaming ---")
    
    async def stream_concurrent(prompt: str, model: str, request_id: int):
        """Stream a request concurrently."""
        try:
            print(f"Request {request_id} starting...")
            response = ""
            
            async for chunk in router.stream(prompt, model=model):
                response += chunk
            
            print(f"Request {request_id} completed: {len(response)} characters")
            return f"Request {request_id} -> {len(response)} chars"
            
        except Exception as e:
            return f"Request {request_id} -> Error: {e}"
    
    # Make multiple concurrent streaming requests
    prompts = [
        "Explain quantum computing",
        "Write a poem about AI",
        "Describe machine learning",
        "Tell a story about robots"
    ]
    
    print("Starting concurrent streaming requests...")
    start_time = time.time()
    
    tasks = [
        stream_concurrent(prompt, "gpt-3.5-turbo", i+1)
        for i, prompt in enumerate(prompts)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"All concurrent requests completed in {end_time - start_time:.2f} seconds")
    
    for result in results:
        print(result)
    
    # Streaming with stop sequences
    print("\n--- Streaming with Stop Sequences ---")
    
    try:
        prompt = "Write a short story that ends with 'The End'"
        
        print(f"Prompt: {prompt}")
        print("Response (will stop at 'The End'):")
        
        async for chunk in router.stream(
            prompt,
            model="gpt-3.5-turbo",
            stop=["The End", "END", "End"]
        ):
            print(chunk, end="", flush=True)
        
        print("\nStreaming stopped by stop sequence")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Performance comparison: streaming vs non-streaming
    print("\n--- Performance Comparison ---")
    
    prompt = "Write a detailed explanation of neural networks"
    
    # Non-streaming
    print("Testing non-streaming completion...")
    try:
        start_time = time.time()
        response = await router.complete(prompt, model="gpt-3.5-turbo")
        end_time = time.time()
        
        print(f"Non-streaming completed in {end_time - start_time:.2f} seconds")
        print(f"Response length: {len(response.content)} characters")
        
    except Exception as e:
        print(f"Non-streaming error: {e}")
    
    # Streaming
    print("Testing streaming completion...")
    try:
        start_time = time.time()
        response_length = 0
        
        async for chunk in router.stream(prompt, model="gpt-3.5-turbo"):
            response_length += len(chunk)
        
        end_time = time.time()
        
        print(f"Streaming completed in {end_time - start_time:.2f} seconds")
        print(f"Response length: {response_length} characters")
        
    except Exception as e:
        print(f"Streaming error: {e}")
    
    # Clean up
    await router.close()
    print("\n=== Streaming demo completed ===")


async def main():
    """Main function to run the streaming demo."""
    await streaming_demo()


if __name__ == "__main__":
    asyncio.run(main()) 