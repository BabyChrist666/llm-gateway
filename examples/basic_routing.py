#!/usr/bin/env python3
"""
LLM Gateway - Basic Routing Example

This example demonstrates multi-provider routing, fallback,
rate limiting, and caching.
"""

import asyncio
from llm_gateway import (
    LLMGateway,
    GatewayConfig,
    Provider,
    ProviderConfig,
    RoutingStrategy,
)


async def main():
    print("=" * 60)
    print("LLM Gateway - Basic Routing Example")
    print("=" * 60)

    # Configure providers
    providers = [
        ProviderConfig(
            name="openai",
            provider_type=Provider.OPENAI,
            api_key="sk-...",  # Set your API key
            models=["gpt-4", "gpt-3.5-turbo"],
            priority=1,
            rate_limit=100,  # requests per minute
        ),
        ProviderConfig(
            name="anthropic",
            provider_type=Provider.ANTHROPIC,
            api_key="sk-ant-...",
            models=["claude-3-opus", "claude-3-sonnet"],
            priority=2,
            rate_limit=60,
        ),
        ProviderConfig(
            name="cohere",
            provider_type=Provider.COHERE,
            api_key="...",
            models=["command-r-plus", "command-r"],
            priority=3,
            rate_limit=100,
        ),
    ]

    # Configure gateway
    config = GatewayConfig(
        providers=providers,
        routing_strategy=RoutingStrategy.PRIORITY,
        enable_fallback=True,
        enable_caching=True,
        cache_ttl=3600,  # 1 hour
        max_retries=3,
        timeout=30,
    )

    gateway = LLMGateway(config)

    # Example 1: Basic completion
    print("\n1. Basic completion with automatic routing...")

    response = await gateway.complete(
        prompt="What is the capital of France?",
        max_tokens=100,
    )

    print(f"   Provider used: {response.provider}")
    print(f"   Model: {response.model}")
    print(f"   Response: {response.text[:100]}...")
    print(f"   Latency: {response.latency_ms}ms")

    # Example 2: Specific provider
    print("\n2. Request specific provider...")

    response = await gateway.complete(
        prompt="Write a haiku about coding.",
        provider="anthropic",
        model="claude-3-sonnet",
    )

    print(f"   Provider: {response.provider}")
    print(f"   Response: {response.text}")

    # Example 3: Fallback demonstration
    print("\n3. Fallback on failure...")

    # Simulate primary provider failure
    response = await gateway.complete(
        prompt="Hello!",
        provider="openai",
        fallback=True,  # Enable fallback
    )

    print(f"   Final provider: {response.provider}")
    print(f"   Fallback used: {response.used_fallback}")

    # Example 4: Cached response
    print("\n4. Caching demonstration...")

    # First request (cache miss)
    response1 = await gateway.complete(
        prompt="What is 2+2?",
        cache=True,
    )
    print(f"   First request - cached: {response1.from_cache}")

    # Second request (cache hit)
    response2 = await gateway.complete(
        prompt="What is 2+2?",
        cache=True,
    )
    print(f"   Second request - cached: {response2.from_cache}")

    # Example 5: Rate limit handling
    print("\n5. Rate limit status...")

    status = gateway.get_rate_limit_status()
    for provider, info in status.items():
        print(f"   {provider}: {info['remaining']}/{info['limit']} remaining")

    # Example 6: Health check
    print("\n6. Provider health check...")

    health = await gateway.health_check()
    for provider, is_healthy in health.items():
        status = "✓ healthy" if is_healthy else "✗ unhealthy"
        print(f"   {provider}: {status}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
