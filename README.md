# LLM Gateway

[![Tests](https://github.com/BabyChrist666/llm-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/llm-gateway/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/llm-gateway/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/llm-gateway)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-provider LLM routing with fallback, rate limiting, and caching.

Route requests across OpenAI, Anthropic, and other providers with automatic failover, intelligent load balancing, and response caching.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from llm_gateway import (
    Gateway, GatewayConfig, Request,
    OpenAIProvider, AnthropicProvider, ProviderConfig,
    RateLimitConfig, CacheConfig,
)

# Configure providers
providers = [
    OpenAIProvider(ProviderConfig(
        name="openai",
        api_key="sk-...",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        priority=1,
    )),
    AnthropicProvider(ProviderConfig(
        name="anthropic",
        api_key="sk-ant-...",
        cost_per_1k_input=0.008,
        cost_per_1k_output=0.024,
        priority=0,  # Fallback
    )),
]

# Create gateway with rate limiting and caching
gateway = Gateway(
    providers=providers,
    config=GatewayConfig(
        rate_limit=RateLimitConfig(requests_per_minute=60),
        cache=CacheConfig(ttl_seconds=3600),
    ),
)

# Make request
async def main():
    response = await gateway.complete(Request(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4",
    ))
    print(response.content)

asyncio.run(main())
```

## Routing Strategies

### Round Robin
Distribute requests evenly across healthy providers:

```python
from llm_gateway import RoundRobinStrategy

gateway = Gateway(providers, strategy=RoundRobinStrategy())
```

### Lowest Latency
Route to the provider with best average response time:

```python
from llm_gateway import LowestLatencyStrategy

gateway = Gateway(providers, strategy=LowestLatencyStrategy())
```

### Cost Optimized
Prefer cheaper providers:

```python
from llm_gateway import CostOptimizedStrategy

gateway = Gateway(providers, strategy=CostOptimizedStrategy())
```

### Priority
Route by provider priority with automatic fallback:

```python
from llm_gateway import PriorityStrategy

gateway = Gateway(providers, strategy=PriorityStrategy())
```

## Rate Limiting

Token bucket and sliding window limiters:

```python
from llm_gateway import RateLimitConfig

config = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    tokens_per_minute=100000,
    burst_multiplier=1.5,  # Allow 1.5x burst
)
```

The gateway raises `RateLimitError` when limits are exceeded:

```python
from llm_gateway import RateLimitError

try:
    response = await gateway.complete(request)
except RateLimitError as e:
    print(f"Rate limited. Wait {e.wait_time:.1f}s")
```

## Response Caching

Cache identical requests to reduce costs:

```python
from llm_gateway import CacheConfig

config = CacheConfig(
    enabled=True,
    ttl_seconds=3600,
    max_entries=1000,
    include_model_in_key=True,
    include_temperature_in_key=True,
)
```

Cache keys are generated from message content, model, and temperature.

## Fallback

Automatic fallback when providers fail:

```python
config = GatewayConfig(
    fallback_enabled=True,
    max_fallback_attempts=3,
    timeout=30.0,
)
```

Providers are tried in priority order. Unhealthy providers (3+ consecutive failures) are temporarily skipped.

## Metrics

Track usage and performance:

```python
metrics = gateway.get_metrics()
# {
#     "total_requests": 1000,
#     "cached_requests": 250,
#     "failed_requests": 5,
#     "cache_hit_rate": 0.25,
#     "avg_latency_ms": 150,
#     "total_cost": 0.42,
#     "providers": {
#         "openai": {"success_rate": 0.995, ...},
#         "anthropic": {"success_rate": 0.99, ...},
#     }
# }

status = gateway.get_provider_status()
# {"openai": {"status": "healthy", ...}}
```

## Custom Providers

Extend `Provider` for custom backends:

```python
from llm_gateway import Provider, ProviderConfig, ToolResult

class LocalLLMProvider(Provider):
    async def complete(self, messages, model, **kwargs):
        # Your implementation
        return {
            "content": "response",
            "model": model,
            "provider": self.name,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
```

## Testing

```bash
pytest tests/ -v
```

104 tests covering:
- Provider management and metrics
- Routing strategies
- Rate limiting (token bucket, sliding window)
- Response caching
- Gateway integration

## License

MIT
