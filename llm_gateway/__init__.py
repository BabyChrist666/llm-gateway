"""
LLM Gateway - Multi-provider routing with fallback, rate limiting, and caching.

A production-ready gateway for routing LLM requests across multiple providers
with intelligent fallback, rate limiting, response caching, and cost tracking.
"""

from .providers import (
    Provider,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
)
from .routing import (
    Router,
    RoutingStrategy,
    RoundRobinStrategy,
    LowestLatencyStrategy,
    CostOptimizedStrategy,
    FallbackChain,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    SlidingWindow,
)
from .cache import (
    Cache,
    CacheConfig,
    MemoryCache,
    CacheKey,
)
from .gateway import (
    Gateway,
    GatewayConfig,
    Request,
    Response,
    GatewayError,
)

__version__ = "0.1.0"

__all__ = [
    # Providers
    "Provider",
    "ProviderConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    # Routing
    "Router",
    "RoutingStrategy",
    "RoundRobinStrategy",
    "LowestLatencyStrategy",
    "CostOptimizedStrategy",
    "FallbackChain",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "TokenBucket",
    "SlidingWindow",
    # Caching
    "Cache",
    "CacheConfig",
    "MemoryCache",
    "CacheKey",
    # Gateway
    "Gateway",
    "GatewayConfig",
    "Request",
    "Response",
    "GatewayError",
]
