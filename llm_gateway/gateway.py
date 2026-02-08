"""
Main gateway implementation.

Combines routing, rate limiting, caching, and provider management.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .providers import Provider, ProviderConfig, MockProvider
from .routing import Router, RoutingStrategy, RoundRobinStrategy, FallbackChain
from .rate_limiter import RateLimiter, RateLimitConfig, CompositeRateLimiter, create_rate_limiter
from .cache import Cache, CacheConfig, MemoryCache, CacheKey


class GatewayError(Exception):
    """Base exception for gateway errors."""
    pass


class RateLimitError(GatewayError):
    """Raised when rate limit is exceeded."""
    def __init__(self, wait_time: float):
        self.wait_time = wait_time
        super().__init__(f"Rate limit exceeded. Wait {wait_time:.2f} seconds.")


class NoProviderError(GatewayError):
    """Raised when no provider is available."""
    pass


@dataclass
class Request:
    """Gateway request."""
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "metadata": self.metadata,
        }


@dataclass
class Response:
    """Gateway response."""
    content: str
    model: str
    provider: str
    cached: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "cached": self.cached,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": round(self.cost, 6),
            "attempts": self.attempts,
        }


@dataclass
class GatewayConfig:
    """Gateway configuration."""
    rate_limit: Optional[RateLimitConfig] = None
    cache: Optional[CacheConfig] = None
    fallback_enabled: bool = True
    max_fallback_attempts: int = 3
    timeout: float = 30.0

    def to_dict(self) -> dict:
        return {
            "rate_limit": self.rate_limit.to_dict() if self.rate_limit else None,
            "cache": self.cache.to_dict() if self.cache else None,
            "fallback_enabled": self.fallback_enabled,
            "max_fallback_attempts": self.max_fallback_attempts,
            "timeout": self.timeout,
        }


class Gateway:
    """
    LLM Gateway with routing, rate limiting, and caching.

    Routes requests to multiple LLM providers with:
    - Intelligent routing strategies
    - Automatic fallback on failure
    - Rate limiting per provider or global
    - Response caching
    - Cost tracking
    - Metrics collection
    """

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        config: Optional[GatewayConfig] = None,
        strategy: Optional[RoutingStrategy] = None,
    ):
        self.config = config or GatewayConfig()
        self.router = Router(
            providers=providers or [],
            strategy=strategy or RoundRobinStrategy(),
            fallback=self.config.fallback_enabled,
            max_fallback_attempts=self.config.max_fallback_attempts,
        )

        # Rate limiting
        self.rate_limiter: Optional[CompositeRateLimiter] = None
        if self.config.rate_limit:
            self.rate_limiter = create_rate_limiter(self.config.rate_limit)

        # Caching
        self.cache: Optional[Cache] = None
        if self.config.cache and self.config.cache.enabled:
            self.cache = MemoryCache(self.config.cache)

        # Metrics
        self.total_requests = 0
        self.cached_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.total_cost = 0.0

    def add_provider(self, provider: Provider) -> None:
        """Add a provider to the gateway."""
        self.router.add_provider(provider)

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name."""
        return self.router.remove_provider(name)

    @property
    def providers(self) -> List[Provider]:
        """Get list of providers from router."""
        return self.router.providers

    async def complete(self, request: Request) -> Response:
        """
        Send a completion request through the gateway.

        Args:
            request: Gateway request

        Returns:
            Gateway response

        Raises:
            RateLimitError: If rate limit exceeded
            NoProviderError: If no provider available
            GatewayError: On other errors
        """
        start_time = time.time()
        self.total_requests += 1

        # Check rate limit
        if self.rate_limiter:
            if not self.rate_limiter.try_acquire():
                wait_time = self.rate_limiter.get_wait_time()
                raise RateLimitError(wait_time)

        # Check cache
        cache_key = None
        if self.cache and self.config.cache:
            cache_key = CacheKey.generate(
                messages=request.messages,
                model=request.model if self.config.cache.include_model_in_key else None,
                temperature=request.temperature if self.config.cache.include_temperature_in_key else None,
            )
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.cached_requests += 1
                latency_ms = (time.time() - start_time) * 1000
                return Response(
                    content=cached_response["content"],
                    model=cached_response["model"],
                    provider=cached_response["provider"],
                    cached=True,
                    latency_ms=latency_ms,
                    input_tokens=cached_response.get("input_tokens", 0),
                    output_tokens=cached_response.get("output_tokens", 0),
                )

        # Execute with fallback
        try:
            async def execute_provider(provider: Provider) -> dict:
                return await asyncio.wait_for(
                    provider.complete(
                        messages=request.messages,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ),
                    timeout=self.config.timeout,
                )

            result, provider, attempts = await self.router.fallback_chain.execute(
                model=request.model,
                func=execute_provider,
            )
        except Exception as e:
            self.failed_requests += 1
            raise GatewayError(f"All providers failed: {e}") from e

        # Build response
        latency_ms = (time.time() - start_time) * 1000
        self.total_latency_ms += latency_ms

        input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        cost = provider.calculate_cost(input_tokens, output_tokens)
        self.total_cost += cost

        response = Response(
            content=result["content"],
            model=result.get("model", request.model),
            provider=provider.name,
            cached=False,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            attempts=attempts,
        )

        # Cache response
        if self.cache and cache_key:
            self.cache.set(cache_key, {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

        return response

    def get_metrics(self) -> dict:
        """Get gateway metrics."""
        avg_latency = 0.0
        if self.total_requests > 0:
            avg_latency = self.total_latency_ms / self.total_requests

        result = {
            "total_requests": self.total_requests,
            "cached_requests": self.cached_requests,
            "failed_requests": self.failed_requests,
            "cache_hit_rate": self.cached_requests / max(1, self.total_requests),
            "avg_latency_ms": round(avg_latency, 2),
            "total_cost": round(self.total_cost, 6),
            "providers": self.router.get_provider_metrics(),
        }

        if self.cache:
            result["cache"] = self.cache.get_stats().to_dict()

        return result

    def get_provider_status(self) -> dict:
        """Get status of all providers."""
        return {
            p.name: {
                "status": p.status.value,
                "metrics": p.metrics.to_dict(),
            }
            for p in self.providers
        }


def create_gateway(
    providers: Optional[List[Provider]] = None,
    rate_limit: Optional[RateLimitConfig] = None,
    cache: Optional[CacheConfig] = None,
    strategy: Optional[RoutingStrategy] = None,
) -> Gateway:
    """
    Create a gateway with common defaults.

    Args:
        providers: List of providers
        rate_limit: Rate limit configuration
        cache: Cache configuration
        strategy: Routing strategy

    Returns:
        Configured Gateway instance
    """
    config = GatewayConfig(
        rate_limit=rate_limit,
        cache=cache,
    )
    return Gateway(
        providers=providers or [],
        config=config,
        strategy=strategy,
    )
