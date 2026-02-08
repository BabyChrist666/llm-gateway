"""
Routing strategies for LLM requests.

Implements various strategies for selecting providers and handling fallback.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from .providers import Provider, ProviderStatus


class RoutingStrategy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        """
        Select a provider for the request.

        Args:
            providers: Available providers
            model: Model to use

        Returns:
            Selected provider or None if no suitable provider found
        """
        pass


class RoundRobinStrategy(RoutingStrategy):
    """Round-robin selection across providers."""

    def __init__(self):
        self._index = 0

    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        healthy = [
            p for p in providers
            if p.status == ProviderStatus.HEALTHY and p.supports_model(model)
        ]

        if not healthy:
            return None

        provider = healthy[self._index % len(healthy)]
        self._index += 1
        return provider


class LowestLatencyStrategy(RoutingStrategy):
    """Select provider with lowest average latency."""

    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        healthy = [
            p for p in providers
            if p.status == ProviderStatus.HEALTHY and p.supports_model(model)
        ]

        if not healthy:
            return None

        # Sort by average latency, providers with no data go last
        return min(
            healthy,
            key=lambda p: p.metrics.avg_latency_ms if p.metrics.total_requests > 0 else float('inf'),
        )


class CostOptimizedStrategy(RoutingStrategy):
    """Select provider with lowest cost per token."""

    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        healthy = [
            p for p in providers
            if p.status == ProviderStatus.HEALTHY and p.supports_model(model)
        ]

        if not healthy:
            return None

        # Calculate average cost per token
        def cost_score(p: Provider) -> float:
            return (
                p.config.cost_per_1k_input +
                p.config.cost_per_1k_output
            ) / 2

        return min(healthy, key=cost_score)


class PriorityStrategy(RoutingStrategy):
    """Select provider by priority."""

    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        healthy = [
            p for p in providers
            if p.status == ProviderStatus.HEALTHY and p.supports_model(model)
        ]

        if not healthy:
            return None

        return max(healthy, key=lambda p: p.config.priority)


class RandomStrategy(RoutingStrategy):
    """Random provider selection."""

    def select(
        self,
        providers: List[Provider],
        model: str,
    ) -> Optional[Provider]:
        healthy = [
            p for p in providers
            if p.status == ProviderStatus.HEALTHY and p.supports_model(model)
        ]

        if not healthy:
            return None

        return random.choice(healthy)


@dataclass
class FallbackChain:
    """
    Manages fallback between providers.

    Tries providers in order, falling back to next on failure.
    """
    providers: List[Provider] = field(default_factory=list)
    max_attempts: int = 3

    def get_ordered_providers(self, model: str) -> List[Provider]:
        """Get providers that support the model, ordered by priority."""
        supporting = [p for p in self.providers if p.supports_model(model)]

        # Sort by priority (descending), then by health status
        def sort_key(p: Provider) -> tuple:
            status_order = {
                ProviderStatus.HEALTHY: 0,
                ProviderStatus.DEGRADED: 1,
                ProviderStatus.UNHEALTHY: 2,
            }
            return (status_order[p.status], -p.config.priority)

        return sorted(supporting, key=sort_key)

    async def execute(
        self,
        model: str,
        func,
        *args,
        **kwargs,
    ) -> tuple:
        """
        Execute function with fallback.

        Args:
            model: Model to use
            func: Async function to call (provider, *args, **kwargs)
            *args, **kwargs: Arguments for func

        Returns:
            Tuple of (result, provider_used, attempts_made)
        """
        providers = self.get_ordered_providers(model)
        last_error = None
        attempts = 0

        for provider in providers[:self.max_attempts]:
            attempts += 1
            try:
                result = await func(provider, *args, **kwargs)
                return result, provider, attempts
            except Exception as e:
                last_error = e
                continue

        raise last_error or Exception("No providers available")


class Router:
    """
    Main router for directing requests to providers.
    """

    def __init__(
        self,
        providers: List[Provider],
        strategy: Optional[RoutingStrategy] = None,
        fallback: bool = True,
        max_fallback_attempts: int = 3,
    ):
        self.providers = list(providers)  # Make a copy
        self.strategy = strategy or RoundRobinStrategy()
        self.fallback_enabled = fallback
        self.fallback_chain = FallbackChain(
            providers=self.providers,  # Share the same list
            max_attempts=max_fallback_attempts,
        )

    def add_provider(self, provider: Provider) -> None:
        """Add a provider."""
        self.providers.append(provider)
        # No need to add to fallback_chain since it shares the list

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name."""
        for i, p in enumerate(self.providers):
            if p.name == name:
                self.providers.pop(i)
                # No need to update fallback_chain since it shares the list
                return True
        return False

    def select_provider(self, model: str) -> Optional[Provider]:
        """Select a provider using the configured strategy."""
        return self.strategy.select(self.providers, model)

    def get_healthy_providers(self) -> List[Provider]:
        """Get all healthy providers."""
        return [p for p in self.providers if p.status == ProviderStatus.HEALTHY]

    def get_provider_metrics(self) -> dict:
        """Get metrics for all providers."""
        return {p.name: p.metrics.to_dict() for p in self.providers}
