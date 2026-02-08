"""Tests for llm_gateway.routing module."""

import pytest

from llm_gateway.providers import ProviderConfig, MockProvider, ProviderStatus
from llm_gateway.routing import (
    Router,
    RoutingStrategy,
    RoundRobinStrategy,
    LowestLatencyStrategy,
    CostOptimizedStrategy,
    PriorityStrategy,
    RandomStrategy,
    FallbackChain,
)


@pytest.fixture
def mock_providers():
    """Create a set of mock providers."""
    configs = [
        ProviderConfig(name="provider1", cost_per_1k_input=0.01, priority=1),
        ProviderConfig(name="provider2", cost_per_1k_input=0.02, priority=2),
        ProviderConfig(name="provider3", cost_per_1k_input=0.005, priority=0),
    ]
    return [MockProvider(c) for c in configs]


class TestRoundRobinStrategy:
    def test_select_cycles(self, mock_providers):
        strategy = RoundRobinStrategy()

        # Should cycle through providers
        selected = [
            strategy.select(mock_providers, "model").name
            for _ in range(6)
        ]

        # Each provider should appear twice
        assert selected.count("provider1") == 2
        assert selected.count("provider2") == 2
        assert selected.count("provider3") == 2

    def test_select_skips_unhealthy(self, mock_providers):
        strategy = RoundRobinStrategy()

        # Mark one as unhealthy
        for _ in range(3):
            mock_providers[0].record_failure("error")

        # Should only return healthy providers
        for _ in range(10):
            selected = strategy.select(mock_providers, "model")
            assert selected.name in ["provider2", "provider3"]


class TestLowestLatencyStrategy:
    def test_select_lowest(self):
        # Create fresh providers to avoid fixture ordering issues
        providers = [
            MockProvider(ProviderConfig(name="p1")),
            MockProvider(ProviderConfig(name="p2")),
            MockProvider(ProviderConfig(name="p3")),
        ]
        strategy = LowestLatencyStrategy()

        # Record success to set proper metrics
        providers[0].record_success(100, 0, 0)  # avg = 100ms
        providers[1].record_success(50, 0, 0)   # avg = 50ms
        providers[2].record_success(200, 0, 0)  # avg = 200ms

        selected = strategy.select(providers, "model")
        # Provider with lowest latency (50ms)
        assert selected.metrics.avg_latency_ms == 50

    def test_select_no_data_last(self, mock_providers):
        strategy = LowestLatencyStrategy()

        # Only one has data
        mock_providers[0].metrics.total_latency_ms = 100
        mock_providers[0].metrics.successful_requests = 1

        selected = strategy.select(mock_providers, "model")
        assert selected.name == "provider1"


class TestCostOptimizedStrategy:
    def test_select_cheapest(self, mock_providers):
        strategy = CostOptimizedStrategy()

        selected = strategy.select(mock_providers, "model")
        assert selected.name == "provider3"  # Lowest cost


class TestPriorityStrategy:
    def test_select_highest_priority(self, mock_providers):
        strategy = PriorityStrategy()

        selected = strategy.select(mock_providers, "model")
        assert selected.name == "provider2"  # Highest priority


class TestRandomStrategy:
    def test_select_returns_valid(self, mock_providers):
        strategy = RandomStrategy()

        for _ in range(10):
            selected = strategy.select(mock_providers, "model")
            assert selected in mock_providers


class TestFallbackChain:
    def test_get_ordered_providers(self, mock_providers):
        chain = FallbackChain(providers=mock_providers)
        ordered = chain.get_ordered_providers("model")

        # Should be ordered by priority (descending)
        assert ordered[0].name == "provider2"

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_providers):
        chain = FallbackChain(providers=mock_providers)

        async def func(provider):
            return f"result from {provider.name}"

        result, provider, attempts = await chain.execute("model", func)

        assert result.startswith("result from")
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, mock_providers):
        chain = FallbackChain(providers=mock_providers)
        call_count = 0

        async def func(provider):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("fail")
            return f"result from {provider.name}"

        result, provider, attempts = await chain.execute("model", func)

        assert attempts == 3
        assert result.startswith("result from")


class TestRouter:
    def test_create(self, mock_providers):
        router = Router(providers=mock_providers)
        assert len(router.providers) == 3

    def test_add_provider(self):
        router = Router(providers=[])
        new_provider = MockProvider(ProviderConfig(name="new"))
        router.add_provider(new_provider)
        assert len(router.providers) == 1

    def test_remove_provider(self, mock_providers):
        router = Router(providers=mock_providers)
        result = router.remove_provider("provider1")
        assert result is True
        assert len(router.providers) == 2

    def test_remove_nonexistent(self, mock_providers):
        router = Router(providers=mock_providers)
        result = router.remove_provider("nonexistent")
        assert result is False

    def test_select_provider(self, mock_providers):
        router = Router(providers=mock_providers)
        provider = router.select_provider("model")
        assert provider in mock_providers

    def test_get_healthy_providers(self, mock_providers):
        router = Router(providers=mock_providers)

        # Mark one as unhealthy
        for _ in range(3):
            mock_providers[0].record_failure("error")

        healthy = router.get_healthy_providers()
        assert len(healthy) == 2

    def test_get_provider_metrics(self, mock_providers):
        router = Router(providers=mock_providers)
        metrics = router.get_provider_metrics()

        assert "provider1" in metrics
        assert "total_requests" in metrics["provider1"]
