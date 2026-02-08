"""Tests for llm_gateway.gateway module."""

import pytest

from llm_gateway.providers import ProviderConfig, MockProvider
from llm_gateway.routing import RoundRobinStrategy
from llm_gateway.rate_limiter import RateLimitConfig
from llm_gateway.cache import CacheConfig
from llm_gateway.gateway import (
    Gateway,
    GatewayConfig,
    Request,
    Response,
    GatewayError,
    RateLimitError,
    NoProviderError,
    create_gateway,
)


@pytest.fixture
def mock_providers():
    """Create mock providers."""
    configs = [
        ProviderConfig(name="provider1", cost_per_1k_input=0.01),
        ProviderConfig(name="provider2", cost_per_1k_input=0.02),
    ]
    return [MockProvider(c, response="test response") for c in configs]


@pytest.fixture
def basic_request():
    """Create a basic request."""
    return Request(
        messages=[{"role": "user", "content": "hello"}],
        model="test-model",
    )


class TestRequest:
    def test_create(self):
        request = Request(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
        )
        assert request.model == "gpt-4"
        assert request.temperature == 0.7

    def test_to_dict(self):
        request = Request(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            temperature=0.5,
        )
        d = request.to_dict()
        assert d["model"] == "gpt-4"
        assert d["temperature"] == 0.5


class TestResponse:
    def test_create(self):
        response = Response(
            content="hello",
            model="gpt-4",
            provider="openai",
            latency_ms=100,
        )
        assert response.content == "hello"
        assert response.cached is False

    def test_to_dict(self):
        response = Response(
            content="hello",
            model="gpt-4",
            provider="openai",
            latency_ms=100,
            cost=0.001,
        )
        d = response.to_dict()
        assert d["provider"] == "openai"
        assert d["cost"] == 0.001


class TestGatewayConfig:
    def test_defaults(self):
        config = GatewayConfig()
        assert config.fallback_enabled is True
        assert config.timeout == 30.0

    def test_with_rate_limit(self):
        config = GatewayConfig(
            rate_limit=RateLimitConfig(requests_per_minute=30),
        )
        assert config.rate_limit.requests_per_minute == 30

    def test_to_dict(self):
        config = GatewayConfig()
        d = config.to_dict()
        assert "fallback_enabled" in d


class TestGateway:
    def test_create(self, mock_providers):
        gateway = Gateway(providers=mock_providers)
        assert len(gateway.providers) == 2

    def test_add_provider(self):
        gateway = Gateway(providers=[])
        new_provider = MockProvider(ProviderConfig(name="new"))
        gateway.add_provider(new_provider)
        assert len(gateway.providers) == 1

    def test_remove_provider(self, mock_providers):
        gateway = Gateway(providers=mock_providers)
        result = gateway.remove_provider("provider1")
        assert result is True
        assert len(gateway.providers) == 1

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_providers, basic_request):
        gateway = Gateway(providers=mock_providers)
        response = await gateway.complete(basic_request)

        assert response.content == "test response"
        assert response.cached is False
        assert response.provider in ["provider1", "provider2"]

    @pytest.mark.asyncio
    async def test_complete_with_fallback(self, basic_request):
        # First provider fails, second succeeds
        providers = [
            MockProvider(ProviderConfig(name="fail"), fail=True),
            MockProvider(ProviderConfig(name="success"), response="ok"),
        ]
        gateway = Gateway(providers=providers)

        response = await gateway.complete(basic_request)

        assert response.content == "ok"
        assert response.provider == "success"
        assert response.attempts == 2

    @pytest.mark.asyncio
    async def test_complete_all_fail(self, basic_request):
        providers = [
            MockProvider(ProviderConfig(name="fail1"), fail=True),
            MockProvider(ProviderConfig(name="fail2"), fail=True),
        ]
        gateway = Gateway(providers=providers)

        with pytest.raises(GatewayError):
            await gateway.complete(basic_request)

    @pytest.mark.asyncio
    async def test_complete_with_cache(self, mock_providers, basic_request):
        config = GatewayConfig(cache=CacheConfig(enabled=True))
        gateway = Gateway(providers=mock_providers, config=config)

        # First request - cache miss
        response1 = await gateway.complete(basic_request)
        assert response1.cached is False

        # Second request - cache hit
        response2 = await gateway.complete(basic_request)
        assert response2.cached is True

    @pytest.mark.asyncio
    async def test_complete_rate_limited(self, mock_providers, basic_request):
        config = GatewayConfig(
            rate_limit=RateLimitConfig(
                requests_per_minute=1,
                burst_multiplier=1.0,
            ),
        )
        gateway = Gateway(providers=mock_providers, config=config)

        # First request should succeed
        await gateway.complete(basic_request)

        # Second should be rate limited
        with pytest.raises(RateLimitError):
            await gateway.complete(basic_request)

    def test_get_metrics(self, mock_providers):
        gateway = Gateway(providers=mock_providers)
        metrics = gateway.get_metrics()

        assert "total_requests" in metrics
        assert "providers" in metrics

    def test_get_provider_status(self, mock_providers):
        gateway = Gateway(providers=mock_providers)
        status = gateway.get_provider_status()

        assert "provider1" in status
        assert "status" in status["provider1"]


class TestCreateGateway:
    def test_basic(self, mock_providers):
        gateway = create_gateway(providers=mock_providers)
        assert len(gateway.providers) == 2

    def test_with_rate_limit(self, mock_providers):
        gateway = create_gateway(
            providers=mock_providers,
            rate_limit=RateLimitConfig(requests_per_minute=60),
        )
        assert gateway.rate_limiter is not None

    def test_with_cache(self, mock_providers):
        gateway = create_gateway(
            providers=mock_providers,
            cache=CacheConfig(enabled=True),
        )
        assert gateway.cache is not None
