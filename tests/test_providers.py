"""Tests for llm_gateway.providers module."""

import pytest

from llm_gateway.providers import (
    ProviderConfig,
    ProviderMetrics,
    ProviderStatus,
    Provider,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
)


class TestProviderConfig:
    def test_create_basic(self):
        config = ProviderConfig(name="test")
        assert config.name == "test"
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_create_with_options(self):
        config = ProviderConfig(
            name="openai",
            api_key="sk-test",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            models=["gpt-4", "gpt-3.5-turbo"],
        )
        assert config.api_key == "sk-test"
        assert config.cost_per_1k_input == 0.01
        assert len(config.models) == 2

    def test_to_dict(self):
        config = ProviderConfig(name="test", priority=5)
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["priority"] == 5


class TestProviderMetrics:
    def test_initial_state(self):
        metrics = ProviderMetrics()
        assert metrics.total_requests == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency(self):
        metrics = ProviderMetrics(
            total_requests=10,
            successful_requests=10,
            total_latency_ms=1000,
        )
        assert metrics.avg_latency_ms == 100.0

    def test_success_rate(self):
        metrics = ProviderMetrics(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
        )
        assert metrics.success_rate == 0.8

    def test_to_dict(self):
        metrics = ProviderMetrics(total_requests=5, successful_requests=5)
        d = metrics.to_dict()
        assert d["total_requests"] == 5
        assert d["success_rate"] == 1.0


class TestMockProvider:
    def test_create(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config)
        assert provider.name == "mock"

    @pytest.mark.asyncio
    async def test_complete(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config, response="test response")

        result = await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )

        assert result["content"] == "test response"
        assert result["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_complete_with_failure(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config, fail=True)

        with pytest.raises(Exception, match="Mock failure"):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="test-model",
            )

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config)

        await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )

        assert provider.metrics.total_requests == 1
        assert provider.metrics.successful_requests == 1
        assert provider.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_failure(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config, fail=True)

        try:
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="test-model",
            )
        except Exception:
            pass

        assert provider.metrics.total_requests == 1
        assert provider.metrics.failed_requests == 1
        assert provider.metrics.consecutive_failures == 1

    def test_supports_model_all(self):
        config = ProviderConfig(name="mock", models=[])
        provider = MockProvider(config)
        assert provider.supports_model("any-model") is True

    def test_supports_model_restricted(self):
        config = ProviderConfig(name="mock", models=["gpt-4"])
        provider = MockProvider(config)
        assert provider.supports_model("gpt-4") is True
        assert provider.supports_model("gpt-3.5") is False

    def test_status_healthy(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config)
        assert provider.status == ProviderStatus.HEALTHY

    def test_status_degraded_after_failure(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config)
        provider.record_failure("error")
        assert provider.status == ProviderStatus.DEGRADED

    def test_status_unhealthy_after_multiple_failures(self):
        config = ProviderConfig(name="mock")
        provider = MockProvider(config)
        for _ in range(3):
            provider.record_failure("error")
        assert provider.status == ProviderStatus.UNHEALTHY

    def test_calculate_cost(self):
        config = ProviderConfig(
            name="mock",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
        )
        provider = MockProvider(config)

        cost = provider.calculate_cost(1000, 500)
        assert cost == 0.01 + 0.015  # 1k input + 0.5k output


class TestOpenAIProvider:
    def test_create(self):
        config = ProviderConfig(name="openai", api_key="sk-test")
        provider = OpenAIProvider(config)
        assert provider.name == "openai"
        assert "openai.com" in provider.base_url

    @pytest.mark.asyncio
    async def test_complete(self):
        config = ProviderConfig(name="openai", api_key="sk-test")
        provider = OpenAIProvider(config)

        # This returns a mock response
        result = await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4",
        )

        assert "content" in result
        assert result["provider"] == "openai"


class TestAnthropicProvider:
    def test_create(self):
        config = ProviderConfig(name="anthropic", api_key="sk-test")
        provider = AnthropicProvider(config)
        assert provider.name == "anthropic"
        assert "anthropic.com" in provider.base_url
