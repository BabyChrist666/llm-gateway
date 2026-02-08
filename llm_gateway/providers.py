"""
LLM Provider implementations.

Abstraction layer for different LLM providers (OpenAI, Anthropic, etc.)
"""

import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    api_key: str = ""
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    models: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = preferred

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "models": self.models,
            "priority": self.priority,
        }


@dataclass
class ProviderMetrics:
    """Metrics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost: float = 0.0
    last_request_time: Optional[float] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_cost": round(self.total_cost, 6),
            "consecutive_failures": self.consecutive_failures,
        }


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.metrics = ProviderMetrics()
        self._status = ProviderStatus.HEALTHY

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def status(self) -> ProviderStatus:
        # Mark as unhealthy after consecutive failures
        if self.metrics.consecutive_failures >= 3:
            return ProviderStatus.UNHEALTHY
        elif self.metrics.consecutive_failures >= 1:
            return ProviderStatus.DEGRADED
        return ProviderStatus.HEALTHY

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response dict with 'content', 'model', 'usage', etc.
        """
        pass

    def supports_model(self, model: str) -> bool:
        """Check if provider supports the given model."""
        if not self.config.models:
            return True  # No restrictions
        return model in self.config.models

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost

    def record_success(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record a successful request."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.total_latency_ms += latency_ms
        self.metrics.total_tokens_input += input_tokens
        self.metrics.total_tokens_output += output_tokens
        self.metrics.total_cost += self.calculate_cost(input_tokens, output_tokens)
        self.metrics.last_request_time = time.time()
        self.metrics.consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        """Record a failed request."""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_error = error
        self.metrics.consecutive_failures += 1
        self.metrics.last_request_time = time.time()


class OpenAIProvider(Provider):
    """OpenAI API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send completion request to OpenAI."""
        # This is a mock implementation - in production, use httpx or aiohttp
        start_time = time.time()

        try:
            # Simulate API call
            # In real implementation: use httpx.AsyncClient

            # Mock response
            response = {
                "content": f"[OpenAI {model}] Mock response",
                "model": model,
                "provider": self.name,
                "usage": {
                    "prompt_tokens": sum(len(m["content"]) // 4 for m in messages),
                    "completion_tokens": 10,
                },
            }

            latency_ms = (time.time() - start_time) * 1000
            self.record_success(
                latency_ms,
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )

            return response
        except Exception as e:
            self.record_failure(str(e))
            raise


class AnthropicProvider(Provider):
    """Anthropic API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send completion request to Anthropic."""
        start_time = time.time()

        try:
            # Mock response
            response = {
                "content": f"[Anthropic {model}] Mock response",
                "model": model,
                "provider": self.name,
                "usage": {
                    "prompt_tokens": sum(len(m["content"]) // 4 for m in messages),
                    "completion_tokens": 10,
                },
            }

            latency_ms = (time.time() - start_time) * 1000
            self.record_success(
                latency_ms,
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )

            return response
        except Exception as e:
            self.record_failure(str(e))
            raise


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(
        self,
        config: ProviderConfig,
        response: Optional[str] = None,
        fail: bool = False,
        latency_ms: float = 0,
    ):
        super().__init__(config)
        self.mock_response = response or "Mock response"
        self.should_fail = fail
        self.mock_latency_ms = latency_ms

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return mock response."""
        start_time = time.time()

        if self.mock_latency_ms > 0:
            import asyncio
            await asyncio.sleep(self.mock_latency_ms / 1000)

        if self.should_fail:
            self.record_failure("Mock failure")
            raise Exception("Mock failure")

        input_tokens = sum(len(m["content"]) // 4 for m in messages)
        output_tokens = len(self.mock_response) // 4

        response = {
            "content": self.mock_response,
            "model": model,
            "provider": self.name,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        }

        latency_ms = (time.time() - start_time) * 1000
        self.record_success(latency_ms, input_tokens, output_tokens)

        return response
