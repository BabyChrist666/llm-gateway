"""
Rate limiting implementations.

Provides token bucket and sliding window rate limiters.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
from threading import Lock


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000
    tokens_per_hour: int = 1000000
    burst_multiplier: float = 1.5

    def to_dict(self) -> dict:
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_hour": self.tokens_per_hour,
            "burst_multiplier": self.burst_multiplier,
        }


class RateLimiter(ABC):
    """Abstract rate limiter."""

    @abstractmethod
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        pass

    @abstractmethod
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait before tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if tokens available now)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter state."""
        pass


class TokenBucket(RateLimiter):
    """
    Token bucket rate limiter.

    Allows bursting up to bucket capacity, refills at steady rate.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now

    def try_acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                return 0.0

            needed = tokens - self.tokens
            return needed / self.refill_rate

    def reset(self) -> None:
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill = time.time()

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens


class SlidingWindow(RateLimiter):
    """
    Sliding window rate limiter.

    Tracks requests in a time window, more accurate than fixed windows.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list = []
        self._lock = Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests outside the window."""
        cutoff = time.time() - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def try_acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._clean_old_requests()

            if len(self.requests) + tokens <= self.max_requests:
                now = time.time()
                for _ in range(tokens):
                    self.requests.append(now)
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        with self._lock:
            self._clean_old_requests()

            if len(self.requests) + tokens <= self.max_requests:
                return 0.0

            # Need to wait for oldest requests to expire
            if not self.requests:
                return 0.0

            # How many requests need to expire
            excess = len(self.requests) + tokens - self.max_requests
            if excess <= 0:
                return 0.0

            # Wait for the oldest `excess` requests to expire
            oldest = sorted(self.requests)[:excess]
            wait_until = oldest[-1] + self.window_seconds
            return max(0, wait_until - time.time())

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()

    @property
    def current_count(self) -> int:
        with self._lock:
            self._clean_old_requests()
            return len(self.requests)


@dataclass
class CompositeRateLimiter:
    """
    Combines multiple rate limiters.

    All limiters must allow for request to proceed.
    """
    limiters: Dict[str, RateLimiter] = field(default_factory=dict)

    def add_limiter(self, name: str, limiter: RateLimiter) -> None:
        """Add a rate limiter."""
        self.limiters[name] = limiter

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire from all limiters."""
        # Check all first
        for limiter in self.limiters.values():
            if not limiter.try_acquire(0):  # Dry run check
                return False

        # Acquire from all
        results = []
        for limiter in self.limiters.values():
            if limiter.try_acquire(tokens):
                results.append(True)
            else:
                # Rollback is not implemented - best effort
                return False

        return all(results)

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get max wait time across all limiters."""
        if not self.limiters:
            return 0.0
        return max(l.get_wait_time(tokens) for l in self.limiters.values())

    def reset(self) -> None:
        """Reset all limiters."""
        for limiter in self.limiters.values():
            limiter.reset()


def create_rate_limiter(config: RateLimitConfig) -> CompositeRateLimiter:
    """Create a composite rate limiter from config."""
    composite = CompositeRateLimiter()

    # Requests per minute
    composite.add_limiter(
        "rpm",
        TokenBucket(
            capacity=int(config.requests_per_minute * config.burst_multiplier),
            refill_rate=config.requests_per_minute / 60,
        ),
    )

    # Tokens per minute
    composite.add_limiter(
        "tpm",
        TokenBucket(
            capacity=int(config.tokens_per_minute * config.burst_multiplier),
            refill_rate=config.tokens_per_minute / 60,
        ),
    )

    return composite
