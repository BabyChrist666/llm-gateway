"""Tests for llm_gateway.rate_limiter module."""

import pytest
import time

from llm_gateway.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    CompositeRateLimiter,
    create_rate_limiter,
)


class TestRateLimitConfig:
    def test_defaults(self):
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000

    def test_custom(self):
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_multiplier=2.0,
        )
        assert config.requests_per_minute == 100
        assert config.burst_multiplier == 2.0

    def test_to_dict(self):
        config = RateLimitConfig()
        d = config.to_dict()
        assert "requests_per_minute" in d


class TestTokenBucket:
    def test_create(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.available_tokens == 10

    def test_acquire(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Should succeed
        assert bucket.try_acquire(5) is True
        assert bucket.available_tokens >= 4.9  # Allow for timing

    def test_acquire_too_many(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Should fail
        assert bucket.try_acquire(15) is False
        assert bucket.available_tokens == 10

    def test_refill(self):
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # Fast refill

        # Drain bucket
        bucket.try_acquire(10)
        assert bucket.available_tokens < 1

        # Wait for refill
        time.sleep(0.1)
        assert bucket.available_tokens >= 5

    def test_get_wait_time_available(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.get_wait_time(5) == 0.0

    def test_get_wait_time_needs_wait(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.try_acquire(10)

        wait = bucket.get_wait_time(5)
        assert wait > 0

    def test_reset(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.try_acquire(10)
        bucket.reset()
        assert bucket.available_tokens == 10


class TestSlidingWindow:
    def test_create(self):
        window = SlidingWindow(max_requests=10, window_seconds=60)
        assert window.max_requests == 10

    def test_acquire(self):
        window = SlidingWindow(max_requests=10, window_seconds=60)

        for _ in range(10):
            assert window.try_acquire() is True

        # Should fail on 11th
        assert window.try_acquire() is False

    def test_window_expiry(self):
        window = SlidingWindow(max_requests=10, window_seconds=0.1)

        # Fill window
        for _ in range(10):
            window.try_acquire()

        assert window.try_acquire() is False

        # Wait for window to expire
        time.sleep(0.15)

        assert window.try_acquire() is True

    def test_current_count(self):
        window = SlidingWindow(max_requests=10, window_seconds=60)

        window.try_acquire()
        window.try_acquire()

        assert window.current_count == 2

    def test_get_wait_time(self):
        window = SlidingWindow(max_requests=2, window_seconds=1.0)

        window.try_acquire()
        window.try_acquire()

        wait = window.get_wait_time()
        assert wait > 0
        assert wait <= 1.0

    def test_reset(self):
        window = SlidingWindow(max_requests=10, window_seconds=60)

        for _ in range(10):
            window.try_acquire()

        window.reset()
        assert window.current_count == 0


class TestCompositeRateLimiter:
    def test_create(self):
        composite = CompositeRateLimiter()
        assert len(composite.limiters) == 0

    def test_add_limiter(self):
        composite = CompositeRateLimiter()
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        composite.add_limiter("test", bucket)
        assert "test" in composite.limiters

    def test_try_acquire_all_pass(self):
        composite = CompositeRateLimiter()
        composite.add_limiter("a", TokenBucket(capacity=10, refill_rate=1.0))
        composite.add_limiter("b", TokenBucket(capacity=10, refill_rate=1.0))

        assert composite.try_acquire(5) is True

    def test_get_wait_time_max(self):
        composite = CompositeRateLimiter()

        bucket1 = TokenBucket(capacity=10, refill_rate=1.0)
        bucket2 = TokenBucket(capacity=10, refill_rate=0.5)

        # Drain both
        bucket1.try_acquire(10)
        bucket2.try_acquire(10)

        composite.add_limiter("fast", bucket1)
        composite.add_limiter("slow", bucket2)

        wait = composite.get_wait_time(5)
        assert wait > 5.0  # Should be based on slower limiter

    def test_reset(self):
        composite = CompositeRateLimiter()
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.try_acquire(10)
        composite.add_limiter("test", bucket)

        composite.reset()
        assert bucket.available_tokens == 10


class TestCreateRateLimiter:
    def test_creates_composite(self):
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=10000,
        )
        limiter = create_rate_limiter(config)

        assert isinstance(limiter, CompositeRateLimiter)
        assert "rpm" in limiter.limiters
        assert "tpm" in limiter.limiters
