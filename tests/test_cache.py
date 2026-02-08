"""Tests for llm_gateway.cache module."""

import pytest
import time

from llm_gateway.cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    CacheKey,
    Cache,
    MemoryCache,
)


class TestCacheConfig:
    def test_defaults(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_entries == 1000

    def test_custom(self):
        config = CacheConfig(
            enabled=False,
            ttl_seconds=60,
            max_entries=100,
        )
        assert config.enabled is False
        assert config.ttl_seconds == 60

    def test_to_dict(self):
        config = CacheConfig()
        d = config.to_dict()
        assert "enabled" in d
        assert "ttl_seconds" in d


class TestCacheEntry:
    def test_create(self):
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time(),
            ttl_seconds=60,
        )
        assert entry.key == "test"
        assert entry.is_expired is False

    def test_expired(self):
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time() - 100,
            ttl_seconds=60,
        )
        assert entry.is_expired is True

    def test_age(self):
        now = time.time()
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=now - 30,
            ttl_seconds=60,
        )
        assert 29 <= entry.age_seconds <= 31


class TestCacheStats:
    def test_hit_rate(self):
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_no_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert abs(d["hit_rate"] - (10 / 15)) < 0.001


class TestCacheKey:
    def test_generate_basic(self):
        messages = [{"role": "user", "content": "hello"}]
        key = CacheKey.generate(messages=messages)
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_with_model(self):
        messages = [{"role": "user", "content": "hello"}]
        key1 = CacheKey.generate(messages=messages, model="gpt-4")
        key2 = CacheKey.generate(messages=messages, model="gpt-3.5")
        assert key1 != key2

    def test_generate_deterministic(self):
        messages = [{"role": "user", "content": "hello"}]
        key1 = CacheKey.generate(messages=messages, model="gpt-4")
        key2 = CacheKey.generate(messages=messages, model="gpt-4")
        assert key1 == key2

    def test_generate_with_kwargs(self):
        messages = [{"role": "user", "content": "hello"}]
        key1 = CacheKey.generate(messages=messages, temperature=0.5)
        key2 = CacheKey.generate(messages=messages, temperature=0.7)
        assert key1 != key2


class TestMemoryCache:
    def test_create(self):
        cache = MemoryCache()
        assert cache.config.enabled is True

    def test_set_and_get(self):
        cache = MemoryCache()
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_get_missing(self):
        cache = MemoryCache()
        result = cache.get("missing")
        assert result is None

    def test_get_expired(self):
        config = CacheConfig(ttl_seconds=0)  # Immediate expiry
        cache = MemoryCache(config)
        cache.set("key1", "value1", ttl=0)

        time.sleep(0.01)
        result = cache.get("key1")
        assert result is None

    def test_delete(self):
        cache = MemoryCache()
        cache.set("key1", "value1")

        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_delete_missing(self):
        cache = MemoryCache()
        result = cache.delete("missing")
        assert result is False

    def test_clear(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_max_entries_eviction(self):
        config = CacheConfig(max_entries=3)
        cache = MemoryCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should trigger eviction

        stats = cache.get_stats()
        assert stats.entries <= 3

    def test_stats_tracking(self):
        cache = MemoryCache()
        cache.set("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1

    def test_cleanup_expired(self):
        config = CacheConfig(ttl_seconds=0)
        cache = MemoryCache(config)

        cache.set("key1", "value1", ttl=0)
        cache.set("key2", "value2", ttl=0)

        time.sleep(0.01)
        removed = cache.cleanup_expired()
        assert removed == 2

    def test_disabled_cache(self):
        config = CacheConfig(enabled=False)
        cache = MemoryCache(config)

        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result is None  # Cache disabled
