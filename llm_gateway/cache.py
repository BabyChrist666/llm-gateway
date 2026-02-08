"""
Response caching for LLM requests.

Provides in-memory caching with TTL and semantic key generation.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from threading import Lock


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    max_entries: int = 1000
    include_model_in_key: bool = True
    include_temperature_in_key: bool = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "include_model_in_key": self.include_model_in_key,
            "include_temperature_in_key": self.include_temperature_in_key,
        }


@dataclass
class CacheEntry:
    """A cached response."""
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": self.entries,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheKey:
    """Generates cache keys from request parameters."""

    @staticmethod
    def generate(
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate a cache key from request parameters.

        Args:
            messages: Chat messages
            model: Model name (optional)
            temperature: Temperature (optional)
            **kwargs: Additional parameters to include in key

        Returns:
            Hash string for cache lookup
        """
        key_parts = {
            "messages": messages,
        }

        if model is not None:
            key_parts["model"] = model
        if temperature is not None:
            key_parts["temperature"] = temperature

        # Add any extra kwargs that affect output
        for k, v in kwargs.items():
            if v is not None:
                key_parts[k] = v

        # Create stable JSON representation
        key_json = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:32]


class Cache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(Cache):
    """
    In-memory cache with TTL and LRU eviction.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        if not self.config.enabled:
            return None

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            entry.hit_count += 1
            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.config.enabled:
            return

        ttl = ttl or self.config.ttl_seconds

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.config.max_entries:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl,
            )
            self._stats.entries = len(self._cache)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.entries = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        with self._lock:
            self._stats.entries = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                entries=self._stats.entries,
            )

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # First remove expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
            self._stats.evictions += 1

        # If still at capacity, remove entry with oldest access
        if len(self._cache) >= self.config.max_entries:
            # Find entry with lowest hit count and oldest age
            oldest = min(
                self._cache.keys(),
                key=lambda k: (self._cache[k].hit_count, -self._cache[k].age_seconds),
            )
            del self._cache[oldest]
            self._stats.evictions += 1

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired]
            for k in expired:
                del self._cache[k]
                self._stats.evictions += 1
            self._stats.entries = len(self._cache)
            return len(expired)
