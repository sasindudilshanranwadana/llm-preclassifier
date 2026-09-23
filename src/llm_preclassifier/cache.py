"""Bounded, in-memory TTL/LRU cache, plus an optional Redis-backed alternative."""
from collections import OrderedDict
from time import monotonic
from typing import Any, Callable, Generic, Protocol, TypeVar

T = TypeVar("T")


class ClassificationCache(Generic[T]):
    """Cache decisions without retaining request content as cache keys."""

    def __init__(self, max_entries: int, ttl_seconds: float, clock: Callable[[], float] = monotonic) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: T) -> None:
        self._entries[key] = (self._clock() + self.ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class _RedisLike(Protocol):
    def get(self, name: str) -> Any: ...
    def set(self, name: str, value: Any, ex: int | None = None) -> Any: ...


class RedisClassificationCache(Generic[T]):
    """Same ``get``/``put`` interface as ``ClassificationCache``, backed by Redis.

    Eviction is TTL-only (no ``max_entries``/LRU) — that's Redis's job across instances.
    """

    def __init__(
        self,
        client: _RedisLike,
        ttl_seconds: float,
        serialize: Callable[[T], bytes | str],
        deserialize: Callable[[bytes], T],
        key_prefix: str = "llm_preclassifier:cache:",
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._client = client
        self._serialize = serialize
        self._deserialize = deserialize
        self._key_prefix = key_prefix

    def get(self, key: str) -> T | None:
        raw = self._client.get(self._key_prefix + key)
        if raw is None:
            return None
        return self._deserialize(raw)

    def put(self, key: str, value: T) -> None:
        self._client.set(self._key_prefix + key, self._serialize(value), ex=max(1, int(self.ttl_seconds)))
