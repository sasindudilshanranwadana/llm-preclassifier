"""Bounded, in-memory TTL/LRU cache."""
from collections import OrderedDict
from time import monotonic
from typing import Callable, Generic, TypeVar

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
