"""Fixed-window per-key rate limiting, in-process only (per instance, not distributed)."""
from __future__ import annotations

import time
from typing import Callable


class RateLimiter:
    """Allows up to ``limit_per_minute`` calls per key in each rolling 60-second window.

    ``limit_per_minute <= 0`` disables limiting entirely (``allow`` always returns True).
    """

    def __init__(self, limit_per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit_per_minute = limit_per_minute
        self._clock = clock
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, key: str) -> bool:
        if self.limit_per_minute <= 0:
            return True
        window = int(self._clock() // 60)
        start, count = self._windows.get(key, (window, 0))
        if start != window:
            start, count = window, 0
        count += 1
        self._windows[key] = (start, count)
        return count <= self.limit_per_minute
