"""Runtime counters, exposed as JSON via ``/status`` and, optionally, as Prometheus text.

Prometheus export is opt-in (``ENABLE_METRICS``) so the dependency is only exercised
when requested; the ``/status`` JSON counters always work, matching prior behaviour.
"""
from __future__ import annotations

from collections import Counter


class Metrics:
    """Tracks named counters, mirroring them into a Prometheus registry when enabled."""

    def __init__(self, names: tuple[str, ...], enable_prometheus: bool = False) -> None:
        self._counts: Counter[str] = Counter({name: 0 for name in names})
        self._prometheus_counters = None
        self._registry = None
        if enable_prometheus:
            from prometheus_client import CollectorRegistry, Counter as PrometheusCounter

            self._registry = CollectorRegistry()
            self._prometheus_counters = {
                name: PrometheusCounter(
                    f"llm_preclassifier_{name}", name.replace("_", " "), registry=self._registry,
                )
                for name in names
            }

    def increment(self, name: str) -> None:
        self._counts[name] += 1
        if self._prometheus_counters is not None:
            self._prometheus_counters[name].inc()

    def status(self) -> dict[str, int]:
        return dict(self._counts)

    def render_prometheus(self) -> tuple[bytes, str]:
        """Return (body, content_type). Raises if Prometheus export isn't enabled."""
        if self._registry is None:
            raise RuntimeError("Prometheus metrics are disabled; set ENABLE_METRICS=true")
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self._registry), CONTENT_TYPE_LATEST
