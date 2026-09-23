from llm_preclassifier.metrics import Metrics

NAMES = ("classifications_total", "feedback_received")


def test_status_reports_zero_counters_before_any_increment():
    metrics = Metrics(NAMES)

    assert metrics.status() == {"classifications_total": 0, "feedback_received": 0}


def test_increment_updates_the_status_counter():
    metrics = Metrics(NAMES)

    metrics.increment("classifications_total")
    metrics.increment("classifications_total")

    assert metrics.status()["classifications_total"] == 2
    assert metrics.status()["feedback_received"] == 0


def test_render_prometheus_is_disabled_by_default():
    metrics = Metrics(NAMES)

    try:
        metrics.render_prometheus()
    except RuntimeError as error:
        assert "disabled" in str(error)
    else:
        raise AssertionError("expected RuntimeError when Prometheus export is disabled")


def test_render_prometheus_reflects_increments_when_enabled():
    metrics = Metrics(NAMES, enable_prometheus=True)

    metrics.increment("classifications_total")
    body, content_type = metrics.render_prometheus()

    assert "text/plain" in content_type
    assert b"llm_preclassifier_classifications_total 1.0" in body
