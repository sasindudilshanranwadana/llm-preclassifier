import httpx
import pytest

from llm_preclassifier.client import PreclassifierClient, PreclassifierError


def make_client(handler, **kwargs) -> PreclassifierClient:
    client = PreclassifierClient("https://example.com", **kwargs)
    headers = {"Authorization": f"Bearer {kwargs['api_key']}"} if "api_key" in kwargs else {}
    client._client = httpx.Client(
        base_url="https://example.com", headers=headers, transport=httpx.MockTransport(handler),
    )
    return client


def _decision_body(**overrides):
    body = {
        "version": "v1",
        "task_type": "summarization",
        "complexity": "simple",
        "tool_requirement": "none",
        "recommended_model_tier": "economy",
        "confidence": 0.9,
        "action": "route",
        "reasons": ["keyword_match:summarization"],
        "policy_version": "2026.09.1",
        "decision_id": "abc123",
        "model_recommendations": [],
    }
    body.update(overrides)
    return body


def test_classify_returns_a_validated_decision():
    def handler(request):
        assert request.url.path == "/v1/classify"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=_decision_body())

    client = make_client(handler, api_key="secret")

    decision = client.classify([{"role": "user", "content": "Summarise this report."}])

    assert decision.task_type == "summarization"
    assert decision.decision_id == "abc123"


def test_classify_strips_available_tools_and_policy_flags_defaults():
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return httpx.Response(200, json=_decision_body())

    client = make_client(handler)
    client.classify([{"role": "user", "content": "hi"}])

    import json

    body = json.loads(captured["body"])
    assert body["available_tools"] == []
    assert body["policy_flags"] == []


def test_feedback_returns_ack_and_is_not_retried_on_failure():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json={"detail": "boom"})

    client = make_client(handler, max_retries=3)

    with pytest.raises(PreclassifierError):
        client.feedback("abc123", "incorrect", corrected_task_type="extraction")

    assert len(calls) == 1


def test_policy_and_model_catalog_and_health():
    def handler(request):
        if request.url.path == "/v1/policy":
            return httpx.Response(200, json={"version": "2026.09.1", "sha256": "x" * 64})
        if request.url.path == "/v1/model-catalog":
            return httpx.Response(200, json={"version": "2026.09.1", "sha256": "x" * 64, "currency": "USD"})
        return httpx.Response(200, json={"status": "ok"})

    client = make_client(handler)

    assert client.policy()["version"] == "2026.09.1"
    assert client.model_catalog()["currency"] == "USD"
    assert client.health() == {"status": "ok"}


def test_retries_on_5xx_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json=_decision_body())

    client = make_client(handler, max_retries=3, backoff_seconds=0)

    decision = client.classify([{"role": "user", "content": "hi"}])

    assert decision.task_type == "summarization"
    assert len(calls) == 3


def test_raises_after_exhausting_retries():
    def handler(request):
        return httpx.Response(500, json={"detail": "down"})

    client = make_client(handler, max_retries=1, backoff_seconds=0)

    with pytest.raises(PreclassifierError):
        client.classify([{"role": "user", "content": "hi"}])


def test_client_side_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(422, json={"detail": "bad request"})

    client = make_client(handler, max_retries=3, backoff_seconds=0)

    with pytest.raises(PreclassifierError):
        client.classify([{"role": "user", "content": "hi"}])

    assert len(calls) == 1


def test_context_manager_closes_the_underlying_client():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    with make_client(handler) as client:
        client.health()

    with pytest.raises(RuntimeError):
        client.health()
