import json

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_preclassifier.api import create_app
from llm_preclassifier.config import Settings
from llm_preclassifier.policy import PolicyError


def client(**overrides):
    settings = Settings(log_decisions=False, **overrides)
    return TestClient(create_app(settings))


def test_healthz_requires_no_dependency():
    response = client().get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_versioned_explainable_decision():
    response = client().post(
        "/v1/classify",
        json={"messages": [{"role": "user", "content": "Summarise this report."}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["task_type"] == "summarization"
    assert payload["recommended_model_tier"] == "economy"
    assert payload["reasons"]
    assert "prompt" not in payload


def test_route_is_a_backward_compatible_alias():
    payload = {"messages": [{"role": "user", "content": "Extract the total from this receipt."}]}

    response = client().post("/v1/route", json=payload)

    assert response.status_code == 200
    assert response.json()["task_type"] == "extraction"


def test_production_requires_client_credentials():
    try:
        create_app(Settings(environment="production", client_api_keys=""))
    except RuntimeError as error:
        assert "CLIENT_API_KEYS" in str(error)
    else:
        raise AssertionError("production mode must fail closed without credentials")


def test_configured_token_protects_classification_endpoint():
    secured = client(client_api_keys="test-token")
    request = {"messages": [{"role": "user", "content": "Hello"}]}

    assert secured.post("/v1/classify", json=request).status_code == 401
    assert secured.post(
        "/v1/classify",
        json=request,
        headers={"Authorization": "Bearer test-token"},
    ).status_code == 200


def test_request_size_limit_rejects_large_bodies():
    limited = client(max_request_bytes=1024)

    response = limited.post("/v1/classify", content=b"x" * 1025, headers={"content-type": "application/json"})

    assert response.status_code == 413


def test_policy_flags_are_not_masked_by_cached_decision():
    api = client()
    messages = [{"role": "user", "content": "Summarise this report."}]

    first = api.post("/v1/classify", json={"messages": messages})
    flagged = api.post("/v1/classify", json={"messages": messages, "policy_flags": ["high_stakes"]})

    assert first.json()["action"] == "route"
    assert flagged.json()["action"] == "escalate"


def test_available_tools_are_part_of_cache_identity():
    api = client()
    messages = [{"role": "user", "content": "Explain why this design is slow."}]

    without_tools = api.post("/v1/classify", json={"messages": messages})
    with_tools = api.post("/v1/classify", json={"messages": messages, "available_tools": ["terminal"]})

    assert without_tools.json()["tool_requirement"] == "none"
    assert with_tools.json()["tool_requirement"] == "optional"


def test_tool_history_outside_recent_turns_is_part_of_cache_identity():
    api = client()
    tail = [{"role": "user", "content": f"note {index}"} for index in range(5)]
    tail.append({"role": "user", "content": "Explain why this design is slow."})
    with_history = [{"role": "assistant", "tool_calls": [{"id": "1"}]}, *tail]

    plain = api.post("/v1/classify", json={"messages": tail})
    historic = api.post("/v1/classify", json={"messages": with_history})

    assert plain.json()["tool_requirement"] == "none"
    assert historic.json()["tool_requirement"] == "optional"


def test_chunked_body_over_limit_is_rejected():
    limited = client(max_request_bytes=1024)

    response = limited.post(
        "/v1/classify",
        content=iter([b"x" * 600, b"x" * 600]),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_chunked_body_within_limit_is_classified():
    body = b'{"messages":[{"role":"user","content":"Summarise this report."}]}'

    response = client().post("/v1/classify", content=iter([body[:20], body[20:]]),
                             headers={"content-type": "application/json"})

    assert response.status_code == 200
    assert response.json()["task_type"] == "summarization"


def test_status_reports_zero_counters_before_traffic():
    response = client().get("/status")

    assert response.json() == {
        "classifications_total": 0,
        "classifications_computed": 0,
        "classifications_cache_hits": 0,
        "decision_logs_written": 0,
        "feedback_received": 0,
        "proxy_requests_total": 0,
    }


def test_semantic_layer_is_off_by_default():
    assert Settings().semantic_model == ""


def test_app_passes_semantic_classifier_to_classify(monkeypatch):
    import llm_preclassifier.api as api

    class AlwaysMedical:
        def assess(self, text):
            from llm_preclassifier.semantic import SemanticVerdict

            return SemanticVerdict("medical", None, 0.0)

    monkeypatch.setattr(api, "_build_semantic", lambda settings, policy: AlwaysMedical())
    response = TestClient(api.create_app(Settings(log_decisions=False))).post(
        "/v1/classify", json={"messages": [{"role": "user", "content": "Summarise this report."}]},
    )

    assert response.json()["action"] == "escalate"
    assert "semantic_high_stakes:medical" in response.json()["reasons"]


def test_policy_endpoint_reports_active_policy_and_requires_auth(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    source = Path(__file__).resolve().parents[1] / "src" / "llm_preclassifier" / "data" / "policy.yaml"
    policy_file.write_text(source.read_text().replace("version: '2026.09.1'", "version: 'acme-7'"))
    api = client(policy_path=str(policy_file), client_api_keys="secret")

    assert api.get("/v1/policy").status_code == 401
    body = api.get("/v1/policy", headers={"Authorization": "Bearer secret"}).json()
    decision = api.post(
        "/v1/classify",
        headers={"Authorization": "Bearer secret"},
        json={"messages": [{"role": "user", "content": "Summarise this report."}]},
    ).json()

    assert body["version"] == "acme-7"
    assert len(body["sha256"]) == 64
    assert decision["policy_version"] == "acme-7"


def test_invalid_policy_fails_at_startup(tmp_path):
    broken = tmp_path / "policy.yaml"
    broken.write_text("version: '1'\n")

    with pytest.raises(PolicyError, match="missing keys"):
        create_app(Settings(policy_path=str(broken)))


def test_feedback_endpoint_disabled_by_default():
    api = client()

    response = api.post("/v1/feedback", json={"decision_id": "abc", "outcome": "correct"})

    assert response.status_code == 404


def test_feedback_endpoint_records_correction_metadata(tmp_path):
    log_path = tmp_path / "feedback.jsonl"
    api = client(enable_feedback=True, feedback_log_path=str(log_path))
    decision = api.post(
        "/v1/classify",
        json={"messages": [{"role": "user", "content": "Summarise this report."}]},
    ).json()

    response = api.post("/v1/feedback", json={
        "decision_id": decision["decision_id"],
        "outcome": "incorrect",
        "corrected_task_type": "extraction",
    })

    assert response.status_code == 200
    assert response.json() == {"status": "recorded"}
    record = json.loads(log_path.read_text())
    assert record["decision_id"] == decision["decision_id"]
    assert record["corrected_task_type"] == "extraction"
    assert "prompt" not in record


def test_feedback_requires_auth_when_configured(tmp_path):
    log_path = tmp_path / "feedback.jsonl"
    api = client(enable_feedback=True, feedback_log_path=str(log_path), client_api_keys="secret")

    response = api.post("/v1/feedback", json={"decision_id": "abc", "outcome": "correct"})

    assert response.status_code == 401


def test_feedback_rejects_unknown_fields(tmp_path):
    log_path = tmp_path / "feedback.jsonl"
    api = client(enable_feedback=True, feedback_log_path=str(log_path))

    response = api.post("/v1/feedback", json={
        "decision_id": "abc", "outcome": "correct", "prompt": "leaked content",
    })

    assert response.status_code == 422


def test_classify_response_includes_model_recommendations():
    response = client().post(
        "/v1/classify",
        json={"messages": [{"role": "user", "content": "Summarise this report."}]},
    )

    recommendations = response.json()["model_recommendations"]
    assert recommendations
    assert {"provider", "model", "input_cost_per_million", "output_cost_per_million", "currency"} <= set(
        recommendations[0]
    )


def test_model_catalog_endpoint_reports_active_catalog_and_requires_auth(tmp_path):
    catalog_file = tmp_path / "model_catalog.yaml"
    source = Path(__file__).resolve().parents[1] / "src" / "llm_preclassifier" / "data" / "model_catalog.yaml"
    catalog_file.write_text(source.read_text().replace("version: '2026.09.1'", "version: 'acme-catalog-1'"))
    api = client(model_catalog_path=str(catalog_file), client_api_keys="secret")

    assert api.get("/v1/model-catalog").status_code == 401
    body = api.get("/v1/model-catalog", headers={"Authorization": "Bearer secret"}).json()

    assert body["version"] == "acme-catalog-1"
    assert len(body["sha256"]) == 64
    assert body["currency"] == "USD"


def test_invalid_model_catalog_fails_at_startup(tmp_path):
    from llm_preclassifier.catalog import ModelCatalogError

    broken = tmp_path / "model_catalog.yaml"
    broken.write_text("version: '1'\n")

    with pytest.raises(ModelCatalogError, match="missing keys"):
        create_app(Settings(model_catalog_path=str(broken)))


def test_proxy_endpoint_disabled_by_default():
    api = client()

    response = api.post("/v1/chat/completions", json={"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 404


def test_proxy_requires_upstream_configuration():
    with pytest.raises(RuntimeError, match="PROXY_UPSTREAM_BASE_URL"):
        create_app(Settings(enable_proxy=True, proxy_upstream_api_key="k"))
    with pytest.raises(RuntimeError, match="PROXY_UPSTREAM_API_KEY"):
        create_app(Settings(enable_proxy=True, proxy_upstream_base_url="https://api.example.com/v1"))


def test_proxy_forwards_unmodified_payload_and_attaches_decision(monkeypatch):
    import llm_preclassifier.api as api_module

    captured = {}

    async def fake_forward(payload, *, base_url, api_key, timeout_seconds):
        captured["payload"] = payload
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return {"id": "chatcmpl-1", "choices": []}, 200

    monkeypatch.setattr(api_module, "forward_chat_completion", fake_forward)
    api = client(
        enable_proxy=True,
        proxy_upstream_base_url="https://api.example.com/v1",
        proxy_upstream_api_key="upstream-secret",
    )

    response = api.post("/v1/chat/completions", json={
        "model": "gpt-x",
        "messages": [{"role": "user", "content": "Summarise this report."}],
        "available_tools": ["terminal"],
    })

    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl-1", "choices": []}
    assert "available_tools" not in captured["payload"]
    assert captured["payload"]["model"] == "gpt-x"
    assert captured["base_url"] == "https://api.example.com/v1"
    decision = json.loads(response.headers["X-Preclassifier-Decision"])
    assert decision["task_type"] == "summarization"


def test_proxy_requires_auth_when_configured():
    api = client(
        enable_proxy=True,
        proxy_upstream_base_url="https://api.example.com/v1",
        proxy_upstream_api_key="upstream-secret",
        client_api_keys="secret",
    )

    response = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 401


def test_proxy_rejects_missing_messages():
    api = client(
        enable_proxy=True,
        proxy_upstream_base_url="https://api.example.com/v1",
        proxy_upstream_api_key="upstream-secret",
    )

    response = api.post("/v1/chat/completions", json={"model": "gpt-x"})

    assert response.status_code == 422


def test_proxy_returns_bad_gateway_on_upstream_failure(monkeypatch):
    import llm_preclassifier.api as api_module
    from llm_preclassifier.proxy import UpstreamError

    async def failing_forward(payload, *, base_url, api_key, timeout_seconds):
        raise UpstreamError("connection refused")

    monkeypatch.setattr(api_module, "forward_chat_completion", failing_forward)
    api = client(
        enable_proxy=True,
        proxy_upstream_base_url="https://api.example.com/v1",
        proxy_upstream_api_key="upstream-secret",
    )

    response = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 502
