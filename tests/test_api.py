from fastapi.testclient import TestClient

from llm_preclassifier.api import create_app
from llm_preclassifier.config import Settings


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
