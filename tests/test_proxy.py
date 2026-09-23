import asyncio

import httpx
import pytest

from llm_preclassifier.proxy import UpstreamError, forward_chat_completion


def test_forward_chat_completion_returns_upstream_body_and_status(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer upstream-key"
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda base_url, timeout: real_async_client(
        base_url=base_url, timeout=timeout, transport=httpx.MockTransport(handler),
    ))

    body, status = asyncio.run(forward_chat_completion(
        {"model": "gpt-x", "messages": []},
        base_url="https://api.example.com/v1",
        api_key="upstream-key",
        timeout_seconds=5,
    ))

    assert status == 200
    assert body == {"id": "chatcmpl-1"}


def test_forward_chat_completion_wraps_transport_errors(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda base_url, timeout: real_async_client(
        base_url=base_url, timeout=timeout, transport=httpx.MockTransport(handler),
    ))

    with pytest.raises(UpstreamError):
        asyncio.run(forward_chat_completion(
            {"model": "gpt-x", "messages": []},
            base_url="https://api.example.com/v1",
            api_key="upstream-key",
            timeout_seconds=5,
        ))


def test_forward_chat_completion_handles_non_json_upstream_response(monkeypatch):
    def handler(request):
        return httpx.Response(502, text="bad gateway")

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda base_url, timeout: real_async_client(
        base_url=base_url, timeout=timeout, transport=httpx.MockTransport(handler),
    ))

    body, status = asyncio.run(forward_chat_completion(
        {"model": "gpt-x", "messages": []},
        base_url="https://api.example.com/v1",
        api_key="upstream-key",
        timeout_seconds=5,
    ))

    assert status == 502
    assert body == {"error": "upstream returned a non-JSON response"}
