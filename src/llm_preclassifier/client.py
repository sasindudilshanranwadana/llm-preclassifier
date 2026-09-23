"""A small, dependency-light client for calling a running llm-preclassifier instance.

This is a thin HTTP wrapper — it does not reimplement classification. Point it at
your own deployment's base URL and, if configured, a client API key.
"""
from __future__ import annotations

import time
from types import TracebackType
from typing import Any

import httpx

from llm_preclassifier.schemas import ClassificationDecision, FeedbackAck


class PreclassifierError(RuntimeError):
    """Raised for non-2xx responses and transport failures, after retries are exhausted."""


class PreclassifierClient:
    """Synchronous client for ``/v1/classify``, ``/v1/feedback``, ``/v1/policy`` and ``/v1/model-catalog``.

    Retries idempotent GETs and ``/v1/classify`` on connection errors and 5xx responses,
    with exponential backoff. ``/v1/feedback`` is not retried automatically, since a retry
    after an ambiguous failure could record a correction twice.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds, headers=headers)

    def __enter__(self) -> "PreclassifierClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def classify(
        self,
        messages: list[dict],
        available_tools: list[str] | None = None,
        policy_flags: list[str] | None = None,
    ) -> ClassificationDecision:
        payload = {
            "messages": messages,
            "available_tools": available_tools or [],
            "policy_flags": policy_flags or [],
        }
        body = self._request("POST", "/v1/classify", retry=True, json=payload)
        return ClassificationDecision.model_validate(body)

    def feedback(
        self,
        decision_id: str,
        outcome: str,
        corrected_task_type: str | None = None,
        corrected_complexity: str | None = None,
        corrected_tool_requirement: str | None = None,
        corrected_recommended_model_tier: str | None = None,
    ) -> FeedbackAck:
        payload = {
            key: value
            for key, value in {
                "decision_id": decision_id,
                "outcome": outcome,
                "corrected_task_type": corrected_task_type,
                "corrected_complexity": corrected_complexity,
                "corrected_tool_requirement": corrected_tool_requirement,
                "corrected_recommended_model_tier": corrected_recommended_model_tier,
            }.items()
            if value is not None
        }
        body = self._request("POST", "/v1/feedback", retry=False, json=payload)
        return FeedbackAck.model_validate(body)

    def policy(self) -> dict[str, str]:
        return self._request("GET", "/v1/policy", retry=True)

    def model_catalog(self) -> dict[str, str]:
        return self._request("GET", "/v1/model-catalog", retry=True)

    def health(self) -> dict[str, str]:
        return self._request("GET", "/healthz", retry=True)

    def _request(self, method: str, path: str, *, retry: bool, json: dict | None = None) -> Any:
        attempts = self.max_retries + 1 if retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, path, json=json)
            except httpx.HTTPError as error:
                last_error = error
            else:
                if response.status_code < 500:
                    if response.status_code >= 400:
                        raise PreclassifierError(
                            f"{method} {path} failed: {response.status_code} {response.text}"
                        )
                    return response.json()
                last_error = PreclassifierError(f"{method} {path} failed: {response.status_code} {response.text}")
            if attempt < attempts - 1:
                time.sleep(self.backoff_seconds * (2 ** attempt))
        raise PreclassifierError(f"{method} {path} failed after {attempts} attempt(s): {last_error}") from last_error
