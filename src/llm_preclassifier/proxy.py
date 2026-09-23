"""Optional OpenAI-compatible proxy: classify locally, then forward unmodified upstream.

The service never rewrites the caller's model choice or message content — it only
attaches the local classification decision as a response header. Streaming
(``stream: true``) is not supported yet; upstream is always called non-streaming.
"""
from __future__ import annotations

import httpx


class UpstreamError(RuntimeError):
    """Raised when the upstream chat completions call fails or times out."""


async def forward_chat_completion(
    payload: dict,
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[dict, int]:
    """POST ``payload`` to ``{base_url}/chat/completions``; return (body, status_code)."""
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds) as client:
        try:
            response = await client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.HTTPError as error:
            raise UpstreamError(str(error)) from error
    try:
        body = response.json()
    except ValueError:
        body = {"error": "upstream returned a non-JSON response"}
    return body, response.status_code
