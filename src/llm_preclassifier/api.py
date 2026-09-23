"""FastAPI application factory with private-by-default production controls."""
from __future__ import annotations

import hmac
from collections import Counter

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_preclassifier.audit import append_decision_log
from llm_preclassifier.cache import ClassificationCache
from llm_preclassifier.classifier import classify
from llm_preclassifier.config import Settings
from llm_preclassifier.policy import Policy, load_policy
from llm_preclassifier.schemas import ClassificationDecision, ClassificationRequest, HealthResponse
from llm_preclassifier.utils import _classification_cache_key

_METRIC_NAMES = (
    "classifications_total",
    "classifications_computed",
    "classifications_cache_hits",
    "decision_logs_written",
)


class RequestSizeLimitMiddleware:
    """Reject oversized bodies, never buffering more than max_bytes of a chunked upload."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await JSONResponse({"detail": "invalid content-length"}, status_code=400)(scope, receive, send)
                return
            if declared > self.max_bytes:
                await _too_large()(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        received = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunk = message.get("body", b"")
            received += len(chunk)
            if received > self.max_bytes:
                await _too_large()(scope, receive, send)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)


def _too_large() -> JSONResponse:
    return JSONResponse({"detail": "request body exceeds MAX_REQUEST_BYTES"}, status_code=413)


def _build_semantic(settings: Settings, policy: Policy):
    if not settings.semantic_model:
        return None
    try:
        from llm_preclassifier.semantic import build_semantic_classifier
    except ImportError as error:
        raise RuntimeError("SEMANTIC_MODEL requires: pip install llm-preclassifier[semantic]") from error
    return build_semantic_classifier(settings.semantic_model, settings.semantic_cache_dir, policy)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate()
    app = FastAPI(
        title="llm-preclassifier",
        version="0.1.0",
        description="Self-hosted, provider-neutral preflight classification for agent requests.",
        docs_url="/docs",
        redoc_url=None,
    )
    cache: ClassificationCache[ClassificationDecision] = ClassificationCache(
        max_entries=settings.cache_max_entries,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    # A bad policy file fails at startup, not on the first request.
    policy = load_policy(settings.policy_path)
    semantic = _build_semantic(settings, policy)
    metrics: Counter[str] = Counter({name: 0 for name in _METRIC_NAMES})

    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    def authenticate(request: Request) -> None:
        if not settings.api_keys:
            return
        authorization = request.headers.get("authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bearer token required")
        supplied = authorization[len(prefix):]
        if not any(hmac.compare_digest(supplied, expected) for expected in settings.api_keys):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")

    @app.get("/healthz", response_model=HealthResponse, tags=["operational"])
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/v1/classify", response_model=ClassificationDecision, tags=["classification"])
    async def classify_request(request: Request, payload: ClassificationRequest) -> ClassificationDecision:
        authenticate(request)
        if len(payload.messages) > settings.max_messages:
            raise HTTPException(status_code=422, detail="messages exceeds MAX_MESSAGES")
        messages = [message.model_dump() for message in payload.messages]
        cache_key = _classification_cache_key(messages, payload.available_tools, payload.policy_flags)
        decision = cache.get(cache_key)
        if decision is None:
            # Embedding is CPU-bound; keep it off the event loop.
            decision = await run_in_threadpool(classify, messages, {
                "available_tools": payload.available_tools,
                "policy_flags": payload.policy_flags,
            }, semantic=semantic, policy=policy)
            cache.put(cache_key, decision)
            metrics["classifications_computed"] += 1
        else:
            metrics["classifications_cache_hits"] += 1
        metrics["classifications_total"] += 1
        if settings.log_decisions:
            append_decision_log(settings.decision_log_path, decision)
            metrics["decision_logs_written"] += 1
        return decision

    @app.post("/v1/route", response_model=ClassificationDecision, include_in_schema=False)
    async def route_alias(request: Request, payload: ClassificationRequest) -> ClassificationDecision:
        return await classify_request(request, payload)

    @app.get("/status", tags=["operational"])
    async def runtime_status(request: Request) -> dict[str, int]:
        authenticate(request)
        return dict(metrics)

    @app.get("/v1/policy", tags=["operational"])
    async def active_policy(request: Request) -> dict[str, str]:
        authenticate(request)
        return {"version": policy.version, "sha256": policy.sha256}

    return app
