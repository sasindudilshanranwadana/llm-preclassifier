"""FastAPI application factory with private-by-default production controls."""
from __future__ import annotations

import hmac

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_preclassifier.audit import append_decision_log, append_feedback_log
from llm_preclassifier.cache import ClassificationCache, RedisClassificationCache
from llm_preclassifier.catalog import load_catalog
from llm_preclassifier.classifier import classify
from llm_preclassifier.config import Settings
from llm_preclassifier.metrics import Metrics
from llm_preclassifier.policy import Policy, load_policy
from llm_preclassifier.proxy import UpstreamError, forward_chat_completion
from llm_preclassifier.ratelimit import RateLimiter
from llm_preclassifier.schemas import (
    ClassificationDecision,
    ClassificationRequest,
    FeedbackAck,
    FeedbackRequest,
    HealthResponse,
)
from llm_preclassifier.utils import _classification_cache_key

_METRIC_NAMES = (
    "classifications_total",
    "classifications_computed",
    "classifications_cache_hits",
    "decision_logs_written",
    "feedback_received",
    "proxy_requests_total",
    "rate_limited_total",
)
_PROXY_ONLY_KEYS = ("available_tools", "policy_flags")


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


def _build_cache(settings: Settings) -> ClassificationCache[ClassificationDecision] | RedisClassificationCache:
    if not settings.redis_url:
        return ClassificationCache(max_entries=settings.cache_max_entries, ttl_seconds=settings.cache_ttl_seconds)
    try:
        import redis
    except ImportError as error:
        raise RuntimeError("REDIS_URL requires: pip install llm-preclassifier[redis]") from error
    client = redis.Redis.from_url(settings.redis_url)
    return RedisClassificationCache(
        client,
        settings.cache_ttl_seconds,
        serialize=lambda decision: decision.model_dump_json(),
        deserialize=ClassificationDecision.model_validate_json,
    )


def _rate_limit_key(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer "):]
    return request.client.host if request.client else "unknown"


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
    cache = _build_cache(settings)
    # A bad policy file fails at startup, not on the first request.
    policy = load_policy(settings.policy_path)
    catalog = load_catalog(settings.model_catalog_path)
    semantic = _build_semantic(settings, policy)
    metrics = Metrics(_METRIC_NAMES, enable_prometheus=settings.enable_metrics)
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    def enforce_rate_limit(request: Request) -> None:
        if not rate_limiter.allow(_rate_limit_key(request)):
            metrics.increment("rate_limited_total")
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")

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
        enforce_rate_limit(request)
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
            }, semantic=semantic, policy=policy, catalog=catalog)
            cache.put(cache_key, decision)
            metrics.increment("classifications_computed")
        else:
            metrics.increment("classifications_cache_hits")
        metrics.increment("classifications_total")
        if settings.log_decisions:
            append_decision_log(settings.decision_log_path, decision)
            metrics.increment("decision_logs_written")
        return decision

    @app.post("/v1/route", response_model=ClassificationDecision, include_in_schema=False)
    async def route_alias(request: Request, payload: ClassificationRequest) -> ClassificationDecision:
        return await classify_request(request, payload)

    @app.get("/status", tags=["operational"])
    async def runtime_status(request: Request) -> dict[str, int]:
        authenticate(request)
        return metrics.status()

    @app.get("/v1/policy", tags=["operational"])
    async def active_policy(request: Request) -> dict[str, str]:
        authenticate(request)
        return {"version": policy.version, "sha256": policy.sha256}

    @app.get("/v1/model-catalog", tags=["operational"])
    async def active_model_catalog(request: Request) -> dict:
        authenticate(request)
        return {"version": catalog.version, "sha256": catalog.sha256, "currency": catalog.currency}

    if settings.enable_metrics:
        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics() -> Response:
            body, content_type = metrics.render_prometheus()
            return Response(body, media_type=content_type)

    if settings.enable_feedback:
        @app.post("/v1/feedback", response_model=FeedbackAck, tags=["classification"])
        async def submit_feedback(request: Request, payload: FeedbackRequest) -> FeedbackAck:
            authenticate(request)
            enforce_rate_limit(request)
            append_feedback_log(settings.feedback_log_path, payload)
            metrics.increment("feedback_received")
            return FeedbackAck(status="recorded")

    if settings.enable_proxy:
        @app.post("/v1/chat/completions", tags=["proxy"])
        async def proxy_chat_completions(request: Request) -> JSONResponse:
            authenticate(request)
            enforce_rate_limit(request)
            body = await request.json()
            messages = body.get("messages") if isinstance(body, dict) else None
            if not isinstance(messages, list) or not messages:
                raise HTTPException(status_code=422, detail="messages is required")
            if len(messages) > settings.max_messages:
                raise HTTPException(status_code=422, detail="messages exceeds MAX_MESSAGES")
            decision = await run_in_threadpool(classify, messages, {
                "available_tools": body.get("available_tools", []),
                "policy_flags": body.get("policy_flags", []),
            }, semantic=semantic, policy=policy, catalog=catalog)
            metrics.increment("proxy_requests_total")
            upstream_payload = {key: value for key, value in body.items() if key not in _PROXY_ONLY_KEYS}
            try:
                upstream_body, upstream_status = await forward_chat_completion(
                    upstream_payload,
                    base_url=settings.proxy_upstream_base_url,
                    api_key=settings.proxy_upstream_api_key,
                    timeout_seconds=settings.proxy_timeout_seconds,
                )
            except UpstreamError as error:
                raise HTTPException(status_code=502, detail=f"upstream error: {error}") from error
            return JSONResponse(
                upstream_body,
                status_code=upstream_status,
                headers={"X-Preclassifier-Decision": decision.model_dump_json()},
            )

    return app
