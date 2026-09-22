"""FastAPI application factory with private-by-default production controls."""
from __future__ import annotations

import hmac
from collections import Counter

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from llm_preclassifier.audit import append_decision_log
from llm_preclassifier.cache import ClassificationCache
from llm_preclassifier.classifier import classify
from llm_preclassifier.config import Settings
from llm_preclassifier.schemas import ClassificationDecision, ClassificationRequest, HealthResponse
from llm_preclassifier.utils import _classification_cache_key


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
    metrics: Counter[str] = Counter()

    @app.middleware("http")
    async def enforce_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                exceeds_limit = int(content_length) > settings.max_request_bytes
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
            if exceeds_limit:
                return JSONResponse({"detail": "request body exceeds MAX_REQUEST_BYTES"}, status_code=413)
        elif request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > settings.max_request_bytes:
                return JSONResponse({"detail": "request body exceeds MAX_REQUEST_BYTES"}, status_code=413)
        return await call_next(request)

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
        cache_key = _classification_cache_key(messages)
        decision = cache.get(cache_key)
        if decision is None:
            decision = classify(messages, {
                "available_tools": payload.available_tools,
                "policy_flags": payload.policy_flags,
            })
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

    return app


app = create_app()
