# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- Optional offline semantic layer (`[semantic]` extra, `SEMANTIC_MODEL`): nearest-exemplar voting over local `bge-small-en-v1.5` embeddings. Escalation is OR-ed with the rules; task type falls back to the semantic vote when rules have no or conflicting signals.
- `runtime-semantic` Docker target with baked-in weights that runs with no network; now the Docker Compose default (memory limit raised to 512m).
- `--semantic` flag for the evaluation CLI, `make eval-semantic`, a semantic CI job, and a second blind set (`blind-v2`) written after the exemplars were frozen.
- Versioned routing policy file (`POLICY_PATH`, default `data/policy.yaml`) holding patterns, category order, confidence values and semantic thresholds. Validated at startup; `python -m llm_preclassifier.validate_policy` checks a file, `--policy` evaluates one, and `GET /v1/policy` reports the active version and SHA-256.
- Optional feedback endpoint (`ENABLE_FEEDBACK`, `FEEDBACK_LOG_PATH`): `POST /v1/feedback` records a correction against a decision's `decision_id`. Metadata only — the schema has no field for prompt content and rejects unknown keys.
- Versioned model catalog (`MODEL_CATALOG_PATH`, default `data/model_catalog.yaml`) mapping each abstract `recommended_model_tier` to concrete provider/model picks with illustrative per-million-token costs. Validated at startup; `python -m llm_preclassifier.validate_catalog` checks a file, `--catalog` attaches recommendations during evaluation, and `GET /v1/model-catalog` reports the active version, SHA-256 and currency.
- Optional OpenAI-compatible proxy mode (`ENABLE_PROXY`, `PROXY_UPSTREAM_BASE_URL`, `PROXY_UPSTREAM_API_KEY`, `PROXY_TIMEOUT_SECONDS`): `POST /v1/chat/completions` classifies locally, then forwards the request unmodified to a configured upstream, returning its response byte-for-byte with the local decision attached as an `X-Preclassifier-Decision` header. Streaming is not supported yet.
- Optional Prometheus metrics (`ENABLE_METRICS`): `GET /metrics` mirrors the `/status` counters in Prometheus text format.
- Optional Redis-backed decision cache (`REDIS_URL`, `redis` extra) so multiple instances can share cached decisions instead of each keeping an independent in-memory LRU.
- Optional per-key rate limiting (`RATE_LIMIT_PER_MINUTE`) on the classification, feedback and proxy endpoints, keyed by bearer token or client IP; returns `429` once exceeded.
- Bundled learned task model (`data/task_model.bin`, `learned` policy section): a hashed n-gram logistic regression trained on seven pinned public datasets, with benchmark prompts excluded. It relabels the rules' `chat`/`classification` fallback when it is at least 60% confident (reason `learned_task_model`), in pure Python with no new runtime dependencies. `training/train_task_model.py` reproduces it (`[train]` extra).
- `llm_preclassifier.client.PreclassifierClient`: a small synchronous SDK wrapping `/v1/classify`, `/v1/feedback`, `/v1/policy`, `/v1/model-catalog` and `/healthz`, with retry/backoff on connection errors and 5xx responses. See [`examples/python/basic_usage.py`](examples/python/basic_usage.py).

### Changed

- Classification runs in a worker thread so CPU-bound embedding never blocks the event loop.
- `policy_version` in decisions is now the policy file's version (`2026.09.1` by default) instead of the fixed `v1`. The default policy reproduces the previous rules exactly.
- Decisions now carry a `decision_id` (opaque, stable across cache hits for the same request) for correlating `/v1/feedback` submissions.
- Decisions now carry `model_recommendations`, populated from the active model catalog for the four actionable tiers (empty for `unknown` and `human_or_policy_review`).
- Default policy version is now `2026.09.2` (adds the `learned` section). Policies without `learned` still validate and use the default.
- New runtime dependencies: `pyyaml`, `httpx`, `prometheus-client`. New optional extra: `redis`.

### Results

- Rules-only blind benchmarks with the learned model: 37.5% → 45.8% and 26.7% → 33.3%.
- Blind benchmarks: 37.5% → 79.2% and 26.7% → 93.3%; high-stakes escalation 15/15 (was 5/15), with 2/8 false escalations on benign look-alikes.

## 0.1.0 — Unreleased

### Added

- Offline deterministic task preclassification API with an inspectable, versioned response.
- Conservative `unknown` and `escalate` outcomes for ambiguity and selected policy-sensitive signals.
- Loopback-bound hardened Docker Compose configuration.
- Request limits, production credential enforcement, metadata-only optional decision logs, and security/privacy documentation.
- Interactive GitHub Pages launch site.
- Offline accuracy evaluation (`python -m llm_preclassifier.evaluation`) with per-class precision/recall, confusion matrix, calibration buckets, a 48-case seed dataset and a CI accuracy gate.
- Held-out and blind evaluation sets; CI gates the regression sets at 95% and reports the blind benchmark.
- `mixed_signals` reason with reduced confidence when several task categories match.

### Fixed

- Cache key now covers `policy_flags`, `available_tools` and full tool history, so a cached `route` can no longer mask a `high_stakes` escalation.
- Chunked request bodies are size-checked while being read instead of after full buffering.
- Settings read environment variables at construction, and importing `llm_preclassifier.api` no longer builds an app (Docker uses `uvicorn --factory`).
- `/status` reports all counters, including zeros.
- Narrowed coding and `send` patterns that misclassified everyday requests.
- Escalation now catches dosage, overdose and self-harm phrasings (`dosage`, `hurt myself`) that were previously routed.
- External actions (issues, pull requests, messages, meetings) classify as `agent_action` even when coding words are present.
- Unrecognised short prompts fall back to `chat` at 0.5 confidence with a `no_category_signal` reason; greetings keep high confidence.
- README configuration table now lists the environment variables the service actually reads.

### Known limitations

- The rules are not a learned model-quality predictor and have not been evaluated as a universal routing benchmark.
- Rules-only blind accuracy is 37.5%; unseen high-stakes phrasings are routed rather than escalated unless the semantic layer is enabled.
- Capability tiers are abstract; applications must map them to their own providers and policies.
- The service does not execute tools or make safety guarantees. Model completions are only proxied when `ENABLE_PROXY` is explicitly turned on, and even then the request is forwarded unmodified — the service never picks a model or rewrites a prompt on the caller's behalf.
