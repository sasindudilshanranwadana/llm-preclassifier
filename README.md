<div align="center">
  <img src="assets/3d-hero.svg" alt="Animated LLM Preclassifier Hero" width="100%">
</div>

<div align="center">
  <a href="https://github.com/sasindudilshanranwadana/llm-preclassifier/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/sasindudilshanranwadana/llm-preclassifier/ci.yml?branch=main&label=CI&color=08D9D6" alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-FF2E63.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.11-7B2FF7.svg" alt="Python">
  </a>
  <a href="Dockerfile">
    <img src="https://img.shields.io/badge/docker-ready-FFD400.svg" alt="Docker">
  </a>
</div>

<br>

> **A self-hosted, provider-neutral preflight classifier for agent requests.** 
> Intercept, classify, and route incoming agent tasks _before_ calling a heavy LLM or executing a tool.

<br>

### THE PROBLEM & THE SOLUTION

| ❌ The Problem | 🟩 The Solution |
|---|---|
| Agents send **all** prompts through the same slow, expensive state-of-the-art LLM gateway. | Route **simple** tasks to fast local/economy models, saving tokens and latency. |
| Tool boundaries are granted blindly based on the user's prompt wrapper. | Detect **explicit tool needs** upfront and fail-fast if unauthorized. |
| Black-box gateways obscure _why_ a model was selected. | Get an **inspectable, deterministic, offline** policy decision before execution. |

<br>

## ⚡ ARCHITECTURE

<div align="center">
  <img src="assets/architecture.svg" alt="Architecture Flowchart" width="100%">
</div>

`llm-preclassifier` is a decision sidecar. It receives your agent's input, applies a configurable standard of rules, and issues a structured routing verdict. **It does not execute tools, forward API keys, or hallucinate.**

<br>

## 🚀 QUICK START

The service is distributed as a hardened, non-root Docker container.

```console
$ docker compose up --build -d
[+] Building 6.2s (13/13) FINISHED
[+] Running 1/1
 ✔ Container llm-preclassifier-runtime-1  Started
```

Check the health status:
```console
$ curl -sS http://127.0.0.1:8802/healthz
{"status":"ok"}
```

<br>

## 📊 DECISION ENGINE (API)

Send a raw prompt array to the `/v1/classify` endpoint. 

**Request:**
```json
curl -X POST http://127.0.0.1:8802/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Summarise this report."}]}'
```

**Verdict:**
```json
{
  "version": "v1",
  "task_type": "summarization",
  "complexity": "simple",
  "tool_requirement": "none",
  "recommended_model_tier": "economy",
  "confidence": 0.88,
  "action": "route",
  "reasons": ["offline_rules_v1"],
  "policy_version": "2026.09.1",
  "decision_id": "3f8e2c1a9d7b4f0e8c6a5d2b1e9f7a3c",
  "model_recommendations": [
    {"provider": "anthropic", "model": "claude-haiku-4-5", "input_cost_per_million": 1.0,
     "output_cost_per_million": 5.0, "currency": "USD", "notes": "Cheapest general-purpose option; good for extraction, summarization, chat."}
  ]
}
```

`decision_id` is an opaque per-decision identifier (stable across cache hits for the same request) for referencing the decision from `/v1/feedback`. `model_recommendations` maps the abstract `recommended_model_tier` to concrete provider/model picks from the versioned, operator-editable [model catalog](src/llm_preclassifier/data/model_catalog.yaml) — empty for `unknown` and `human_or_policy_review`.

<br>

## ⚙️ CAPABILITIES

> [!NOTE]
> **V0.1 represents the Offline Tier.** Deterministic logic rules are applied in-memory. Zero outbound requests are made to any provider.

<details>
<summary><b>View supported configuration variables</b></summary>
<br>

Configuration is entirely environment-driven. Do not commit `.env` files.

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PRECLASSIFIER_ENV` | `development` | `development` or `production`; production refuses to start without `CLIENT_API_KEYS`. |
| `CLIENT_API_KEYS` | `""` | Comma-separated bearer tokens. Empty disables auth (development only). |
| `MAX_REQUEST_BYTES` | `65536` | Request body limit, including chunked uploads (minimum `1024`). |
| `MAX_MESSAGES` | `32` | Maximum messages per request. |
| `CACHE_TTL_SECONDS` | `300` | Decision cache time-to-live. |
| `CACHE_MAX_ENTRIES` | `1024` | Decision cache capacity (LRU). |
| `LOG_DECISIONS` | `false` | Append decision metadata (never prompt content) as JSONL. |
| `DECISION_LOG_PATH` | `""` | JSONL path; required when `LOG_DECISIONS=true`. |
| `SEMANTIC_MODEL` | `""` | Embedding model for the semantic layer; empty disables it. Requires the `semantic` extra. |
| `SEMANTIC_CACHE_DIR` | `""` | Where model weights are cached (`/opt/models` in the `runtime-semantic` image). |
| `POLICY_PATH` | `""` | Custom routing policy YAML; empty uses the bundled [`policy.yaml`](src/llm_preclassifier/data/policy.yaml). Validated at startup. |
| `ENABLE_FEEDBACK` | `false` | Expose `POST /v1/feedback` for recording corrections against a `decision_id`. |
| `FEEDBACK_LOG_PATH` | `""` | JSONL path; required when `ENABLE_FEEDBACK=true`. |
| `MODEL_CATALOG_PATH` | `""` | Custom model catalog YAML; empty uses the bundled [`model_catalog.yaml`](src/llm_preclassifier/data/model_catalog.yaml). Validated at startup. |
| `ENABLE_PROXY` | `false` | Expose `POST /v1/chat/completions`: classify, then forward the request unmodified to `PROXY_UPSTREAM_BASE_URL`. |
| `PROXY_UPSTREAM_BASE_URL` | `""` | OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`); required when `ENABLE_PROXY=true`. |
| `PROXY_UPSTREAM_API_KEY` | `""` | Bearer key sent to the upstream; required when `ENABLE_PROXY=true`. Never accepted from the client. |
| `PROXY_TIMEOUT_SECONDS` | `60` | Upstream request timeout. |

The container listens on port `8802` (`uvicorn --factory llm_preclassifier.api:create_app`). Docker Compose builds the `runtime-semantic` target; use `--target runtime` for the lean rules-only image.

</details>

<details>
<summary><b>View privacy and security guarantees</b></summary>
<br>

- **In-memory Processing:** Prompts are processed strictly in RAM.
- **Zero Prompt Logging:** `LOG_DECISIONS=true` logs the _decision_ (e.g. `complexity: simple`), but strips all prompt content, user messages, and headers.
- **Hardened Runtime:** The Docker container drops all capabilities (`--cap-drop ALL`), runs read-only (`--read-only`), prevents privilege escalation (`no-new-privileges`), and uses a non-root user.

</details>

<br>

## 🧠 SEMANTIC LAYER (OPTIONAL, OFFLINE)

Keyword rules only catch the wording they were written for. The semantic layer embeds each request locally with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (ONNX via `fastembed`, CPU-only, no PyTorch) and votes among the nearest labeled examples in [`data/exemplars.json`](src/llm_preclassifier/data/exemplars.json).

- **Escalation is OR-ed:** a request escalates if the rules *or* the nearest examples indicate medical, legal, financial or self-harm risk (`semantic_high_stakes:<category>` reason).
- **Task type:** the semantic vote is used only when the rules have no signal or conflicting signals (`semantic_task_vote` reason); clear rule matches still win.
- **Private:** the model runs in-process and the `runtime-semantic` image bakes the weights in with `HF_HUB_OFFLINE=1`, so it works with `--network none`.
- **Cost:** ~320 MiB RAM, ~630 MB image, ~50 req/s with 8 concurrent clients on a shared 6-vCPU VPS (unpinned).

```bash
pip install "llm-preclassifier[semantic]"
SEMANTIC_MODEL=BAAI/bge-small-en-v1.5 uvicorn --factory llm_preclassifier.api:create_app
```

## 🗂️ ROUTING POLICY

Patterns, category order, confidence values and semantic thresholds live in a versioned YAML file, not in code. Copy [`policy.yaml`](src/llm_preclassifier/data/policy.yaml), edit it, bump `version`, and point `POLICY_PATH` at it.

```bash
python -m llm_preclassifier.validate_policy my-policy.yaml                                  # validate: OK version=... sha256=...
python -m llm_preclassifier.evaluation eval/holdout.jsonl --policy my-policy.yaml  # measure before deploying
```

- Every decision carries `policy_version`; `GET /v1/policy` (authenticated) returns the active version and SHA-256.
- Invalid files (unknown or missing keys, bad regexes, out-of-range values) stop the service at startup with the offending key named, e.g. `patterns.coding[3]`.
- `semantic.exemplars` can point at your own exemplar bank, relative to the policy file.

## 🔁 FEEDBACK LOOP (OPTIONAL)

Set `ENABLE_FEEDBACK=true` and `FEEDBACK_LOG_PATH` to accept corrections against a decision, referenced by its opaque `decision_id`. The endpoint accepts structured fields only — there is no field for prompt content, and the schema rejects unknown keys.

```bash
curl -X POST http://127.0.0.1:8802/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"decision_id":"3f8e2c1a9d7b4f0e8c6a5d2b1e9f7a3c","outcome":"incorrect","corrected_task_type":"extraction"}'
```

Use this to build a labeled dataset from real traffic for `--policy` evaluation, without ever capturing what was actually asked.

## 💰 MODEL CATALOG

`recommended_model_tier` is deliberately abstract (`economy`, `standard`, `capable`, `reasoning`) so the rules never hardcode a specific vendor. The concrete mapping — provider, model name, and illustrative per-million-token cost — lives in a separate versioned file, [`model_catalog.yaml`](src/llm_preclassifier/data/model_catalog.yaml), so pricing can be kept current without a code release.

```bash
python -m llm_preclassifier.validate_catalog my-catalog.yaml                        # validate: OK version=... sha256=...
python -m llm_preclassifier.evaluation eval/holdout.jsonl --catalog my-catalog.yaml  # attach recommendations while evaluating
```

- Every decision's `model_recommendations` reflects the active catalog; `GET /v1/model-catalog` (authenticated) reports its version, SHA-256 and currency.
- Set `MODEL_CATALOG_PATH` to point at your own file, edited for your actual contracted rates. Invalid files (unknown/missing keys, negative costs, empty tiers) stop the service at startup.
- These are **not fetched live** and are not a pricing guarantee — treat them as a starting point for your own cost model.

## 🔌 OPENAI-COMPATIBLE PROXY MODE (OPTIONAL)

By default the service only classifies — it never calls a model. Set `ENABLE_PROXY=true`, `PROXY_UPSTREAM_BASE_URL` and `PROXY_UPSTREAM_API_KEY` to expose `POST /v1/chat/completions`: the request is classified locally, then forwarded **unmodified** (model, messages, and all other OpenAI fields untouched) to your configured upstream.

```bash
curl -X POST http://127.0.0.1:8802/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-x","messages":[{"role":"user","content":"Summarise this report."}]}'
```

- The response body is the upstream's response, byte-for-byte; the local decision is attached only as the `X-Preclassifier-Decision` response header, so existing OpenAI SDK clients keep working unchanged.
- `available_tools` and `policy_flags` may be included in the request to inform classification; both are stripped before forwarding, since upstream chat completions APIs don't know them.
- The upstream API key is operator-configured server-side and is never accepted from the caller.
- Streaming (`stream: true`) is not supported yet — the upstream call is always non-streaming.
- This mode does not change routing decisions or pick a model on the caller's behalf; it forwards whatever `model` the client already requested. Use `model_recommendations` (from the `X-Preclassifier-Decision` header) to inform your own client-side model choice.

## 🎯 ACCURACY EVALUATION

Each JSONL line sets `prompt` (or `messages`), optional `available_tools` / `policy_flags`, and the `expected` decision fields to check.

| Set | Cases | Role | Rules only | Rules + semantic |
|---|---:|---|---:|---:|
| `eval/dataset.jsonl` | 48 | Regression, gated in CI at 95% | 100% | 97.9% |
| `eval/holdout.jsonl` | 32 | Regression, gated in CI at 95% (rules were tuned after first run) | 100% | 100% |
| `eval/blind.jsonl` | 24 | Blind benchmark, report only | 37.5% | 79.2% |
| `eval/blind-v2.jsonl` | 30 | Blind benchmark written after exemplars were frozen, report only | 26.7% | **93.3%** |

High-stakes escalation across both blind sets: **15/15 caught with the semantic layer** (5/15 with rules only), at the cost of 2 false escalations out of 8 benign look-alikes (`compound interest` → financial, `how a bill becomes law` → legal). The layer is biased towards escalating on purpose.

```bash
make eval            # gated regression sets, rules only
make eval-blind      # blind benchmarks, rules only
make eval-semantic   # everything with the semantic layer (downloads the model once)
```

> [!WARNING]
> With the rules-only default, unseen phrasings of medical, legal and financial risk are frequently **routed instead of escalated**. Run with the semantic layer (the default Docker Compose target) if you rely on `escalate`, and keep provider-side safeguards in place either way: these are small synthetic benchmarks, not a safety guarantee.

Reports include accuracy for each field, precision and recall for each class, a confusion matrix, confidence calibration and every miss. All sets are synthetic; extend them with anonymised real traffic.

## 📖 PROJECT RESOURCES

- [⚖️ License (Apache 2.0)](LICENSE)
- [🛡️ Security Policy](SECURITY.md)
- [🤝 Contributing Guidelines](CONTRIBUTING.md)
- [📝 Changelog](CHANGELOG.md)

## SUPPORT

[![Support on Ko-fi](https://img.shields.io/badge/Support_on-Ko--fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/sasiverse)

> [!WARNING]
> This service acts as an advisory sidecar. It is **not** a universal security firewall. It does not determine whether a prompt is fundamentally harmless, authorized by identity, or structurally safe against jailbreaks. Apply proper boundary controls at your tool-execution layer.

