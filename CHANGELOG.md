# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- Optional offline semantic layer (`[semantic]` extra, `SEMANTIC_MODEL`): nearest-exemplar voting over local `bge-small-en-v1.5` embeddings. Escalation is OR-ed with the rules; task type falls back to the semantic vote when rules have no or conflicting signals.
- `runtime-semantic` Docker target with baked-in weights that runs with no network; now the Docker Compose default (memory limit raised to 512m).
- `--semantic` flag for the evaluation CLI, `make eval-semantic`, a semantic CI job, and a second blind set (`blind-v2`) written after the exemplars were frozen.

### Changed

- Classification runs in a worker thread so CPU-bound embedding never blocks the event loop.

### Results

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
- The service does not proxy model completions, execute tools, or make safety guarantees.
