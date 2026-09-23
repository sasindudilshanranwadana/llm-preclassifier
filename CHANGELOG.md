# Changelog

All notable changes to this project are documented here.

## 0.1.0 — Unreleased

### Added

- Offline deterministic task preclassification API with an inspectable, versioned response.
- Conservative `unknown` and `escalate` outcomes for ambiguity and selected policy-sensitive signals.
- Loopback-bound hardened Docker Compose configuration.
- Request limits, production credential enforcement, metadata-only optional decision logs, and security/privacy documentation.
- Interactive GitHub Pages launch site.
- Offline accuracy evaluation (`python -m llm_preclassifier.evaluation`) with per-class precision/recall, confusion matrix, calibration buckets, a 48-case seed dataset and a CI accuracy gate.
- `mixed_signals` reason with reduced confidence when several task categories match.

### Fixed

- Cache key now covers `policy_flags`, `available_tools` and full tool history, so a cached `route` can no longer mask a `high_stakes` escalation.
- Chunked request bodies are size-checked while being read instead of after full buffering.
- Settings read environment variables at construction, and importing `llm_preclassifier.api` no longer builds an app (Docker uses `uvicorn --factory`).
- `/status` reports all counters, including zeros.
- Narrowed coding and `send` patterns that misclassified everyday requests.

### Known limitations

- The rules are not a learned model-quality predictor and have not been evaluated as a universal routing benchmark.
- Capability tiers are abstract; applications must map them to their own providers and policies.
- The service does not proxy model completions, execute tools, or make safety guarantees.
