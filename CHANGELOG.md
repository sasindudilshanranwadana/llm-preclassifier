# Changelog

All notable changes to this project are documented here.

## 0.1.0 — Unreleased

### Added

- Offline deterministic task preclassification API with an inspectable, versioned response.
- Conservative `unknown` and `escalate` outcomes for ambiguity and selected policy-sensitive signals.
- Loopback-bound hardened Docker Compose configuration.
- Request limits, production credential enforcement, metadata-only optional decision logs, and security/privacy documentation.
- Interactive GitHub Pages launch site.

### Known limitations

- The rules are not a learned model-quality predictor and have not been evaluated as a universal routing benchmark.
- Capability tiers are abstract; applications must map them to their own providers and policies.
- The service does not proxy model completions, execute tools, or make safety guarantees.
