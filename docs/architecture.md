# Architecture

## Goal

`llm-preclassifier` is a standalone decision service. It accepts an agent request and returns an explainable classification before an application makes a model, tool, or human-review decision.

It deliberately does not proxy completions, store prompt history, manage provider credentials, execute tools, or select a vendor model.

## Request flow

```text
OpenAI-style messages + optional available_tools
                 │
                 ▼
          request validation
                 │
                 ▼
        deterministic signal rules
                 │
                 ├─ explicit tool/action signals
                 ├─ task-shape signals
                 ├─ multi-step/artifact signals
                 ├─ policy-sensitive signals
                 └─ ambiguity signals
                 │
                 ▼
         versioned decision response
                 │
                 ▼
 application-owned routing policy
```

## Decision contract

The public response is intentionally provider-neutral:

- `task_type` identifies a coarse request shape.
- `complexity` is a deterministic policy classification, not a model-quality prediction.
- `tool_requirement` captures explicit or provided tool signals.
- `recommended_model_tier` is an abstract policy input.
- `action` permits `route`, `escalate`, or `unknown`.
- `reasons` records stable, inspectable rule outcomes.
- `version` and `policy_version` enable safe evolution.

A consuming application owns the consequence of every decision. It should map tiers to its own provider, cost, privacy, and reliability policy.

## Privacy boundary

The classifier receives request content in memory to make a decision. It does not persist that content. Optional JSONL logging is disabled by default and accepts only a fixed metadata allowlist. The service does not call external models in V0.1.

## Extension direction

Future classifiers may be added behind the same contract only when they preserve explicit configuration, documented data flows, and reproducible evaluation. A local model backend must remain opt-in and should never silently transmit request content to a remote endpoint.
