# Configuration

Configuration is read from environment variables. Copy `.env.example` for local development, but do not commit real credentials.

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PRECLASSIFIER_ENV` | `development` | Set to `production` to enforce configured client credentials at startup. |
| `CLIENT_API_KEYS` | empty | Comma-separated bearer credentials accepted by protected endpoints. Use a secret manager in production. |
| `MAX_REQUEST_BYTES` | `65536` | Maximum accepted HTTP request body size. Minimum: 1024. |
| `MAX_MESSAGES` | `32` | Maximum messages accepted per classification request. Minimum: 1. |
| `CACHE_TTL_SECONDS` | `300` | Reserved for caller-side/cache integrations. |
| `CACHE_MAX_ENTRIES` | `1024` | Reserved for caller-side/cache integrations. |
| `LOG_DECISIONS` | `false` | Enables metadata-only JSONL decision logging. |
| `DECISION_LOG_PATH` | empty | Required if `LOG_DECISIONS=true`; choose an access-controlled path. |

## Authentication

When `CLIENT_API_KEYS` is set, `/v1/classify`, `/v1/route`, and `/status` require:

```http
Authorization: Bearer your-configured-token
```

`/healthz` remains unauthenticated for container liveness checks. Bind the service privately and place it behind a TLS-terminating reverse proxy if it must be reached over a network.

## Logging

Logging is disabled by default. A log record contains only:

- UTC timestamp;
- request identifier supplied by the caller, if any;
- versioned decision labels;
- confidence and action.

It never accepts request text as an argument and never writes message content, headers, model output, stack traces, or exception text. Operators remain responsible for filesystem permissions, retention, rotation, and deletion.
