# Privacy

## Default behaviour

`llm-preclassifier` processes supplied request content in memory and does not persist it. V0.1 uses deterministic local rules and makes no outbound provider call.

No telemetry, analytics SDK, account system, or cloud storage is included.

## Optional logs

Decision logging is disabled by default. If an operator enables it, the application writes only an allowlisted metadata record. It does not log prompts, messages, credentials, request headers, upstream responses, or exception strings.

The operator controls storage, access, retention, backup, and deletion. Treat any deployment that receives personal or confidential content as a system requiring an appropriate privacy assessment and security controls.

## Operator responsibilities

- Minimise the request content sent to the service.
- Use TLS and private network boundaries for non-local deployments.
- Store client credentials in a secret manager.
- Set a retention period for any enabled logs.
- Review downstream provider and gateway data handling separately; this project does not control them.

This document describes the default software behaviour, not legal advice or a complete compliance programme.