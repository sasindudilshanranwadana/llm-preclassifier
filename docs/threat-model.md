# Threat model

## Security boundary

`llm-preclassifier` is a policy-input service, not a safety enforcement product. It does not determine whether a prompt is harmless, authorised, factually correct, or suitable for autonomous execution.

## Assets

- request content in process memory;
- client credentials;
- decision metadata when optional logging is enabled;
- service availability and routing integrity.

## Primary threats and controls

| Threat | Control in V0.1 | Operator responsibility |
|---|---|---|
| Unauthorised use | Bearer credentials; production fails closed without configured keys | Keep keys in a secret manager; rotate/revoke compromised keys. |
| Oversized request denial of service | Request-byte and message-count limits | Use reverse-proxy rate/concurrency limits and network isolation. |
| Prompt retention | No persistence by default; metadata-only optional log writer | Keep logging disabled unless justified; secure and rotate logs. |
| Public exposure | Loopback-only Compose binding | Terminate TLS and restrict ingress before exposing service. |
| Container privilege escalation | Non-root user, read-only filesystem, dropped capabilities, no-new-privileges | Apply runtime patching, image scanning, and resource policies. |
| Misuse of a decision | Explicit `unknown`/`escalate` actions; documented limitations | Evaluate the policy against your own tasks and require review where appropriate. |

## Out of scope

- malicious model-output detection;
- prompt-injection prevention;
- provider-account security;
- tool sandboxing;
- legal, medical, financial, or other high-stakes advice validation;
- universal cost or quality optimisation.

## Deployment baseline

Use a private network, a TLS reverse proxy where needed, a secret manager, application-level rate limits, resource limits, and observability that does not capture raw prompts. Review the [configuration guide](configuration.md) before deployment.