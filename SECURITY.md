# Security policy

## Supported versions

Only the latest released version receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private security advisory workflow for this repository, or contact the maintainer through the email listed on the GitHub profile with:

- a concise description and impact;
- affected version and deployment assumptions;
- reproduction steps or a minimal proof of concept;
- any suggested mitigation.

Do not include credentials, customer prompts, personal data, or destructive payloads. We will acknowledge a good-faith report, investigate, and coordinate a fix before public disclosure where appropriate.

## Deployment warning

This project is a classification sidecar, not a security boundary. Run it privately by default, protect it with TLS and authentication where required, and evaluate its output before using it in any policy-sensitive workflow.
