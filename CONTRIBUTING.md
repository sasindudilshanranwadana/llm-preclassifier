# Contributing

Thanks for considering a contribution.

## Before opening a pull request

1. Open an issue for substantial changes so the scope can be discussed first.
2. Keep changes narrow and preserve the public decision contract unless the issue explicitly calls for a versioned change.
3. Do not add real prompts, credentials, personal data, provider-account details, or unverified performance claims to the repository.
4. Add or update focused tests for behavioural changes.
5. Run:

   ```bash
   python3 -m pip install -e '.[dev]'
   make test
   make build
   ```

## Design principles

- Prefer deterministic and explainable behaviour over opaque routing claims.
- Return `unknown` or `escalate` when the rules cannot support a safe policy recommendation.
- Keep the classifier provider-neutral and self-hostable.
- Treat logging and outbound data flow as explicit opt-ins.
- Document measurable evidence before making performance, accuracy, or cost claims.

## Pull requests

Use a focused title, explain the user-visible change, list verification performed, and note any API or documentation impact. Maintainers may ask for smaller commits, fixtures that use synthetic data, or a release-note entry.
