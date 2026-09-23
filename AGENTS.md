# Release policy

The user explicitly requires these rules for this repository:

- Start the public releases at `0.1`.
- Increment a minor release by decimal `0.01` and a major release by decimal
  `0.10`. These terms describe the user's decimal policy, not SemVer.
- Keep the initial version exactly `0.1`; render subsequent versions with two
  decimal digits, such as `0.11`, `0.19`, `0.20`, and `0.30`. This preserves
  increasing PyPI/PEP 440 order. Do not shorten `0.20` to `0.2`.
- **Do not create, tag, or publish version `1.0` or above unless the user
  explicitly approves that change in a future instruction.** Do not infer
  approval from a generic request to release or make a major update.
- Use `python scripts/release_version.py check` before building, and
  `python scripts/release_version.py check --tag vVERSION` before publishing.
  Use the script's `bump minor` or `bump major` commands for version changes.
- Keep `pyproject.toml` and `src/pollard_jev/__init__.py` versions identical.

See [docs/RELEASING.md](docs/RELEASING.md) for the release workflow. Never commit
credentials or copy publishing tokens into documentation, issues, or logs.
