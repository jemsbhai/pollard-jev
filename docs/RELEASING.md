# Releases

The first public package version is **0.1**, with Git tag **v0.1**. Earlier
milestone verification notes that mention `0.1.0` record a local, unpublished
build; those historical notes do not set the public release version.

## Version policy

The user defines a minor increment as decimal **0.01** and a major increment as
decimal **0.10**. This is a project-specific decimal policy, not SemVer.

| Current | Increment | Next |
| --- | --- | --- |
| 0.1 | minor | 0.11 |
| 0.1 | major | 0.20 |
| 0.19 | minor | 0.20 |
| 0.20 | major | 0.30 |
| 0.99 | minor | refused |
| 0.90 | major | refused |

Except for the exact initial `0.1`, use two decimal digits. Python package
versions compare numeric components, so `0.20` follows `0.19`, while spelling
the same decimal value as `0.2` would sort before `0.19`. Do not remove trailing
zeroes. The release helper uses decimal arithmetic and checks both version
sources to avoid floating-point rounding or metadata mismatches. See the
[Python version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#final-releases).

**Never create, tag, or publish version 1.0 or above without future explicit
user approval.** Generic authorization for a release or major update is not
approval to cross that boundary. The helper has no bypass flag: if such approval
is given later, the policy, helper, tests, and release checks must be deliberately
updated together before that release.

## Prepare and verify

Run these commands from the repository root with the project environment active.
For the initial release, retain `0.1`. For a subsequent authorized release,
choose exactly one increment:

```text
python scripts/release_version.py bump minor
python scripts/release_version.py bump major
```

The command updates `pyproject.toml` and `src/pollard_jev/__init__.py`. Review the
change and update the release notes before continuing.

```text
python scripts/release_version.py check
python -m pytest -q
python -m build
python -m twine check dist/*
```

Ensure `dist/` contains only the intended release's wheel and source archive.
Install the wheel into a clean environment and run the offline demo before
publishing. The demo is simulation-only and does not require the optional model
backend or physical devices.

## Tag and publish

Commit the verified source. Choose the exact tag `vVERSION`, then validate it
against the version in the committed checkout:

```text
python scripts/release_version.py check --tag v0.1
```

For later releases, substitute their checked version. Tags such as `0.1`,
`v0.1.0`, or a tag for a different package version are rejected. Create and push
the matching tag only after tests and package checks pass. GitHub CI checks
versions, runs tests, and builds distributions; it does not automatically
publish to PyPI. Use an authenticated Twine upload of the two verified files.
Keep authentication in the publishing service or local
credential storage; never place credentials in repository files or command logs.

After upload, verify the PyPI version and install the released package from
PyPI into a clean environment. Published version files cannot be replaced; any
follow-up change needs the next approved version under this policy.
