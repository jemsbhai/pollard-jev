"""Check or advance the project's explicitly bounded decimal release version."""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
APPROVAL_LIMIT = Decimal("1.0")
INCREMENTS = {"minor": Decimal("0.01"), "major": Decimal("0.10")}
VERSION_PATTERN = re.compile(r"0\.(?:1|[1-9][0-9])\Z")


class VersionError(ValueError):
    """The requested release violates the repository's version policy."""


def validate_version(version: str) -> Decimal:
    """Accept initial 0.1, then canonical two-decimal versions through 0.99."""
    try:
        number = Decimal(version)
    except InvalidOperation as exc:
        raise VersionError(f"Invalid release version: {version!r}") from exc
    if number.is_finite() and number >= APPROVAL_LIMIT:
        raise VersionError(
            "Versions at or above 1.0 require explicit user approval. "
            "This helper cannot authorize or produce them."
        )
    if not VERSION_PATTERN.fullmatch(version) or version == "0.10":
        raise VersionError(
            "Use initial version 0.1 or two decimal digits from 0.11 to 0.99 "
            "(for example, 0.20)."
        )
    return number


def next_version(current: str, increment: str) -> str:
    if increment not in INCREMENTS:
        raise VersionError("Increment must be 'minor' or 'major'.")
    candidate = format(validate_version(current) + INCREMENTS[increment], ".2f")
    validate_version(candidate)
    return candidate


def _text(path: Path) -> str:
    # Preserve existing LF/CRLF line endings when replacing only a version.
    return path.read_bytes().decode("utf-8")


def read_version(root: Path = REPO_ROOT) -> str:
    """Require matching canonical versions in package metadata and runtime."""
    try:
        project = tomllib.loads(_text(root / "pyproject.toml"))
        metadata_version = project["project"]["version"]
        module = ast.parse(_text(root / "src/pollard_jev/__init__.py"))
        assignments = [
            node for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__"
                    for target in node.targets)
        ]
        if len(assignments) != 1:
            raise VersionError("Expected one literal __version__ assignment.")
        runtime_version = ast.literal_eval(assignments[0].value)
    except (OSError, KeyError, SyntaxError, tomllib.TOMLDecodeError) as exc:
        raise VersionError(f"Cannot read release versions: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise VersionError(f"Cannot read a literal runtime version: {exc}") from exc
    if not isinstance(metadata_version, str) or not isinstance(runtime_version, str):
        raise VersionError("Both release versions must be strings.")
    if metadata_version != runtime_version:
        raise VersionError(
            f"Version mismatch: pyproject.toml has {metadata_version!r}; "
            f"pollard_jev.__version__ has {runtime_version!r}."
        )
    validate_version(metadata_version)
    return metadata_version


def check_tag(version: str, tag: str) -> None:
    validate_version(version)
    expected = f"v{version}"
    if tag != expected:
        raise VersionError(f"Release tag must be exactly {expected!r}; got {tag!r}.")


def bump_version(increment: str, root: Path = REPO_ROOT) -> str:
    """Update both source versions only after every policy check succeeds."""
    current = read_version(root)
    following = next_version(current, increment)
    metadata_path = root / "pyproject.toml"
    runtime_path = root / "src/pollard_jev/__init__.py"
    metadata = _text(metadata_path)
    runtime = _text(runtime_path)
    section = re.search(r"(?ms)^\[project\][^\S\r\n]*\r?\n(.*?)(?=^\[|\Z)", metadata)
    if section is None:
        raise VersionError("Cannot locate the [project] metadata section.")

    def replace_version(text: str, name: str) -> str:
        pattern = rf"(?m)^({name}\s*=\s*)([\"']){re.escape(current)}\2"
        result, count = re.subn(
            pattern, lambda match: f"{match[1]}{match[2]}{following}{match[2]}", text,
        )
        if count != 1:
            raise VersionError(f"Expected one editable {name} assignment.")
        return result

    project_section = replace_version(section[1], "version")
    new_metadata = metadata[:section.start(1)] + project_section + metadata[section.end(1):]
    new_runtime = replace_version(runtime, "__version__")
    metadata_path.write_bytes(new_metadata.encode("utf-8"))
    runtime_path.write_bytes(new_runtime.encode("utf-8"))
    return following


def main(argv: list[str] | None = None, *, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check both versions and an optional release tag")
    check.add_argument("--tag", help="Require the exact release tag, for example v0.1")
    bump = commands.add_parser("bump", help="Advance both versions within the approved range")
    bump.add_argument("increment", choices=tuple(INCREMENTS))
    args = parser.parse_args(argv)
    try:
        if args.command == "bump":
            print(bump_version(args.increment, root))
        else:
            version = read_version(root)
            if args.tag is not None:
                check_tag(version, args.tag)
            print(f"Release version checked: {version}")
    except (VersionError, OSError) as exc:
        print(f"Release version error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
