"""The release policy must preserve PyPI ordering and its approval boundary."""

import importlib.util
from pathlib import Path

from packaging.version import Version
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_version.py"
SPEC = importlib.util.spec_from_file_location("release_version", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def checkout(tmp_path):
    def create(version="0.1", runtime_version=None):
        runtime_version = version if runtime_version is None else runtime_version
        (tmp_path / "src/pollard_jev").mkdir(parents=True, exist_ok=True)
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "pollard-jev"\n'
            f'version = "{version}"\n\n'
            '[tool.example]\nversion = "unrelated"\n', encoding="utf-8",
        )
        (tmp_path / "src/pollard_jev/__init__.py").write_text(
            f'"""Package."""\n__version__ = "{runtime_version}"\n', encoding="utf-8",
        )
        return tmp_path
    return create


@pytest.mark.parametrize(("current", "increment", "expected"), [
    ("0.1", "minor", "0.11"),
    ("0.1", "major", "0.20"),
    ("0.19", "minor", "0.20"),
    ("0.20", "major", "0.30"),
    ("0.89", "major", "0.99"),
    ("0.98", "minor", "0.99"),
])
def test_decimal_increment(current, increment, expected):
    assert release.next_version(current, increment) == expected


def test_every_accepted_bump_increases_pypi_version():
    versions = ["0.1"] + [f"0.{value:02d}" for value in range(11, 100)]
    for current in versions:
        for increment in ("minor", "major"):
            try:
                following = release.next_version(current, increment)
            except release.VersionError:
                continue
            assert Version(following) > Version(current), (current, following)


@pytest.mark.parametrize("version", [
    "0.1.0", "0.10", "0.2", "0.01", "0.00", "0.100", "0.11rc1", "NaN",
    "Infinity", "garbage", "1.0", "1.00", "2.0",
])
def test_noncanonical_or_unapproved_versions_rejected(version):
    with pytest.raises(release.VersionError):
        release.validate_version(version)


@pytest.mark.parametrize(("current", "increment"), [
    ("0.99", "minor"), ("0.90", "major"), ("0.99", "major"),
])
def test_approval_boundary_cannot_modify_checkout(checkout, current, increment):
    root = checkout(current)
    paths = [root / "pyproject.toml", root / "src/pollard_jev/__init__.py"]
    before = [path.read_bytes() for path in paths]
    with pytest.raises(release.VersionError, match="explicit user approval"):
        release.bump_version(increment, root)
    assert [path.read_bytes() for path in paths] == before


def test_bump_updates_both_versions_preserves_other_metadata_and_line_endings(checkout):
    root = checkout("0.19")
    paths = [root / "pyproject.toml", root / "src/pollard_jev/__init__.py"]
    for path in paths:
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    before = [path.read_bytes() for path in paths]
    assert release.bump_version("minor", root) == "0.20"
    assert release.read_version(root) == "0.20"
    assert [path.read_bytes() for path in paths] == [
        content.replace(b'"0.19"', b'"0.20"') for content in before
    ]


def test_mismatched_sources_prevent_check_and_bump(checkout):
    root = checkout("0.11", "0.12")
    with pytest.raises(release.VersionError, match="mismatch"):
        release.read_version(root)
    with pytest.raises(release.VersionError, match="mismatch"):
        release.bump_version("minor", root)
    assert 'version = "0.11"' in (root / "pyproject.toml").read_text()


@pytest.mark.parametrize("tag", ["0.1", "v0.1.0", "v0.11", "v1.0", "v0.10"])
def test_tag_must_match_exactly(tag):
    with pytest.raises(release.VersionError, match="exactly"):
        release.check_tag("0.1", tag)


def test_cli_checks_tag_and_reports_errors(checkout, capsys):
    root = checkout()
    assert release.main(["check", "--tag", "v0.1"], root=root) == 0
    assert "0.1" in capsys.readouterr().out
    assert release.main(["check", "--tag", "v1.0"], root=root) == 1
    assert "exactly" in capsys.readouterr().err
    assert release.main(["bump", "minor"], root=root) == 0
    assert release.read_version(root) == "0.11"


def test_repository_sources_match_approved_initial_version():
    # Future releases may advance, but can never silently cross the boundary.
    release.validate_version(release.read_version(SCRIPT.parents[1]))
