"""The real-run entry point preserves local evidence and blocks network access."""

from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_local.py"


def test_existing_result_directory_is_not_overwritten(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    evidence = output / "summary.json"
    evidence.write_text("existing evidence", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(cache), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "existing evidence is preserved" in result.stderr
    assert evidence.read_text(encoding="utf-8") == "existing evidence"
    assert list(output.iterdir()) == [evidence]


def test_missing_cache_fails_before_creating_output_or_loading_models(tmp_path):
    output = tmp_path / "results"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(tmp_path / "absent"), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "must already exist" in result.stderr
    assert not output.exists()


def test_offline_process_blocks_connections_without_importing_model_runtime():
    # Isolate the intentional process-wide socket guard from the pytest process.
    code = '''
import os, runpy, socket, sys
runner = runpy.run_path(sys.argv[1])
assert not any(name in sys.modules for name in ('torch', 'transformers', 'huggingface_hub'))
runner['disable_network']()
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
for connect in (socket.create_connection, socket.socket().connect, socket.socket().connect_ex):
    try:
        connect(('127.0.0.1', 9))
    except RuntimeError as error:
        assert 'disabled' in str(error)
    else:
        raise AssertionError('socket connection allowed')
'''
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("error", [FileNotFoundError("git absent"), PermissionError("git unavailable")])
def test_runtime_metadata_survives_unavailable_optional_git(monkeypatch, error):
    spec = importlib.util.spec_from_file_location("benchmark_local_metadata_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner.subprocess, "run", unavailable)
    metadata = runner.runtime_metadata()
    assert metadata["git_commit"] is None
    assert metadata["git_status"] is None
    assert metadata["python"] == sys.version
    assert metadata["packages"]
    assert "scripts/benchmark_local.py" in metadata["source_sha256"]


def test_git_metadata_does_not_swallow_unrelated_programming_errors(monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_local_error_test", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def invalid_call(*args, **kwargs):
        raise TypeError("unexpected implementation error")

    monkeypatch.setattr(runner.subprocess, "run", invalid_call)
    with pytest.raises(TypeError, match="unexpected implementation error"):
        runner.git_output("status", "--short")
