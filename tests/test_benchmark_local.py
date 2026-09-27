"""The real-run entry point preserves local evidence and blocks network access."""

from pathlib import Path
import subprocess
import sys


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
