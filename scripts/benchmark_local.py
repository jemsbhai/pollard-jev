"""Run the synthetic robotics benchmark using an existing local OpenJev cache.

Invoke with an existing Python environment containing the optional dependencies.
This command never installs packages or downloads model files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def disable_network():
    """Defense in depth in this dedicated process, before model imports."""
    for name in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_PROGRESS_BARS",
    ):
        os.environ[name] = "1"

    def denied(*args, **kwargs):
        raise RuntimeError("Network connections are disabled in the local benchmark")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_output(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_metadata():
    # Record versions only, never package source URLs or environment variables.
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    sources = sorted((ROOT / "src" / "pollard_jev").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path, help="Existing Hugging Face cache; no downloads")
    parser.add_argument("--output", required=True, type=Path, help="New or empty result directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--cases-per-scenario", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--representation", choices=("json-v1", "text-v1", "robot-rules-v1"), default="json-v1")
    args = parser.parse_args(argv)
    if not args.cache_dir.is_dir():
        parser.error("--cache-dir must already exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty so existing evidence is preserved")
    if args.cases_per_scenario < 1 or args.max_length < 1:
        parser.error("case count and context length must be positive")
    import math
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    disable_network()
    sys.path.insert(0, str(ROOT / "src"))

    from pollard_jev.benchmark import make_corpus, run_benchmark, calibrate_thresholds, evaluate_saved
    from pollard_jev.contracts import PolicyConfig
    from pollard_jev.providers.context import ContextCheckedEncoder
    from pollard_jev.providers.openjev import REPOSITORY, REVISION, CHECKPOINT, OpenJevProvider
    from huggingface_hub import snapshot_download
    import torch

    started = datetime.now(timezone.utc).isoformat()
    environment = runtime_metadata()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected existing environment")
    environment.update({"torch_cuda": torch.version.cuda, "device": args.device})
    if args.device == "cuda":
        environment.update({"gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory})
    save(args.output / "environment.json", environment)
    print(f"Using existing {CHECKPOINT} on {args.device}; network disabled", flush=True)
    snapshot = Path(snapshot_download(
        repo_id=REPOSITORY, revision=REVISION, cache_dir=args.cache_dir,
        local_files_only=True, token=False,
        allow_patterns=["modeling_openjev.py", f"{CHECKPOINT}/*"],
    ))
    files = [snapshot / "modeling_openjev.py", *(
        snapshot / CHECKPOINT / name for name in
        ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    )]
    manifest = {path.relative_to(snapshot).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}
    save(args.output / "model-manifest.json", {"repository": REPOSITORY, "revision": REVISION, "checkpoint": CHECKPOINT, "files": manifest})
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    provider = OpenJevProvider.from_local_cache(cache_dir=args.cache_dir, device=args.device,
                                              max_length=args.max_length, representation=args.representation)
    load_seconds = time.perf_counter() - load_started
    guard = ContextCheckedEncoder(provider._encoder)
    provider._encoder = guard
    print(f"Model loaded in {load_seconds:.2f}s; starting calibration split", flush=True)

    corpus = make_corpus(seed=args.seed, cases_per_scenario=args.cases_per_scenario)
    save(args.output / "corpus.json", corpus.to_dict())
    calibration = run_benchmark(
        provider, corpus.calibration, timeout_s=args.timeout_s, warmup=1,
        records_path=args.output / "calibration-records.jsonl",
        pollard_path=args.output / "calibration-ledger.db",
    )
    save(args.output / "calibration-run.json", calibration.to_dict())
    selection = calibrate_thresholds(calibration)
    save(args.output / "calibration-grid.json", selection.to_dict())
    print(f"Calibration complete; selected threshold={selection.policy.acceptance_threshold}, margin={selection.policy.minimum_margin}; starting held-out split", flush=True)
    heldout = run_benchmark(
        provider, corpus.test, policy=selection.policy, timeout_s=args.timeout_s, warmup=0,
        records_path=args.output / "heldout-records.jsonl",
        pollard_path=args.output / "heldout-ledger.db",
    )
    save(args.output / "heldout-run.json", heldout.to_dict())
    default_replay = evaluate_saved(heldout, PolicyConfig())
    save(args.output / "heldout-default-policy-replay.json", default_replay)
    save(args.output / "context-measurements.json", guard.measurements)
    summary = {
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": provider.identity.model_dump(mode="json"),
        "seed": args.seed, "cases_per_scenario_per_split": args.cases_per_scenario,
        "corpus_fingerprint": corpus.fingerprint,
        "load_seconds": load_seconds,
        "calibration": calibration.metrics(), "heldout": heldout.metrics(),
        "heldout_default_policy_replay": default_replay["metrics"],
        "selected_policy": selection.policy.model_dump(mode="json"),
        "encoder_forward_attempts": guard.forward_calls,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "scope": {
            "observations": "seeded synthetic structured scenarios; not recorded sensor data",
            "clock": "fixed per-case observation clock; real inference/deadline timing",
            "actions": "bounded simulator only",
            "calibration": "threshold selection on calibration split only; exploratory, not deployment validation",
            "comparison": "pinned 4B model versus deterministic state machine; 0.8B not run",
            "latency": "load and warmup separate; end-to-end includes policy, simulator and record persistence",
            "energy": "unmeasured",
            "network": "HF offline flags and Python socket connections blocked; no downloads or installs",
        },
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
