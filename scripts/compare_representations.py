"""Compare frozen JSON and factual-text inputs using an existing local model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from benchmark_local import ROOT, disable_network, runtime_metadata, save, sha256


def run_paired(providers, cases, configs, output, phase, *, warmup=False):
    """Alternate execution order on identical cases, keeping ledgers separate."""
    from pollard_jev.benchmark import BenchmarkRun, make_corpus, run_benchmark
    rows = {name: [] for name in providers}
    warmups = {name: () for name in providers}
    if warmup:
        case = replace(next(case for case in make_corpus(seed=0, cases_per_scenario=1).calibration
                            if case.scenario == "clear"), split="warmup")
        for name, provider in providers.items():
            run = run_benchmark(provider, (case,), policy=configs[name], warmup=0, timeout_s=120)
            warmups[name] = run.warmup
    for index, case in enumerate(cases):
        order = list(providers)
        if index % 2:
            order.reverse()
        for name in order:
            folder = output / name
            folder.mkdir(exist_ok=True)
            run = run_benchmark(
                providers[name], (case,), policy=configs[name], warmup=0, timeout_s=120,
                records_path=folder / f"{phase}-records.jsonl",
                pollard_path=folder / f"{phase}-ledger.db",
            )
            rows[name].extend(run.rows)
        if (index + 1) % 16 == 0 or index + 1 == len(cases):
            print(f"{phase}: {index + 1}/{len(cases)} paired cases", flush=True)
    runs = {name: BenchmarkRun(tuple(rows[name]), warmups[name]) for name in providers}
    for name, run in runs.items():
        save(output / name / f"{phase}-run.json", run.to_dict())
    return runs


def compare(runs):
    from pollard_jev.benchmark import BenchmarkRun
    json_rows = runs["json-v1"].rows
    candidates = [name for name in runs if name != "json-v1"]
    if len(candidates) != 1:
        raise ValueError("Comparison requires JSON and exactly one candidate")
    text_rows = runs[candidates[0]].rows
    improved = regressed = both_correct = both_wrong = 0
    for left, right in zip(json_rows, text_rows, strict=True):
        if left.case != right.case:
            raise ValueError("Paired comparison requires identical requests, labels and evaluation times")
        expected = left.case.expected_action
        if expected is None:
            continue
        a = left.record.provider_result is not None and left.record.provider_result.proposed_action == expected
        b = right.record.provider_result is not None and right.record.provider_result.proposed_action == expected
        improved += int(not a and b)
        regressed += int(a and not b)
        both_correct += int(a and b)
        both_wrong += int(not a and not b)
    return {
        "candidate": candidates[0],
        "metrics": {name: run.metrics() for name, run in runs.items()},
        "per_scenario": {
            name: {scenario: BenchmarkRun(tuple(row for row in run.rows if row.case.scenario == scenario), ()).metrics()
                   for scenario in sorted({row.case.scenario for row in run.rows})}
            for name, run in runs.items()
        },
        "paired_proposal_counts_on_eligible": {
            "candidate_improves": improved, "candidate_regresses": regressed,
            "both_correct": both_correct, "both_wrong": both_wrong,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "evaluation"), required=True)
    parser.add_argument("--cases-per-scenario", type=int, default=8)
    parser.add_argument("--candidate", choices=("text-v1", "robot-rules-v1"), default="text-v1")
    args = parser.parse_args(argv)
    if not args.cache_dir.is_dir():
        parser.error("cache must already exist; this command never downloads")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output directory must be new or empty")
    if args.cases_per_scenario < 1:
        parser.error("cases-per-scenario must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    disable_network()
    sys.path.insert(0, str(ROOT / "src"))
    from pollard_jev.benchmark import make_corpus, calibrate_thresholds, evaluate_saved
    from pollard_jev.contracts import PolicyConfig
    from pollard_jev.providers.context import ContextCheckedEncoder
    from pollard_jev.providers.openjev import REPOSITORY, REVISION, CHECKPOINT, OpenJevProvider
    from huggingface_hub import snapshot_download
    import torch

    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
        "representations": ["json-v1", args.candidate],
        "candidate_information": "Fixed demo task rules plus factual readings" if args.candidate == "robot-rules-v1" else "Same readings as factual prose; no task rules added",
        "development": "Original seed 20260927 calibration cases only; already observed exploratory data",
        "evaluation_seed": 20260928, "stress_seed": 20260929,
        "cases_per_scenario_per_evaluation_split": args.cases_per_scenario,
        "selection": "Same predefined calibration grid, separately per representation; no held-out tuning",
        "execution": "Identical requests; alternating representation order per case; shared weights and guarded encoder",
        "scope": "Synthetic templates, fixed evidence clock, simulated dispatch; not physical or recorded-sensor validation",
        "default": "json-v1 and package policy defaults remain unchanged",
        "representation_source_sha256": sha256(ROOT / "src/pollard_jev/providers/representations.py"),
        "runner_source_sha256": sha256(Path(__file__)),
    }
    save(args.output / "protocol.json", protocol)
    env = runtime_metadata()
    env["source_sha256"]["scripts/compare_representations.py"] = sha256(Path(__file__))
    if not torch.cuda.is_available():
        raise RuntimeError("Existing environment has no available CUDA device")
    env.update({"torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    save(args.output / "environment.json", env)
    snapshot = Path(snapshot_download(repo_id=REPOSITORY, revision=REVISION, cache_dir=args.cache_dir,
                                     local_files_only=True, token=False))
    files = [snapshot / "modeling_openjev.py", *(snapshot / CHECKPOINT / name for name in
             ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"))]
    save(args.output / "model-manifest.json", {
        "repository": REPOSITORY, "revision": REVISION, "checkpoint": CHECKPOINT,
        "files": {path.relative_to(snapshot).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
    })
    torch.manual_seed(20260928)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = OpenJevProvider.from_local_cache(cache_dir=args.cache_dir, device="cuda", max_length=2048)
    load_seconds = time.perf_counter() - started
    guard = ContextCheckedEncoder(base._encoder)
    providers = {name: OpenJevProvider(guard, model=base.identity.model, model_version=base.identity.model_version,
                                     representation=name) for name in protocol["representations"]}
    configs = {name: PolicyConfig() for name in providers}
    print(f"Existing local model loaded in {load_seconds:.2f}s; {args.stage}; network disabled", flush=True)
    summary = {"protocol": protocol, "load_seconds": load_seconds,
               "providers": {name: provider.identity.model_dump(mode="json") for name, provider in providers.items()}}
    if args.stage == "development":
        corpus = make_corpus(seed=20260927, cases_per_scenario=4)
        save(args.output / "development-cases.json", [case.to_dict() for case in corpus.calibration])
        runs = run_paired(providers, corpus.calibration, configs, args.output, "development", warmup=True)
        summary["development"] = compare(runs)
    else:
        from pollard_jev.benchmark_stress import make_stress_cases, serialize_stress_cases
        corpus = make_corpus(seed=20260928, cases_per_scenario=args.cases_per_scenario)
        save(args.output / "corpus.json", corpus.to_dict())
        calibration = run_paired(providers, corpus.calibration, configs, args.output, "calibration", warmup=True)
        for name, run in calibration.items():
            selection = calibrate_thresholds(run)
            configs[name] = selection.policy
            save(args.output / name / "calibration-grid.json", selection.to_dict())
        # Both representation and threshold choices are fixed before test inference.
        save(args.output / "selected-policies.json", {name: config.model_dump(mode="json") for name, config in configs.items()})
        heldout = run_paired(providers, corpus.test, configs, args.output, "heldout")
        stress_cases = make_stress_cases(seed=20260929, cases_per_scenario=4)
        save(args.output / "stress-cases.json", serialize_stress_cases(stress_cases, seed=20260929))
        stress = run_paired(providers, stress_cases, configs, args.output, "stress")
        summary.update({"corpus_fingerprint": corpus.fingerprint,
                        "selected_policies": {name: config.model_dump(mode="json") for name, config in configs.items()},
                        "calibration_default_policy": compare(calibration),
                        "heldout": compare(heldout), "stress": compare(stress)})
        for phase, runs in (("heldout", heldout), ("stress", stress)):
            summary[f"{phase}_default_policy_replay"] = {}
            for name, run in runs.items():
                replay = evaluate_saved(run, PolicyConfig())
                save(args.output / name / f"{phase}-default-policy-replay.json", replay)
                summary[f"{phase}_default_policy_replay"][name] = replay["metrics"]
    summary.update({"completed_at": datetime.now(timezone.utc).isoformat(),
                    "encoder_forward_attempts": guard.forward_calls,
                    "max_pair_tokens": max(max(row["tokenized_pair_lengths"]) for row in guard.measurements),
                    "context_overflows": sum(row["context_limit_exceeded"] for row in guard.measurements),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
    save(args.output / "context-measurements.json", guard.measurements)
    save(args.output / "summary.json", summary)
    import json
    for phase in ("development", "heldout", "stress"):
        if phase in summary:
            print(json.dumps({phase: summary[phase]["metrics"], "paired": summary[phase]["paired_proposal_counts_on_eligible"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
