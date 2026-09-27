"""Seeded synthetic feature benchmarks, with calibration isolated from testing.

Labels describe this toy supervisor's intended outcomes, not observed physical
success. Model inputs contain opaque identifiers rather than scenario labels.
Evidence uses frozen case clocks; deadlines and latency use real wall time.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from .contracts import DecisionRecord, DecisionRequest, Observation, PolicyConfig, PolicyOutcome
from .loop import DecisionLoop
from .policy import DecisionPolicy
from .providers.base import DecisionProvider
from .simulator import RobotSimulator, baseline, default_choices

CORPUS_VERSION = "synthetic-robot-features-v1"
SCENARIOS = ("clear", "obstacle", "low_battery", "stuck", "noisy_coherent", "conflicting", "missing", "stale")
_EXPECTED = dict(zip(SCENARIOS, ("continue", "inspect", "request_help", "recover", "continue", "need_evidence", "need_evidence", "defer"), strict=True))
_NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


@dataclass(frozen=True)
class BenchmarkCase:
    scenario: str
    split: Literal["calibration", "test", "warmup"]
    request: DecisionRequest
    now: datetime
    expected_outcome: str

    @property
    def expected_action(self) -> str | None:
        return self.expected_outcome if self.expected_outcome not in ("need_evidence", "defer") else None

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "split": self.split, "now": self.now.isoformat(),
                "expected_outcome": self.expected_outcome, "request": self.request.model_dump(mode="json")}


@dataclass(frozen=True)
class BenchmarkCorpus:
    seed: int
    calibration: tuple[BenchmarkCase, ...]
    test: tuple[BenchmarkCase, ...]

    def to_dict(self) -> dict:
        data = {"version": CORPUS_VERSION, "seed": self.seed,
                "calibration": [case.to_dict() for case in self.calibration],
                "test": [case.to_dict() for case in self.test]}
        data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return data

    @property
    def fingerprint(self) -> str:
        return self.to_dict()["fingerprint"]


def _case(seed: int, split: str, scenario: str, index: int) -> BenchmarkCase:
    token = hashlib.sha256(f"{CORPUS_VERSION}:{seed}:{split}:{scenario}:{index}".encode()).hexdigest()
    rng = random.Random(int(token, 16))
    clearance = rng.uniform(0.9, 2.5)
    values = {"front_range_m": clearance, "camera_clearance_m": clearance + rng.uniform(-0.09, 0.09),
              "battery_pct": rng.uniform(30, 95), "stuck": 0.0}
    if scenario == "obstacle":
        values["front_range_m"] = rng.uniform(0.08, 0.35)
        values["camera_clearance_m"] = values["front_range_m"] + rng.uniform(0.01, 0.09)
    elif scenario == "low_battery":
        values["battery_pct"] = rng.uniform(1, 8)
    elif scenario == "stuck":
        values["stuck"] = 1.0
    elif scenario == "noisy_coherent":
        values["front_range_m"] = rng.uniform(0.65, 1.15)
        values["camera_clearance_m"] = values["front_range_m"] + rng.uniform(0.2, 0.35)
    elif scenario == "conflicting":
        values["front_range_m"] = rng.uniform(1.2, 2.5)
        values["camera_clearance_m"] = rng.uniform(0.05, 0.3)
    values = {key: round(value, 5) for key, value in values.items()}
    units = {"front_range_m": "m", "camera_clearance_m": "m", "battery_pct": "%", "stuck": "bool"}
    observed_at = _NOW - timedelta(seconds=60 if scenario == "stale" else 1)
    observations = tuple(Observation(
        observation_id=f"o-{token[:16]}-{i}", source="synthetic-sensors-v1", feature=feature,
        value=value, unit=units[feature], observed_at=observed_at,
        valid_until=observed_at + timedelta(seconds=30),
    ) for i, (feature, value) in enumerate(values.items())
        if not (scenario == "missing" and feature == "front_range_m"))
    request = DecisionRequest(
        request_id=f"r-{token[:24]}", question="Which bounded supervisory skill should the mobile robot run next?",
        created_at=_NOW - timedelta(seconds=1), valid_until=_NOW + timedelta(seconds=30),
        observations=observations, choices=default_choices(),
    )
    return BenchmarkCase(scenario, split, request, _NOW, _EXPECTED[scenario])


def make_corpus(seed: int = 20260927, cases_per_scenario: int = 4) -> BenchmarkCorpus:
    """Produce independent, stratified calibration and test samples.

    Expected labels are specified by scenario before inputs are sampled. Neither
    the state-machine baseline nor a provider is called to obtain the labels.
    The noise stratum varies engineered range features, not raw sensor signals.
    """
    if type(seed) is not int or type(cases_per_scenario) is not int or cases_per_scenario < 1:
        raise ValueError("seed must be an integer and cases_per_scenario a positive integer")
    splits = []
    for split in ("calibration", "test"):
        cases = [_case(seed, split, scenario, i) for scenario in SCENARIOS for i in range(cases_per_scenario)]
        random.Random(f"{seed}:{split}:order").shuffle(cases)
        splits.append(tuple(cases))
    return BenchmarkCorpus(seed, *splits)


@dataclass(frozen=True)
class BenchmarkRow:
    case: BenchmarkCase
    record: DecisionRecord
    end_to_end_s: float
    baseline_outcome: str
    baseline_elapsed_s: float

    def to_dict(self) -> dict:
        return {"case": self.case.to_dict(), "record": self.record.model_dump(mode="json"),
                "end_to_end_s": self.end_to_end_s, "baseline_outcome": self.baseline_outcome,
                "baseline_elapsed_s": self.baseline_elapsed_s}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latencies(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_s": None, "p50_s": None, "p95_s": None, "max_s": None}
    ordered = sorted(values)
    return {"count": len(values), "mean_s": statistics.mean(values), "p50_s": statistics.median(values),
            "p95_s": ordered[math.ceil(len(values) * 0.95) - 1], "max_s": max(values)}


def _metrics(rows: tuple[BenchmarkRow, ...], outcomes: tuple[PolicyOutcome, ...]) -> dict:
    total = len(rows)
    eligible = sum(row.case.expected_action is not None for row in rows)
    accepted = correct_accepted = eligible_accepted = proposals_correct = proposals_present = 0
    outcomes_correct = required_abstention_correct = unexpected_abstentions = 0
    for row, outcome in zip(rows, outcomes, strict=True):
        expected_action = row.case.expected_action
        result = row.record.provider_result
        if expected_action is not None:
            proposals_present += int(result is not None and result.proposed_action is not None)
            proposals_correct += int(result is not None and result.proposed_action == expected_action)
        is_accepted = outcome.disposition == "accept"
        accepted += int(is_accepted)
        eligible_accepted += int(is_accepted and expected_action is not None)
        correct_accepted += int(is_accepted and expected_action is not None and outcome.action == expected_action)
        actual = outcome.action if is_accepted else outcome.disposition
        outcomes_correct += int(actual == row.case.expected_outcome)
        required_abstention_correct += int(expected_action is None and not is_accepted)
        unexpected_abstentions += int(expected_action is not None and not is_accepted)
    return {
        "total_cases": total, "eligible_action_cases": eligible,
        "proposal_correct_count": proposals_correct,
        "proposal_accuracy_on_eligible": _rate(proposals_correct, eligible),
        "proposal_coverage_on_eligible": _rate(proposals_present, eligible),
        "accepted_count": accepted, "accepted_correct_count": correct_accepted,
        "accepted_wrong_count": accepted - correct_accepted,
        "accepted_action_accuracy": _rate(correct_accepted, accepted),
        "coverage": _rate(accepted, total), "eligible_action_coverage": _rate(eligible_accepted, eligible),
        "correct_action_coverage_on_eligible": _rate(correct_accepted, eligible),
        "abstention_rate": _rate(total - accepted, total),
        "unexpected_abstention_count": unexpected_abstentions,
        "required_abstention_cases": total - eligible,
        "required_abstention_correct_count": required_abstention_correct,
        "policy_outcome_accuracy": _rate(outcomes_correct, total),
        "baseline_outcome_accuracy": _rate(sum(row.baseline_outcome == row.case.expected_outcome for row in rows), total),
        "provider_status_counts": {status: sum(row.record.provider_status == status for row in rows)
                                   for status in sorted({row.record.provider_status for row in rows})},
        "end_to_end_latency": _latencies([row.end_to_end_s for row in rows]),
        "inference_loop_latency": _latencies([row.record.inference_elapsed_s for row in rows]),
        "baseline_decision_latency": _latencies([row.baseline_elapsed_s for row in rows]),
    }


@dataclass(frozen=True)
class BenchmarkRun:
    rows: tuple[BenchmarkRow, ...]
    warmup: tuple[dict, ...]

    def metrics(self) -> dict:
        metrics = _metrics(self.rows, tuple(row.record.policy for row in self.rows))
        metrics["completed_simulated_actions"] = sum(
            row.record.action is not None and row.record.action.status == "completed" for row in self.rows)
        return metrics

    def to_dict(self) -> dict:
        return {"metrics": self.metrics(), "warmup": list(self.warmup),
                "rows": [row.to_dict() for row in self.rows],
                "measurement_notes": [
                    "Synthetic engineered features and scenario labels; no physical or raw sensor validation.",
                    "Each case has a frozen evidence clock; inference deadlines use real monotonic time.",
                    "End-to-end timings include inference, policy, simulated dispatch and audit recording, but exclude loop setup and model load.",
                    "Inference-loop timings include Pollard model-call and thread overhead; they are not pure model kernel latency.",
                    "Baseline timings measure the state-machine decision only, without inference, dispatch or audit overhead.",
                    "Warmups are separate cases excluded from all accuracy, coverage and latency aggregates.",
                    "Proposal accuracy counts absent proposals as incorrect on cases labeled with an action; required-abstention cases are excluded.",
                    "Policy and baseline outcome accuracy require the labeled action or the exact labeled abstention disposition.",
                ]}


def run_benchmark(
    provider: DecisionProvider, cases: tuple[BenchmarkCase, ...], *, policy: PolicyConfig | None = None,
    timeout_s: float = 120.0, warmup: int = 1, records_path: str | Path | None = None,
    pollard_path: str | Path | None = None,
) -> BenchmarkRun:
    """Run each case through the real loop and bounded in-memory simulator.

    Timings exclude model construction. Each case has its own simulator and
    loop, so no previous simulated movement changes a later case's inputs.
    A timed-out worker cannot be cancelled by this thread-based loop: stop this
    benchmark immediately on timeout rather than accumulate model workers.
    """
    cases = tuple(cases)
    if not cases or len({case.request.request_id for case in cases}) != len(cases):
        raise ValueError("benchmark requires nonempty cases with unique request IDs")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    config = policy or PolicyConfig()
    warmups = tuple(_case(0, "warmup", "clear", i) for i in range(warmup))
    rows, warmup_results = [], []
    for case in (*warmups, *cases):
        started = time.perf_counter()
        baseline_outcome = baseline(case.request, case.now, config)
        baseline_elapsed = time.perf_counter() - started
        with DecisionLoop(provider, max_requests=1, timeout_s=timeout_s, policy=config,
                          simulator=RobotSimulator(), clock=lambda: case.now,
                          records_path=records_path, pollard_path=pollard_path) as loop:
            started = time.perf_counter()
            record = loop.decide(case.request)
            elapsed = time.perf_counter() - started
        if record.provider_status in ("timeout", "busy"):
            raise TimeoutError("Benchmark stopped after inference timeout; worker may still be computing")
        if case.split == "warmup":
            warmup_results.append({"request_id": case.request.request_id, "provider_status": record.provider_status,
                                   "end_to_end_s": elapsed, "inference_loop_s": record.inference_elapsed_s})
        else:
            rows.append(BenchmarkRow(case, record, elapsed, baseline_outcome, baseline_elapsed))
    return BenchmarkRun(tuple(rows), tuple(warmup_results))


@dataclass(frozen=True)
class CalibrationResult:
    policy: PolicyConfig
    candidates: tuple[dict, ...]
    calibration_request_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"policy": self.policy.model_dump(mode="json"), "candidates": list(self.candidates),
                "calibration_request_ids": list(self.calibration_request_ids),
                "selection_rule": "Minimize accepted wrong count, then maximize accepted correct count; ties prefer higher threshold, then margin.",
                "limitation": "Calibration selects only on synthetic calibration cases; held-out results are required and do not establish physical safety."}


def evaluate_saved(run: BenchmarkRun, policy: PolicyConfig) -> dict:
    """Replay a policy on recorded outputs without inference or new dispatch.

    This is a counterfactual decision comparison. The latency measurements still
    describe the original run, and replayed accepts are not simulated actions.
    Baseline outcomes also retain the original run's policy configuration; this
    is a comparison to that recorded baseline, even if physical limits change.
    """
    evaluator = DecisionPolicy(policy)
    outcomes = tuple(evaluator.evaluate(row.case.request, row.record.provider_result, row.case.now)
                     if row.record.provider_status == "ok" else evaluator.outcome(
                         row.case.request, row.case.now, "defer", row.record.provider_status)
                     for row in run.rows)
    return {"mode": "saved_policy_replay_no_dispatch", "policy": policy.model_dump(mode="json"),
            "metrics": _metrics(run.rows, outcomes),
            "latency_note": "Latency measurements are from the original run, not this policy replay.",
            "baseline_note": "Baseline outcomes and timings retain the original run's policy configuration.",
            "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes]}


def calibrate_thresholds(
    calibration_run: BenchmarkRun, *, thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0),
    margins: tuple[float, ...] = (0.0, 0.05, 0.15, 0.3, 1.0),
) -> CalibrationResult:
    """Choose thresholds using saved calibration predictions only; never infer.

    Latencies belong to the original inference run, so candidate entries contain
    decision metrics only. Abstaining on every eligible case earns zero correct
    acceptances; coverage remains visible even when accepted wrong count is zero.
    """
    rows = calibration_run.rows
    if not rows or any(row.case.split != "calibration" for row in rows):
        raise ValueError("threshold selection requires exclusively calibration cases")
    if not thresholds or not margins:
        raise ValueError("calibration grids must be nonempty")
    base = rows[0].record.policy_config
    if any(row.record.policy_config != base for row in rows):
        raise ValueError("calibration records must use the same original policy")
    candidates = []
    for threshold in thresholds:
        for margin in margins:
            config = PolicyConfig.model_validate({**base.model_dump(), "acceptance_threshold": threshold, "minimum_margin": margin})
            evaluator = DecisionPolicy(config)
            outcomes = tuple(evaluator.evaluate(row.case.request, row.record.provider_result, row.case.now)
                             if row.record.provider_status == "ok" else row.record.policy for row in rows)
            metrics = _metrics(rows, outcomes)
            candidates.append({"acceptance_threshold": threshold, "minimum_margin": margin,
                               **{key: value for key, value in metrics.items() if "latency" not in key}})
    selected = min(candidates, key=lambda item: (item["accepted_wrong_count"], -item["accepted_correct_count"],
                                                -item["acceptance_threshold"], -item["minimum_margin"]))
    selected_policy = PolicyConfig.model_validate({**base.model_dump(),
        "acceptance_threshold": selected["acceptance_threshold"], "minimum_margin": selected["minimum_margin"]})
    return CalibrationResult(selected_policy, tuple(candidates), tuple(row.case.request.request_id for row in rows))
