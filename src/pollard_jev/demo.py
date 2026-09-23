"""Credential-free offline scenarios and read-only record commands."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .contracts import DecisionRequest, Observation
from .loop import DecisionLoop
from .providers.fixture import FixtureProvider
from .records import inspect_history, simulate_policy
from .simulator import baseline, default_choices


def example_request(
    now: datetime, *, request_id: str = "example", conflict: bool = False, missing: bool = False,
) -> DecisionRequest:
    values = {"front_range_m": (1.5, "m"), "camera_clearance_m": (0.2 if conflict else 1.4, "m"),
              "battery_pct": (75.0, "%"), "stuck": (0.0, "bool")}
    observations = tuple(Observation(
        observation_id=f"{request_id}-{feature}", source="synthetic-sensors-v1", feature=feature,
        value=value, unit=unit, observed_at=now, valid_until=now + timedelta(seconds=30),
    ) for feature, (value, unit) in values.items() if not (missing and feature == "front_range_m"))
    return DecisionRequest(
        request_id=request_id, question="Which bounded supervisory skill should the mobile robot run next?",
        created_at=now, valid_until=now + timedelta(seconds=30), observations=observations, choices=default_choices(),
    )


class ScenarioClock:
    def __init__(self):
        self.now = datetime.now(timezone.utc)

    def __call__(self):
        return self.now


class ExpiringFixture(FixtureProvider):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def infer(self, requests):
        results = super().infer(requests)
        # Deterministic virtual passage of time during inference; no timing race.
        self.clock.now += timedelta(seconds=31)
        return results


def run_demo(output: Path) -> tuple:
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    records_path = output / f"decisions-{run_id}.jsonl"
    pollard_path = output / f"pollard-{run_id}.db"
    records = []
    print("Offline mobile-robot supervisor | SYNTHETIC scores | SIMULATED actions")
    print("Demonstration thresholds: support >= 0.80; top-score margin >= 0.15")
    print("Budget unit: one logical provider batch attempt (not tokens or joules)")
    print(f"{'scenario':<27} {'proposal':<14} {'policy':<15} {'baseline':<15} result")
    for name in ("fresh_coherent", "conflicting_observations", "missing_evidence", "stale_during_inference", "exhausted_budget", "provider_failure"):
        clock = ScenarioClock()
        request = example_request(clock(), request_id=name, conflict=name == "conflicting_observations", missing=name == "missing_evidence")
        provider = ExpiringFixture(clock) if name == "stale_during_inference" else FixtureProvider(failure=name == "provider_failure")
        with DecisionLoop(
            provider, max_requests=0 if name == "exhausted_budget" else 1, clock=clock,
            records_path=records_path, pollard_path=pollard_path,
        ) as loop:
            record = loop.decide(request)
        records.append(record)
        proposal = record.provider_result.proposed_action if record.provider_result else "(none)"
        baseline_action = baseline(request, record.policy.evaluated_at)
        result = f"simulated {record.action.action}" if record.action and record.action.status == "completed" else record.policy.reason
        print(f"{name:<27} {proposal:<14} {record.policy.disposition:<15} {baseline_action:<15} {result}")
    print(f"Records: {records_path.resolve()}")
    print(f"Pollard ledger: {pollard_path.resolve()}")
    return tuple(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run all six offline scenarios")
    demo.add_argument("--output", type=Path, default=Path("artifacts"))
    for name in ("history", "simulate-policy"):
        command = commands.add_parser(name)
        command.add_argument("records", type=Path)
    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo(args.output)
    elif args.command == "history":
        for record in inspect_history(args.records):
            print(record.model_dump_json())
    else:
        for record in inspect_history(args.records):
            print(simulate_policy(record).model_dump_json())


if __name__ == "__main__":
    main()
