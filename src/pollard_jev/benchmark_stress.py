"""Independent synthetic challenge specifications for the toy supervisor.

These cases are test-only: confidence calibration and representation selection
must use separate development data. Labels specify the desired priority of
bounded skills, not measured physical outcomes. Generation never calls a model,
the baseline, or the policy to assign those labels.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone

from .benchmark import BenchmarkCase
from .contracts import DecisionRequest, Observation
from .simulator import default_choices

STRESS_VERSION = "synthetic-robot-stress-v1"
STRESS_SCENARIOS = (
    "stress_clearance_below", "stress_clearance_at", "stress_clearance_above",
    "stress_battery_below", "stress_battery_at", "stress_battery_above",
    "stress_low_battery_stuck", "stress_obstacle_stuck",
    "stress_order_original", "stress_order_permuted",
    "stress_unknown_evidence", "stress_duplicate_evidence",
    "stress_future_observation", "stress_future_request", "stress_excluded_preferred",
)
_FIXED_LABELS = {
    "stress_clearance_below": "inspect", "stress_clearance_at": "continue",
    "stress_clearance_above": "continue", "stress_battery_below": "request_help",
    "stress_battery_at": "continue", "stress_battery_above": "continue",
    "stress_low_battery_stuck": "request_help", "stress_obstacle_stuck": "recover",
    "stress_unknown_evidence": "need_evidence", "stress_duplicate_evidence": "need_evidence",
    "stress_future_observation": "defer", "stress_future_request": "defer",
    "stress_excluded_preferred": "defer",
}
_MODES = ("continue", "inspect", "recover", "request_help")
_UNITS = {"front_range_m": "m", "camera_clearance_m": "m", "battery_pct": "%", "stuck": "bool"}
_NOW = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def _token(seed: int, scenario: str, index: int) -> str:
    return hashlib.sha256(f"{STRESS_VERSION}:{seed}:{scenario}:{index}".encode()).hexdigest()


def _apply_mode(values: dict[str, float], mode: str, rng: random.Random) -> None:
    if mode == "inspect":
        values["front_range_m"] = rng.uniform(0.1, 0.3)
        values["camera_clearance_m"] = values["front_range_m"] + rng.uniform(0.01, 0.1)
    elif mode == "recover":
        values["stuck"] = 1.0
    elif mode == "request_help":
        values["battery_pct"] = rng.uniform(1, 8)


def _make_case(seed: int, scenario: str, index: int) -> BenchmarkCase:
    token = _token(seed, scenario, index)
    # Request IDs distinguish audit records. Matched order pairs share complete
    # observation objects, so order is the only change to their model inputs.
    feature_scenario = "order_pair" if scenario in ("stress_order_original", "stress_order_permuted") else scenario
    observation_token = _token(seed, feature_scenario, index)
    rng = random.Random(int(observation_token, 16))
    clearance = rng.uniform(1, 2.5)
    values = {"front_range_m": clearance, "camera_clearance_m": clearance + rng.uniform(-0.1, 0.1),
              "battery_pct": rng.uniform(30, 95), "stuck": 0.0}
    expected = _FIXED_LABELS.get(scenario, _MODES[index % len(_MODES)])
    delta = (0.00001, 0.0001, 0.001, 0.01)[index % 4]
    if scenario.startswith("stress_clearance_"):
        offset = {"below": -delta, "at": 0.0, "above": delta}[scenario.rsplit("_", 1)[1]]
        first, second = ("front_range_m", "camera_clearance_m") if index % 2 else ("camera_clearance_m", "front_range_m")
        values[first] = 0.5 + offset
        values[second] = 0.5 + rng.uniform(0.02, 0.2)
    elif scenario.startswith("stress_battery_"):
        offset = {"below": -delta, "at": 0.0, "above": delta}[scenario.rsplit("_", 1)[1]]
        values["battery_pct"] = 10.0 + offset
    elif scenario == "stress_low_battery_stuck":
        _apply_mode(values, "request_help", rng)
        values["stuck"] = 1.0
    elif scenario == "stress_obstacle_stuck":
        _apply_mode(values, "inspect", rng)
        values["stuck"] = 1.0
    elif scenario in ("stress_order_original", "stress_order_permuted", "stress_excluded_preferred"):
        _apply_mode(values, _MODES[index % len(_MODES)], rng)
    values = {feature: round(value, 5) for feature, value in values.items()}
    observed_at = _NOW - timedelta(seconds=1)
    observations = []
    unknown_feature = tuple(_UNITS)[index % len(_UNITS)]
    for i, (feature, value) in enumerate(values.items()):
        unknown = scenario == "stress_unknown_evidence" and feature == unknown_feature
        timestamp = _NOW + timedelta(seconds=1) if scenario == "stress_future_observation" and feature == unknown_feature else observed_at
        observations.append(Observation(
            observation_id=f"o-{observation_token[:16]}-{i}", source="synthetic-sensors-v1", feature=feature,
            value=None if unknown else value, unit=_UNITS[feature], status="unknown" if unknown else "known",
            observed_at=timestamp, valid_until=timestamp + timedelta(seconds=30),
        ))
    if scenario == "stress_duplicate_evidence":
        feature = tuple(_UNITS)[index % len(_UNITS)]
        # Even agreeing duplicate required features are deliberately ambiguous.
        value = values[feature]
        if index % 2:
            value = 1.0 - value if feature == "stuck" else value + 0.1
        observations.append(Observation(
            observation_id=f"o-{observation_token[:16]}-4", source="synthetic-sensors-v1", feature=feature,
            value=value, unit=_UNITS[feature], observed_at=observed_at,
            valid_until=observed_at + timedelta(seconds=30),
        ))
    elif scenario == "stress_order_permuted":
        # Always change order, including small case counts, without drawing
        # randomness that could change any sampled numeric feature.
        shift = 1 + index % (len(observations) - 1)
        observations = observations[shift:] + observations[:shift]
    choices = default_choices()
    if scenario == "stress_excluded_preferred":
        omitted = _MODES[index % len(_MODES)]
        choices = tuple(choice for choice in choices if choice.name != omitted)
    created_at = _NOW + timedelta(seconds=1) if scenario == "stress_future_request" else observed_at
    request = DecisionRequest(
        request_id=f"r-{token[:24]}", question="Which bounded supervisory skill should the mobile robot run next?",
        created_at=created_at, valid_until=created_at + timedelta(seconds=30),
        observations=tuple(observations), choices=choices,
    )
    return BenchmarkCase(scenario, "test", request, _NOW, expected)


def make_stress_cases(seed: int = 20260929, cases_per_scenario: int = 4) -> tuple[BenchmarkCase, ...]:
    """Generate 15 strata of held-out challenges with opaque model-facing IDs.

    Thresholds use the fixed demo specification: clearance >=0.5m and battery
    >=10% suffice. Low battery takes priority over stuck; stuck takes priority
    over a close obstacle. If the preferred skill is excluded, the required
    outcome is abstention, not substitution of another permitted skill.

    Order-original and order-permuted strata form numeric-feature matched pairs.
    Cases are all test-only and must not be used for threshold selection.
    """
    if type(seed) is not int or type(cases_per_scenario) is not int or cases_per_scenario < 1:
        raise ValueError("seed must be an integer and cases_per_scenario a positive integer")
    cases = [_make_case(seed, scenario, i) for scenario in STRESS_SCENARIOS for i in range(cases_per_scenario)]
    random.Random(f"{STRESS_VERSION}:{seed}:order").shuffle(cases)
    return tuple(cases)


def stress_fingerprint(cases: tuple[BenchmarkCase, ...]) -> str:
    """Hash the version and complete ordered cases, including labels and clocks."""
    data = {"version": STRESS_VERSION, "cases": [case.to_dict() for case in cases]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def serialize_stress_cases(cases: tuple[BenchmarkCase, ...], seed: int = 20260929) -> dict:
    """Return the labeled corpus manifest for local, auditable benchmark output."""
    return {"version": STRESS_VERSION, "seed": seed, "fingerprint": stress_fingerprint(cases),
            "cases": [case.to_dict() for case in cases],
            "scope": "Synthetic test-only engineered-feature challenges; no recorded sensor or physical outcome validation."}
