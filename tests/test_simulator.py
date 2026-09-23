from datetime import datetime, timedelta, timezone

import pytest

from pollard_jev.contracts import DecisionRequest, Observation, PolicyConfig
from pollard_jev.providers.fixture import FixtureProvider
from pollard_jev.simulator import RobotSimulator, baseline, default_choices, inspect_evidence


NOW = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)


def request(**overrides):
    features = {
        "front_range_m": (2.0, "m"),
        "camera_clearance_m": (2.1, "m"),
        "battery_pct": (80.0, "%"),
        "stuck": (0.0, "bool"),
    }
    features.update(overrides)
    observations = tuple(
        Observation(
            observation_id=f"obs-{name}", source="fixture-sensor", feature=name,
            value=value, unit=unit, observed_at=NOW,
            valid_until=NOW + timedelta(seconds=5),
            status="known" if value is not None else "unknown",
        )
        for name, (value, unit) in features.items()
    )
    return DecisionRequest(
        request_id="request-1", question="Which simulated skill should run?",
        created_at=NOW, valid_until=NOW + timedelta(seconds=5),
        observations=observations, choices=default_choices(),
    )


def test_simulated_skills_report_observed_state():
    simulator = RobotSimulator()
    outcomes = [
        simulator.execute("request-1", choice.name, choice.parameters, NOW)
        for choice in default_choices()
    ]
    assert outcomes[0].observed["position"] == 20.0
    assert outcomes[1].observed["measurements"] == 2.0
    assert outcomes[2].observed["position"] == 10.0
    assert outcomes[2].observed["recoveries"] == 1.0
    assert outcomes[3].observed["help_requests"] == 1.0
    assert all(outcome.simulated and outcome.status == "completed" for outcome in outcomes)
    assert simulator.dispatch_history == tuple(outcomes)


@pytest.mark.parametrize("action,parameters", [
    ("fly", {}), ("continue", {}), ("continue", {"distance_cm": 1, "speed": 1}),
    ("continue", {"distance_cm": 0}), ("continue", {"distance_cm": 51}),
    ("continue", {"distance_cm": True}), ("continue", {"distance_cm": 20.0}),
    ("inspect", {"samples": 6}), ("recover", {"distance_cm": 21}),
    ("request_help", {"retries": 2}),
])
def test_invalid_dispatch_has_no_side_effects(action, parameters):
    simulator = RobotSimulator()
    with pytest.raises(ValueError):
        simulator.execute("request-1", action, parameters, NOW)
    assert simulator.position_cm == 0.0
    assert simulator.dispatch_history == ()


@pytest.mark.parametrize("identifier,now", [("", NOW), ("request-1", NOW.replace(tzinfo=None))])
def test_invalid_outcome_metadata_has_no_side_effects(identifier, now):
    simulator = RobotSimulator()
    with pytest.raises(ValueError):
        simulator.execute(identifier, "continue", {"distance_cm": 20}, now)
    assert simulator.position_cm == 0.0
    assert simulator.dispatch_history == ()


@pytest.mark.parametrize("features,expected", [
    ({}, "continue"),
    ({"stuck": (1.0, "bool")}, "recover"),
    ({"stuck": (1.0, "bool"), "battery_pct": (5.0, "%")}, "request_help"),
    ({"front_range_m": (0.2, "m"), "camera_clearance_m": (0.3, "m")}, "inspect"),
])
def test_fixture_and_baseline_choose_from_identical_inputs(features, expected):
    item = request(**features)
    result, = FixtureProvider().infer((item,))
    assert baseline(item, NOW) == expected
    assert result.proposed_action == expected
    assert result.evidence == "sufficient"
    assert result.identity.synthetic
    assert result.semantics == "synthetic_support"
    assert sum(result.scores.values()) != pytest.approx(1.0)
    assert result.parameters == next(c.parameters for c in item.choices if c.name == expected)


@pytest.mark.parametrize("features,reason,evidence", [
    ({"stuck": (None, "bool")}, "missing_evidence", "unknown"),
    ({"stuck": (0.5, "bool")}, "invalid_evidence", "unknown"),
    ({"front_range_m": (-1.0, "m")}, "invalid_evidence", "unknown"),
    ({"battery_pct": (101.0, "%")}, "invalid_evidence", "unknown"),
    ({"battery_pct": (80.0, "volts")}, "invalid_evidence", "unknown"),
    ({"camera_clearance_m": (0.1, "m")}, "conflicting_evidence", "conflicting"),
])
def test_bad_evidence_prevents_baseline_action(features, reason, evidence):
    item = request(**features)
    assert inspect_evidence(item, PolicyConfig())[1] == reason
    assert baseline(item, NOW) == "need_evidence"
    assert FixtureProvider().infer((item,))[0].evidence == evidence


def test_missing_and_duplicate_features_are_rejected():
    item = request()
    missing = item.model_copy(update={"observations": item.observations[:-1]})
    assert inspect_evidence(missing, PolicyConfig())[1] == "missing_evidence"
    repeated = item.observations[0].model_copy(update={"observation_id": "duplicate-sensor"})
    duplicate = item.model_copy(update={"observations": (*item.observations, repeated)})
    assert inspect_evidence(duplicate, PolicyConfig())[1] == "invalid_evidence"
    assert baseline(duplicate, NOW) == "need_evidence"


def test_baseline_expires_at_exclusive_validity_boundary():
    assert baseline(request(), NOW + timedelta(seconds=5)) == "defer"
    assert baseline(request(), NOW - timedelta(microseconds=1)) == "defer"


def test_baseline_respects_available_choices_and_allowlist():
    item = request()
    assert baseline(item, NOW, PolicyConfig(allowlist=("inspect",))) == "defer"
    assert baseline(item.model_copy(update={"choices": item.choices[1:]}), NOW) == "defer"


def test_fixture_batch_is_reproducible_and_supports_failure_injection():
    provider = FixtureProvider()
    batch = (request(), request(stuck=(1.0, "bool")))
    assert provider.infer(batch) == provider.infer(batch)
    assert len(provider.infer(batch)) == 2
    with pytest.raises(RuntimeError, match="Synthetic fixture"):
        FixtureProvider(failure=True).infer(batch)
    assert FixtureProvider(proposed_action="fly").infer(batch)[0].proposed_action == "fly"


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), True])
def test_fixture_rejects_invalid_delay(delay):
    with pytest.raises(ValueError):
        FixtureProvider(delay_s=delay)
