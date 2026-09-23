"""Consequential edge cases found in the independent milestone review."""

from datetime import datetime, timedelta, timezone
import threading

import pytest

from pollard_jev.contracts import (
    DecisionRequest, Observation, PolicyConfig, ProviderIdentity, ProviderResult,
)
from pollard_jev.loop import DecisionLoop
from pollard_jev.policy import DecisionPolicy
from pollard_jev.simulator import RobotSimulator, default_choices


NOW = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)
IDENTITY = ProviderIdentity(
    provider="review-fixture", provider_version="1", model="synthetic",
    model_version="1", synthetic=True,
)


def request(request_id="review-request"):
    return DecisionRequest(
        request_id=request_id, question="Select a simulated supervisory skill.",
        created_at=NOW, valid_until=NOW + timedelta(seconds=30),
        observations=tuple(
            Observation(
                observation_id=feature, source="review-fixture", feature=feature,
                value=value, unit=unit, observed_at=NOW,
                valid_until=NOW + timedelta(seconds=5),
            )
            for feature, value, unit in (
                ("front_range_m", 1.2, "m"),
                ("camera_clearance_m", 1.2, "m"),
                ("battery_pct", 80.0, "%"),
                ("stuck", 0.0, "bool"),
            )
        ),
        choices=default_choices(),
    )


def result(item):
    return ProviderResult(
        request_id=item.request_id, identity=IDENTITY, semantics="synthetic_support",
        scores={choice.name: 0.95 if choice.name == "continue" else 0.01
                for choice in item.choices},
        proposed_action="continue", parameters={"distance_cm": 20},
        evidence="sufficient",
    )


class Provider:
    identity = IDENTITY

    def infer(self, requests):
        return tuple(result(item) for item in requests)


@pytest.mark.parametrize("change", ["cancel", "timeout", "expiry"])
def test_dispatch_rechecks_state_after_policy_evaluation(monkeypatch, change):
    """A valid preliminary evaluation does not authorize a later stale dispatch."""
    cancel = threading.Event()
    clocks = {"monotonic": 0.0, "wall": NOW}
    monkeypatch.setattr("pollard_jev.loop.time.monotonic", lambda: clocks["monotonic"])
    simulator = RobotSimulator()
    with DecisionLoop(
        Provider(), simulator=simulator, timeout_s=1.0,
        clock=lambda: clocks["wall"],
    ) as loop:
        evaluate = loop.policy.evaluate
        calls = 0

        def crossing_boundary(item, output, now):
            nonlocal calls
            calls += 1
            outcome = evaluate(item, output, now)
            if calls == 2:  # Final policy evaluation inside registered handler.
                if change == "cancel":
                    cancel.set()
                elif change == "timeout":
                    clocks["monotonic"] = 2.0
                else:
                    clocks["wall"] = NOW + timedelta(seconds=6)
            return outcome

        monkeypatch.setattr(loop.policy, "evaluate", crossing_boundary)
        record = loop.decide(request(), cancel=cancel)

    assert calls >= 2
    assert simulator.dispatch_history == ()
    assert record.policy.disposition != "accept"
    assert record.action is None or record.action.status == "refused"


def test_failed_dispatch_uses_batchs_single_action_attempt():
    """A failed actuator response cannot prove the first attempt had no effect."""
    class FailingSimulator(RobotSimulator):
        attempts = 0

        def execute(self, request_id, action, parameters, now):
            self.attempts += 1
            raise RuntimeError("simulated outcome unavailable")

    simulator = FailingSimulator()
    with DecisionLoop(Provider(), simulator=simulator, clock=lambda: NOW) as loop:
        records = loop.decide_batch((request("first"), request("second")))

    assert simulator.attempts == 1
    assert records[0].action.status == "failed"
    assert records[1].policy.reason == "batch_requires_new_observation"
    assert records[1].action is None


def test_narrow_required_feature_configuration_does_not_crash_policy():
    """Either reject incompatible robot configuration or return a safe outcome."""
    try:
        config = PolicyConfig(required_units={})
    except ValueError:
        return  # Explicit validation is also a valid boundary for this case.
    item = request()
    outcome = DecisionPolicy(config).evaluate(item, result(item), NOW)
    assert outcome.disposition in {"need_evidence", "defer"}


def test_robot_thresholds_cannot_silently_treat_centimetres_as_metres():
    """Configurable labels cannot redefine fixed-unit feasibility thresholds."""
    units = {
        "front_range_m": "cm", "camera_clearance_m": "cm",
        "battery_pct": "%", "stuck": "bool",
    }
    try:
        config = PolicyConfig(required_units=units)
    except ValueError:
        return
    original = request()
    item = DecisionRequest.model_validate({
        **original.model_dump(),
        "observations": tuple(
            obs.model_copy(update={"unit": units[obs.feature]})
            for obs in original.observations
        ),
    })
    # 1.2 cm is less than the 0.5 m movement clearance threshold.
    outcome = DecisionPolicy(config).evaluate(item, result(item), NOW)
    assert outcome.disposition in {"need_evidence", "defer"}
