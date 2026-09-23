import json
import threading
from datetime import timedelta

import pytest
from pollard import verify
from pollard.errors import PolicyViolation
from pydantic import ValidationError

from pollard_jev.contracts import DecisionRequest, Observation, PolicyConfig, ProviderResult
from pollard_jev.demo import example_request, run_demo
from pollard_jev.loop import DecisionLoop
from pollard_jev.providers.fixture import FixtureProvider
from pollard_jev.records import inspect_history, live_reevaluate, simulate_policy


class TransformProvider(FixtureProvider):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def infer(self, requests):
        return tuple(self.transform(r.model_dump()) for r in super().infer(requests))


def change(**updates):
    def apply(result):
        return {**result, **updates}
    return apply


def test_fresh_full_loop_and_pollard_integrity(request_data, now, tmp_path):
    path = tmp_path / "records.jsonl"
    with DecisionLoop(FixtureProvider(), clock=lambda: now, records_path=path, pollard_path=tmp_path / "ledger.db") as loop:
        record = loop.decide(request_data)
        assert record.policy.disposition == "accept"
        assert record.action.status == "completed"
        assert record.action.simulated
        assert loop.spent_requests == 1
        assert verify(loop.runtime.store, loop.run.cursor_id).ok
        assert len(loop.simulator.dispatch_history) == 1
    assert inspect_history(path) == (record,)


def test_unauthorized_action_never_dispatches(request_data, now):
    with DecisionLoop(TransformProvider(change(proposed_action="launch")), clock=lambda: now) as loop:
        record = loop.decide(request_data)
        assert record.policy.reason == "unauthorized_action"
        assert not loop.simulator.dispatch_history


@pytest.mark.parametrize("name,params", [("unknown", {}), ("continue", {"distance_cm": 51}), ("continue", {"distance_cm": -1}), ("inspect", {"samples": True}), ("continue", {"distance_cm": 10, "extra": 1})])
def test_real_pollard_registry_refuses_bad_tools(name, params):
    with DecisionLoop(FixtureProvider()) as loop:
        with pytest.raises(PolicyViolation):
            loop.run.tool_call(name, params)
        assert not loop.simulator.dispatch_history
        assert loop.spent_requests == 0


def test_registry_cannot_bypass_dispatch_context():
    with DecisionLoop(FixtureProvider()) as loop:
        node = loop.run.tool_call("continue", {"distance_cm": 20})
        assert node.result == {"status": "refused", "reason": "no_dispatch_context"}
        assert not loop.simulator.dispatch_history


def test_allowlist_narrows_valid_choices(request_data, now):
    with DecisionLoop(FixtureProvider(), clock=lambda: now, policy=PolicyConfig(allowlist=("inspect",))) as loop:
        record = loop.decide(request_data)
        assert record.policy.reason == "unauthorized_action"
        assert not loop.simulator.dispatch_history


def test_evidence_rechecked_inside_dispatch(request_data, now):
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        return now if calls == 1 else now + timedelta(seconds=31)
    with DecisionLoop(FixtureProvider(), clock=clock) as loop:
        record = loop.decide(request_data)
        assert record.provider_result.proposed_action == "continue"
        assert record.policy.reason == "stale_or_future_evidence"
        assert record.action.status == "refused"
        assert not loop.simulator.dispatch_history


@pytest.mark.parametrize("updates", [
    {"scores": {"continue": float("nan")}}, {"scores": {"continue": float("inf")}},
    {"scores": {"continue": -0.1}}, {"scores": {"continue": True}},
    {"parameters": {"distance_cm": True}}, {"request_id": "wrong-request"},
    {"unexpected_secret": "never-store-this"},
])
def test_malformed_outputs_defer(request_data, now, updates):
    with DecisionLoop(TransformProvider(change(**updates)), clock=lambda: now) as loop:
        record = loop.decide(request_data)
        assert record.provider_status == "malformed"
        assert record.provider_result is None
        assert record.policy.disposition == "defer"
        assert not loop.simulator.dispatch_history
        assert loop.spent_requests == 1


def test_parameters_must_obey_request_and_global_bounds(request_data, now):
    for distance, expected in [(100, "invalid_action_parameters"), (30, "parameters_not_permitted_by_request")]:
        with DecisionLoop(TransformProvider(change(parameters={"distance_cm": distance})), clock=lambda: now) as loop:
            record = loop.decide(request_data)
            assert record.policy.reason == expected
            assert not loop.simulator.dispatch_history


class BlockingProvider(FixtureProvider):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def infer(self, requests):
        self.started.set()
        self.release.wait(3)
        try:
            return super().infer(requests)
        finally:
            self.finished.set()


def test_timeout_drops_late_response_and_bounds_workers(request_data, now):
    provider = BlockingProvider()
    with DecisionLoop(provider, timeout_s=0.03, clock=lambda: now) as loop:
        try:
            record = loop.decide(request_data)
            assert provider.started.is_set()
            assert record.provider_status == "timeout"
            assert loop.decide(request_data).provider_status == "busy"
            assert loop.spent_requests == 1
        finally:
            provider.release.set()
            assert provider.finished.wait(2)
            loop._worker.join(2)
        assert not loop.simulator.dispatch_history
        assert record.action is None


def test_cancelled_before_inference_is_free(request_data):
    cancel = threading.Event()
    cancel.set()
    with DecisionLoop(FixtureProvider()) as loop:
        record = loop.decide(request_data, cancel=cancel)
        assert record.provider_status == "cancelled"
        assert loop.spent_requests == 0
        assert not loop.simulator.dispatch_history


def test_cancellation_during_inference_blocks_late_execution(request_data, now):
    cancel = threading.Event()
    class CancellingProvider(FixtureProvider):
        def infer(self, requests):
            result = super().infer(requests)
            cancel.set()
            return result
    with DecisionLoop(CancellingProvider(), clock=lambda: now) as loop:
        record = loop.decide(request_data, cancel=cancel)
        assert record.provider_status == "cancelled"
        assert loop.spent_requests == 1
        assert not loop.simulator.dispatch_history


def test_cancellation_rechecked_at_dispatch(request_data, now, monkeypatch):
    cancel = threading.Event()
    with DecisionLoop(FixtureProvider(), clock=lambda: now) as loop:
        original = loop.run.tool_call
        def tool_call(*args, **kwargs):
            cancel.set()
            return original(*args, **kwargs)
        monkeypatch.setattr(loop.run, "tool_call", tool_call)
        record = loop.decide(request_data, cancel=cancel)
        assert record.policy.reason == "cancelled"
        assert not loop.simulator.dispatch_history


def test_batch_is_one_request_and_next_attempt_refused(now):
    with DecisionLoop(FixtureProvider(), max_requests=1, clock=lambda: now) as loop:
        records = loop.decide_batch((example_request(now, request_id="a"), example_request(now, request_id="b")))
        assert len(records) == 2
        assert records[0].pollard_model_node_id == records[1].pollard_model_node_id
        assert all(r.budget_spent_requests == 1 for r in records)
        assert records[1].policy.reason == "batch_requires_new_observation"
        assert len(loop.simulator.dispatch_history) == 1
        exhausted = loop.decide(example_request(now, request_id="c"))
        assert exhausted.provider_status == "budget_exhausted"
        assert loop.spent_requests == 1


def test_provider_failure_is_charged_and_exception_text_not_recorded(request_data, now, tmp_path):
    secret = "fake-api-key-never-store"
    class FailingProvider(FixtureProvider):
        def infer(self, requests):
            raise RuntimeError(secret)
    path = tmp_path / "decisions.jsonl"
    with DecisionLoop(FailingProvider(), max_requests=1, clock=lambda: now, records_path=path) as loop:
        record = loop.decide(request_data)
        assert record.provider_status == "failure"
        assert loop.spent_requests == 1
        assert secret not in str([n for n in loop.runtime.store.walk(loop.run.root_id)])
    assert secret not in path.read_text()


def test_history_and_policy_simulation_have_no_side_effects(request_data, now, tmp_path):
    path = tmp_path / "decisions.jsonl"
    with DecisionLoop(FixtureProvider(), clock=lambda: now, records_path=path) as loop:
        record = loop.decide(request_data)
        before = tuple(loop.simulator.dispatch_history)
        assert inspect_history(path) == (record,)
        assert simulate_policy(record) == record.policy
        assert simulate_policy(record, PolicyConfig(acceptance_threshold=1.0)).disposition == "defer"
        assert simulate_policy(record, at=now + timedelta(seconds=31)).reason == "stale_or_future_evidence"
        assert tuple(loop.simulator.dispatch_history) == before
        assert loop.spent_requests == 1


def test_explicit_live_reevaluation_charges_but_cannot_dispatch(request_data, now):
    with DecisionLoop(FixtureProvider(), clock=lambda: now) as loop:
        record = loop.decide(request_data)
        updated = live_reevaluate(record, loop)
        assert loop.spent_requests == 2
        assert len(loop.simulator.dispatch_history) == 1
        assert updated.mode == "live_reevaluation"
        assert updated.reevaluates_record_id == record.record_id
        assert updated.action is None


def test_independent_scores_are_not_normalized(request_data, now):
    scores = {"continue": 0.95, "inspect": 0.5, "recover": 0.4, "request_help": 0.3}
    with DecisionLoop(TransformProvider(change(scores=scores)), clock=lambda: now) as loop:
        record = loop.decide(request_data)
        assert record.provider_result.scores == scores
        assert sum(scores.values()) > 1
        assert record.policy.disposition == "accept"


def test_demo_scenarios_are_offline(tmp_path, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in offline demo")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    records = run_demo(tmp_path)
    assert [r.policy.disposition for r in records] == ["accept", "need_evidence", "need_evidence", "defer", "defer", "defer"]
    assert sum(r.action is not None and r.action.status == "completed" for r in records) == 1


def test_contracts_reject_invalid_observations(request_data):
    data = request_data.observations[0].model_dump()
    for updates in ({"value": float("nan")}, {"observed_at": data["observed_at"].replace(tzinfo=None)},
                    {"valid_until": data["observed_at"]}, {"status": "unknown"}):
        with pytest.raises(ValidationError):
            Observation.model_validate({**data, **updates})
