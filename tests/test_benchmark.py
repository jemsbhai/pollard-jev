from collections import Counter

import pytest

from pollard_jev.benchmark import SCENARIOS, calibrate_thresholds, evaluate_saved, make_corpus, run_benchmark
from pollard_jev.contracts import PolicyConfig
from pollard_jev.providers.fixture import FixtureProvider


def test_corpus_is_seeded_stratified_disjoint_and_opaque():
    corpus = make_corpus(seed=17, cases_per_scenario=3)
    assert corpus == make_corpus(seed=17, cases_per_scenario=3)
    assert corpus.fingerprint == make_corpus(seed=17, cases_per_scenario=3).fingerprint
    assert corpus.fingerprint != make_corpus(seed=18, cases_per_scenario=3).fingerprint
    assert len(corpus.calibration) == len(corpus.test) == 24
    assert Counter(case.scenario for case in corpus.calibration) == {name: 3 for name in SCENARIOS}
    assert Counter(case.scenario for case in corpus.test) == {name: 3 for name in SCENARIOS}
    calibration_ids = {case.request.request_id for case in corpus.calibration}
    assert calibration_ids.isdisjoint(case.request.request_id for case in corpus.test)
    calibration_values = {tuple(obs.value for obs in case.request.observations) for case in corpus.calibration}
    test_values = {tuple(obs.value for obs in case.request.observations) for case in corpus.test}
    assert calibration_values.isdisjoint(test_values)
    for case in (*corpus.calibration, *corpus.test):
        assert case.scenario not in case.request.request_id
        assert all(case.scenario not in obs.observation_id for obs in case.request.observations)


def test_fixture_reaches_independent_labels_and_dispatches_through_loop():
    corpus = make_corpus(cases_per_scenario=1)
    run = run_benchmark(FixtureProvider(), corpus.test, warmup=1)
    metrics = run.metrics()
    assert metrics["total_cases"] == 8
    assert metrics["proposal_accuracy_on_eligible"] == 1.0
    assert metrics["policy_outcome_accuracy"] == 1.0
    assert metrics["baseline_outcome_accuracy"] == 1.0
    assert metrics["accepted_wrong_count"] == 0
    assert metrics["completed_simulated_actions"] == 5
    assert metrics["coverage"] == 5 / 8
    assert metrics["required_abstention_correct_count"] == 3
    assert metrics["end_to_end_latency"]["count"] == 8
    assert len(run.warmup) == 1
    assert run.warmup[0]["request_id"] not in {case.request.request_id for case in corpus.test}
    for row in run.rows:
        assert row.record.pollard_model_node_id is not None
        assert row.record.recorded_at == row.case.now
        assert row.record.action is None or row.record.action.simulated
    assert len(run.to_dict()["rows"]) == 8


def test_confident_wrong_provider_is_counted_even_when_policy_allows_action():
    corpus = make_corpus(cases_per_scenario=1)
    # Inspect is feasible even when it is not the scenario's preferred action.
    run = run_benchmark(FixtureProvider(proposed_action="inspect"), corpus.test, warmup=0)
    metrics = run.metrics()
    assert metrics["proposal_accuracy_on_eligible"] == 1 / 5
    assert metrics["accepted_wrong_count"] == 4
    assert metrics["accepted_action_accuracy"] == 1 / 5
    assert metrics["coverage"] == 5 / 8


class AbstainingProvider(FixtureProvider):
    def infer(self, requests):
        return tuple(result.model_copy(update={"proposed_action": None, "parameters": {}, "evidence": "unknown"})
                     for result in super().infer(requests))


def test_all_abstaining_provider_does_not_appear_perfect():
    run = run_benchmark(AbstainingProvider(), make_corpus(cases_per_scenario=1).test, warmup=0)
    metrics = run.metrics()
    assert metrics["accepted_wrong_count"] == 0
    assert metrics["accepted_action_accuracy"] is None
    assert metrics["coverage"] == 0.0
    assert metrics["eligible_action_coverage"] == 0.0
    assert metrics["proposal_accuracy_on_eligible"] == 0.0
    assert metrics["proposal_coverage_on_eligible"] == 0.0
    assert metrics["abstention_rate"] == 1.0
    assert metrics["unexpected_abstention_count"] == 5
    assert metrics["policy_outcome_accuracy"] == 3 / 8


def test_calibration_rejects_test_rows_and_never_calls_provider_again():
    class CountingProvider(FixtureProvider):
        calls = 0

        def infer(self, requests):
            self.calls += 1
            return super().infer(requests)

    corpus = make_corpus(cases_per_scenario=1)
    provider = CountingProvider()
    calibration = run_benchmark(provider, corpus.calibration, warmup=0)
    selected = calibrate_thresholds(calibration, thresholds=(0.8, 1.0), margins=(0.15, 1.0))
    assert provider.calls == 8
    assert selected.policy.acceptance_threshold == 0.8
    assert selected.policy.minimum_margin == 0.15
    assert set(selected.calibration_request_ids).isdisjoint(case.request.request_id for case in corpus.test)
    heldout = run_benchmark(provider, corpus.test, policy=selected.policy, warmup=0)
    assert provider.calls == 16
    assert heldout.metrics()["accepted_wrong_count"] == 0
    with pytest.raises(ValueError, match="exclusively calibration"):
        calibrate_thresholds(heldout)


def test_calibration_can_reject_confident_wrong_predictions_and_exposes_lost_coverage():
    run = run_benchmark(FixtureProvider(proposed_action="inspect"), make_corpus(cases_per_scenario=1).calibration, warmup=0)
    selected = calibrate_thresholds(run, thresholds=(0.8, 1.0), margins=(0.15,))
    assert selected.policy.acceptance_threshold == 1.0
    chosen = next(item for item in selected.candidates if item["acceptance_threshold"] == 1.0)
    assert chosen["accepted_wrong_count"] == chosen["accepted_correct_count"] == 0
    assert chosen["coverage"] == 0.0
    assert "latency" not in str(chosen.keys())


def test_saved_policy_replay_changes_decisions_without_mutating_dispatch_records():
    run = run_benchmark(FixtureProvider(), make_corpus(cases_per_scenario=1).test, warmup=0)
    before = run.to_dict()
    replay = evaluate_saved(run, PolicyConfig(acceptance_threshold=1.0))
    assert replay["mode"] == "saved_policy_replay_no_dispatch"
    assert replay["metrics"]["accepted_count"] == 0
    assert replay["metrics"]["unexpected_abstention_count"] == 5
    assert "completed_simulated_actions" not in replay["metrics"]
    assert replay["metrics"]["end_to_end_latency"] == run.metrics()["end_to_end_latency"]
    assert run.to_dict() == before
    assert run.metrics()["completed_simulated_actions"] == 5


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_corpus_size_is_rejected(count):
    with pytest.raises(ValueError):
        make_corpus(cases_per_scenario=count)
