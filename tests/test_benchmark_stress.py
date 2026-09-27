from collections import Counter
import json

import pytest

from pollard_jev.benchmark import calibrate_thresholds, make_corpus, run_benchmark
from pollard_jev.benchmark_stress import (
    STRESS_SCENARIOS, make_stress_cases, serialize_stress_cases, stress_fingerprint,
)
from pollard_jev.providers.fixture import FixtureProvider
from pollard_jev.providers.openjev import OpenJevProvider
from pollard_jev.simulator import baseline


def _values(case):
    return {obs.feature: obs.value for obs in case.request.observations}


def test_stress_is_seeded_stratified_test_only_and_separate_from_primary_corpus():
    cases = make_stress_cases()
    assert cases == make_stress_cases()
    assert stress_fingerprint(cases) == stress_fingerprint(make_stress_cases())
    assert stress_fingerprint(cases) != stress_fingerprint(make_stress_cases(seed=17))
    assert len(cases) == 60
    assert Counter(case.scenario for case in cases) == {name: 4 for name in STRESS_SCENARIOS}
    assert all(case.split == "test" for case in cases)
    corpus = make_corpus()
    ids = {case.request.request_id for case in cases}
    assert len(ids) == len(cases)
    assert ids.isdisjoint(case.request.request_id for case in (*corpus.calibration, *corpus.test))
    manifest = serialize_stress_cases(cases)
    assert manifest["fingerprint"] == stress_fingerprint(cases)
    assert len(json.loads(json.dumps(manifest))["cases"]) == len(cases)


def test_independent_stress_labels_match_declared_toy_specification():
    cases = make_stress_cases()
    # Baseline is a cross-check after labels exist, never their source.
    for case in cases:
        assert baseline(case.request, case.now) == case.expected_outcome
        values = _values(case)
        if case.scenario.startswith("stress_clearance_"):
            distance = min(values["front_range_m"], values["camera_clearance_m"])
            if case.scenario.endswith("below"):
                assert distance < 0.5 and case.expected_action == "inspect"
            elif case.scenario.endswith("at"):
                assert distance == 0.5 and case.expected_action == "continue"
            else:
                assert distance > 0.5 and case.expected_action == "continue"
        if case.scenario == "stress_low_battery_stuck":
            assert values["battery_pct"] < 10 and values["stuck"] == 1
            assert case.expected_action == "request_help"
        if case.scenario == "stress_obstacle_stuck":
            assert min(values["front_range_m"], values["camera_clearance_m"]) < 0.5
            assert values["stuck"] == 1 and values["battery_pct"] >= 10
            assert case.expected_action == "recover"


def test_permuted_cases_are_feature_matched_and_keep_expected_labels():
    cases = make_stress_cases()
    originals = [case for case in cases if case.scenario == "stress_order_original"]
    permuted = [case for case in cases if case.scenario == "stress_order_permuted"]
    for original in originals:
        matches = [case for case in permuted if _values(case) == _values(original)]
        assert len(matches) == 1
        paired = matches[0]
        assert paired.expected_outcome == original.expected_outcome
        assert paired.request.request_id != original.request.request_id
        assert [obs.feature for obs in paired.request.observations] != [obs.feature for obs in original.request.observations]
        assert sorted((obs.model_dump() for obs in paired.request.observations), key=lambda item: item["observation_id"]) == sorted(
            (obs.model_dump() for obs in original.request.observations), key=lambda item: item["observation_id"])


def test_unknown_duplicate_and_future_evidence_and_exclusions_are_represented():
    for case in make_stress_cases():
        if case.scenario == "stress_unknown_evidence":
            assert sum(obs.value is None and obs.status == "unknown" for obs in case.request.observations) == 1
            assert case.expected_outcome == "need_evidence"
        elif case.scenario == "stress_duplicate_evidence":
            assert len(case.request.observations) == 5
            assert len({obs.feature for obs in case.request.observations}) == 4
            assert len({obs.observation_id for obs in case.request.observations}) == 5
            assert case.expected_outcome == "need_evidence"
        elif case.scenario == "stress_future_observation":
            assert any(obs.observed_at > case.now for obs in case.request.observations)
            assert case.expected_outcome == "defer"
        elif case.scenario == "stress_future_request":
            assert case.request.created_at > case.now
            assert case.expected_outcome == "defer"
        elif case.scenario == "stress_excluded_preferred":
            assert len(case.request.choices) == 3
            assert case.expected_outcome == "defer"


def test_model_premises_never_receive_stress_labels_or_split_metadata():
    class CapturingEncoder:
        def __init__(self):
            self.premises = []

        def predict_hypotheses(self, premise, hypotheses):
            self.premises.append(premise)
            return [[0.2, 0.6, 0.2] for _ in hypotheses]

    encoder = CapturingEncoder()
    provider = OpenJevProvider(encoder, synthetic=True)
    cases = make_stress_cases(cases_per_scenario=1)
    provider.infer(tuple(case.request for case in cases))
    for case, premise in zip(cases, encoder.premises, strict=True):
        assert case.scenario not in premise
        assert case.scenario not in case.request.request_id
        assert "expected_outcome" not in premise
        assert '"split"' not in premise
        assert '"test"' not in premise


def test_stress_executes_fixture_through_loop_and_cannot_be_calibration_data():
    run = run_benchmark(FixtureProvider(), make_stress_cases(cases_per_scenario=1), warmup=0)
    assert run.metrics()["policy_outcome_accuracy"] == 1.0
    assert run.metrics()["baseline_outcome_accuracy"] == 1.0
    assert run.metrics()["accepted_wrong_count"] == 0
    with pytest.raises(ValueError, match="exclusively calibration"):
        calibrate_thresholds(run)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_stress_size_is_rejected(count):
    with pytest.raises(ValueError):
        make_stress_cases(cases_per_scenario=count)
