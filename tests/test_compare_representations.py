"""Paired representation comparisons using only local synthetic providers."""

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pollard_jev.benchmark import make_corpus, run_benchmark
from pollard_jev.contracts import PolicyConfig
from pollard_jev.providers.fixture import FixtureProvider


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_representations")


def one_case(scenario="clear", seed=20260927):
    return next(case for case in make_corpus(seed=seed, cases_per_scenario=1).test
                if case.scenario == scenario)


def provider_for(action):
    return FixtureProvider(failure=action is None, proposed_action=action)


@pytest.mark.parametrize("candidate", ["text-v1", "robot-rules-v1"])
@pytest.mark.parametrize("left,right,expected", [
    ("inspect", "continue", "candidate_improves"),
    ("continue", "inspect", "candidate_regresses"),
    ("continue", "continue", "both_correct"),
    ("inspect", "recover", "both_wrong"),
    (None, "continue", "candidate_improves"),
    ("continue", None, "candidate_regresses"),
    (None, None, "both_wrong"),
])
def test_paired_proposal_counts_include_absent_results(comparison, left, right, expected, candidate):
    case = one_case()
    runs = {
        "json-v1": run_benchmark(provider_for(left), (case,), warmup=0),
        candidate: run_benchmark(provider_for(right), (case,), warmup=0),
    }
    result = comparison.compare(runs)
    assert result["candidate"] == candidate
    assert result["paired_proposal_counts_on_eligible"] == {
        name: int(name == expected)
        for name in ("candidate_improves", "candidate_regresses", "both_correct", "both_wrong")
    }
    for name, run in runs.items():
        assert result["metrics"][name] == run.metrics()
        assert result["per_scenario"][name]["clear"] == run.metrics()
    for name, action in (("json-v1", left), (candidate, right)):
        if action is None:
            assert runs[name].rows[0].record.provider_result is None
            assert result["metrics"][name]["provider_status_counts"] == {"failure": 1}


@pytest.mark.parametrize("scenario", ["missing", "conflicting", "stale"])
def test_required_abstentions_excluded_from_proposal_counts_but_kept_in_metrics(comparison, scenario):
    case = one_case(scenario)
    runs = {
        "json-v1": run_benchmark(FixtureProvider(proposed_action="continue"), (case,), warmup=0),
        "text-v1": run_benchmark(FixtureProvider(failure=True), (case,), warmup=0),
    }
    result = comparison.compare(runs)
    assert all(count == 0 for count in result["paired_proposal_counts_on_eligible"].values())
    for name in runs:
        assert result["metrics"][name]["total_cases"] == 1
        assert result["metrics"][name]["required_abstention_cases"] == 1
        assert result["metrics"][name]["eligible_action_cases"] == 0
        assert result["per_scenario"][name][scenario]["total_cases"] == 1


def test_comparison_rejects_different_requests_even_with_same_scenario_label(comparison):
    first, second = one_case(seed=1), one_case(seed=2)
    assert first.scenario == second.scenario
    assert first.request != second.request
    runs = {
        "json-v1": run_benchmark(FixtureProvider(), (first,), warmup=0),
        "text-v1": run_benchmark(FixtureProvider(), (second,), warmup=0),
    }
    with pytest.raises(ValueError, match="identical requests"):
        comparison.compare(runs)


def test_comparison_rejects_unequal_number_of_paired_cases(comparison):
    cases = make_corpus(cases_per_scenario=1).test[:2]
    runs = {
        "json-v1": run_benchmark(FixtureProvider(), cases, warmup=0),
        "text-v1": run_benchmark(FixtureProvider(), cases[:1], warmup=0),
    }
    with pytest.raises(ValueError):
        comparison.compare(runs)


def test_comparison_rejects_same_request_with_different_expected_label(comparison):
    first = one_case()
    second = replace(first, expected_outcome="inspect")
    assert first.request == second.request
    runs = {
        "json-v1": run_benchmark(FixtureProvider(), (first,), warmup=0),
        "robot-rules-v1": run_benchmark(FixtureProvider(), (second,), warmup=0),
    }
    with pytest.raises(ValueError, match="identical requests, labels and evaluation times"):
        comparison.compare(runs)


def test_run_paired_alternates_same_cases_and_separates_warmups(comparison, monkeypatch, tmp_path):
    import pollard_jev.benchmark as benchmark

    calls = []
    providers = {"json-v1": FixtureProvider(), "text-v1": FixtureProvider()}
    configs = {
        "json-v1": PolicyConfig(),
        "text-v1": PolicyConfig(acceptance_threshold=0.9),
    }
    cases = make_corpus(cases_per_scenario=1).test[:3]
    before = [case.to_dict() for case in cases]
    real_run = benchmark.run_benchmark

    def observed_run(provider, supplied_cases, **kwargs):
        name = next(name for name, candidate in providers.items() if candidate is provider)
        calls.append((name, tuple(supplied_cases), kwargs))
        return real_run(provider, supplied_cases, **kwargs)

    monkeypatch.setattr(benchmark, "run_benchmark", observed_run)
    runs = comparison.run_paired(providers, cases, configs, tmp_path, "trial", warmup=True)
    assert [name for name, _, _ in calls] == [
        "json-v1", "text-v1",  # Separate warmups.
        "json-v1", "text-v1", "text-v1", "json-v1", "json-v1", "text-v1",
    ]
    assert calls[0][1] == calls[1][1]
    warmup_case = calls[0][1][0]
    assert warmup_case.split == "warmup"
    assert warmup_case.request.request_id not in {case.request.request_id for case in cases}
    for index, case in enumerate(cases):
        first, second = calls[2 + index * 2:4 + index * 2]
        assert first[1] == second[1] == (case,)
    for name, _, kwargs in calls:
        assert kwargs["policy"] == configs[name]
        assert kwargs["warmup"] == 0
        assert kwargs["timeout_s"] == 120
    for name, run in runs.items():
        assert [row.case for row in run.rows] == list(cases)
        assert run.metrics()["total_cases"] == 3
        assert len(run.warmup) == 1
        assert run.warmup[0]["request_id"] == warmup_case.request.request_id
        assert all(row.record.policy_config == configs[name] for row in run.rows)
        records = (tmp_path / name / "trial-records.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(records) == 3
        assert [json.loads(line)["request"]["request_id"] for line in records] == [case.request.request_id for case in cases]
        saved = json.loads((tmp_path / name / "trial-run.json").read_text(encoding="utf-8"))
        assert saved["metrics"]["total_cases"] == 3
        assert len(saved["warmup"]) == 1
        assert (tmp_path / name / "trial-ledger.db").is_file()
    assert [case.to_dict() for case in cases] == before
