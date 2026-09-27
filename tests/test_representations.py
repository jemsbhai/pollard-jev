"""Factual input rendering must preserve evidence without supplying answers."""

import builtins
from datetime import datetime, timedelta, timezone
import json
import re

import pytest

from pollard_jev.contracts import ActionChoice, DecisionRequest, Observation
from pollard_jev.demo import example_request
from pollard_jev.loop import DecisionLoop
from pollard_jev.providers.openjev import OpenJevProvider
from pollard_jev.providers.representations import ROBOT_RULES, render_premise


NOW = datetime(2026, 9, 27, 12, 34, 56, 123456, tzinfo=timezone(timedelta(hours=-4)))


def observation(identifier="reading-a", feature="front_range_m", value=0.25, **changes):
    values = dict(
        observation_id=identifier, source="range-sensor", feature=feature,
        value=value, unit="m", observed_at=NOW - timedelta(seconds=60),
        valid_until=NOW - timedelta(seconds=30),
    )
    values.update(changes)
    return Observation(**values)


def request(observations=None, question="Select a skill."):
    return DecisionRequest(
        request_id="case-opaque-1", question=question,
        created_at=NOW, valid_until=NOW + timedelta(seconds=30),
        observations=observations if observations is not None else (observation(),),
        choices=(
            ActionChoice(name="action-alpha", hypothesis="Hypothesis alpha.", parameters={"steps": 2}),
            ActionChoice(name="action-beta", hypothesis="Hypothesis beta.", parameters={"samples": 1}),
        ),
    )


class CapturingEncoder:
    def __init__(self):
        self.calls = []

    def predict_hypotheses(self, premise, hypotheses):
        self.calls.append((premise, hypotheses))
        return [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]


def quoted(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def test_json_v1_is_byte_for_byte_legacy_serialization():
    sample = request((
        observation(source='sensor "a"\nroom é'),
        observation("reading-b", "battery_pct", None, status="unknown", unit="%"),
    ), question='Select "next"\tstep? λ')
    # The pre-representation adapter's exact serialization, including whitespace,
    # ASCII escaping, sorted keys, and the two existing timestamp conventions.
    legacy = json.dumps({
        "question": sample.question,
        "created_at": sample.created_at.isoformat(),
        "valid_until": sample.valid_until.isoformat(),
        "observations": [item.model_dump(mode="json") for item in sample.observations],
    }, sort_keys=True, allow_nan=False)
    assert render_premise(sample).encode("utf-8") == legacy.encode("utf-8")
    assert render_premise(sample, "json-v1").encode("utf-8") == legacy.encode("utf-8")
    assert "\\u00e9" in legacy
    assert '"value": null' in legacy


def test_text_keeps_individual_duplicate_readings_unknown_and_absent_features():
    sample = request((
        observation("reading-a", value=0.25),
        observation("reading-b", value=2.5, source="another-range-sensor"),
        observation("reading-c", "battery_pct", None, status="unknown", unit="%"),
    ))
    text = render_premise(sample, "text-v1")
    lines = text.splitlines()
    assert len(lines) == 3 + len(sample.observations)
    assert quoted(sample.question) in lines[0]
    assert quoted(sample.created_at.isoformat()) in lines[1]
    assert quoted(sample.valid_until.isoformat()) in lines[1]
    for index, (item, line) in enumerate(zip(sample.observations, lines[3:], strict=True), 1):
        assert line.startswith(f"{index}. ")
        for field, value in (
            ("Feature", item.feature), ("status", item.status), ("unit", item.unit),
            ("observation", item.observation_id), ("source", item.source),
            ("observed at", item.observed_at.isoformat()), ("valid until", item.valid_until.isoformat()),
        ):
            assert f"{field} {quoted(value)}" in line
    assert '"Forward range" is 0.25' in lines[3]
    assert '"Forward range" is 2.5' in lines[4]
    assert '"Battery charge" is null (unknown)' in lines[5]
    assert "camera_clearance_m" not in text
    assert "Camera clearance" not in text
    assert "Robot stuck indicator" not in text


def test_arbitrary_strings_stay_quoted_on_one_line_per_observation():
    unusual = 'custom "field"\nextra line\t\\sensor\rλ'
    sample = request((observation(
        identifier=unusual, feature=unusual, source=unusual, unit=unusual,
    ),), question=unusual)
    text = render_premise(sample, "text-v1")
    assert len(text.splitlines()) == 4
    assert unusual not in text
    assert quoted(unusual) in text
    for field in ("Feature", "source", "unit", "observation"):
        assert f"{field} {quoted(unusual)}" in text.splitlines()[3]


def test_text_reports_facts_without_policy_adjectives_or_action_labels():
    sample = request((
        observation("reading-a", value=0.1),
        observation("reading-b", "camera_clearance_m", 2.5),
        observation("reading-c", "battery_pct", 1.0, unit="%"),
        observation("reading-d", "stuck", 1.0, unit="bool"),
    ))
    text = render_premise(sample, "text-v1")
    # These expired, conflicting and small readings deliberately invite policy
    # conclusions. The renderer must preserve the numbers without adding them.
    assert '"Forward range" is 0.1' in text
    assert '"Camera clearance" is 2.5' in text
    assert '"Battery charge" is 1.0' in text
    assert '"Robot stuck indicator" is 1.0' in text
    assert not re.search(r"\b(stale|fresh|conflicting|sufficient|insufficient|low|clear|feasible|unsafe|threshold)\b", text)
    for choice in sample.choices:
        assert choice.name not in text
        assert choice.hypothesis not in text


@pytest.mark.parametrize("representation", ["json-v1", "text-v1", "robot-rules-v1"])
def test_provider_identity_and_mapping_preserve_selected_representation(representation):
    encoder = CapturingEncoder()
    provider = OpenJevProvider(encoder, synthetic=True, representation=representation)
    sample = request((observation(), observation("unknown-id", "battery_pct", None, status="unknown", unit="%")))
    result, = provider.infer((sample,))
    assert provider.identity.settings["input_representation"] == representation
    assert result.identity == provider.identity
    premise, hypotheses = encoder.calls[0]
    assert hypotheses == [choice.hypothesis for choice in sample.choices]
    if representation != "json-v1":
        assert '"Forward range" is 0.25' in premise
        assert '"Battery charge" is null (unknown)' in premise
        assert 'observation "unknown-id"' in premise
    else:
        assert json.loads(premise)["observations"] == [item.model_dump(mode="json") for item in sample.observations]
    assert result.proposed_action == "action-alpha"
    assert result.parameters == {"steps": 2}


def test_default_provider_identity_records_legacy_representation():
    assert OpenJevProvider(CapturingEncoder(), synthetic=True).identity.settings == {"input_representation": "json-v1"}


@pytest.mark.parametrize("representation", ["json-v1", "text-v1", "robot-rules-v1"])
def test_representation_cannot_change_after_loop_captures_audit_identity(representation):
    class CoherentEncoder(CapturingEncoder):
        def predict_hypotheses(self, premise, hypotheses):
            self.calls.append((premise, hypotheses))
            return [[0.02, 0.96, 0.02]] + [[0.85, 0.10, 0.05] for _ in hypotheses[1:]]

    encoder = CoherentEncoder()
    provider = OpenJevProvider(encoder, synthetic=True, representation=representation)
    sample = example_request(NOW)
    replacement = "robot-rules-v1" if representation == "json-v1" else "json-v1"
    with DecisionLoop(provider, clock=lambda: NOW) as loop:
        with pytest.raises(AttributeError):
            provider.representation = replacement
        record = loop.decide(sample)

    assert provider.representation == representation
    assert record.provider_status == "ok"
    assert record.policy.disposition == "accept"
    assert record.provider_identity.settings["input_representation"] == representation
    assert record.provider_result.identity == record.provider_identity == provider.identity
    assert encoder.calls[0][0] == render_premise(sample, representation)


@pytest.mark.parametrize("representation", ["", "future-v2", None, True])
def test_invalid_representation_rejected_before_encoder_use(representation):
    encoder = CapturingEncoder()
    with pytest.raises(ValueError, match="Unknown input representation"):
        OpenJevProvider(encoder, synthetic=True, representation=representation)
    with pytest.raises(ValueError, match="Unknown input representation"):
        render_premise(request(), representation)
    assert encoder.calls == []


def test_factory_rejects_invalid_representation_before_optional_import(monkeypatch):
    original_import = builtins.__import__
    optional = {"huggingface_hub", "torch", "transformers", "numpy"}
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in optional:
            attempted.append(name)
            raise AssertionError("Optional model dependency import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="Unknown input representation"):
        OpenJevProvider.from_local_cache(representation="unsupported")
    assert attempted == []


def test_robot_rules_prefix_is_static_task_information_and_preserves_factual_suffix(monkeypatch):
    import pollard_jev.contracts as contracts
    import pollard_jev.policy as policy
    import pollard_jev.simulator as simulator

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must not compute an outcome or inspect evidence")

    monkeypatch.setattr(simulator, "baseline", forbidden)
    monkeypatch.setattr(simulator, "inspect_evidence", forbidden)
    monkeypatch.setattr(policy.DecisionPolicy, "evaluate", forbidden)
    monkeypatch.setattr(contracts, "fresh", forbidden)
    samples = (
        request((observation(value=0.1), observation("battery", "battery_pct", 1.0, unit="%"))),
        request((observation(value=2.5), observation("battery", "battery_pct", 80.0, unit="%"))),
        request((observation(), observation("duplicate", value=2.5),
                 observation("unknown", "stuck", None, unit="bool", status="unknown"))),
    )
    factual_parts = []
    for sample in samples:
        factual = render_premise(sample, "text-v1")
        with_rules = render_premise(sample, "robot-rules-v1")
        assert with_rules == ROBOT_RULES + "\n" + factual
        assert with_rules.count(ROBOT_RULES) == 1
        # The extra context describes the task uniformly, never a per-case answer.
        assert "Robot demonstration operating rules" not in factual
        assert sample.choices[0].hypothesis not in with_rules
        factual_parts.append(factual)
    assert len(set(factual_parts)) == len(samples)


def test_robot_rules_keep_original_numeric_boundaries_and_precedence():
    lines = ROBOT_RULES.splitlines()
    numbered = [line for line in lines if re.match(r"^[1-4]\. ", line)]
    assert len(numbered) == 4
    assert "Battery charge below 10 percent" in numbered[0]
    assert "request assistance" in numbered[0]
    assert "at least 10 percent" in numbered[1]
    assert "stuck indicator of 1" in numbered[1]
    assert "recovery maneuver" in numbered[1]
    assert "stuck indicator 0" in numbered[2]
    assert "forward range or camera clearance below 0.5 metres" in numbered[2]
    assert "additional measurements" in numbered[2]
    assert "both forward range and camera clearance at least 0.5 metres" in numbered[3]
    assert "continue moving" in numbered[3]
    assert "Unknown, conflicting or expired observations require withholding movement" in ROBOT_RULES
