"""Adapter mapping tests; these deliberately never load real model weights."""

from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
import types

import pytest

from pollard_jev.contracts import ActionChoice, DecisionRequest, Observation, ProviderResult
from pollard_jev.providers.openjev import CHECKPOINT, REPOSITORY, REVISION, OpenJevProvider


class FakeEncoder:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict_hypotheses(self, premise, hypotheses):
        self.calls.append((premise, hypotheses))
        return self.rows


def make_request(request_id="request-1"):
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    return DecisionRequest(
        request_id=request_id, question="Which supervisory skill is supported?",
        created_at=now, valid_until=now + timedelta(seconds=10),
        observations=(Observation(
            observation_id="range-1", source="range_sensor", feature="front_range_m",
            value=1.2, unit="m", observed_at=now,
            valid_until=now + timedelta(seconds=3),
        ),),
        choices=(
            ActionChoice(name="continue", hypothesis="The robot should continue.", parameters={"duration_ms": 100}),
            ActionChoice(name="inspect", hypothesis="The robot should inspect.", parameters={"samples": 2}),
        ),
    )


def provider(encoder):
    return OpenJevProvider(encoder, model="mapping-test-double", model_version="test-v1", synthetic=True)


def test_maps_hypotheses_preserves_nli_scores_and_typed_observations():
    encoder = FakeEncoder([[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]])
    request = make_request()
    result, = provider(encoder).infer((request,))
    premise, hypotheses = encoder.calls[0]
    payload = json.loads(premise)
    assert payload["question"] == request.question
    assert payload["observations"] == [request.observations[0].model_dump(mode="json")]
    assert payload["valid_until"] == request.valid_until.isoformat()
    assert hypotheses == [choice.hypothesis for choice in request.choices]
    assert result.scores == {"continue": 0.9, "inspect": 0.85}
    assert sum(result.scores.values()) > 1  # Not a categorical action distribution.
    assert result.semantics == "independent_entailment"
    assert result.nli_scores["inspect"].contradiction == 0.1
    assert result.proposed_action == "continue"
    assert result.parameters == {"duration_ms": 100}
    assert result.identity.synthetic is True
    assert ProviderResult.model_validate_json(result.model_dump_json()) == result


def test_batch_preserves_question_ids_and_order():
    encoder = FakeEncoder([[0.01, 0.98, 0.01], [0.8, 0.1, 0.1]])
    requests = (make_request("first"), make_request("second"))
    results = provider(encoder).infer(requests)
    assert [result.request_id for result in results] == ["first", "second"]
    assert len(encoder.calls) == 2


def test_missing_observations_are_unknown_without_calling_encoder():
    encoder = FakeEncoder(None)
    request = make_request().model_copy(update={"observations": ()})
    result, = provider(encoder).infer((request,))
    assert result.evidence == "unknown"
    assert result.proposed_action is None
    assert result.scores == {}
    assert encoder.calls == []


@pytest.mark.parametrize("rows", [
    [[0.1, 0.8, 0.1]],
    [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]],
    [[0.1, 0.8], [0.1, 0.8, 0.1]],
    [[0.1, float("nan"), 0.1], [0.1, 0.8, 0.1]],
    [[0.1, float("inf"), 0.1], [0.1, 0.8, 0.1]],
    [[-0.1, 1.0, 0.1], [0.1, 0.8, 0.1]],
    [[0.1, 0.1, 0.1], [0.1, 0.8, 0.1]],
    [[True, 0.0, 0.0], [0.1, 0.8, 0.1]],
    [["0.1", 0.8, 0.1], [0.1, 0.8, 0.1]],
    None,
])
def test_malformed_nli_outputs_rejected(rows):
    with pytest.raises(ValueError):
        provider(FakeEncoder(rows)).infer((make_request(),))


def test_array_mapping_without_numpy_dependency():
    class Array:
        def tolist(self):
            return [[0.1, 0.8, 0.1], [0.01, 0.98, 0.01]]

    result, = provider(FakeEncoder(Array())).infer((make_request(),))
    assert result.proposed_action == "inspect"
    assert result.parameters == {"samples": 2}


def test_import_does_not_import_model_dependencies():
    code = (
        "import sys; import pollard_jev.providers.openjev; "
        "assert not any(name in sys.modules for name in "
        "('torch', 'transformers', 'numpy', 'huggingface_hub'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_explicit_factory_uses_only_pinned_local_cache(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    (tmp_path / CHECKPOINT).mkdir()
    (tmp_path / CHECKPOINT / "config.json").write_text(json.dumps({
        "id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"},
    }), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / CHECKPOINT / name).write_bytes(b"test-double")
    (tmp_path / "modeling_openjev.py").write_text(
        "class OpenJevCrossEncoder:\n"
        "    def __init__(self, path, **kwargs):\n"
        "        self.path = path\n"
        "        self.kwargs = kwargs\n"
        "        self.device = kwargs['device'] or 'cpu'\n",
        encoding="utf-8",
    )
    adapter = OpenJevProvider.from_local_cache(cache_dir=tmp_path, device="cpu", max_length=512)
    assert calls == [{
        "repo_id": REPOSITORY, "revision": REVISION, "cache_dir": tmp_path,
        "local_files_only": True,
        "allow_patterns": ["modeling_openjev.py", f"{CHECKPOINT}/*"],
    }]
    assert adapter._encoder.path == str(tmp_path)
    assert adapter._encoder.kwargs == {"subfolder": CHECKPOINT, "device": "cpu", "max_len": 512}
    assert adapter.identity.synthetic is False
    assert REVISION in adapter.identity.model_version
    assert "max_len=512" in adapter.identity.model_version
    automatic = OpenJevProvider.from_local_cache()
    assert "device=cpu" in automatic.identity.model_version


def test_incomplete_cached_snapshot_fails_without_importing_helper(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda **kw: str(tmp_path)))
    with pytest.raises(FileNotFoundError, match="already be cached"):
        OpenJevProvider.from_local_cache()


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
def test_partial_snapshot_rejected_before_helper_executes(tmp_path, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda **kw: str(tmp_path)))
    (tmp_path / CHECKPOINT).mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if name != missing:
            (tmp_path / CHECKPOINT / name).write_bytes(b"test-double")
    (tmp_path / "modeling_openjev.py").write_text("raise AssertionError('must not execute')", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="already be cached"):
        OpenJevProvider.from_local_cache()


def test_wrong_cached_label_order_rejected_before_helper_executes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda **kw: str(tmp_path)))
    (tmp_path / CHECKPOINT).mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / CHECKPOINT / name).write_bytes(b"test-double")
    (tmp_path / CHECKPOINT / "config.json").write_text(json.dumps({
        "id2label": {"0": "entailment", "1": "contradiction", "2": "neutral"},
    }), encoding="utf-8")
    (tmp_path / "modeling_openjev.py").write_text("raise AssertionError('must not execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="NLI label order"):
        OpenJevProvider.from_local_cache()
