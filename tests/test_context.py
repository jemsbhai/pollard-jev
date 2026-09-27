"""Context overflow tests with no model libraries or weights."""

import pytest

from pollard_jev.providers.context import ContextCheckedEncoder


class FakeEncoder:
    template = "Premise: {premise}\nHypothesis: {hypothesis}"
    max_len = 8

    def __init__(self):
        self.calls = []
        self.tokenizer_calls = []

    def tok(self, text, **kwargs):
        self.tokenizer_calls.append((text, kwargs))
        return {"input_ids": list(range(len(text.split())))}

    def predict_hypotheses(self, premise, hypotheses):
        self.calls.append((premise, hypotheses))
        return [[0.1, 0.8, 0.1] for _ in hypotheses]


def test_complete_pairs_measured_without_padding_or_truncation():
    encoder = FakeEncoder()
    guarded = ContextCheckedEncoder(encoder)
    premise, hypotheses = "  clear  ", ["  proceed  ", "inspect area"]
    assert guarded.predict_hypotheses(premise, hypotheses) == [[0.1, 0.8, 0.1]] * 2
    assert encoder.calls == [(premise, hypotheses)]
    assert encoder.tokenizer_calls == [
        ("Premise: clear\nHypothesis: proceed", {"truncation": False, "padding": False}),
        ("Premise: clear\nHypothesis: inspect area", {"truncation": False, "padding": False}),
    ]
    assert guarded.measurements == [{
        "tokenized_pair_lengths": [4, 5], "unshared_input_tokens": 9,
        "maximum_context_tokens": 8, "input_context_truncated": False,
        "context_limit_exceeded": False, "tokenization_basis": "complete_unpadded_pairs",
    }]
    guarded.predict_hypotheses("clear", ["proceed"])
    assert guarded.forward_calls == 2
    assert len(guarded.measurements) == 2


def test_overflow_of_any_pair_prevents_helper_call_and_keeps_measurement():
    encoder = FakeEncoder()
    guarded = ContextCheckedEncoder(encoder)
    with pytest.raises(ValueError, match="exceeds encoder context limit"):
        guarded.predict_hypotheses("clear", ["proceed", "one two three four five six"])
    assert encoder.calls == []
    assert guarded.forward_calls == 0
    assert guarded.measurements[0]["tokenized_pair_lengths"] == [4, 9]
    assert guarded.measurements[0]["context_limit_exceeded"] is True


def test_exact_limit_is_allowed():
    encoder = FakeEncoder()
    guarded = ContextCheckedEncoder(encoder)
    guarded.predict_hypotheses("clear", ["one two three four five"])
    assert guarded.forward_calls == 1
    assert guarded.measurements[0]["tokenized_pair_lengths"] == [8]


@pytest.mark.parametrize("field,value", [
    ("tok", None), ("template", None), ("template", "{missing}"),
    ("max_len", True), ("max_len", 0), ("max_len", -1), ("max_len", 1.5),
])
def test_malformed_metadata_fails_before_helper(field, value):
    encoder = FakeEncoder()
    setattr(encoder, field, value)
    guarded = ContextCheckedEncoder(encoder)
    with pytest.raises(ValueError):
        guarded.predict_hypotheses("clear", ["proceed"])
    assert encoder.calls == []
    assert guarded.measurements == []
    assert guarded.forward_calls == 0


@pytest.mark.parametrize("encoded", [
    None, [], {}, {"input_ids": []}, {"input_ids": [True]},
    {"input_ids": [1.0]}, {"input_ids": ["1"]}, {"input_ids": [[1]]},
    {"input_ids": (1, 2)},
])
def test_malformed_token_ids_fail_before_helper(encoded):
    encoder = FakeEncoder()
    encoder.tok = lambda *args, **kwargs: encoded
    guarded = ContextCheckedEncoder(encoder)
    with pytest.raises(ValueError, match="integer token IDs"):
        guarded.predict_hypotheses("clear", ["proceed"])
    assert encoder.calls == []
    assert guarded.measurements == []
    assert guarded.forward_calls == 0


def test_helper_failure_remains_an_attempt_with_measured_input():
    encoder = FakeEncoder()

    def fail(*args):
        raise RuntimeError("model failure")

    encoder.predict_hypotheses = fail
    guarded = ContextCheckedEncoder(encoder)
    with pytest.raises(RuntimeError, match="model failure"):
        guarded.predict_hypotheses("clear", ["proceed"])
    assert guarded.forward_calls == 1
    assert len(guarded.measurements) == 1
