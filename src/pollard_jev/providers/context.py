"""Measure complete NLI inputs and reject silent upstream truncation.

This wrapper imports no model libraries. Measurements count unpadded text-pair
tokens, not GPU work or hosted billing tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ContextCheckedEncoder:
    """Guard the inspected OpenJev helper's truncating tokenizer path.

    ``forward_calls`` counts helper invocation attempts, including failed calls;
    it does not count the helper's internal GPU forward passes. Valid complete
    measurements are retained even when a context overflow prevents inference.
    Use one instance serially, as with the underlying encoder.
    """

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.measurements: list[dict[str, Any]] = []
        self.forward_calls = 0

    def predict_hypotheses(self, premise: str, hypotheses: list[str]) -> Any:
        tokenizer = getattr(self.encoder, "tok", None)
        template = getattr(self.encoder, "template", None)
        maximum = getattr(self.encoder, "max_len", None)
        if not callable(tokenizer) or not isinstance(template, str):
            raise ValueError("Encoder must expose a tokenizer and text template")
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Encoder max_len must be a positive integer")
        if not isinstance(premise, str) or not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("Expected a text premise and nonempty hypothesis list")
        if any(not isinstance(hypothesis, str) for hypothesis in hypotheses):
            raise ValueError("Hypotheses must be text")

        lengths = []
        for hypothesis in hypotheses:
            try:
                text = template.format(premise=premise.strip(), hypothesis=hypothesis.strip())
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError("Encoder template cannot format premise and hypothesis") from exc
            encoded = tokenizer(text, truncation=False, padding=False)
            tokens = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
                raise ValueError("Tokenizer must expose a nonempty list of integer token IDs")
            lengths.append(len(tokens))

        overflow = any(length > maximum for length in lengths)
        self.measurements.append({
            "tokenized_pair_lengths": lengths,
            "unshared_input_tokens": sum(lengths),
            "maximum_context_tokens": maximum,
            "input_context_truncated": False,
            "context_limit_exceeded": overflow,
            "tokenization_basis": "complete_unpadded_pairs",
        })
        if overflow:
            raise ValueError("Complete NLI pair exceeds encoder context limit")

        self.forward_calls += 1
        return self.encoder.predict_hypotheses(premise, hypotheses)
