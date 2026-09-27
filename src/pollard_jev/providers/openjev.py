"""Optional adapter for the inspected AlexWortega/OpenJev NLI helper.

Imports and normal tests need no model dependencies. ``from_local_cache`` is
explicit and only loads a previously downloaded, commit-pinned snapshot.
"""

from __future__ import annotations

import importlib.util
import json
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from ..contracts import DecisionRequest, NLIScores, ProviderIdentity, ProviderResult
from .representations import REPRESENTATIONS, render_premise

REPOSITORY = "AlexWortega/openjev"
REVISION = "552759daad712f1af6c4c13dabcb1e047886fc9c"
CHECKPOINT = "qwen3.5-4b-nli-v2"
ADAPTER_VERSION = "openjev-nli-adapter-v1"
_CHECKPOINT_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
_NLI_LABELS = {"0": "contradiction", "1": "entailment", "2": "neutral"}


class HypothesisEncoder(Protocol):
    def predict_hypotheses(self, premise: str, hypotheses: list[str]) -> Any: ...


class OpenJevProvider:
    """Map typed observations and hypotheses to independent entailment scores.

    An injected encoder must implement the upstream ``predict_hypotheses``
    signature. Its caller supplies honest model identity, including whether it
    is synthetic. Action feasibility remains the deterministic policy's job.
    Input defaults to the original JSON. Experimental ``robot-rules-v1`` adds
    fixed default-demo task rules; these do not track custom policy settings.
    """

    def __init__(
        self,
        encoder: HypothesisEncoder,
        *,
        model: str = f"{REPOSITORY}/{CHECKPOINT}",
        model_version: str = REVISION,
        synthetic: bool = False,
        representation: str = "json-v1",
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown input representation: {representation!r}")
        self._representation = representation
        self._encoder = encoder
        self.identity = ProviderIdentity(
            provider="alexwortega-openjev",
            provider_version=ADAPTER_VERSION,
            model=model,
            model_version=model_version,
            synthetic=synthetic,
            settings={"input_representation": representation},
        )

    @property
    def representation(self) -> str:
        """Keep the rendered input format consistent with the audit identity."""
        return self._representation

    @classmethod
    def from_local_cache(
        cls,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        max_length: int = 4096,
        representation: str = "json-v1",
    ) -> OpenJevProvider:
        """Load the inspected helper and weights from the HF cache, offline.

        Downloading is a separate explicit step documented in docs/openjev.md.
        The upstream helper is executable Python, loaded only by this method.
        """
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown input representation: {representation!r}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install pollard-jev[openjev] to load a real encoder") from exc

        snapshot = Path(snapshot_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=["modeling_openjev.py", f"{CHECKPOINT}/*"],
        ))
        helper_path = snapshot / "modeling_openjev.py"
        checkpoint_path = snapshot / CHECKPOINT
        if not helper_path.is_file() or any(
            not (checkpoint_path / name).is_file() for name in _CHECKPOINT_FILES
        ):
            raise FileNotFoundError("The inspected helper and checkpoint must already be cached")
        # local_files_only can return a partial snapshot. Check the inspected
        # checkpoint's files and label order before executing its Python helper.
        config = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("id2label") != _NLI_LABELS:
            raise ValueError("Cached OpenJev config does not have the inspected NLI label order")
        spec = importlib.util.spec_from_file_location("_pollard_jev_openjev_helper", helper_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the cached OpenJev helper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.OpenJevCrossEncoder(
            str(snapshot), subfolder=CHECKPOINT, device=device, max_len=max_length,
        )
        return cls(
            encoder,
            model_version=(
                f"{REVISION};max_len={max_length};dtype=bfloat16;"
                f"device={getattr(encoder, 'device', device or 'auto')}"
            ),
            representation=representation,
        )

    def infer(self, requests: tuple[DecisionRequest, ...]) -> tuple[ProviderResult, ...]:
        """One logical budgeted invocation; score each question's hypotheses.

        No adapter dispatches actions. A timed-out worker may finish computing,
        but only the supervisor can decide whether a result is still usable.
        """
        return tuple(self._infer_one(request) for request in requests)

    def _infer_one(self, request: DecisionRequest) -> ProviderResult:
        if not request.observations:
            return ProviderResult(
                request_id=request.request_id, identity=self.identity,
                semantics="independent_entailment", scores={},
                proposed_action=None, evidence="unknown",
            )
        # Preserve timestamps, units, unknown markers and validity limits.
        # This is a text representation of engineered features, not a sensor encoder.
        premise = render_premise(request, self.representation)
        hypotheses = [choice.hypothesis for choice in request.choices]
        raw = self._encoder.predict_hypotheses(premise, hypotheses)
        rows = _validated_rows(raw, len(request.choices))
        details = {choice.name: row for choice, row in zip(request.choices, rows, strict=True)}
        scores = {name: row.entailment for name, row in details.items()}
        # A stable tie retains the first supplied choice; policy margin can defer it.
        chosen = max(request.choices, key=lambda choice: scores[choice.name])
        return ProviderResult(
            request_id=request.request_id, identity=self.identity,
            semantics="independent_entailment", scores=scores,
            proposed_action=chosen.name, parameters=dict(chosen.parameters),
            evidence="sufficient", nli_scores=details,
        )


def _validated_rows(raw: Any, count: int) -> list[NLIScores]:
    # ndarray.tolist() avoids importing numpy in the offline installation.
    rows = raw.tolist() if callable(getattr(raw, "tolist", None)) else raw
    if not isinstance(rows, (list, tuple)) or len(rows) != count:
        raise ValueError("OpenJev must return exactly one NLI row per hypothesis")
    output = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Each OpenJev row must contain contradiction, entailment, neutral")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in row):
            raise ValueError("OpenJev scores must be numeric, not booleans or strings")
        # Pydantic rejects non-finite, out-of-range and non-normalized label values.
        output.append(NLIScores(
            contradiction=float(row[0]), entailment=float(row[1]), neutral=float(row[2]),
        ))
    return output
