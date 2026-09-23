"""Read-only history and policy simulation intentionally have no dispatcher."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from .contracts import DecisionRecord, PolicyConfig, PolicyOutcome
from .policy import DecisionPolicy

if TYPE_CHECKING:
    from .loop import DecisionLoop


class RecordStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def append(self, record: DecisionRecord) -> None:
        text = record.model_dump_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")
                stream.flush()


def inspect_history(path: str | Path) -> tuple[DecisionRecord, ...]:
    """Historical inspection: parse recorded events, with no execution object."""
    with Path(path).open(encoding="utf-8") as stream:
        return tuple(DecisionRecord.model_validate_json(line) for line in stream if line.strip())


def simulate_policy(
    record: DecisionRecord, config: PolicyConfig | None = None, *, at: datetime | None = None,
) -> PolicyOutcome:
    """Evaluate evidence/support at a chosen time, never rerun inference or skills.

    Defaults to the recorded policy evaluation time. This is a policy experiment,
    not a counterfactual trajectory, budget replay, or claim of action success.
    """
    policy = DecisionPolicy(config or record.policy_config)
    now = at or record.policy.evaluated_at
    if record.provider_status != "ok":
        return policy.outcome(record.request, now, "defer", record.provider_status)
    return policy.evaluate(record.request, record.provider_result, now)


def live_reevaluate(record: DecisionRecord, loop: DecisionLoop) -> DecisionRecord:
    """Explicit new, charged provider call on historical inputs; dispatch disabled.

    Timestamp validity is preserved. Old evidence will normally defer. Results
    live in a new record and never replace the historical model observation.
    """
    return loop.decide_batch(
        (record.request,), dispatch=False, mode="live_reevaluation",
        reevaluates_record_id=record.record_id,
    )[0]
