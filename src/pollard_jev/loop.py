"""Pollard-backed supervisory loop; inference workers can only return data."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable
from uuid import uuid4

from pollard import ActionSpec, Budget, Registry, Runtime
from pollard.errors import BudgetExceeded, PolicyViolation

from .contracts import ActionOutcome, DecisionRecord, DecisionRequest, PolicyConfig, ProviderIdentity, ProviderResult, fresh
from .policy import DecisionPolicy
from .providers.base import DecisionProvider
from .records import RecordStore
from .simulator import RobotSimulator, SKILL_PARAMETERS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value) -> str:
    # Pollard identities prohibit float JSON values. A canonical JSON *string*
    # preserves numeric precision and full typed inputs without inventing an API.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RequestMeter:
    """One logical provider batch attempt, including failure/timeout, costs one."""
    name = "requests"

    def precheck_estimate(self, node_kind, payload):
        return int(node_kind == "model_call")

    def charge(self, node_kind, payload, result, meta):
        return int(node_kind == "model_call")


class DecisionLoop:
    def __init__(
        self, provider: DecisionProvider, *, max_requests: int = 10,
        timeout_s: float = 1.0, policy: PolicyConfig | None = None,
        simulator: RobotSimulator | None = None, records_path: str | Path | None = None,
        pollard_path: str | Path | None = None, clock: Callable[[], datetime] = utc_now,
    ):
        if type(max_requests) is not int or max_requests < 0:
            raise ValueError("max_requests must be a nonnegative integer")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.provider = provider
        self.identity = ProviderIdentity.model_validate(provider.identity.model_dump())
        self.policy = DecisionPolicy(policy)
        self.simulator = simulator or RobotSimulator()
        self.max_requests, self.timeout_s, self.clock = max_requests, timeout_s, clock
        self.records = RecordStore(records_path) if records_path is not None else None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._dispatch_context = None
        self._closed = False
        self._registry = Registry([
            ActionSpec(
                name, "sim-skill-v1", "Bounded simulated robot skill",
                {"type": "object", "properties": {
                    key: {"type": "integer", "minimum": lo, "maximum": hi}
                    for key, (lo, hi) in bounds.items()
                }, "required": list(bounds), "additionalProperties": False},
                True, handler=self._handler(name),
            ) for name, bounds in SKILL_PARAMETERS.items() if name in self.policy.config.allowlist
        ])
        if pollard_path is not None:
            Path(pollard_path).parent.mkdir(parents=True, exist_ok=True)
        self.runtime = Runtime(pollard_path, meters=[RequestMeter()], registry=self._registry,
                               refuse_duplicate_recordings=True)
        self.run = self.runtime.run("pollard-jev-" + uuid4().hex, budget=Budget(extra={"requests": max_requests}))
        self.run.note({"configuration_json": canonical_json({
            "policy": self.policy.config.model_dump(mode="json"),
            "provider": self.identity.model_dump(mode="json"),
            "max_requests": max_requests, "timeout_s": timeout_s,
            "budget_unit": "logical_provider_batch_attempts", "simulator": "sim-skill-v1",
        })})

    @property
    def spent_requests(self) -> int:
        return int(self.run.report()["spent"].get("requests", 0))

    def _handler(self, action):
        def handle(parameters):
            # This guard is inside the registered callable, after Pollard's own
            # schema/budget gate, immediately before the simulator is invoked.
            context = self._dispatch_context
            if context is None:
                return {"status": "refused", "reason": "no_dispatch_context"}
            request, result, cancel, deadline = context
            now = self.clock()
            if cancel.is_set():
                return {"status": "refused", "reason": "cancelled"}
            if time.monotonic() >= deadline:
                return {"status": "refused", "reason": "timeout"}
            check = self.policy.evaluate(request, result, now)
            if check.disposition != "accept" or check.action != action or check.parameters != parameters:
                return {"status": "refused", "reason": check.reason}
            # Validation itself can consume time. Check the cheap mutable gates
            # again at the actual dispatch boundary, including wall-clock expiry.
            now = self.clock()
            if cancel.is_set():
                return {"status": "refused", "reason": "cancelled"}
            if time.monotonic() >= deadline:
                return {"status": "refused", "reason": "timeout"}
            if not fresh(request, now):
                return {"status": "refused", "reason": "stale_or_future_evidence"}
            try:
                outcome = self.simulator.execute(request.request_id, action, parameters, now)
                return {"status": "completed", "outcome": outcome.model_dump(mode="json")}
            except Exception:
                # Never persist exception strings, which can contain credentials.
                return {"status": "failed", "reason": "simulator_failure"}
        return handle

    def _infer(self, requests, cancel, deadline):
        replies = queue.Queue(maxsize=1)
        provider = self.provider
        # Pass a detached validated copy so a provider cannot mutate policy inputs.
        inputs = tuple(DecisionRequest.model_validate(r.model_dump()) for r in requests)

        def work():
            try:
                output = provider.infer(inputs)
            except Exception:
                replies.put(("failure", None))
                return
            try:
                if not isinstance(output, (tuple, list)) or len(output) != len(requests):
                    raise ValueError("wrong batch shape")
                results = tuple(ProviderResult.model_validate(
                    item.model_dump() if isinstance(item, ProviderResult) else item
                ) for item in output)
                if [r.request_id for r in results] != [r.request_id for r in requests]:
                    raise ValueError("wrong request IDs/order")
                if any(r.identity != self.identity for r in results):
                    raise ValueError("wrong provider identity")
                replies.put(("ok", [r.model_dump(mode="json") for r in results]))
            except Exception:
                # Malformed raw output is deliberately not stored.
                replies.put(("malformed", None))

        self._worker = threading.Thread(target=work, daemon=True, name="pollard-jev-inference")
        self._worker.start()
        while True:
            if cancel.is_set():
                return {"status": "cancelled", "results": None}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "timeout", "results": None}
            try:
                status, result = replies.get(timeout=min(remaining, 0.01))
            except queue.Empty:
                continue
            # A queued reply may itself have arrived too late.
            if cancel.is_set():
                return {"status": "cancelled", "results": None}
            if time.monotonic() >= deadline:
                return {"status": "timeout", "results": None}
            return {"status": status, "results": result}

    def decide(self, request: DecisionRequest, *, cancel: threading.Event | None = None) -> DecisionRecord:
        return self.decide_batch((request,), cancel=cancel)[0]

    def decide_batch(
        self, requests: tuple[DecisionRequest, ...], *, cancel: threading.Event | None = None,
        dispatch: bool = True, mode: str = "decision", reevaluates_record_id: str | None = None,
    ) -> tuple[DecisionRecord, ...]:
        if mode not in ("decision", "live_reevaluation"):
            raise ValueError("invalid record mode")
        if mode == "live_reevaluation" and dispatch:
            raise ValueError("live reevaluation must disable dispatch")
        if mode == "live_reevaluation" and not reevaluates_record_id:
            raise ValueError("live reevaluation must reference a historical record")
        requests = tuple(DecisionRequest.model_validate(r.model_dump()) for r in requests)
        if not requests or len({r.request_id for r in requests}) != len(requests):
            raise ValueError("a batch requires distinct request IDs")
        cancel = cancel if cancel is not None else threading.Event()
        with self._lock:
            if self._closed:
                raise RuntimeError("decision loop is closed")
            batch_id = uuid4().hex
            started = time.monotonic()
            deadline = started + self.timeout_s
            node_id = None
            if cancel.is_set():
                envelope = {"status": "cancelled", "results": None}
            elif self._worker is not None and self._worker.is_alive():
                # A timed-out native call may still be working. Bound this loop
                # to one worker rather than accumulating abandoned inference.
                envelope = {"status": "busy", "results": None}
            else:
                try:
                    node = self.run.model_call(
                        {"batch_id": batch_id, "requests_json": canonical_json([r.model_dump(mode="json") for r in requests]),
                         "provider_json": canonical_json(self.identity.model_dump(mode="json"))},
                        fn=lambda _: self._infer(requests, cancel, deadline),
                    )
                    node_id, envelope = node.id, node.result
                except BudgetExceeded as exc:
                    node_id = exc.refusal_id
                    envelope = {"status": "budget_exhausted", "results": None}
            elapsed = time.monotonic() - started
            status = envelope["status"]
            results = envelope["results"] or [None] * len(requests)
            records = []
            did_dispatch = False
            for request, raw in zip(requests, results, strict=True):
                result = ProviderResult.model_validate(raw) if raw is not None else None
                now = self.clock()
                outcome = self.policy.evaluate(request, result, now) if status == "ok" else self.policy.outcome(request, now, "defer", status)
                if cancel.is_set():
                    outcome = self.policy.outcome(request, now, "defer", "cancelled")
                elif status == "ok" and time.monotonic() >= deadline:
                    outcome = self.policy.outcome(request, now, "defer", "timeout")
                action = None
                action_node_id = None
                if outcome.disposition == "accept" and dispatch:
                    if did_dispatch:
                        outcome = self.policy.outcome(request, now, "defer", "batch_requires_new_observation")
                    else:
                        self.run.note({"dispatch_intent_json": outcome.model_dump_json()})
                        self._dispatch_context = (request, result, cancel, deadline)
                        # A failed response cannot prove that the first action
                        # had no effect. Permit at most one attempt per batch.
                        did_dispatch = True
                        try:
                            action_node = self.run.tool_call(outcome.action, outcome.parameters, version="sim-skill-v1")
                            action_node_id = action_node.id
                            if action_node.result["status"] == "completed":
                                action = ActionOutcome.model_validate(action_node.result["outcome"])
                            else:
                                outcome = self.policy.outcome(request, self.clock(), "defer", action_node.result["reason"])
                                action = ActionOutcome(
                                    request_id=request.request_id, action=result.proposed_action,
                                    status=action_node.result["status"], started_at=now, completed_at=self.clock(),
                                    parameters=result.parameters,
                                )
                        except (PolicyViolation, BudgetExceeded) as exc:
                            action_node_id = exc.refusal_id
                            outcome = self.policy.outcome(request, self.clock(), "defer", "pollard_refusal")
                        finally:
                            self._dispatch_context = None
                record = DecisionRecord(
                    record_id=uuid4().hex, batch_id=batch_id, recorded_at=self.clock(), request=request,
                    provider_identity=self.identity, provider_result=result, provider_status=status,
                    policy_config=self.policy.config, policy=outcome, action=action,
                    pollard_version=version("pollard"), pollard_root_id=self.run.root_id,
                    pollard_model_node_id=node_id, pollard_action_node_id=action_node_id,
                    budget_limit_requests=self.max_requests, budget_spent_requests=self.spent_requests,
                    inference_elapsed_s=elapsed, inference_timeout_s=self.timeout_s,
                    mode=mode, reevaluates_record_id=reevaluates_record_id,
                )
                self.run.note({"decision_record_json": record.model_dump_json()})
                if self.records is not None:
                    self.records.append(record)
                records.append(record)
            return tuple(records)

    def close(self):
        with self._lock:
            if not self._closed:
                close = getattr(self.runtime.store, "close", None)
                if close is not None:
                    close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
