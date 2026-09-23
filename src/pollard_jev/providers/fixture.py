"""A reproducible synthetic provider; no model, network, or actuator access."""

from __future__ import annotations

import math
import time

from ..contracts import DecisionRequest, PolicyConfig, ProviderIdentity, ProviderResult
from ..simulator import inspect_evidence


class FixtureProvider:
    """Synthetic independent support scores for offline tests and demonstrations.

    Scores intentionally do not sum to one. They are hand-authored evidence
    support values, not model predictions or probabilities of physical success.
    Delay and failure injection let the loop demonstrate its deadline handling.
    """

    identity = ProviderIdentity(
        provider="fixture",
        provider_version="1",
        model="deterministic-robot-rules",
        model_version="1",
        synthetic=True,
    )

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        failure: bool = False,
        proposed_action: str | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        if isinstance(delay_s, bool) or not isinstance(delay_s, (int, float)) or (
            not math.isfinite(delay_s) or delay_s < 0
        ):
            raise ValueError("delay_s must be a finite non-negative duration")
        if type(failure) is not bool:
            raise ValueError("failure must be a boolean")
        self.delay_s = float(delay_s)
        self.failure = failure
        self.proposed_action = proposed_action
        self.config = config or PolicyConfig()
        self.identity = self.identity.model_copy(update={"settings": {
            "delay_s": self.delay_s, "failure": failure,
            "proposed_action_override": proposed_action or "",
            "fixture_policy_json": self.config.model_dump_json(),
        }})

    def infer(self, requests: tuple[DecisionRequest, ...]) -> tuple[ProviderResult, ...]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.failure:
            raise RuntimeError("Synthetic fixture provider failure")
        return tuple(self._result(request) for request in requests)

    def _result(self, request: DecisionRequest) -> ProviderResult:
        values, reason = inspect_evidence(request, self.config)
        required = {"front_range_m", "camera_clearance_m", "battery_pct", "stuck"}
        if reason == "conflicting_evidence":
            evidence = "conflicting"
            action = "inspect"
        elif reason is not None or not required <= values.keys():
            evidence = "unknown"
            action = "inspect"
        else:
            evidence = "sufficient"
            if values["battery_pct"] < self.config.minimum_battery_pct:
                action = "request_help"
            elif values["stuck"] == 1.0:
                action = "recover"
            elif min(values["front_range_m"], values["camera_clearance_m"]) < self.config.minimum_clearance_m:
                action = "inspect"
            else:
                action = "continue"
        if self.proposed_action is not None:
            action = self.proposed_action
        choices = {choice.name: choice for choice in request.choices}
        return ProviderResult(
            request_id=request.request_id,
            identity=self.identity,
            semantics="synthetic_support",
            scores={name: 0.94 if name == action else 0.12 for name in choices},
            proposed_action=action,
            parameters=dict(choices[action].parameters) if action in choices else {},
            evidence=evidence,
        )
