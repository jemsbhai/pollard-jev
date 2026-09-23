"""A bounded, in-memory robot skill set and a demonstration state machine.

No code in this module connects to a controller or operates physical hardware.
The thresholds and resulting observations describe this toy simulator only.
"""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from threading import Lock

from .contracts import (
    ActionChoice, ActionOutcome, DecisionRequest, PolicyConfig, fresh,
)


SKILL_PARAMETERS: dict[str, dict[str, tuple[int, int]]] = {
    "continue": {"distance_cm": (1, 50)},
    "inspect": {"samples": (1, 5)},
    "recover": {"distance_cm": (1, 20)},
    "request_help": {"retries": (1, 1)},
}


def default_choices() -> tuple[ActionChoice, ...]:
    """The available simulated skills, with explicit bounded default arguments."""
    return (
        ActionChoice(
            name="continue",
            hypothesis="The robot has sufficient battery, is not stuck, and its path is clear.",
            parameters={"distance_cm": 20},
        ),
        ActionChoice(
            name="inspect",
            hypothesis="The robot should collect additional measurements before moving.",
            parameters={"samples": 2},
        ),
        ActionChoice(
            name="recover",
            hypothesis="The robot is stuck and has sufficient battery for a recovery maneuver.",
            parameters={"distance_cm": 10},
        ),
        ActionChoice(
            name="request_help",
            hypothesis="The robot should request assistance because its battery is too low.",
            parameters={"retries": 1},
        ),
    )


def inspect_evidence(
    request: DecisionRequest, config: PolicyConfig,
) -> tuple[dict[str, float], str | None]:
    """Check structured evidence without scoring or testing its timestamp.

    Duplicate required features are ambiguous even when their values agree.
    Freshness is evaluated separately using the dispatch-time clock.
    """
    values: dict[str, float] = {}
    seen: set[str] = set()
    missing = False
    for observation in request.observations:
        feature = observation.feature
        if feature not in config.required_units:
            continue
        if feature in seen:
            return values, "invalid_evidence"
        seen.add(feature)
        if observation.unit != config.required_units[feature]:
            return values, "invalid_evidence"
        if observation.status != "known" or observation.value is None:
            missing = True
            continue
        value = observation.value
        if not isfinite(value):
            return values, "invalid_evidence"
        if feature in ("front_range_m", "camera_clearance_m") and value < 0:
            return values, "invalid_evidence"
        if feature == "battery_pct" and not 0 <= value <= 100:
            return values, "invalid_evidence"
        if feature == "stuck" and value not in (0.0, 1.0):
            return values, "invalid_evidence"
        values[feature] = value
    if missing or set(config.required_units) - values.keys():
        return values, "missing_evidence"
    if {"front_range_m", "camera_clearance_m"} <= values.keys() and (
        abs(values["front_range_m"] - values["camera_clearance_m"])
        > config.conflict_tolerance_m
    ):
        return values, "conflicting_evidence"
    return values, None


def baseline(
    request: DecisionRequest, now: datetime, config: PolicyConfig | None = None,
) -> str:
    """A deterministic state machine using the same inputs as the provider.

    Returns a skill name, ``need_evidence``, or ``defer``; it never dispatches.
    Its thresholds are demonstration settings, not a calibrated robot policy.
    """
    config = config or PolicyConfig()
    if not fresh(request, now):
        return "defer"
    values, reason = inspect_evidence(request, config)
    if reason is not None:
        return "need_evidence"
    # The robotics baseline needs these four features even if a caller replaces
    # required_units with a narrower configuration.
    required = {"front_range_m", "camera_clearance_m", "battery_pct", "stuck"}
    if not required <= values.keys():
        return "need_evidence"
    if values["battery_pct"] < config.minimum_battery_pct:
        action = "request_help"
    elif values["stuck"] == 1.0:
        action = "recover"
    elif min(values["front_range_m"], values["camera_clearance_m"]) < config.minimum_clearance_m:
        action = "inspect"
    else:
        action = "continue"
    if action not in config.allowlist or action not in {choice.name for choice in request.choices}:
        return "defer"
    return action


class RobotSimulator:
    """Synchronous simulated dispatch with validation before every side effect."""

    def __init__(self) -> None:
        self._position_cm = 0.0
        self._measurements = 0
        self._recoveries = 0
        self._help_requests = 0
        self._dispatch_history: list[ActionOutcome] = []
        self._lock = Lock()

    @property
    def position_cm(self) -> float:
        return self._position_cm

    @property
    def dispatch_history(self) -> tuple[ActionOutcome, ...]:
        """A snapshot of completed simulated dispatches."""
        with self._lock:
            return tuple(self._dispatch_history)

    def execute(
        self, request_id: str, action: str, parameters: dict[str, int], now: datetime,
    ) -> ActionOutcome:
        if action not in SKILL_PARAMETERS:
            raise ValueError(f"Unsupported simulated action: {action!r}")
        bounds = SKILL_PARAMETERS[action]
        if not isinstance(parameters, dict) or parameters.keys() != bounds.keys():
            raise ValueError(f"Parameters for {action!r} must be exactly {tuple(bounds)}")
        parameters = parameters.copy()
        for name, (minimum, maximum) in bounds.items():
            value = parameters[name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")

        with self._lock:
            position = self._position_cm
            measurements = self._measurements
            recoveries = self._recoveries
            help_requests = self._help_requests
            if action == "continue":
                position += parameters["distance_cm"]
            elif action == "inspect":
                measurements += parameters["samples"]
            elif action == "recover":
                position -= parameters["distance_cm"]
                recoveries += 1
            elif action == "request_help":
                help_requests += parameters["retries"]

            # Construct/validate the complete outcome before committing state,
            # including request identifiers and timezone-aware timestamps.
            outcome = ActionOutcome(
                request_id=request_id,
                action=action,
                status="completed",
                started_at=now,
                completed_at=now,
                parameters=parameters,
                observed={
                    "position": position,
                    "position_unit": "cm",
                    "measurements": float(measurements),
                    "recoveries": float(recoveries),
                    "help_requests": float(help_requests),
                },
            )
            self._position_cm = position
            self._measurements = measurements
            self._recoveries = recoveries
            self._help_requests = help_requests
            self._dispatch_history.append(outcome)
            return outcome
