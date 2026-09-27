"""Versioned, factual model inputs; rendering never selects an action."""

from __future__ import annotations

import json

from ..contracts import DecisionRequest

REPRESENTATIONS = ("json-v1", "text-v1", "robot-rules-v1")
# Experimental, fixed task context. These are the original robot demonstration
# rules, supplied uniformly, never a computed per-request label or recommendation.
# They do not automatically track a caller's custom PolicyConfig.
ROBOT_RULES = (
    "Robot demonstration operating rules (apply in this priority order):\n"
    "1. Battery charge below 10 percent is too low; the robot should request assistance.\n"
    "2. With battery charge at least 10 percent, a stuck indicator of 1 means the robot is stuck "
    "and should perform a recovery maneuver.\n"
    "3. With battery charge at least 10 percent and stuck indicator 0 (not stuck), "
    "a forward range or camera clearance below 0.5 metres means the robot should collect "
    "additional measurements before moving.\n"
    "4. With battery charge at least 10 percent, stuck indicator 0, and both forward range "
    "and camera clearance at least 0.5 metres, the robot has sufficient battery and room "
    "to continue moving.\n"
    "Unknown, conflicting or expired observations require withholding movement. "
    "Every observation below is reported separately; these rules do not establish that it is valid."
)
_FEATURE_NAMES = {
    "front_range_m": "Forward range",
    "camera_clearance_m": "Camera clearance",
    "battery_pct": "Battery charge",
    "stuck": "Robot stuck indicator",
}


def render_premise(request: DecisionRequest, representation: str = "json-v1") -> str:
    """Preserve readings, order, duplicates, units and timestamps without policy.

    ``text-v1`` expands feature names, but does not calculate thresholds,
    freshness, feasibility or the preferred action. All readings remain separate.
    Values and metadata use JSON quoting to preserve arbitrary strings exactly.
    ``robot-rules-v1`` additionally supplies fixed demonstration task rules. It
    is an information intervention, not merely another serialization format.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown input representation: {representation!r}")
    if representation == "json-v1":
        return json.dumps({
            "question": request.question,
            "created_at": request.created_at.isoformat(),
            "valid_until": request.valid_until.isoformat(),
            "observations": [obs.model_dump(mode="json") for obs in request.observations],
        }, sort_keys=True, allow_nan=False)
    quote = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
    lines = [
        f"Question: {quote(request.question)}",
        f"Request created at {quote(request.created_at.isoformat())}; valid until {quote(request.valid_until.isoformat())}.",
        "The following observations are reported independently:",
    ]
    if representation == "robot-rules-v1":
        lines.insert(0, ROBOT_RULES)
    for index, obs in enumerate(request.observations, 1):
        label = _FEATURE_NAMES.get(obs.feature, obs.feature)
        value = "null (unknown)" if obs.status == "unknown" else quote(obs.value)
        lines.append(
            f"{index}. {quote(label)} is {value}, in unit {quote(obs.unit)}. "
            f"Feature {quote(obs.feature)}; status {quote(obs.status)}; "
            f"observation {quote(obs.observation_id)}; source {quote(obs.source)}; "
            f"observed at {quote(obs.observed_at.isoformat())}; valid until {quote(obs.valid_until.isoformat())}."
        )
    return "\n".join(lines)
