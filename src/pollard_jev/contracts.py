"""Versioned JSON contracts. Scores describe evidence, never physical success."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictInt,
    StrictStr, field_validator, model_validator,
)

Name = Annotated[str, Field(min_length=1, max_length=200)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Observation(Contract):
    observation_id: Name
    source: Name
    feature: Name
    value: Number | None
    unit: Name
    observed_at: AwareDatetime
    valid_until: AwareDatetime
    status: Literal["known", "unknown"] = "known"

    @model_validator(mode="after")
    def valid_observation(self):
        if self.valid_until <= self.observed_at:
            raise ValueError("valid_until must follow observed_at")
        if (self.status == "known") != (self.value is not None):
            raise ValueError("known observations require a value; unknown ones require None")
        return self


class ActionChoice(Contract):
    name: Name
    hypothesis: Annotated[str, Field(min_length=1, max_length=4000)]
    parameters: dict[str, StrictInt] = Field(default_factory=dict)


class DecisionRequest(Contract):
    request_id: Name
    question: Annotated[str, Field(min_length=1, max_length=8000)]
    created_at: AwareDatetime
    valid_until: AwareDatetime
    observations: tuple[Observation, ...]
    choices: Annotated[tuple[ActionChoice, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def valid_request(self):
        if self.valid_until <= self.created_at:
            raise ValueError("request validity must be positive")
        names = [choice.name for choice in self.choices]
        ids = [obs.observation_id for obs in self.observations]
        if len(names) != len(set(names)) or len(ids) != len(set(ids)):
            raise ValueError("choice names and observation IDs must be unique")
        return self


class ProviderIdentity(Contract):
    provider: Name
    provider_version: Name
    model: Name
    model_version: Name
    synthetic: StrictBool
    settings: dict[str, Number | StrictStr | StrictBool] = Field(default_factory=dict)


class NLIScores(Contract):
    contradiction: Score
    entailment: Score
    neutral: Score

    @model_validator(mode="after")
    def distribution(self):
        if abs(self.contradiction + self.entailment + self.neutral - 1) > 1e-5:
            raise ValueError("NLI labels must form a distribution for each hypothesis")
        return self


class ProviderResult(Contract):
    request_id: Name
    identity: ProviderIdentity
    semantics: Literal["synthetic_support", "independent_entailment", "categorical"]
    scores: dict[str, Score]
    proposed_action: Name | None
    parameters: dict[str, StrictInt] = Field(default_factory=dict)
    evidence: Literal["sufficient", "insufficient", "unknown", "conflicting"]
    nli_scores: dict[str, NLIScores] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_scores(self):
        if self.semantics == "synthetic_support" and not self.identity.synthetic:
            raise ValueError("synthetic support must have a synthetic provider identity")
        if self.semantics == "categorical" and self.scores:
            if abs(sum(self.scores.values()) - 1) > 1e-5:
                raise ValueError("categorical scores must sum to one")
        if self.nli_scores:
            if self.semantics != "independent_entailment" or self.scores.keys() != self.nli_scores.keys():
                raise ValueError("NLI detail must match independent entailment scores")
            if any(abs(self.scores[k] - v.entailment) > 1e-6 for k, v in self.nli_scores.items()):
                raise ValueError("entailment detail disagrees with score")
        return self


class PolicyConfig(Contract):
    version: Name = "robot-demo-policy-v1"
    allowlist: tuple[Name, ...] = ("continue", "inspect", "recover", "request_help")
    acceptance_threshold: Score = 0.8
    minimum_margin: Score = 0.15
    required_units: dict[str, str] = Field(default_factory=lambda: {
        "front_range_m": "m", "camera_clearance_m": "m", "battery_pct": "%", "stuck": "bool",
    })
    conflict_tolerance_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.4
    minimum_clearance_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.5
    minimum_battery_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] = 10.0

    @field_validator("required_units")
    @classmethod
    def fixed_robot_units(cls, units):
        required = {"front_range_m": "m", "camera_clearance_m": "m", "battery_pct": "%", "stuck": "bool"}
        if any(units.get(feature) != unit for feature, unit in required.items()):
            raise ValueError("robot features and their physical units cannot be removed or redefined")
        if any(not key or not value for key, value in units.items()):
            raise ValueError("feature and unit labels cannot be empty")
        return units


class PolicyOutcome(Contract):
    request_id: Name
    policy_version: Name
    disposition: Literal["accept", "need_evidence", "defer"]
    reason: Name
    evaluated_at: AwareDatetime
    action: Name | None = None
    parameters: dict[str, StrictInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def accepted_action(self):
        if (self.disposition == "accept") != (self.action is not None):
            raise ValueError("only an accepted outcome specifies a dispatchable action")
        if self.disposition != "accept" and self.parameters:
            raise ValueError("a deferred outcome cannot specify action parameters")
        return self


class ActionOutcome(Contract):
    request_id: Name
    action: Name
    action_version: Name = "sim-skill-v1"
    status: Literal["completed", "refused", "failed"]
    started_at: AwareDatetime
    completed_at: AwareDatetime
    simulated: Literal[True] = True
    parameters: dict[str, StrictInt]
    observed: dict[str, Number | StrictStr | StrictBool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_times(self):
        if self.completed_at < self.started_at:
            raise ValueError("completion precedes start")
        return self


class DecisionRecord(Contract):
    schema_version: Literal["1"] = "1"
    record_id: Name
    batch_id: Name
    recorded_at: AwareDatetime
    request: DecisionRequest
    provider_identity: ProviderIdentity
    provider_result: ProviderResult | None
    provider_status: Literal["ok", "failure", "malformed", "timeout", "cancelled", "budget_exhausted", "busy"]
    policy_config: PolicyConfig
    policy: PolicyOutcome
    action: ActionOutcome | None
    pollard_version: Name
    pollard_root_id: Name
    pollard_model_node_id: Name | None
    pollard_action_node_id: Name | None
    budget_limit_requests: Annotated[StrictInt, Field(ge=0)]
    budget_spent_requests: Annotated[StrictInt, Field(ge=0)]
    inference_elapsed_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    inference_timeout_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    mode: Literal["decision", "live_reevaluation"] = "decision"
    reevaluates_record_id: Name | None = None

    @model_validator(mode="after")
    def linked_record(self):
        if self.policy.request_id != self.request.request_id or self.policy.policy_version != self.policy_config.version:
            raise ValueError("policy must match the recorded request and configuration")
        if (self.provider_status == "ok") != (self.provider_result is not None):
            raise ValueError("only successful inference has a validated provider result")
        if self.provider_result is not None and (
            self.provider_result.request_id != self.request.request_id or self.provider_result.identity != self.provider_identity
        ):
            raise ValueError("provider result must match the request and identity")
        if self.action is not None:
            if self.action.request_id != self.request.request_id:
                raise ValueError("action must match the request")
            if self.action.status == "completed" and (
                self.policy.disposition != "accept" or self.action.action != self.policy.action
                or self.action.parameters != self.policy.parameters
            ):
                raise ValueError("completed action must match its accepted policy")
        if self.budget_spent_requests > self.budget_limit_requests:
            raise ValueError("logical request budget cannot be exceeded")
        if self.mode == "live_reevaluation" and (self.action is not None or self.reevaluates_record_id is None):
            raise ValueError("live reevaluation must link a record and cannot dispatch")
        return self


def fresh(request: DecisionRequest, now: datetime) -> bool:
    """All supplied evidence must be valid now; expiry is an exclusive bound."""
    return request.created_at <= now < request.valid_until and all(
        obs.observed_at <= now < obs.valid_until for obs in request.observations
    )
