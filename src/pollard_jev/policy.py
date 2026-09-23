"""Small deterministic evidence and feasibility gate for the demonstration robot."""

from datetime import datetime

from .contracts import DecisionRequest, PolicyConfig, PolicyOutcome, ProviderResult, fresh
from .simulator import SKILL_PARAMETERS, inspect_evidence


class DecisionPolicy:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = PolicyConfig.model_validate((config or PolicyConfig()).model_dump())

    def outcome(self, request, now, disposition, reason, action=None, parameters=None):
        return PolicyOutcome(
            request_id=request.request_id, policy_version=self.config.version,
            disposition=disposition, reason=reason, evaluated_at=now,
            action=action, parameters=parameters or {},
        )

    def evaluate(self, request: DecisionRequest, result: ProviderResult | None, now: datetime) -> PolicyOutcome:
        if not fresh(request, now):
            return self.outcome(request, now, "defer", "stale_or_future_evidence")
        values, evidence_problem = inspect_evidence(request, self.config)
        if evidence_problem:
            return self.outcome(request, now, "need_evidence", evidence_problem)
        if result is None:
            return self.outcome(request, now, "defer", "no_provider_result")
        choices = {choice.name: choice for choice in request.choices}
        action = result.proposed_action
        if result.request_id != request.request_id:
            return self.outcome(request, now, "defer", "mismatched_request")
        if action is not None and (action not in choices or action not in self.config.allowlist or action not in SKILL_PARAMETERS):
            return self.outcome(request, now, "defer", "unauthorized_action")
        if result.evidence != "sufficient" or action is None:
            return self.outcome(request, now, "need_evidence", "provider_insufficient_evidence")
        if result.scores.keys() != choices.keys():
            return self.outcome(request, now, "defer", "score_choices_mismatch")
        score = result.scores[action]
        others = [v for k, v in result.scores.items() if k != action]
        if score < self.config.acceptance_threshold or (others and score - max(others) < self.config.minimum_margin):
            return self.outcome(request, now, "defer", "insufficient_support_or_margin")
        bounds = SKILL_PARAMETERS[action]
        params = result.parameters
        if params.keys() != bounds.keys() or any(
            type(params[k]) is not int or not lo <= params[k] <= hi for k, (lo, hi) in bounds.items()
        ):
            return self.outcome(request, now, "defer", "invalid_action_parameters")
        # Request choices narrow the callable skill parameters as well as its name.
        if params != choices[action].parameters:
            return self.outcome(request, now, "defer", "parameters_not_permitted_by_request")
        if action == "continue" and (
            values["battery_pct"] < self.config.minimum_battery_pct or values["stuck"] != 0
            or min(values["front_range_m"], values["camera_clearance_m"]) < self.config.minimum_clearance_m
        ):
            return self.outcome(request, now, "defer", "action_infeasible")
        if action == "recover" and (
            values["stuck"] != 1 or values["battery_pct"] < self.config.minimum_battery_pct
        ):
            return self.outcome(request, now, "defer", "action_infeasible")
        return self.outcome(request, now, "accept", "supported_and_feasible", action, params)
