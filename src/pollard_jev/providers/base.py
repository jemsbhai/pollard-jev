from typing import Protocol

from ..contracts import DecisionRequest, ProviderIdentity, ProviderResult


class DecisionProvider(Protocol):
    """One infer invocation is one budgeted batch, regardless of question count.

    Implementations return data only and must not control actuators. The loop
    discards late responses; it cannot forcibly stop arbitrary native inference.
    """

    identity: ProviderIdentity

    def infer(self, requests: tuple[DecisionRequest, ...]) -> tuple[ProviderResult, ...]: ...
