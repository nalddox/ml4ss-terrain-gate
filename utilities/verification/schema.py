"""Common verification substrate for ECE 532 Fall 2026.

Shared vocabulary across A2, A3, A4, and final projects:
observation -> claim -> obligation -> checker -> warrant -> action gate -> receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionOutcome(str, Enum):
    ACT = "act"
    DEFER = "defer"
    REQUEST_EVIDENCE = "request_evidence"
    HAND_OFF = "hand_off"
    REPAIR = "repair"


class CheckerKind(str, Enum):
    DETERMINISTIC = "deterministic"
    STATISTICAL = "statistical"
    PROTOCOL = "protocol"
    HUMAN_REVIEW = "human_review"  # must be operationalized; not a default pass


@dataclass
class Observation:
    """Raw or processed evidence with provenance."""

    source: str
    modality: str
    timestamp: str
    payload_ref: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class TypedClaim:
    """Action-relevant statement derived from evidence."""

    claim_id: str
    claim_type: str  # e.g. geometric, causal, statistical, protocol
    statement: str
    confidence: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
    risk_tier: RiskTier = RiskTier.MEDIUM


@dataclass
class VerificationObligation:
    """Required check before a claim may support an action."""

    obligation_id: str
    claim_id: str
    checker_id: str
    description: str
    required: bool = True


@dataclass
class CheckerResult:
    """Outcome of an executable checker."""

    checker_id: str
    checker_kind: CheckerKind
    passed: bool
    message: str
    scope: str
    input_contract: dict[str, Any] = field(default_factory=dict)
    unavailable: bool = False


@dataclass
class BoundedWarrant:
    """Explicit bounds on what a claim supports."""

    claim_id: str
    supports: str
    excludes: str
    calibration: float | None = None
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ActionGate:
    """Fail-closed admission policy."""

    claim_id: str
    risk_tier: RiskTier
    outcome: ActionOutcome
    reason: str
    obligations_met: bool
    fail_closed: bool = True


@dataclass
class ExecutionReceipt:
    """Auditable record for grading and reproducibility."""

    run_id: str
    assignment: str
    environment: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    obligations: list[dict[str, Any]] = field(default_factory=list)
    checker_results: list[dict[str, Any]] = field(default_factory=list)
    warrants: list[dict[str, Any]] = field(default_factory=list)
    action_gates: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    failures_injected: list[str] = field(default_factory=list)
    outcome: ActionOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.outcome is not None:
            data["outcome"] = self.outcome.value
        return data


def evaluate_gate(
    obligations: list[VerificationObligation],
    results: list[CheckerResult],
    claim: TypedClaim,
) -> ActionGate:
    """Fail closed when a required checker is missing or fails."""
    result_by_id = {r.checker_id: r for r in results}
    required = [o for o in obligations if o.required and o.claim_id == claim.claim_id]

    for obligation in required:
        result = result_by_id.get(obligation.checker_id)
        if result is None or result.unavailable:
            return ActionGate(
                claim_id=claim.claim_id,
                risk_tier=claim.risk_tier,
                outcome=ActionOutcome.DEFER,
                reason=f"Required checker unavailable: {obligation.checker_id}",
                obligations_met=False,
                fail_closed=True,
            )
        if not result.passed:
            return ActionGate(
                claim_id=claim.claim_id,
                risk_tier=claim.risk_tier,
                outcome=ActionOutcome.REQUEST_EVIDENCE,
                reason=result.message,
                obligations_met=False,
                fail_closed=True,
            )

    return ActionGate(
        claim_id=claim.claim_id,
        risk_tier=claim.risk_tier,
        outcome=ActionOutcome.ACT,
        reason="All required obligations satisfied",
        obligations_met=True,
        fail_closed=True,
    )
