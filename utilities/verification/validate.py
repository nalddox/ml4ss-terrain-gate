"""CLI receipt validator for ECE 532 Fall 2026.

Usage:
    python -m utilities.verification.validate experiments/receipt.json
    python -m utilities.verification.validate --strict experiments/receipt.json

Exit codes:
    0  receipt is valid
    1  receipt is invalid (structural or semantic errors)
    2  file not found or unparseable JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

VALID_ASSIGNMENTS = {"A1", "A2", "A3", "A4", "project"}
VALID_RISK_TIERS = {"low", "medium", "high", "critical"}
VALID_OUTCOMES = {"act", "defer", "request_evidence", "hand_off", "repair"}
VALID_CHECKER_KINDS = {"deterministic", "statistical", "protocol", "human_review"}
VALID_CLAIM_TYPES = {
    "geometric", "causal", "statistical", "protocol",
    "representation", "calibration", "fusion", "identification",
    "observational", "interventional", "counterfactual", "deferred",
}


def _errors_for_observation(obs: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"observations[{idx}]"
    for key in ("source", "modality", "timestamp", "payload_ref"):
        if key not in obs:
            errors.append(f"{prefix}: missing required field '{key}'")
        elif not isinstance(obs[key], str) or not obs[key].strip():
            errors.append(f"{prefix}.{key}: must be a non-empty string")
    return errors


def _errors_for_claim(claim: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"claims[{idx}]"
    for key in ("claim_id", "claim_type", "statement"):
        if key not in claim:
            errors.append(f"{prefix}: missing required field '{key}'")
        elif not isinstance(claim[key], str) or not claim[key].strip():
            errors.append(f"{prefix}.{key}: must be a non-empty string")
    if "risk_tier" in claim and claim["risk_tier"] not in VALID_RISK_TIERS:
        errors.append(
            f"{prefix}.risk_tier: '{claim['risk_tier']}' not in {sorted(VALID_RISK_TIERS)}"
        )
    if "confidence" in claim and claim["confidence"] is not None:
        c = claim["confidence"]
        if not isinstance(c, (int, float)) or c < 0 or c > 1:
            errors.append(f"{prefix}.confidence: must be a number in [0, 1] or null")
    return errors


def _errors_for_obligation(obl: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"obligations[{idx}]"
    for key in ("obligation_id", "claim_id", "checker_id", "description"):
        if key not in obl:
            errors.append(f"{prefix}: missing required field '{key}'")
    return errors


def _errors_for_checker_result(cr: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"checker_results[{idx}]"
    for key in ("checker_id", "checker_kind", "passed", "message", "scope"):
        if key not in cr:
            errors.append(f"{prefix}: missing required field '{key}'")
    if "checker_kind" in cr and cr["checker_kind"] not in VALID_CHECKER_KINDS:
        errors.append(
            f"{prefix}.checker_kind: '{cr['checker_kind']}' not in {sorted(VALID_CHECKER_KINDS)}"
        )
    if "passed" in cr and not isinstance(cr["passed"], bool):
        errors.append(f"{prefix}.passed: must be a boolean")
    return errors


def _errors_for_action_gate(gate: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"action_gates[{idx}]"
    for key in ("claim_id", "risk_tier", "outcome", "reason", "obligations_met"):
        if key not in gate:
            errors.append(f"{prefix}: missing required field '{key}'")
    if "outcome" in gate and gate["outcome"] not in VALID_OUTCOMES:
        errors.append(
            f"{prefix}.outcome: '{gate['outcome']}' not in {sorted(VALID_OUTCOMES)}"
        )
    if "risk_tier" in gate and gate["risk_tier"] not in VALID_RISK_TIERS:
        errors.append(
            f"{prefix}.risk_tier: '{gate['risk_tier']}' not in {sorted(VALID_RISK_TIERS)}"
        )
    return errors


def _semantic_checks(
    receipt: dict[str, Any], strict: bool = False
) -> list[str]:
    """Cross-field checks that enforce the fail-closed contract."""
    errors = []

    claim_ids = {c.get("claim_id") for c in receipt.get("claims", [])}
    checker_ids = {r.get("checker_id") for r in receipt.get("checker_results", [])}

    for i, obl in enumerate(receipt.get("obligations", [])):
        cid = obl.get("claim_id")
        chk = obl.get("checker_id")
        required = obl.get("required", True)

        if cid and cid not in claim_ids:
            errors.append(
                f"obligations[{i}].claim_id: references unknown claim '{cid}'"
            )
        if required and chk and chk not in checker_ids:
            errors.append(
                f"obligations[{i}]: required checker '{chk}' has no corresponding "
                f"checker_result — receipt must fail closed"
            )

    for i, gate in enumerate(receipt.get("action_gates", [])):
        cid = gate.get("claim_id")
        outcome = gate.get("outcome")
        met = gate.get("obligations_met", False)
        if cid and cid not in claim_ids:
            errors.append(
                f"action_gates[{i}].claim_id: references unknown claim '{cid}'"
            )
        if outcome == "act" and not met:
            errors.append(
                f"action_gates[{i}]: outcome is 'act' but obligations_met is false "
                f"— violates fail-closed contract"
            )

    unavailable_required = []
    for obl in receipt.get("obligations", []):
        if not obl.get("required", True):
            continue
        chk = obl.get("checker_id")
        for cr in receipt.get("checker_results", []):
            if cr.get("checker_id") == chk and cr.get("unavailable"):
                unavailable_required.append(chk)
    for chk in unavailable_required:
        admitted = any(
            g.get("outcome") == "act"
            for g in receipt.get("action_gates", [])
        )
        if admitted:
            errors.append(
                f"Required checker '{chk}' is unavailable but receipt admits an 'act' "
                f"outcome — violates fail-closed contract"
            )

    if strict:
        for i, claim in enumerate(receipt.get("claims", [])):
            evidence = claim.get("evidence_ids", [])
            obs_sources = {
                o.get("source") for o in receipt.get("observations", [])
            }
            for eid in evidence:
                if eid not in obs_sources:
                    errors.append(
                        f"claims[{i}].evidence_ids: references unknown source '{eid}'"
                    )

    return errors


def _assignment_specific_checks(
    receipt: dict[str, Any], strict: bool = False
) -> list[str]:
    """Per-assignment validation rules, enforced only in strict mode."""
    if not strict:
        return []

    errors: list[str] = []
    assignment = receipt.get("assignment", "")

    if assignment == "A1":
        failures = receipt.get("failures_injected", [])
        if not failures:
            errors.append(
                "A1: failures_injected must be non-empty "
                "(Q4.c requires at least one injected failure)"
            )
        has_failed_checker = any(
            cr.get("passed") is False
            for cr in receipt.get("checker_results", [])
        )
        if not has_failed_checker:
            errors.append(
                "A1: at least one checker_results entry must have "
                "passed=false (failure injection must be detected)"
            )
        has_non_act_gate = any(
            g.get("outcome") != "act"
            for g in receipt.get("action_gates", [])
        )
        if not has_non_act_gate:
            errors.append(
                "A1: at least one action_gates entry must have "
                "outcome != 'act' (injected failure must trigger deferral)"
            )

    elif assignment == "A2":
        if not receipt.get("checker_results"):
            errors.append(
                "A2: at least one checker_results entry is required "
                "(fusion requires at least one checker)"
            )
        has_failed_checker = any(
            cr.get("passed") is False
            for cr in receipt.get("checker_results", [])
        )
        if not has_failed_checker:
            errors.append(
                "A2: at least one checker_results entry must have "
                "passed=false (Q4.c failure injection must be recorded)"
            )
        has_non_act_gate = any(
            g.get("outcome") != "act"
            for g in receipt.get("action_gates", [])
        )
        if not has_non_act_gate:
            errors.append(
                "A2: at least one action_gates entry must have "
                "outcome != 'act' (failed modality must trigger deferral)"
            )

    elif assignment == "A3":
        claims = receipt.get("claims", [])
        causal_types = {"observational", "interventional", "counterfactual", "deferred"}
        typed_claims = [
            c for c in claims if c.get("claim_type") in causal_types
        ]
        if not typed_claims:
            errors.append(
                "A3: at least one claims entry must have claim_type in "
                f"{sorted(causal_types)}"
            )
        has_deferred = any(c.get("claim_type") == "deferred" for c in claims)
        if not has_deferred:
            errors.append(
                "A3: at least one claims entry must have claim_type='deferred' "
                "(the assignment requires one non-identifiable query)"
            )

    elif assignment == "A4":
        observations = receipt.get("observations", [])
        if len(observations) < 5:
            errors.append(
                f"A4: observations has {len(observations)} entries but the "
                f"5-step workflow minimum requires at least 5"
            )
        failures = receipt.get("failures_injected", [])
        if len(failures) < 3:
            errors.append(
                f"A4: failures_injected has {len(failures)} entries but at "
                f"least 3 are required"
            )
        gates = receipt.get("action_gates", [])
        has_non_act = any(g.get("outcome") != "act" for g in gates)
        if not has_non_act:
            errors.append(
                "A4: at least one action_gates entry must have outcome != 'act' "
                "(failure must be caught)"
            )

    return errors


def validate_receipt(
    receipt: dict[str, Any], strict: bool = False
) -> list[str]:
    """Validate a receipt dict. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    for key in ("run_id", "assignment"):
        if key not in receipt:
            errors.append(f"Missing required top-level field '{key}'")
        elif not isinstance(receipt[key], str) or not receipt[key].strip():
            errors.append(f"Top-level field '{key}' must be a non-empty string")

    assignment = receipt.get("assignment", "")
    if assignment and assignment not in VALID_ASSIGNMENTS:
        errors.append(
            f"assignment: '{assignment}' not in {sorted(VALID_ASSIGNMENTS)}"
        )

    list_fields = [
        "observations", "claims", "obligations",
        "checker_results", "action_gates",
    ]
    for lf in list_fields:
        val = receipt.get(lf)
        if val is not None and not isinstance(val, list):
            errors.append(f"'{lf}' must be a list")

    for i, obs in enumerate(receipt.get("observations", [])):
        errors.extend(_errors_for_observation(obs, i))

    for i, claim in enumerate(receipt.get("claims", [])):
        errors.extend(_errors_for_claim(claim, i))

    for i, obl in enumerate(receipt.get("obligations", [])):
        errors.extend(_errors_for_obligation(obl, i))

    for i, cr in enumerate(receipt.get("checker_results", [])):
        errors.extend(_errors_for_checker_result(cr, i))

    for i, gate in enumerate(receipt.get("action_gates", [])):
        errors.extend(_errors_for_action_gate(gate, i))

    if "outcome" in receipt and receipt["outcome"] is not None:
        if receipt["outcome"] not in VALID_OUTCOMES:
            errors.append(
                f"outcome: '{receipt['outcome']}' not in {sorted(VALID_OUTCOMES)}"
            )

    errors.extend(_semantic_checks(receipt, strict=strict))
    errors.extend(_assignment_specific_checks(receipt, strict=strict))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m utilities.verification.validate",
        description="Validate an ECE 532 execution receipt.",
    )
    parser.add_argument("receipt", type=Path, help="Path to receipt JSON file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode: also check evidence_id cross-references",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress success message",
    )
    args = parser.parse_args(argv)

    if not args.receipt.exists():
        print(f"Error: file not found: {args.receipt}", file=sys.stderr)
        return 2

    try:
        with open(args.receipt) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print("Error: receipt must be a JSON object", file=sys.stderr)
        return 1

    errors = validate_receipt(data, strict=args.strict)

    if errors:
        print(f"INVALID — {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        return 1

    if not args.quiet:
        assignment = data.get("assignment", "?")
        run_id = data.get("run_id", "?")
        outcome = data.get("outcome", "not set")
        n_claims = len(data.get("claims", []))
        n_checkers = len(data.get("checker_results", []))
        n_gates = len(data.get("action_gates", []))
        print(
            f"VALID — {assignment} run {run_id}: "
            f"{n_claims} claim(s), {n_checkers} checker(s), "
            f"{n_gates} gate(s), outcome={outcome}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
