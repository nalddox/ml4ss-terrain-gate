"""Deterministic checkers shared by the tests and the receipt builder (from ECE 532 A2 Q4.a-b).

Each check returns a list of violation strings; an empty list means it passed.
"""
import numpy as np


def check_mask(mask, num_modalities):
    """Q4.a: binary entries, at least one modality per window, [N,M] or [M] shape."""
    a = np.asarray(mask)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2 or a.shape[1] != num_modalities:
        return [f'mask shape {np.shape(mask)} does not match {num_modalities} configured modalities']
    errors = []
    if a.dtype != bool and not np.isin(a, [0, 1]).all():
        errors.append(f'mask has non-binary entries: {np.unique(a[~np.isin(a, [0, 1])])[:5].tolist()}')
    empty = np.flatnonzero(a.astype(bool).sum(1) < 1)
    if len(empty):
        errors.append(f'{len(empty)} window(s) have no available modality, e.g. rows {empty[:5].tolist()}')
    return errors


def check_probabilities(probs, atol=1e-5):
    """Finite, in [0,1], rows sum to one."""
    p = np.asarray(probs, dtype=np.float64)
    errors = []
    if not np.isfinite(p).all():
        errors.append(f'{int((~np.isfinite(p)).any(-1).sum())} row(s) contain NaN or inf')
    elif (p < 0).any() or (p > 1).any():
        errors.append('probabilities outside [0,1]')
    elif not np.allclose(p.sum(-1), 1, atol=atol):
        errors.append(f'max |sum-1| = {np.abs(p.sum(-1) - 1).max():.2e}')
    return errors


def check_decision(policy, probabilities, mask, decision):
    """Q4.b: act only at p* >= theta_act, defer below theta_defer, acquire only a missing modality.

    theta_act comes from policy.act_threshold(mask) when defined (per-count policies)."""
    p, a = np.asarray(probabilities, dtype=np.float64), np.asarray(mask).astype(bool)
    confidence = p.max()
    theta_act = policy.act_threshold(a) if hasattr(policy, 'act_threshold') else policy.theta_act
    outcome, modality = decision['outcome'], decision.get('modality')
    errors = []
    if outcome not in {'act', 'acquire', 'defer'}:
        errors.append(f'unknown outcome {outcome!r}')
    if outcome == 'act' and not confidence >= theta_act:
        errors.append(f'act at p*={confidence:.4f} < theta_act={theta_act}')
    if confidence < policy.theta_defer and outcome != 'defer':
        errors.append(f'{outcome} at p*={confidence:.4f} < theta_defer={policy.theta_defer}')
    if outcome == 'acquire':
        if modality not in policy.modalities:
            errors.append(f'acquire names unknown modality {modality!r}')
        elif a[policy.modalities.index(modality)]:
            errors.append(f'acquire requests {modality}, which is already available')
    elif modality is not None:
        errors.append(f'{outcome} must not name a modality (got {modality!r})')
    return errors
