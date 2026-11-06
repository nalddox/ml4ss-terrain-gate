"""Confidence-threshold action policies, carried over from ECE 532 A2 (Q3.b and graduate track).

Starting point for the project gate: act / acquire / defer becomes Act / Defer-and-approach
(acquire = take the next, closer frame of the edge) / Repair, with risk tiers and checkers C1/C2.

Return {"outcome": "act"|"acquire"|"defer", "modality": str|None}.
The probabilities and mask passed to decide describe ONE window.
voi maps modality names to estimates obtained from training data, in the same
units as costs. The runner supplies an empty dict until you provide estimates.
An empty dict is not a learned estimate; it makes acquisition unavailable.
"""
import numpy as np


def mask_key(mask):
    """Availability pattern as a string, e.g. [1,0,1] -> '101'."""
    return ''.join('1' if a else '0' for a in mask)


class ActionPolicy:
    """Static threshold policy with a training-derived VOI lookup.

    voi[name] is either a scalar or a dict {mask_key: value}, where mask_key is
    the availability pattern *before* acquiring `name`. Values are expected gains
    in accuracy (probability of a correct decision), the same dimensionless scale
    as the teaching costs. A missing entry means no estimate, so no acquisition.
    """

    def __init__(self, costs, theta_act=0.85, theta_defer=0.5, voi=None):
        if not 0 <= theta_defer < theta_act <= 1:
            raise ValueError('Require 0 <= theta_defer < theta_act <= 1')
        self.costs = dict(costs)
        self.modalities = list(self.costs)
        self.theta_act = theta_act
        self.theta_defer = theta_defer
        self.voi = dict(voi or {})

    def act_threshold(self, mask):
        """Confidence needed to act given this availability mask."""
        return self.theta_act

    def voi_for(self, modality, mask):
        value = self.voi.get(modality)
        if isinstance(value, dict):
            value = value.get(mask_key(mask))
        return None if value is None else float(value)

    def _check(self, probabilities, mask):
        p = np.asarray(probabilities, dtype=np.float64)
        a = np.asarray(mask)
        if p.ndim != 1 or a.shape != (len(self.modalities),):
            raise ValueError(f'Expected one window: [C] probabilities and [{len(self.modalities)}] mask')
        if not np.isin(a, [0, 1]).all():
            raise ValueError('Mask entries must be binary')
        return p, a.astype(bool)

    def _valid(self, p, a):
        """Fail closed on unusable evidence: nonfinite or unnormalized output, empty mask."""
        return a.any() and np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1) <= 1e-4

    def choose_acquisition(self, p, a):
        """Missing modality with the largest positive VOI - cost, or None."""
        best, best_net = None, 0.0
        for m, name in enumerate(self.modalities):
            if a[m]:
                continue
            voi = self.voi_for(name, a)
            if voi is None:
                continue
            net = voi - self.costs[name]
            if net > best_net:
                best, best_net = name, net
        return best

    def decide(self, probabilities, mask):
        p, a = self._check(probabilities, mask)
        if not self._valid(p, a):
            return {'outcome': 'defer', 'modality': None}
        confidence = p.max()
        if confidence >= self.act_threshold(a):
            return {'outcome': 'act', 'modality': None}
        if confidence >= self.theta_defer:
            name = self.choose_acquisition(p, a)
            if name is not None:
                return {'outcome': 'acquire', 'modality': name}
        return {'outcome': 'defer', 'modality': None}

# G.a–G.d: add your alternative and comparison policies here. Use the supplied
# evaluation conditions and preserve the same decide() return interface.

# Decision costs shared by the graduate policies. A wrong action costs 1 (one lost
# correct decision, the unit of the VOI/accuracy-gain scale) and a deferral costs
# C_DEFER. Acting when P(error) <= C_DEFER minimizes expected cost, so C_DEFER = 0.15
# is the trade-off the static theta_act = 0.85 implies for a calibrated model.
C_ERROR = 1.0
C_DEFER = 0.15
THRESHOLD_GRID = tuple(round(0.5 + 0.01 * i, 2) for i in range(51))  # 0.50 ... 1.00


def fit_threshold(confidence, correct, theta_defer=0.5, c_error=C_ERROR, c_defer=C_DEFER, grid=THRESHOLD_GRID):
    """Threshold minimizing empirical decision cost: acted errors * c_error + non-acted * c_defer.

    Ties go to the lowest threshold (most coverage). Returns (theta, cost per window).
    """
    confidence, correct = np.asarray(confidence, dtype=np.float64), np.asarray(correct, dtype=bool)
    if len(confidence) == 0:
        raise ValueError('cannot fit a threshold on zero windows')
    best = None
    for theta in (t for t in grid if t >= theta_defer):
        act = confidence >= theta
        cost = (c_error * (act & ~correct).sum() + c_defer * (~act).sum()) / len(confidence)
        if best is None or cost < best[1] - 1e-12:
            best = (theta, float(cost))
    return best


class CountThresholdPolicy(ActionPolicy):
    """G.a alternative: cost-sensitive confidence threshold per number of available modalities.

    theta_by_count maps k = sum(mask) to theta_act(k), each fit on validation with
    fit_threshold. Acquisition is unchanged from ActionPolicy (VOI - cost > 0), so
    any difference from the static policy comes from the per-count act threshold.
    """

    def __init__(self, costs, theta_by_count, theta_defer=0.5, voi=None):
        thetas = {int(k): float(v) for k, v in theta_by_count.items()}
        super().__init__(costs, max(thetas.values()), theta_defer, voi)
        if set(thetas) != set(range(1, len(self.modalities) + 1)):
            raise ValueError(f'theta_by_count needs keys 1..{len(self.modalities)}')
        if any(not theta_defer <= t <= 1 for t in thetas.values()):
            raise ValueError('Each theta_act(k) must lie in [theta_defer, 1]')
        self.theta_by_count = thetas

    def act_threshold(self, mask):
        return self.theta_by_count[int(np.sum(mask))]


class AlwaysActPolicy(ActionPolicy):
    """G.b baseline: static full fusion - always act on whatever evidence is available."""

    def decide(self, probabilities, mask):
        self._check(probabilities, mask)
        return {'outcome': 'act', 'modality': None}


class RandomAcquisitionPolicy(ActionPolicy):
    """G.b baseline: at each decision, acquire a uniformly random missing modality with
    probability `probability`, otherwise act. Never defers. Seeded for replay."""

    def __init__(self, costs, probability=0.3, seed=0):
        super().__init__(costs)
        self.probability = probability
        self.rng = np.random.default_rng(seed)

    def decide(self, probabilities, mask):
        _, a = self._check(probabilities, mask)
        missing = np.flatnonzero(~a)
        if len(missing) and self.rng.random() < self.probability:
            return {'outcome': 'acquire', 'modality': self.modalities[int(self.rng.choice(missing))]}
        return {'outcome': 'act', 'modality': None}


class CostBlindPolicy(ActionPolicy):
    """G.b baseline: theta_act = 0.85 confidence policy that ignores acquisition cost and
    requests the missing modality with the highest positive VOI. Costs are still
    recorded so replay can charge them."""

    def choose_acquisition(self, p, a):
        best, best_voi = None, 0.0
        for m, name in enumerate(self.modalities):
            voi = None if a[m] else self.voi_for(name, a)
            if voi is not None and voi > best_voi:
                best, best_voi = name, voi
        return best
