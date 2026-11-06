"""Copied A2 checkers and policy: mask rules and the act/defer threshold invariant."""
import numpy as np
import pytest

from src.checks import check_decision, check_mask, check_probabilities
from src.evidence_policy import ActionPolicy

COSTS = {'camera': 0.05, 'lidar': 0.05, 'map': 0.0}


def test_check_mask():
    assert check_mask(np.ones((4, 3), bool), 3) == []
    assert check_mask([[1, 0, 0], [0, 0, 0]], 3)
    assert check_mask([[1, 2, 0]], 3)
    assert check_mask(np.ones((4, 2)), 3)


def test_check_probabilities():
    assert check_probabilities([[0.2, 0.8]]) == []
    assert check_probabilities([[0.2, 0.7]])
    assert check_probabilities([[np.nan, 1.0]])


@pytest.mark.parametrize('theta_act,theta_defer', [(0.85, 0.5), (0.95, 0.6)])
def test_policy_decisions_satisfy_checker(theta_act, theta_defer):
    rng = np.random.default_rng(0)
    policy = ActionPolicy(COSTS, theta_act, theta_defer, voi={'camera': 0.2, 'lidar': 0.2})
    for _ in range(2000):
        p = rng.dirichlet(np.full(3, 0.5))
        mask = rng.integers(0, 2, 3)
        if not mask.any():
            mask[rng.integers(3)] = 1
        decision = policy.decide(p, mask)
        assert check_decision(policy, p, mask, decision) == []


def test_policy_fails_closed_on_invalid_probabilities():
    policy = ActionPolicy(COSTS)
    assert policy.decide([np.nan, 1.0, 0.0], [1, 1, 1])['outcome'] == 'defer'
    assert policy.decide([0.99, 0.0, 0.0], [1, 1, 1])['outcome'] == 'defer'
