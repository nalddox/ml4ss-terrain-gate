"""Training loop smoke tests on a small separable synthetic problem."""
import pytest
import torch

from src.fusion import AttentionFusion, EarlyFusion, EvidentialFusion, LateFusion
from src.train import train_model

CFG = {'seed': 0, 'lr': 0.01, 'epochs': 40, 'batch_size': 64, 'modality_drop_prob': 0.2, 'kl_anneal_epochs': 10}


def synthetic(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(3, (n,), generator=g)
    centers = torch.eye(3) * 3
    x = [centers[y] + torch.randn(n, 3, generator=g) for _ in range(3)]
    return x, torch.ones(n, 3, dtype=torch.bool), y


@pytest.mark.parametrize('cls', [EarlyFusion, LateFusion, AttentionFusion, EvidentialFusion])
def test_learns_separable_problem(cls):
    torch.manual_seed(0)
    x, mask, y = synthetic()
    model = cls([3, 3, 3], 3, 16)
    train_model(model, x, mask, y, CFG)
    with torch.no_grad():
        _, probs = model(x, mask)
    assert (probs.argmax(1) == y).float().mean() > 0.9


def test_evidential_uncertainty_higher_off_distribution():
    torch.manual_seed(0)
    x, mask, y = synthetic()
    model = EvidentialFusion([3, 3, 3], 3, 16)
    train_model(model, x, mask, y, CFG)
    with torch.no_grad():
        u_in = model.opinions(x, mask)['u'].mean()
        # Views pointing at different classes: in-distribution inputs never look like this.
        far = [torch.tensor([[3.0, 0, 0]]), torch.tensor([[0, 3.0, 0]]), torch.tensor([[0, 0, 3.0]])]
        o = model.opinions(far, torch.ones(1, 3, dtype=torch.bool))
    assert o['conflict'].item() > 0.5
    assert u_in < 0.5


def test_dropout_never_enables_unobserved_view():
    x, mask, y = synthetic(64)
    mask[:, 2] = False
    x[2] = torch.full_like(x[2], float('nan'))  # would poison training if ever switched on
    model = EvidentialFusion([3, 3, 3], 3, 8)
    train_model(model, x, mask, y, {**CFG, 'epochs': 3})
    assert all(torch.isfinite(p).all() for p in model.parameters())
