"""Shared fusion interface (carried over from A2) and TMC evidential fusion properties."""
import pytest
import torch

from src.fusion import (AttentionFusion, EarlyFusion, EvidentialFusion, LateFusion, combine_opinions,
                        evidence_from_opinion, evidential_loss, opinion_from_evidence)

DIMS, C, N = [6, 4, 3], 3, 32
MODELS = [EarlyFusion, LateFusion, AttentionFusion, EvidentialFusion]


def make(cls, seed=0):
    torch.manual_seed(seed)
    return cls(DIMS, C, 16)


def inputs(n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, d, generator=g) for d in DIMS]


@pytest.mark.parametrize('cls', MODELS)
@pytest.mark.parametrize('mask', [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
def test_shapes_and_normalized(cls, mask):
    logits, probs = make(cls)(inputs(), torch.tensor(mask))
    assert logits.shape == probs.shape == (N, C)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(1), torch.ones(N), atol=1e-5)


@pytest.mark.parametrize('cls', MODELS)
def test_rejects_empty_row_and_bad_mask(cls):
    model, x = make(cls), inputs()
    mask = torch.ones(N, 3, dtype=torch.bool)
    mask[5] = False
    with pytest.raises(ValueError):
        model(x, mask)
    with pytest.raises(ValueError):
        model(x, torch.tensor([1, 2, 1]))
    with pytest.raises(ValueError):
        model(x[:2], torch.tensor([1, 1]))


@pytest.mark.parametrize('cls', MODELS)
def test_missing_view_values_are_ignored(cls):
    model, x = make(cls), inputs()
    mask = torch.tensor([1, 0, 1])
    y = [t.clone() for t in x]
    y[1] = torch.full_like(y[1], float('nan'))
    assert torch.allclose(model(x, mask)[1], model(y, mask)[1])


# ------------------------------------------------------------------ evidential


def random_opinion(n=64, k=C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return opinion_from_evidence(torch.rand(n, k, generator=g) * 20)


def test_opinion_sums_to_one_and_inverts():
    e = torch.rand(10, C) * 5
    b, u = opinion_from_evidence(e)
    assert torch.allclose(b.sum(-1) + u, torch.ones(10))
    assert torch.allclose(evidence_from_opinion(b, u), e, atol=1e-5)


def test_vacuous_opinion_is_identity():
    b, u = random_opinion()
    b2, u2, conflict = combine_opinions(b, u, torch.zeros_like(b), torch.ones_like(u))
    assert torch.allclose(b2, b) and torch.allclose(u2, u) and torch.all(conflict == 0)


def test_combination_is_commutative_normalized_and_reduces_uncertainty():
    b1, u1 = random_opinion(seed=1)
    b2, u2 = random_opinion(seed=2)
    ba, ua, ca = combine_opinions(b1, u1, b2, u2)
    bb, ub, cb = combine_opinions(b2, u2, b1, u1)
    assert torch.allclose(ba, bb) and torch.allclose(ua, ub) and torch.allclose(ca, cb)
    assert torch.allclose(ba.sum(-1) + ua, torch.ones_like(ua), atol=1e-6)
    assert torch.all(ua <= torch.minimum(u1, u2) + 1e-7)
    assert torch.all((ca >= 0) & (ca < 1))


def test_disagreeing_confident_views_have_high_conflict():
    strong = 100.0
    b1, u1 = opinion_from_evidence(torch.tensor([[strong, 0.0, 0.0]]))
    b2, u2 = opinion_from_evidence(torch.tensor([[0.0, 0.0, strong]]))
    _, _, conflict = combine_opinions(b1, u1, b2, u2)
    assert conflict.item() > 0.9


def test_single_available_view_equals_that_view():
    model, x = make(EvidentialFusion), inputs()
    o = model.opinions(x, torch.tensor([0, 1, 0]))
    assert torch.allclose(o['alpha'], o['view_alpha'][:, 1], atol=1e-4)
    assert torch.all(o['conflict'] == 0)


def test_opinions_fields():
    o = make(EvidentialFusion).opinions(inputs(), torch.tensor([1, 1, 1]))
    assert o['view_alpha'].shape == (N, 3, C) and o['alpha'].shape == (N, C)
    assert torch.all(o['view_alpha'] >= 1) and torch.all(o['alpha'] >= 1 - 1e-5)
    assert torch.all((o['u'] > 0) & (o['u'] <= 1))
    assert torch.allclose(o['pairwise_conflict'], o['pairwise_conflict'].transpose(1, 2))


def test_evidential_loss_prefers_correct_evidence():
    labels = torch.tensor([0])
    right = evidential_loss(torch.tensor([[20.0, 1.0, 1.0]]), labels)
    wrong = evidential_loss(torch.tensor([[1.0, 20.0, 1.0]]), labels)
    assert right.item() < wrong.item()
