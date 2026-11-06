"""Fusion models over per-view evidence.

Interface (from ECE 532 A2): forward(features, mask) takes M tensors [N,d_m] and a mask [N,M] or [M],
and returns (logits, probabilities), both [N,C]. Missing views contribute nothing; all-empty rows raise
ValueError. In this project a view is camera, LiDAR, or map prior, and a missing view is a dropped
sensor or one with too little data at that range.

EarlyFusion, LateFusion and AttentionFusion are the softmax baselines carried over from A2.
EvidentialFusion is trusted multi-view classification (TMC; Han et al., TPAMI 2023): each view
outputs non-negative evidence for a Dirichlet, and views are combined with the reduced Dempster
rule of subjective logic.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _prepare(features, mask, num_modalities):
    """Validate inputs and return (features, boolean mask [N,M])."""
    features = list(features)
    if len(features) != num_modalities:
        raise ValueError(f'Expected {num_modalities} modalities, got {len(features)}')
    n = features[0].shape[0]
    if any(x.dim() != 2 or x.shape[0] != n for x in features):
        raise ValueError('Each modality must be a [N,d_m] tensor with a shared N')
    mask = torch.as_tensor(mask, device=features[0].device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(n, -1)
    if mask.shape != (n, num_modalities):
        raise ValueError(f'Mask must have shape [{num_modalities}] or [{n},{num_modalities}], got {tuple(mask.shape)}')
    if mask.dtype != torch.bool:
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError('Mask entries must be binary')
        mask = mask.bool()
    if not mask.any(1).all():
        raise ValueError('Every window needs at least one available modality')
    return features, mask


def _zero_fill(features, mask):
    """Zero missing modalities; torch.where also discards NaN in missing slots."""
    return [torch.where(mask[:, m, None], x, torch.zeros_like(x)) for m, x in enumerate(features)]


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class EarlyFusion(nn.Module):
    """Zero-filled concatenation -> Linear -> ReLU -> Linear."""

    def __init__(self, input_dims, num_classes, fusion_embed_dim=64):
        super().__init__()
        self.input_dims = list(input_dims)
        self.num_classes = num_classes
        self.classifier = _mlp(sum(self.input_dims), fusion_embed_dim, num_classes)

    def forward(self, features, mask):
        features, mask = _prepare(features, mask, len(self.input_dims))
        x = torch.cat(_zero_fill(features, mask), dim=1)
        logits = self.classifier(x)
        return logits, logits.softmax(dim=1)


class LateFusion(nn.Module):
    """One-hidden-layer head per modality; fused logits are the mean over available heads."""

    def __init__(self, input_dims, num_classes, fusion_embed_dim=64):
        super().__init__()
        self.input_dims = list(input_dims)
        self.num_classes = num_classes
        self.heads = nn.ModuleList(_mlp(d, fusion_embed_dim, num_classes) for d in self.input_dims)

    def modality_logits(self, features, mask):
        """Per-modality logits [N,M,C]; rows for missing modalities are zero."""
        features, mask = _prepare(features, mask, len(self.input_dims))
        logits = torch.stack([h(x) for h, x in zip(self.heads, _zero_fill(features, mask))], dim=1)
        return torch.where(mask[:, :, None], logits, torch.zeros_like(logits)), mask

    def forward(self, features, mask):
        per_modality, mask = self.modality_logits(features, mask)
        counts = mask.sum(dim=1, keepdim=True).to(per_modality.dtype)
        logits = per_modality.sum(dim=1) / counts
        return logits, logits.softmax(dim=1)


class AttentionFusion(nn.Module):
    """Per-modality embeddings weighted by masked scaled dot-product attention with a learned query."""

    def __init__(self, input_dims, num_classes, fusion_embed_dim=64):
        super().__init__()
        self.input_dims = list(input_dims)
        self.num_classes = num_classes
        self.embed_dim = fusion_embed_dim
        self.embedders = nn.ModuleList(_mlp(d, fusion_embed_dim, fusion_embed_dim) for d in self.input_dims)
        self.query = nn.Parameter(torch.randn(fusion_embed_dim) / fusion_embed_dim ** 0.5)
        self.head = _mlp(fusion_embed_dim, fusion_embed_dim, num_classes)

    def attend(self, features, mask):
        """Return fused z [N,d], attention weights alpha [N,M] and embeddings e [N,M,d]."""
        features, mask = _prepare(features, mask, len(self.input_dims))
        e = torch.stack([f(x) for f, x in zip(self.embedders, _zero_fill(features, mask))], dim=1)
        e = torch.where(mask[:, :, None], e, torch.zeros_like(e))
        scores = (e @ self.query) / self.embed_dim ** 0.5
        scores = scores.masked_fill(~mask, float('-inf'))
        alpha = scores.softmax(dim=1)
        z = (alpha[:, :, None] * e).sum(dim=1)
        return z, alpha, e

    def forward(self, features, mask):
        z, _, _ = self.attend(features, mask)
        logits = self.head(z)
        return logits, logits.softmax(dim=1)


# ------------------------------------------------------------------ evidential (TMC)

def opinion_from_evidence(evidence):
    """Evidence [...,K] >= 0 -> (belief [...,K], uncertainty [...]) with b.sum(-1) + u = 1."""
    k = evidence.shape[-1]
    strength = evidence.sum(-1) + k  # Dirichlet strength S = sum(alpha), alpha = e + 1
    return evidence / strength[..., None], k / strength


def evidence_from_opinion(belief, uncertainty):
    """Inverse of opinion_from_evidence: S = K / u, e = b * S."""
    k = belief.shape[-1]
    return belief * (k / uncertainty)[..., None]


def opinion_conflict(b1, b2):
    """Conflict mass C = sum over i != j of b1_i * b2_j."""
    return b1.sum(-1) * b2.sum(-1) - (b1 * b2).sum(-1)


def combine_opinions(b1, u1, b2, u2):
    """Reduced Dempster combination of two opinions (Han et al., Def. 4).

    Returns (belief, uncertainty, conflict). The vacuous opinion (b = 0, u = 1) is the identity,
    which is how missing views are handled. The normalizer 1 - C is at least u1 * u2 > 0 for
    finite evidence, so the rule is always defined; it degrades (renormalizes away conflict) as
    C -> 1, which is the known weakness under a stale map.
    """
    conflict = opinion_conflict(b1, b2)
    scale = 1.0 / (1.0 - conflict)
    belief = scale[..., None] * (b1 * b2 + b1 * u2[..., None] + b2 * u1[..., None])
    return belief, scale * u1 * u2, conflict


def dirichlet_kl_to_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) per row."""
    k = alpha.shape[-1]
    strength = alpha.sum(-1)
    return (torch.lgamma(strength) - torch.lgamma(torch.tensor(float(k), device=alpha.device))
            - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength)[..., None])).sum(-1))


def evidential_loss(alpha, labels, kl_weight=1.0):
    """TMC per-sample loss: integrated cross-entropy plus KL on misleading evidence (Han et al., Eq. 9-11).

    kl_weight is the annealing coefficient lambda_t = min(1, t / T). Returns [N].
    """
    y = F.one_hot(labels, alpha.shape[-1]).to(alpha.dtype)
    strength = alpha.sum(-1, keepdim=True)
    ace = (y * (torch.digamma(strength) - torch.digamma(alpha))).sum(-1)
    alpha_tilde = y + (1 - y) * alpha  # remove evidence for the true class
    return ace + kl_weight * dirichlet_kl_to_uniform(alpha_tilde)


class EvidentialFusion(nn.Module):
    """TMC: per-view evidence heads, Dirichlet opinions, Dempster combination.

    forward returns (log expected probabilities, expected probabilities alpha / S) so it satisfies
    the shared interface; opinions() exposes beliefs, uncertainty and pairwise conflict.
    """

    def __init__(self, input_dims, num_classes, fusion_embed_dim=64):
        super().__init__()
        self.input_dims = list(input_dims)
        self.num_classes = num_classes
        self.heads = nn.ModuleList(_mlp(d, fusion_embed_dim, num_classes) for d in self.input_dims)

    def view_evidence(self, features, mask):
        """Per-view evidence [N,M,K]; zero (a vacuous opinion) for missing views, plus the bool mask."""
        features, mask = _prepare(features, mask, len(self.input_dims))
        evidence = torch.stack([F.softplus(h(x)) for h, x in zip(self.heads, _zero_fill(features, mask))], dim=1)
        return torch.where(mask[:, :, None], evidence, torch.zeros_like(evidence)), mask

    def opinions(self, features, mask):
        """Dict with view_alpha [N,M,K], view_u [N,M], alpha [N,K], u [N],
        pairwise_conflict [N,M,M] (zero where either view is missing) and conflict [N] (max over pairs)."""
        evidence, mask = self.view_evidence(features, mask)
        view_b, view_u = opinion_from_evidence(evidence)
        b, u = view_b[:, 0], view_u[:, 0]
        for m in range(1, evidence.shape[1]):
            b, u, _ = combine_opinions(b, u, view_b[:, m], view_u[:, m])
        pairwise = opinion_conflict(view_b[:, :, None, :], view_b[:, None, :, :])
        pairwise = pairwise * (~torch.eye(evidence.shape[1], dtype=torch.bool, device=pairwise.device))
        return {
            'mask': mask,
            'view_alpha': evidence + 1,
            'view_u': view_u,
            'alpha': evidence_from_opinion(b, u) + 1,
            'u': u,
            'pairwise_conflict': pairwise,
            'conflict': pairwise.flatten(1).max(1).values,
        }

    def forward(self, features, mask):
        alpha = self.opinions(features, mask)['alpha']
        probs = alpha / alpha.sum(-1, keepdim=True)
        return probs.log(), probs

    def loss(self, features, mask, labels, kl_weight=1.0):
        """Mean over rows of fused loss plus each available view's loss (Han et al., Eq. 12)."""
        o = self.opinions(features, mask)
        total = evidential_loss(o['alpha'], labels, kl_weight)
        for m in range(o['view_alpha'].shape[1]):
            total = total + o['mask'][:, m] * evidential_loss(o['view_alpha'][:, m], labels, kl_weight)
        return total.mean()
