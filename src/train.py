"""Training loop, adapted from ECE 532 A2 (src/study.py: train_model).

Models with a .loss(features, mask, labels, kl_weight) method (EvidentialFusion) are trained on that
loss with the KL weight annealed as min(1, epoch / kl_anneal_epochs); other models use cross-entropy
on their logits. Modality dropout draws a fresh replayable mask each epoch and is intersected with
the recorded availability mask, so a view that was never observed is never switched back on.
"""
import torch
from torch import nn

from src.corruption import modality_dropout


def train_model(model, features, mask, labels, cfg, device='cpu'):
    """Train in place on the training split only. cfg keys: seed, lr, epochs, batch_size,
    modality_drop_prob (default 0), kl_anneal_epochs (default 10)."""
    n, num_views = len(labels), len(features)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    p_drop = cfg.get('modality_drop_prob', 0.0)
    anneal = max(1, cfg.get('kl_anneal_epochs', 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    ce = nn.CrossEntropyLoss()
    shuffle = torch.Generator().manual_seed(cfg['seed'])
    model.train()
    for epoch in range(cfg['epochs']):
        epoch_mask = mask
        if p_drop > 0:
            drop = modality_dropout(n, num_views, p_drop, seed=cfg['seed'] + epoch, device=device)
            # Keep at least one originally available view per row.
            kept = mask & drop
            epoch_mask = torch.where(kept.any(1, keepdim=True), kept, mask)
        order = torch.randperm(n, generator=shuffle).to(device)
        for start in range(0, n, cfg['batch_size']):
            idx = order[start:start + cfg['batch_size']]
            batch = [x[idx] for x in features]
            if hasattr(model, 'loss'):
                loss = model.loss(batch, epoch_mask[idx], labels[idx], kl_weight=min(1.0, epoch / anneal))
            else:
                logits, _ = model(batch, epoch_mask[idx])
                loss = ce(logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
