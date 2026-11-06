"""Provided ECE and NLL helpers. Do not modify."""
import numpy as np

def checked(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if p.ndim != 2 or len(p) == 0 or y.shape != (len(p),):
        raise ValueError('Expected nonempty [N,C] probabilities and [N] labels')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability vectors')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Labels must be integer class indices')
    return p,y

def reliability_bins(probabilities, labels, bins=15):
    p,y = checked(probabilities, labels)
    if not isinstance(bins,int) or bins < 1:
        raise ValueError('bins must be a positive integer')
    confidence=p.max(1); correct=(p.argmax(1)==y)
    # Right-closed bins (0,1/B], ...; confidence=1 belongs to the last bin.
    indices=np.clip(np.ceil(confidence*bins).astype(int)-1,0,bins-1)
    rows=[]
    for b in range(bins):
        selected=indices==b; n=int(selected.sum())
        rows.append({'count': n, 'confidence': float(confidence[selected].mean()) if n else None,
                     'accuracy': float(correct[selected].mean()) if n else None})
    return rows

def compute_ece(probabilities, labels, bins=15):
    rows=reliability_bins(probabilities,labels,bins)
    return sum(row['count']*abs(row['accuracy']-row['confidence']) for row in rows if row['count'])/len(labels)

def compute_nll(probabilities, labels):
    p,y=checked(probabilities,labels)
    return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())

