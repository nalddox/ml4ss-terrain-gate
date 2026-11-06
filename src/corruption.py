"""Deterministic missingness and corruption utilities (from ECE 532 A2).

Project-specific injections (blur, low light, LiDAR thinning, map flip) are added here as they are built."""
import torch

def drop_modalities(features, indices):
    mask=torch.ones((len(features[0]),len(features)),dtype=torch.bool,device=features[0].device)
    mask[:,list(indices)]=False
    return [torch.where(mask[:,m,None],x,torch.zeros_like(x)) for m,x in enumerate(features)],mask

def gaussian_corruption(features, modality, sigma=0.5, seed=0):
    if sigma < 0:
        raise ValueError('sigma must be nonnegative')
    result=[x.clone() for x in features]
    generator=torch.Generator(device=result[modality].device).manual_seed(seed)
    noise=torch.randn(result[modality].shape,generator=generator,device=result[modality].device)
    result[modality] += sigma*noise
    return result

def modality_dropout(n, modalities, probability=0.2, seed=0, device='cpu'):
    if n < 1 or modalities < 1 or not 0 <= probability <= 1:
        raise ValueError('Invalid dropout arguments')
    g=torch.Generator(device=device).manual_seed(seed)
    mask=torch.rand((n,modalities),generator=g,device=device)>=probability
    empty=~mask.any(1)
    mask[empty,torch.randint(modalities,(int(empty.sum()),),generator=g,device=device)]=True
    return mask
