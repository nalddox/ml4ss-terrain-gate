"""Loader for the cached per-view evidence features (the input contract of the fusion stage).

Layout, same as ECE 532 A2 plus a mask and per-row context:
  <cache_dir>/features.npz
      {split}_x{m}      float32 [N, d_m]   frozen features of view m (camera, lidar, map)
      {split}_mask      bool    [N, M]     view available (camera dropped, LiDAR too sparse, ...)
      {split}_y         int64   [N]        class index (0 flat, 1 step/curb, 2 stairs)
      {split}_edge      int64   [N]        edge id
      {split}_range_m   float32 [N]        distance from robot to edge when the frame was taken
      {split}_location  int64   [N]        location id used for the by-location split
  <cache_dir>/metadata.json
      {"views": [...], "input_dims": [...], "classes": [...], "splits": [...]}

The adapter that builds this cache from the GND bags is not written yet. Splits are by location,
with GTown as a separate held-out split.
"""
from pathlib import Path
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_KEYS = ('edge', 'range_m', 'location')


def load_metadata(cache_dir):
    return json.loads((ROOT / cache_dir / 'metadata.json').read_text())


def load_split(split, cache_dir, limit=None, device='cpu'):
    """Return (features [M x [N,d_m]], mask [N,M] bool, labels [N], context dict, metadata)."""
    metadata = load_metadata(cache_dir)
    if split not in metadata['splits']:
        raise ValueError(f'Unknown split {split!r}; cache has {metadata["splits"]}')
    with np.load(ROOT / cache_dir / 'features.npz', allow_pickle=False) as data:
        x = [torch.tensor(data[f'{split}_x{m}'][:limit], device=device) for m in range(len(metadata['views']))]
        mask = torch.tensor(data[f'{split}_mask'][:limit], dtype=torch.bool, device=device)
        y = torch.tensor(data[f'{split}_y'][:limit], dtype=torch.long, device=device)
        context = {k: data[f'{split}_{k}'][:limit] for k in CONTEXT_KEYS if f'{split}_{k}' in data}
    if [t.shape[1] for t in x] != list(metadata['input_dims']):
        raise ValueError('Feature widths do not match metadata input_dims')
    return x, mask, y, context, metadata
