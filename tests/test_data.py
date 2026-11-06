"""Round trip of the cached-feature contract in src/data.py."""
import json

import numpy as np
import pytest

from src import data


def write_cache(root, n=8, dims=(5, 4, 3)):
    cache = root / 'cache'
    cache.mkdir()
    arrays = {}
    for m, d in enumerate(dims):
        arrays[f'train_x{m}'] = np.random.default_rng(m).normal(size=(n, d)).astype(np.float32)
    arrays['train_mask'] = np.ones((n, 3), bool)
    arrays['train_y'] = np.arange(n) % 2
    arrays['train_edge'] = np.arange(n)
    arrays['train_range_m'] = np.linspace(25, 3, n).astype(np.float32)
    arrays['train_map_age_y'] = np.full(n, 5.0, np.float32)
    np.savez(cache / 'features.npz', **arrays)
    (cache / 'metadata.json').write_text(json.dumps(
        {'views': ['camera', 'lidar', 'map'], 'input_dims': list(dims),
         'classes': ['traversable', 'hazard'], 'splits': ['train']}))
    return cache


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    write_cache(tmp_path)
    x, mask, y, context, meta = data.load_split('train', 'cache', limit=5)
    assert [t.shape for t in x] == [(5, 5), (5, 4), (5, 3)]
    assert mask.shape == (5, 3) and y.tolist() == [0, 1, 0, 1, 0]
    assert set(context) == {'edge', 'range_m', 'map_age_y'} and meta['views'][0] == 'camera'
    with pytest.raises(ValueError):
        data.load_split('test', 'cache')
