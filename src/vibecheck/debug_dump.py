"""Env-gated debug dump mirroring vibecheck2/core/debug.py.

Set VC_DEBUG_DUMP=<path.pkl> to record per-instance intermediate
artifacts from the v1 pipeline (root intermediate bounds, root spec
lower bounds + query rows, input linearization, input-split
trajectory) in the same schema vc2 writes under VC2_DEBUG_DUMP, so
scratch/clean_slate/compare_debug.py can diff the two step by step.

Zero cost when the env var is unset."""
from __future__ import annotations

import atexit
import os
import pickle

_STORE = None


def enabled():
    return bool(os.environ.get('VC_DEBUG_DUMP'))


def add(key, value):
    global _STORE
    if not enabled():
        return
    if _STORE is None:
        _STORE = {}
        atexit.register(_save)
    _STORE[key] = _to_np(value)


def add_seq(key, value):
    global _STORE
    if not enabled():
        return
    if _STORE is None:
        _STORE = {}
        atexit.register(_save)
    _STORE.setdefault(key, []).append(_to_np(value))


def _to_np(v):
    import torch
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {k: _to_np(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_to_np(x) for x in v)
    return v


def _save():
    path = os.environ.get('VC_DEBUG_DUMP')
    if path and _STORE is not None:
        with open(path, 'wb') as f:
            pickle.dump(_STORE, f)
