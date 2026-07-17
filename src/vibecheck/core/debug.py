"""Env-gated debug dump for step-by-step comparison against v1.

Set VC2_DEBUG_DUMP=<path.pkl> and the pipeline records a dict of
intermediate artifacts (root intermediate bounds per nonlin edge, root
spec lower bounds + query rows, the input linearization (A, b), the
input-split trajectory: chosen split dims and queue sizes). v1 writes
the same schema under VC_DEBUG_DUMP via vibecheck/debug_dump.py;
scratch/clean_slate/compare_debug.py aligns and diffs the two.

Zero cost when the env var is unset (every hook is a no-op behind one
dict lookup)."""
from __future__ import annotations

import atexit
import os
import pickle

_STORE = None


def enabled():
    return bool(os.environ.get('VC2_DEBUG_DUMP'))


def add(key, value):
    """Record one artifact. Tensors are detached to CPU numpy; lists
    accumulate under list-valued keys via add_seq."""
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
    path = os.environ.get('VC2_DEBUG_DUMP')
    if path and _STORE is not None:
        with open(path, 'wb') as f:
            pickle.dump(_STORE, f)
