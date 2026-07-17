"""The one memory-budget service (design 2.1).

Every batched kernel in the core sizes its work through `chunked`. Predictive
sizing first (declared bytes/item vs free memory with a safety factor), then
ONE narrow OOM fallback: catch CUDA OOM here only, halve, log, retry, and
re-raise loudly at the floor. Nothing else in the core may catch OOM (CLAUDE.md).
"""
from __future__ import annotations

import sys

import torch

SAFETY = 0.5          # use at most this fraction of free memory per chunk
_MIN_CHUNK = 1        # below this, the OOM is real: re-raise


def free_bytes(device) -> int:
    dev = torch.device(device)
    if dev.type == 'cuda':
        free, _total = torch.cuda.mem_get_info(dev)
        # the caching allocator's reserved-but-unallocated blocks are
        # reusable by the next torch alloc, but mem_get_info counts them
        # as USED: after a few 1M-domain BaB rounds `reserved` grows to
        # most of the card, driver-free collapses, and chunk_size starves
        # every later batch (lsnc quadrotor2d_55: rounds fell to ~40k
        # domains/s from a 1M/s start while the allocator sat on reusable
        # cache; abcrown drains the same 14.2M-domain tree in 14.6s)
        cached = (torch.cuda.memory_reserved(dev)
                  - torch.cuda.memory_allocated(dev))
        return int(free) + max(0, int(cached))
    # CPU: keep chunks modest rather than probing the OS; 4 GB nominal.
    return 4 << 30


def chunk_size(n_items: int, bytes_per_item: float, device) -> int:
    """Predicted #items per chunk. Always in [1, n_items]."""
    if bytes_per_item <= 0:
        return n_items
    fit = int(free_bytes(device) * SAFETY / bytes_per_item)
    return max(1, min(n_items, fit))


def chunked_indices(fn, idx: torch.Tensor, bytes_per_item: float):
    """Apply `fn(index_chunk)` over chunks of an index vector, predictively
    sized with the same halve-on-OOM backstop as `chunked`. fn's outputs are
    the caller's to place (it typically scatters into a result); returns None.
    """
    n = idx.numel()
    cs = chunk_size(n, bytes_per_item, idx.device)
    i = 0
    while i < n:
        try:
            fn(idx[i:i + cs])
            i += cs
        except torch.cuda.OutOfMemoryError:
            if cs <= _MIN_CHUNK:
                raise
            torch.cuda.empty_cache()
            cs = max(_MIN_CHUNK, cs // 2)
            print(f'[memory] CUDA OOM at chunk={2*cs}; retrying with {cs}',
                  file=sys.stderr, flush=True)


def attempt(fn, tag: str):
    """Run `fn()` once; on CUDA OOM, log LOUDLY, free the cache, and return
    None so the caller skips its OPTIONAL phase and continues the pipeline.

    For whole-phase admission where no predictive estimate is faithful:
    the forward zonotope's true generator count depends on band widths
    that only the propagation itself knows (measured on vit: a shape
    estimate said 130 GiB, an interval-replay estimate said 130 GiB, the
    real pass took 10 GiB). Centralized here per the one-OOM-catch rule."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'[memory] CUDA OOM in optional phase {tag!r}; phase skipped',
              file=sys.stderr, flush=True)
        return None


def chunked(fn, X: torch.Tensor, bytes_per_item: float):
    """Apply `fn` over the leading dim of X in memory-budgeted chunks.

    fn maps (b, ...) -> (b, ...); results are concatenated on dim 0.
    The ONLY sanctioned CUDA-OOM catch in the core lives here.
    """
    n = X.shape[0]
    cs = chunk_size(n, bytes_per_item, X.device)
    outs = []
    i = 0
    while i < n:
        try:
            outs.append(fn(X[i:i + cs]))
            i += cs
        except torch.cuda.OutOfMemoryError:
            if cs <= _MIN_CHUNK:
                raise
            torch.cuda.empty_cache()
            cs = max(_MIN_CHUNK, cs // 2)
            print(f'[memory] CUDA OOM at chunk={2*cs}; retrying with {cs}',
                  file=sys.stderr, flush=True)
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
