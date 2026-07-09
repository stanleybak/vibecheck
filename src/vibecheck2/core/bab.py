"""Shared branch-and-bound core (design section 2.3).

One Domain, one refutation rule, one split->clamp materialization -- the
concepts the two frontier drivers (relu_split_bab's heap for the
thousands-of-rich-domains regime, input_split_bab's host-tensor frontier
for the millions-of-boxes regime) used to each re-implement. The two
loops remain because their scale regimes are genuinely different (like
LinMap's dense/patches/sparse layouts behind one interface), but the
Domain, the disjunct algebra, and the action->clamp contract are
single-sourced here.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class Domain(NamedTuple):
    """One BaB domain on a heap frontier: the design's Domain in
    heap-entry form. Tuple order is (lb, tick, ...) so heap comparisons
    resolve on the bound then the unique tick and never reach the payload.

    lb      parent-inherited worst bound (heap key: worst-first)
    tick    unique push counter (tie-break)
    splits  ((edge, j, sign|range), ...) -- the domain's accumulated
            actions; sign in {+1,-1} is a relu phase fix, a (lo,hi) tuple
            is a smooth/range split
    floor   parent per-query bounds (np array) | None -- a child is a
            SUBSET of its parent so the parent bound holds; flooring keeps
            bounds monotone down the tree
    betas   {(edge, j): optimized beta} | {} -- transferred split duals
    alphas  {edge: (qd, n) fp16 numpy} | None -- transferred relu slopes
    """
    lb: float
    tick: int
    splits: tuple
    floor: object = None
    betas: dict = {}
    alphas: object = None


def disjunct_selector(disj_idx, q, dev):
    """(D, sel) where sel[d, r] marks that query row r belongs to
    disjunct d. A disjunct is refuted when ANY of its rows is provably
    positive; the spec is unsat when every disjunct is refuted."""
    D = int(disj_idx.max()) + 1 if disj_idx.numel() else 0
    sel = torch.zeros(D, q, device=dev, dtype=torch.bool)
    if q:
        sel[disj_idx, torch.arange(q, device=disj_idx.device)] = True
    return D, sel


def refuted_matrix(lbq, bias, sel):
    """(B, D) bool: entry (b, d) true iff domain b refutes disjunct d,
    i.e. some row of d has a FINITE strictly-positive lower bound (a +inf
    bound is an arithmetic artifact, never a proof). Identical algebra in
    both drivers -- the single source of the unsat rule inside BaB."""
    dev = lbq.device
    B = lbq.shape[0]
    D = sel.shape[0]
    lbb = lbq + bias
    pos = (lbb > 0) & torch.isfinite(lbb)
    out = torch.zeros(B, D, device=dev, dtype=torch.bool)
    for d in range(D):
        out[:, d] = (pos & sel[d]).any(dim=1)
    return out


def materialize_clamps(batch_splits, n_of, B, dev):
    """Build (clamps, range_clamps) from a batch of domains' split lists.

    batch_splits: list (len B) of split tuples ((edge, j, spec), ...)
    n_of(edge) -> element count of that op.

    clamps[edge]        (B, n) int8 in {-1,0,+1}   relu sign fixes
    range_clamps[edge]  ((B,n) lo, (B,n) hi)        smooth-op range splits

    A relu split's spec is an int sign; a range split's spec is a
    (lo, hi) tuple. Multiple splits of the same neuron intersect (max lo,
    min hi) -- a child accumulates every ancestor's constraint.
    """
    clamps = {}
    range_clamps = {}
    for bi, splits in enumerate(batch_splits):
        for nm, j, spec in splits:
            if isinstance(spec, tuple):
                if nm not in range_clamps:
                    n_e = n_of(nm)
                    range_clamps[nm] = (
                        torch.full((B, n_e), -torch.inf, device=dev),
                        torch.full((B, n_e), torch.inf, device=dev))
                rlo, rhi = range_clamps[nm]
                rlo[bi, j] = max(float(rlo[bi, j]), spec[0])
                rhi[bi, j] = min(float(rhi[bi, j]), spec[1])
            else:
                if nm not in clamps:
                    clamps[nm] = torch.zeros(B, n_of(nm), device=dev,
                                             dtype=torch.int8)
                clamps[nm][bi, j] = spec
    return clamps, range_clamps


def merge_intermediates(base, reforward, B, keep_valid=False):
    """Intersect root/base pre-activation bounds (tight, clamp-blind) with
    a per-domain reforward (clamp-aware, looser at the root): best of both
    regimes. base/reforward map edge -> flat (lo, hi[, lo2, hi2 ...])
    tuples. keep_valid clamps each hi up to its merged lo so a
    floating-point inversion never produces an empty interval (the
    input-split driver's guard; the relu driver historically omitted it).
    """
    out = {}
    for k, v in base.items():
        rv = tuple(t.expand(B, -1) for t in v)
        iv = reforward[k]
        merged = []
        for j in range(0, len(rv), 2):
            lo = torch.maximum(rv[j], iv[j])
            hi = torch.minimum(rv[j + 1], iv[j + 1])
            if keep_valid:
                hi = torch.maximum(hi, lo)
            merged.append(lo)
            merged.append(hi)
        out[k] = tuple(merged)
    return out
