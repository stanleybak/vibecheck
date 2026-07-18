"""Closed-form box + 1-halfspace LP (core/box_halfspace).

lagrangian_min/max claim the EXACT LP optimum of d.e + c0 over
e in [-1,1]^n, a.e <= beta; scipy linprog is the independent referee.
tighten_layer must only ever shrink bounds and never cut off a feasible
point (sampling soundness).
"""
import numpy as np
from scipy.optimize import linprog

from vibecheck.core.box_halfspace import (
    lagrangian_max, lagrangian_min, tighten_layer)

_rng = np.random.default_rng(3)


def _lp_min(d, c0, a, beta):
    n = len(d)
    r = linprog(d, A_ub=a.reshape(1, -1), b_ub=[beta],
                bounds=[(-1, 1)] * n, method='highs')
    assert r.status == 0, r.message
    return r.fun + c0


def test_lagrangian_min_max_match_lp_exactly():
    for _ in range(25):
        n = int(_rng.integers(2, 9))
        d = _rng.normal(size=n)
        a = _rng.normal(size=n)
        c0 = float(_rng.normal())
        # beta in a range where the halfspace actually cuts the box but
        # stays feasible: a.e over the box spans +-sum|a|
        span = np.abs(a).sum()
        beta = float(_rng.uniform(-0.6 * span, 0.9 * span))
        got_min = lagrangian_min(d, c0, a, beta)
        assert np.isclose(got_min, _lp_min(d, c0, a, beta), atol=1e-8)
        got_max = lagrangian_max(d, c0, a, beta)
        assert np.isclose(got_max, -_lp_min(-d, -c0, a, beta), atol=1e-8)


def test_lagrangian_min_redundant_halfspace_is_box_min():
    d = np.array([1.0, -2.0, 0.5])
    a = np.array([1.0, 1.0, 1.0])
    box_min = 4.0 - np.abs(d).sum()
    assert np.isclose(lagrangian_min(d, 4.0, a, beta=100.0), box_min)
    # a = 0 rows: constraint vacuous (0 <= beta)
    assert np.isclose(lagrangian_min(d, 4.0, np.zeros(3), beta=1.0), box_min)


def test_lagrangian_min_zero_objective():
    a = np.array([1.0, -1.0])
    assert np.isclose(lagrangian_min(np.zeros(2), 2.5, a, beta=0.3), 2.5)


def test_tighten_layer_sound_and_monotone():
    """Tightened bounds contain every sampled zonotope point satisfying the
    halfspace, never loosen, and only unstable neurons move."""
    n_out, g = 6, 5
    c = _rng.normal(size=n_out)
    G = _rng.normal(size=(n_out, g))
    c[0] = np.abs(G[0]).sum() + 0.2         # neuron 0 genuinely stable-positive
    lo = c - np.abs(G).sum(1)               # exact zonotope box bounds
    hi = c + np.abs(G).sum(1)
    a = _rng.normal(size=g)
    beta = 0.2 * np.abs(a).sum()
    new_lo, new_hi = tighten_layer(c, G, lo, hi, a, beta)
    assert (new_lo >= lo - 1e-12).all() and (new_hi <= hi + 1e-12).all()
    assert new_lo[0] == lo[0] and new_hi[0] == hi[0]
    e = _rng.uniform(-1, 1, size=(4000, g))
    e = e[e @ a <= beta]
    z = c[None, :] + e @ G.T
    assert (z >= new_lo[None, :] - 1e-9).all()
    assert (z <= new_hi[None, :] + 1e-9).all()


def test_tighten_layer_pads_new_generators():
    """n_gens > G columns: appended generators can't affect this layer, so
    results equal the unpadded call."""
    n_out, g = 3, 4
    c = _rng.normal(size=n_out)
    G = _rng.normal(size=(n_out, g))
    lo = c - np.abs(G).sum(1)
    hi = c + np.abs(G).sum(1)
    a_small = _rng.normal(size=g)
    beta = 0.1
    ref = tighten_layer(c, G, lo, hi, a_small, beta)
    a_pad = np.concatenate([a_small, np.zeros(3)])
    got = tighten_layer(c, G, lo, hi, a_pad, beta, n_gens=g + 3)
    assert np.allclose(ref[0], got[0]) and np.allclose(ref[1], got[1])
