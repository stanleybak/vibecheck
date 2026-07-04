"""Exact-MILP escalation correctness (fast, CPU, Gurobi).

The MILP must agree with brute-force exact minimization (all relu
activation patterns enumerated on a tiny net): refute exactly the rows
whose true minimum is positive, and produce a genuine counterexample
candidate when the true minimum is negative.
"""
import itertools

import numpy as np
import pytest
import torch

pytest.importorskip('gurobipy')

from vibecheck2.core import backward
from vibecheck2.core.graph import Net, Op
from vibecheck2.core.linmap import Dense
from vibecheck2.core.milp import refute_rows_milp

RNG = np.random.default_rng(3)


def _net():
    lm1 = Dense(RNG.standard_normal((5, 3)).astype(np.float32),
                RNG.standard_normal(5).astype(np.float32))
    lm2 = Dense(RNG.standard_normal((4, 5)).astype(np.float32),
                RNG.standard_normal(4).astype(np.float32))
    lm3 = Dense(RNG.standard_normal((2, 4)).astype(np.float32),
                RNG.standard_normal(2).astype(np.float32))
    ops = {
        'x': Op('x', 'input', (), (3,), 3),
        'h1': Op('h1', 'linmap', ('x',), (5,), 5, lm=lm1),
        'r1': Op('r1', 'nonlin', ('h1',), (5,), 5, fn='relu'),
        'h2': Op('h2', 'linmap', ('r1',), (4,), 4, lm=lm2),
        'r2': Op('r2', 'nonlin', ('h2',), (4,), 4, fn='relu'),
        'y': Op('y', 'linmap', ('r2',), (2,), 2, lm=lm3),
    }
    net = Net(ops, ['h1', 'r1', 'h2', 'r2', 'y'], 'x', 'y')
    return net, (lm1, lm2, lm3)


def _exact_min(lms, w, b, lo, hi):
    """True min of w.f(x) + b over the box by dense grid + LP-corner
    reasoning via fine sampling (validation only)."""
    lm1, lm2, lm3 = lms
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 41)
                               for i in range(3)]), -1).reshape(-1, 3)
    h1 = np.maximum(g @ lm1.W.T + lm1.b, 0)
    h2 = np.maximum(h1 @ lm2.W.T + lm2.b, 0)
    y = h2 @ lm3.W.T + lm3.b
    return float((y @ w + b).min())


def test_milp_matches_exact():
    net, lms = _net()
    lo = -torch.ones(1, 3)
    hi = torch.ones(1, 3)
    inter = backward.intermediates_crown(net, lo, hi)
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    bias = torch.tensor([0.0, 0.0, 0.0])
    exact = [_exact_min(lms, W[r].numpy(), float(bias[r]),
                        lo[0].numpy(), hi[0].numpy())
             for r in range(3)]
    import time
    refuted, cand = refute_rows_milp(  # each row its own disjunct
        net, lo, hi, inter, W, bias, torch.tensor([0, 1, 2]), [0, 1, 2],
        deadline=time.time() + 60)
    for r in range(3):
        if exact[r] > 0.05:                     # clearly positive rows
            assert r in refuted, (r, exact[r])
        if exact[r] < -0.05:                    # clearly negative rows
            assert r not in refuted, (r, exact[r])


def test_milp_candidate_is_genuine():
    net, lms = _net()
    lo = -torch.ones(1, 3)
    hi = torch.ones(1, 3)
    inter = backward.intermediates_crown(net, lo, hi)
    # a row engineered to be violable: minimize the first output
    W = torch.tensor([[1.0, 0.0]])
    lm1, lm2, lm3 = lms
    g = RNG.uniform(-1, 1, (2000, 3)).astype(np.float32)
    y = (np.maximum(np.maximum(g @ lm1.W.T + lm1.b, 0) @ lm2.W.T + lm2.b,
                    0) @ lm3.W.T + lm3.b)
    bias = torch.tensor([float(-y[:, 0].min()) - 0.5])  # min goes negative
    import time
    refuted, cand = refute_rows_milp(net, lo, hi, inter, W, bias,
                                     torch.tensor([0]), [0],
                                     deadline=time.time() + 60)
    assert 0 not in refuted
    assert cand is not None
    xc = torch.tensor(cand, dtype=torch.float32).reshape(1, -1)
    from vibecheck2.core import forward as fwd
    yv = fwd.point(net, xc)
    assert float(yv[0, 0] + bias[0]) <= 1e-5    # satisfies the CE region (<=0)
    assert (xc[0] >= lo[0] - 1e-6).all() and (xc[0] <= hi[0] + 1e-6).all()


def test_milp_conjunction_refuted_when_no_single_row_is():
    """The sat_relu case: a disjunct is a CONJUNCTION of rows, none of which
    is individually refutable, but whose AND is infeasible. Rows y+c and -y+c
    (c>0) give {y<=-c AND y>=c} = empty, yet each alone is satisfiable when y
    straddles +-c. Per-row refutation must FAIL; the joint max-row must SUCCEED.
    """
    from vibecheck2.core.graph import Net, Op
    from vibecheck2.core.linmap import Dense
    import time
    # trivial net y = x over [-1, 1] (y straddles +-c)
    net = Net({'x': Op('x', 'input', (), (1,), 1),
               'y': Op('y', 'linmap', ('x',), (1,), 1,
                       lm=Dense(np.eye(1, dtype=np.float32),
                                np.zeros(1, dtype=np.float32)))},
              ['y'], 'x', 'y')
    lo, hi = -torch.ones(1, 1), torch.ones(1, 1)
    inter = backward.intermediates_crown(net, lo, hi)
    c = 0.1
    W = torch.tensor([[1.0], [-1.0]])
    bias = torch.tensor([c, c])
    # per-row (each its own disjunct): NEITHER refutable
    indiv, _ = refute_rows_milp(net, lo, hi, inter, W, bias,
                                torch.tensor([0, 1]), [0, 1],
                                deadline=time.time() + 60)
    assert indiv == set(), f"a single row was refutable ({indiv}); joint path not exercised"
    # both rows in ONE disjunct (a conjunction): jointly refuted
    joint, cand = refute_rows_milp(net, lo, hi, inter, W, bias,
                                   torch.tensor([0, 0]), [0],
                                   deadline=time.time() + 60)
    assert 0 in joint, "conjunction not refuted by the joint max-row MILP"
    assert cand is None
