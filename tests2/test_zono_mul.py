"""Correlation-exact zonotope mul: soundness (sampling VALIDATES, never
defines) and the quadratic-form tightness the box collapse cannot give.
"""
import numpy as np
import torch

from vibecheck2.core import forward
from vibecheck2.core.graph import Net, Op
from vibecheck2.core.linmap import Dense


def _quad_net(P):
    """y = sum(x .* (P x)) as a flat DAG: x -> p=Px -> m=x.*p -> y=1.m"""
    n = P.shape[0]
    ops = {
        'x': Op('x', 'input', (), (n,), n),
        'p': Op('p', 'linmap', ('x',), (n,), n,
                lm=Dense(P.astype(np.float32), np.zeros(n, np.float32))),
        'm': Op('m', 'mul', ('x', 'p'), (n,), n),
        'y': Op('y', 'linmap', ('m',), (1,), 1,
                lm=Dense(np.ones((1, n), np.float32),
                         np.zeros(1, np.float32))),
    }
    return Net(ops, ['p', 'm', 'y'], 'x', 'y')


def test_zono_mul_square_elementwise_nonneg():
    """z = x .* x over [-1, 1]^n must bound to ~[0, 1] per element (the
    old box collapse gave [-1, 1]: sign-symmetric garbage)."""
    n = 4
    ops = {
        'x': Op('x', 'input', (), (n,), n),
        'm': Op('m', 'mul', ('x', 'x'), (n,), n),
    }
    net = Net(ops, ['m'], 'x', 'm')
    lo = -torch.ones(1, n)
    hi = torch.ones(1, n)
    zlo, zhi = forward.zono(net, lo, hi)
    assert float(zlo.min()) >= -1e-6, zlo
    assert float(zhi.max()) <= 1.0 + 1e-6, zhi


def test_zono_mul_quadratic_form_sum_tight():
    """y = x^T P x with P PSD is >= 0; the correlated zono must keep the
    sum's lower bound near 0 where the box collapse scales with -n."""
    rng = np.random.default_rng(7)
    n = 6
    A = rng.normal(size=(n, n))
    P = (A @ A.T / n).astype(np.float32)          # PSD
    net = _quad_net(P)
    lo = -torch.ones(1, n)
    hi = torch.ones(1, n)
    zlo, _ = forward.zono(net, lo, hi)
    # the old box collapse floor: per element the corner product reaches
    # -sum_j |P_nj| (a is [-1,1], b centered), so the summed lower bound
    # is -sum |P|. The correlated form keeps the e^2 diagonal exactly and
    # must be strictly tighter (off-diagonal cross terms stay boxed, as
    # in v1's elementwise algebra).
    box_floor = -float(np.abs(P).sum())
    assert float(zlo[0, 0]) > 0.75 * box_floor, (float(zlo[0, 0]), box_floor)
    # soundness vs sampled minimum (validation only)
    xs = torch.rand(20000, n) * 2 - 1
    ys = (xs * (xs @ torch.tensor(P).T)).sum(dim=1)
    assert float(zlo[0, 0]) <= float(ys.min()) + 1e-5


def test_zono_mul_sound_on_random_correlated_branches():
    """Random mul of two correlated affine branches: zono bounds must
    bracket every sampled output (both dense and box-remainder modes)."""
    rng = np.random.default_rng(3)
    n = 5
    W1 = rng.normal(size=(n, n)).astype(np.float32)
    W2 = rng.normal(size=(n, n)).astype(np.float32)
    b1 = rng.normal(size=n).astype(np.float32)
    b2 = rng.normal(size=n).astype(np.float32)
    ops = {
        'x': Op('x', 'input', (), (n,), n),
        'a': Op('a', 'linmap', ('x',), (n,), n, lm=Dense(W1, b1)),
        'b': Op('b', 'linmap', ('x',), (n,), n, lm=Dense(W2, b2)),
        'm': Op('m', 'mul', ('a', 'b'), (n,), n),
    }
    net = Net(ops, ['a', 'b', 'm'], 'x', 'm')
    lo = torch.tensor([[-1.0, -0.5, 0.0, -2.0, 0.3]])
    hi = torch.tensor([[0.5, 1.0, 0.7, -1.0, 0.9]])
    xs = lo + (hi - lo) * torch.rand(50000, n)
    ys = (xs @ torch.tensor(W1).T + torch.tensor(b1)) \
        * (xs @ torch.tensor(W2).T + torch.tensor(b2))
    for br in (False, 'all'):
        zlo, zhi = forward.zono(net, lo, hi, box_remainder=br)
        assert bool((zlo[0] <= ys.min(dim=0).values + 1e-4).all()), br
        assert bool((zhi[0] >= ys.max(dim=0).values - 1e-4).all()), br
