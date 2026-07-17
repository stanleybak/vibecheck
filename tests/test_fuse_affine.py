"""fuse_affine: single-consumer linmap chains compose into one Dense.
Function preserved (fp tolerance), bounds machinery agrees, big maps and
multi-consumer forks untouched.
"""
import numpy as np
import torch

from vibecheck.core import backward, forward
from vibecheck.core.graph import Net, Op
from vibecheck.core.graph_opt import fuse_affine
from vibecheck.core.linmap import Dense


def _chain_net():
    rng = np.random.default_rng(11)
    def d(m, n):
        return Dense(rng.normal(size=(m, n)).astype(np.float32),
                     rng.normal(size=m).astype(np.float32))
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'a': Op('a', 'linmap', ('x',), (8,), 8, lm=d(8, 4)),
        'b': Op('b', 'linmap', ('a',), (6,), 6, lm=d(6, 8)),
        'r': Op('r', 'nonlin', ('b',), (6,), 6, fn='relu'),
        'c': Op('c', 'linmap', ('r',), (5,), 5, lm=d(5, 6)),
        'e': Op('e', 'linmap', ('c',), (3,), 3, lm=d(3, 5)),
        'f': Op('f', 'linmap', ('e',), (2,), 2, lm=d(2, 3)),
    }
    return Net(ops, ['a', 'b', 'r', 'c', 'e', 'f'], 'x', 'f')


def test_fuse_affine_composes_chains_exactly():
    net = _chain_net()
    xs = torch.rand(64, 4) * 2 - 1
    y0 = forward.point(net, xs)
    n_ops0 = len(net.order)
    net = fuse_affine(net)
    # a->b fuses into one map; c->e->f fuses into one map: 6 -> 3 ops
    assert len(net.order) == 3, net.order
    y1 = forward.point(net, xs)
    assert float((y0 - y1).abs().max()) < 1e-4
    lo, hi = -torch.ones(1, 4), torch.ones(1, 4)
    W = torch.eye(2)
    lb = backward.crown(net, lo, hi, W)
    # soundness of the fused bounds vs sampled outputs (validation only)
    xs2 = lo + (hi - lo) * torch.rand(20000, 4)
    ys = forward.point(net, xs2)
    assert bool((lb[0] <= ys.min(dim=0).values + 1e-5).all())


def test_fuse_affine_fork_collapses_exactly():
    rng = np.random.default_rng(5)
    def d(m, n):
        return Dense(rng.normal(size=(m, n)).astype(np.float32), None)
    # 'a' forks into 'b' and 'c', which re-join in 's': the add rule fuses
    # b+c into one map of 'a', then the chain rule folds 'a' -- one linmap,
    # same function
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'a': Op('a', 'linmap', ('x',), (4,), 4, lm=d(4, 4)),
        'b': Op('b', 'linmap', ('a',), (4,), 4, lm=d(4, 4)),
        'c': Op('c', 'linmap', ('a',), (4,), 4, lm=d(4, 4)),
        's': Op('s', 'add', ('b', 'c'), (4,), 4),
    }
    net = Net(ops, ['a', 'b', 'c', 's'], 'x', 's')
    xs = torch.rand(64, 4) * 2 - 1
    y0 = forward.point(net, xs)
    net = fuse_affine(net)
    assert len(net.order) == 1, net.order
    assert float((y0 - forward.point(net, xs)).abs().max()) < 1e-4
    # oversized maps: untouched
    ops2 = {
        'x': Op('x', 'input', (), (4,), 4),
        'a': Op('a', 'linmap', ('x',), (4,), 4, lm=d(4, 4)),
        'b': Op('b', 'linmap', ('a',), (4,), 4, lm=d(4, 4)),
    }
    net2 = Net(ops2, ['a', 'b'], 'x', 'b')
    net2 = fuse_affine(net2, max_n=2)
    assert 'a' in net2.ops


def test_fuse_affine_add_concat_same_source():
    """add/concat of single-consumer linmaps of one shared source fuse to
    a single linmap, then chain-fuse further; function preserved."""
    rng = np.random.default_rng(9)
    def d(m, n):
        return Dense(rng.normal(size=(m, n)).astype(np.float32),
                     rng.normal(size=m).astype(np.float32))
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'a': Op('a', 'linmap', ('x',), (5,), 5, lm=d(5, 4)),
        'b': Op('b', 'linmap', ('x',), (5,), 5, lm=d(5, 4)),
        's': Op('s', 'add', ('a', 'b'), (5,), 5),
        'c': Op('c', 'linmap', ('s',), (3,), 3, lm=d(3, 5)),
    }
    net = Net(ops, ['a', 'b', 's', 'c'], 'x', 'c')
    xs = torch.rand(64, 4) * 2 - 1
    y0 = forward.point(net, xs)
    net = fuse_affine(net)
    assert len(net.order) == 1, net.order       # everything collapses
    y1 = forward.point(net, xs)
    assert float((y0 - y1).abs().max()) < 1e-4
