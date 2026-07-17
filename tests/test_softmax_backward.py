"""Difference-form softmax backward-CROWN: soundness vs sampling (which
VALIDATES, never defines) and alpha improvement."""
import numpy as np
import torch

from vibecheck.core import backward
from vibecheck.core.graph import Net, Op
from vibecheck.core.linmap import Dense


def _softmax_net(rng, n_in=6, pre=3, k=4):
    n = pre * k
    W1 = rng.normal(size=(n, n_in)).astype(np.float32)
    b1 = rng.normal(size=n).astype(np.float32)
    W2 = rng.normal(size=(5, n)).astype(np.float32)
    ops = {
        'x': Op('x', 'input', (), (n_in,), n_in),
        'a': Op('a', 'linmap', ('x',), (n,), n, lm=Dense(W1, b1)),
        's': Op('s', 'nonlin', ('a',), (n,), n, fn='softmax',
                params={'pre': pre, 'k': k, 'post': 1}),
        'o': Op('o', 'linmap', ('s',), (5,), 5, lm=Dense(W2, None)),
    }
    return Net(ops, ['a', 's', 'o'], 'x', 'o'), W1, b1, W2


def test_softmax_backward_sound_and_alpha_improves():
    rng = np.random.default_rng(3)
    net, W1, b1, W2 = _softmax_net(rng)
    lo, hi = -torch.ones(1, 6) * 0.8, torch.ones(1, 6) * 0.8
    W = torch.eye(5)
    inter = backward.intermediates(net, lo, hi)
    lb = backward.crown(net, lo, hi, W, inter)
    lba = backward.alpha_crown(net, lo, hi, W, inter, iters=40)
    xs = lo + (hi - lo) * torch.rand(100000, 6)
    a = xs @ torch.tensor(W1).T + torch.tensor(b1)
    sm = torch.softmax(a.reshape(-1, 3, 4), dim=2).reshape(-1, 12)
    mins = (sm @ torch.tensor(W2).T).min(dim=0).values
    assert bool((lb[0] <= mins + 1e-4).all())
    assert bool((torch.maximum(lb, lba)[0] <= mins + 1e-4).all())
    # alpha must actually move the composite (fixed planes left the
    # attention stack invisible to the optimizer)
    assert float((lba - lb).max()) > 0.1
