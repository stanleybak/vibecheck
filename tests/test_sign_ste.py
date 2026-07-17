"""Sign-net STE plumbing (traffic_signs QConv class).

The STE/pass-through backward is a GRADIENT heuristic for the attack --
verdicts still flow through the ORT chokepoint -- but the graph tagging
and the bound relaxations must stay exact: planes/band for sign are
verified against samples here.
"""
import numpy as np
import pytest
import torch

from vibecheck.core import forward
from vibecheck.core.graph import Net, Op, _tag_merged_signs
from vibecheck.core import linmap as lm
from vibecheck.core.relax import SignFn


def _sign_merge_net():
    """input -> dense -> sign -> scaleshift(+1) -> sign -> dense."""
    n = 4
    rng = np.random.default_rng(0)
    ops = {
        'in': Op('in', 'input', (), n=n),
        'd1': Op('d1', 'linmap', ('in',), n=n,
                 lm=lm.Dense(rng.normal(size=(n, n)).astype(np.float32),
                             np.zeros(n, np.float32))),
        's1': Op('s1', 'nonlin', ('d1',), n=n, fn='sign'),
        'add': Op('add', 'linmap', ('s1',), n=n,
                  lm=lm.ScaleShift(np.ones(n, np.float32),
                                   np.ones(n, np.float32), n)),
        's2': Op('s2', 'nonlin', ('add',), n=n, fn='sign'),
        'd2': Op('d2', 'linmap', ('s2',), n=n,
                 lm=lm.Dense(rng.normal(size=(n, n)).astype(np.float32),
                             np.zeros(n, np.float32))),
    }
    order = [k for k in ops if k != 'in']   # topo order excludes the input
    net = Net(ops=ops, order=order, input_name='in', output_name='d2')
    return net


def test_merged_sign_tagged_and_grad_flows():
    net = _sign_merge_net()
    _tag_merged_signs(net)
    assert not net.ops['s1'].params.get('ste_pass')
    assert net.ops['s2'].params.get('ste_pass')
    x = (torch.randn(8, 4) * 10).requires_grad_(True)
    y = forward.point(net, x)
    y.sum().backward()
    # the second sign's input is +-1 + 1 in {0, 2}: a clipped STE there
    # zeroes everything; pass-through keeps the chain alive
    assert float(x.grad.abs().max()) > 0


def test_sign_planes_band_bracket_samples():
    fn = SignFn()
    lo = torch.tensor([[-2.0, -0.5, 0.3, -4.0]])
    hi = torch.tensor([[-0.1, 1.5, 2.0, 4.0]])
    sl, bl, su, bu = fn.planes(lo, hi)
    lam, mu, delta = fn.band(lo, hi)
    for t in torch.linspace(0, 1, 17):
        x = lo + t * (hi - lo)
        y = torch.sign(x)
        assert bool((y >= sl * x + bl - 1e-6).all())
        assert bool((y <= su * x + bu + 1e-6).all())
        assert bool((y >= lam * x + mu - delta - 1e-6).all())
        assert bool((y <= lam * x + mu + delta + 1e-6).all())


def test_adaptive_ste_grad_scales_with_preact():
    fn = SignFn()
    # pre-acts in the hundreds (QConv scale): the fixed |x|<=1 window
    # had zero live crossings; the adaptive window keeps some alive
    x = (torch.randn(64) * 300).requires_grad_(True)
    y = fn.point(x, {'ste_frac': 0.5})
    y.sum().backward()
    assert float(x.grad.abs().max()) > 0
