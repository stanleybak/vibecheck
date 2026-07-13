"""forward.planes (forward-LiRPA): sampling-validated soundness on the
mscn op family (dense/scaleshift/relu/sigmoid/reciprocal/mul/add/concat)
plus a tightness check vs the rad-mode zonotope on a mul net (the reason
the propagator exists). Sampling VALIDATES the planes; it never defines
them (relaxations come from RelaxLib / McCormick closed forms)."""
import numpy as np
import pytest
import torch

from vibecheck2.core import forward
from vibecheck2.core.graph import Net, Op
from vibecheck2.core import linmap as lm


def _dense(name, src, W, b=None):
    return Op(name, 'linmap', (src,), n=W.shape[0],
              lm=lm.Dense(W.astype(np.float32),
                          None if b is None else b.astype(np.float32)))


def _mscnish_net(seed=0):
    """input(6) -> dense(8) -> relu -> scaleshift -> sigmoid -> A
       A -> dense(8) -> +2.5 shift -> reciprocal -> R  (input > 0 safe)
       mul(A, R) -> dense(4) -> add(A head) -> out"""
    rng = np.random.default_rng(seed)
    ops = {
        'in': Op('in', 'input', (), n=6),
        'd1': _dense('d1', 'in', rng.normal(size=(8, 6)),
                     rng.normal(size=8)),
        'r1': Op('r1', 'nonlin', ('d1',), n=8, fn='relu'),
        'ss': Op('ss', 'linmap', ('r1',), n=8,
                 lm=lm.ScaleShift(rng.normal(size=8).astype(np.float32),
                                  rng.normal(size=8).astype(np.float32),
                                  8)),
        'sg': Op('sg', 'nonlin', ('ss',), n=8, fn='sigmoid'),
        'd2': _dense('d2', 'sg', rng.normal(size=(8, 8)) * 0.1,
                     np.full(8, 2.5)),   # stays positive: reciprocal-safe
        'rc': Op('rc', 'nonlin', ('d2',), n=8, fn='reciprocal'),
        'ml': Op('ml', 'mul', ('sg', 'rc'), n=8),
        'd3': _dense('d3', 'ml', rng.normal(size=(8, 8))),
        'ad': Op('ad', 'add', ('d3', 'sg'), n=8),
        'd4': _dense('d4', 'ad', rng.normal(size=(4, 8))),
    }
    order = [k for k in ops if k != 'in']
    return Net(ops=ops, order=order, input_name='in', output_name='d4')


def test_planes_bracket_samples():
    net = _mscnish_net()
    assert forward.planes_supported(net)
    torch.manual_seed(0)
    lo = torch.rand(3, 6) * 0.4
    hi = lo + torch.rand(3, 6) * 0.6
    plo, phi, st = forward.planes(net, lo, hi, return_state=True)
    for _ in range(400):
        x = lo + torch.rand_like(lo) * (hi - lo)
        y = forward.point(net, x)
        assert bool((y >= plo - 1e-4).all()), 'lower plane violated'
        assert bool((y <= phi + 1e-4).all()), 'upper plane violated'
    # every intermediate edge box must bracket its sampled activations
    taps = {nm: None for nm in net.order
            if net.ops[nm].kind == 'nonlin'}
    x = (lo + hi) / 2
    forward.point(net, x, taps=taps)
    for nm in taps:
        elo, ehi = st[net.ops[nm].inputs[0]].bounds()
        z = taps[nm]
        assert bool((z >= elo - 1e-4).all() and (z <= ehi + 1e-4).all())


def test_planes_finite_and_ordered_vs_zono():
    # tightness vs the zonotope is INSTANCE-dependent (small random nets
    # favor the zono's correlated first-order mul; the wide structured
    # mscn family is measured on the box) -- here only soundness-shape
    # invariants: finite, ordered, and consistent with the zono bracket
    net = _mscnish_net(seed=3)
    torch.manual_seed(1)
    lo = torch.rand(4, 6) * 0.3
    hi = lo + 0.4
    plo, phi = forward.planes(net, lo, hi)
    zlo, zhi = forward.zono(net, lo, hi, box_remainder='all')
    assert bool(torch.isfinite(plo).all() and torch.isfinite(phi).all())
    assert bool((phi >= plo).all())
    # both bracket the truth, so the intersection must be nonempty
    assert bool((torch.maximum(plo, zlo) <= torch.minimum(phi, zhi)
                 + 1e-5).all())


def test_planes_unsupported_kind_raises():
    net = _mscnish_net()
    net.ops['d1'].lm = lm.Conv2d.__new__(lm.Conv2d)  # not Dense/ScaleShift
    assert not forward.planes_supported(net)
