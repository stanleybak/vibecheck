"""Attention-stack tightness and infinity-robustness (fast, CPU).

Covers the four mechanisms that make attention nets bound honestly:
  1. simplex hull: softmax @ V outputs live in the coordinatewise hull
     of V's rows (interval AND crown), not the token-count blowup;
  2. inf-safe interval: exp past fp32 max yields inf, never NaN, and
     the bounds still bracket sampled exact evaluations;
  3. no-poison refinement: intermediates_crown through non-finite
     planes is a no-op, never a NaN writer;
  4. non-finite dual states are refused at build (the NaN-certification
     false-unsat path stays dead).
"""
import numpy as np
import pytest
import torch

from vibecheck2.core import backward, dual_lp
from vibecheck2.core import forward as fwd
from vibecheck2.core.graph import Net, Op, tag_simplex_bmm
from vibecheck2.core.linmap import Dense, Scale, Select, SumAxis

RNG = np.random.default_rng(11)


def _softmax_np(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _attention_net(k=4, p=3):
    """input(4) -> logits(k*k) & V(k*p) -> softmax(k,k) @ V(k,p) -> out."""
    lmq = _d(4, k * k)
    lmv = _d(4, k * p)
    # difference-form softmax over rows of the (k, k) logits
    pi = np.arange(k)[:, None, None]
    ii = np.arange(k)[None, :, None]
    jj = np.arange(k)[None, None, :]
    idx_j = np.ascontiguousarray(
        np.broadcast_to(pi * k + jj, (k, k, k))).reshape(-1)
    idx_i = np.ascontiguousarray(
        np.broadcast_to(pi * k + ii, (k, k, k))).reshape(-1)
    n_d = k * k * k
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'q': Op('q', 'linmap', ('x',), (k * k,), k * k, lm=lmq),
        'v': Op('v', 'linmap', ('x',), (k * p,), k * p, lm=lmv),
        'sj': Op('sj', 'linmap', ('q',), (n_d,), n_d,
                 lm=Select(idx_j, k * k)),
        'si': Op('si', 'linmap', ('q',), (n_d,), n_d,
                 lm=Select(idx_i, k * k)),
        'ni': Op('ni', 'linmap', ('si',), (n_d,), n_d,
                 lm=Scale(-1.0, n_d)),
        'dd': Op('dd', 'add', ('sj', 'ni'), (n_d,), n_d),
        'e': Op('e', 'nonlin', ('dd',), (n_d,), n_d, fn='exp'),
        's': Op('s', 'linmap', ('e',), (k * k,), k * k,
                lm=SumAxis(k * k, k, 1)),
        'w': Op('w', 'nonlin', ('s',), (k * k,), k * k, fn='reciprocal',
                params={'out_lo': 0.0, 'out_hi': 1.0,
                        'softmax_axis_len': k, 'softmax_post': 1}),
        'av': Op('av', 'bmm', ('w', 'v'), (k * p,), k * p,
                 params={'a_shape': (k, k), 'b_shape': (k, p)}),
    }
    net = tag_simplex_bmm(Net(ops, [nm for nm in ops if nm != 'x'],
                              'x', 'av'))

    def ref(x):
        q = (x @ lmq.W.T + lmq.b).reshape(-1, k, k)
        v = (x @ lmv.W.T + lmv.b).reshape(-1, k, p)
        return (_softmax_np(q) @ v).reshape(-1, k * p)

    return net, ref, lmv


def _d(n_in, n_out):
    W = RNG.standard_normal((n_out, n_in)).astype(np.float32)
    b = RNG.standard_normal(n_out).astype(np.float32)
    return Dense(W, b)


def test_simplex_tag():
    net, _, _ = _attention_net()
    assert net.ops['av'].params.get('simplex_left') is True


def test_simplex_hull_interval_and_crown():
    net, ref, lmv = _attention_net()
    lo = -torch.ones(1, 4) * 0.5
    hi = torch.ones(1, 4) * 0.5
    xs = torch.tensor(RNG.uniform(-0.5, 0.5, (256, 4)).astype(np.float32))
    ys = ref(xs.numpy())

    il, ih = fwd.interval(net, lo, hi)
    # soundness
    assert (il.numpy()[0] <= ys.min(0) + 1e-4).all()
    assert (ih.numpy()[0] >= ys.max(0) - 1e-4).all()
    # tightness: within the hull of V's interval rows (the whole point --
    # an independent-[0,1]-weights treatment would exceed it k-fold)
    vl, vh = fwd.interval(net, lo, hi)  # net output IS av
    ivl, ivh = fwd.interval(_subnet_v(net), lo, hi)
    k, p = 4, 3
    hull_lo = ivl.numpy()[0].reshape(k, p).min(0)
    hull_hi = ivh.numpy()[0].reshape(k, p).max(0)
    out_lo = il.numpy()[0].reshape(k, p)
    out_hi = ih.numpy()[0].reshape(k, p)
    assert (out_lo >= np.tile(hull_lo, (k, 1)) - 1e-4).all()
    assert (out_hi <= np.tile(hull_hi, (k, 1)) + 1e-4).all()

    # crown: identity rows bracket and respect the hull too
    W = torch.eye(net.n_out)
    Wb = torch.cat([W, -W])
    lb = backward.crown(net, lo, hi, Wb).numpy()[0]
    cl, ch = lb[:net.n_out], -lb[net.n_out:]
    assert (cl <= ys.min(0) + 1e-4).all()
    assert (ch >= ys.max(0) - 1e-4).all()
    assert (cl.reshape(k, p) >= np.tile(hull_lo, (k, 1)) - 1e-3).all()
    assert (ch.reshape(k, p) <= np.tile(hull_hi, (k, 1)) + 1e-3).all()


def _subnet_v(net):
    ops = {'x': net.ops['x'], 'v': net.ops['v']}
    return Net(ops, ['v'], 'x', 'v')


def _exp_overflow_net():
    """dense scaled so the exp pre-activation range passes fp32 max."""
    W = (RNG.standard_normal((4, 4)) * 500).astype(np.float32)
    lm1 = Dense(W, np.zeros(4, dtype=np.float32))
    lm2 = _d(4, 2)
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'h': Op('h', 'linmap', ('x',), (4,), 4, lm=lm1),
        'e': Op('e', 'nonlin', ('h',), (4,), 4, fn='exp'),
        'y': Op('y', 'linmap', ('e',), (2,), 2, lm=lm2),
    }
    net = Net(ops, ['h', 'e', 'y'], 'x', 'y')

    def ref(x):
        return np.exp(np.clip(x @ W.T, -700, 700)) @ lm2.W.T + lm2.b

    return net, ref


def test_interval_inf_never_nan():
    net, ref = _exp_overflow_net()
    lo, hi = -torch.ones(1, 4), torch.ones(1, 4)
    l, h = fwd.interval(net, lo, hi)
    assert not torch.isnan(l).any() and not torch.isnan(h).any()
    # finite sampled outputs stay inside (inf bounds allowed, NaN not)
    xs = RNG.uniform(-0.02, 0.02, (64, 4)).astype(np.float32)
    ys = ref(xs)
    fin = np.isfinite(ys).all(axis=1)
    assert (l.numpy()[0] <= ys[fin].min(0) + 1e-3).all()
    assert (h.numpy()[0] >= ys[fin].max(0) - 1e-3).all()


def test_refinement_never_poisons():
    net, _ = _exp_overflow_net()
    lo, hi = -torch.ones(1, 4), torch.ones(1, 4)
    inter = backward.intermediates_crown(net, lo, hi)
    for nm, ent in inter.items():
        for t in ent:
            assert not torch.isnan(t).any(), f'NaN in inter[{nm}]'


def test_dual_refuses_nonfinite_state():
    net, _ = _exp_overflow_net()
    lo, hi = -torch.ones(1, 4), torch.ones(1, 4)
    inter = backward.intermediates_crown(net, lo, hi)
    slopes = {'e': torch.full((4,), 0.5)}
    with pytest.raises(NotImplementedError):
        dual_lp.build_state_backward(net, lo, hi, inter, slopes=slopes,
                                     device='cpu')
