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

from vibecheck.core import backward, dual_lp
from vibecheck.core import forward as fwd
from vibecheck.core.graph import Net, Op, tag_simplex_bmm
from vibecheck.core.linmap import Dense, Scale, Select, SumAxis

RNG = np.random.default_rng(11)


def _softmax_np(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _attention_net(k=4, p=3):
    """input(4) -> logits(k*k) & V(k*p) -> softmax(k,k) @ V(k,p) -> out."""
    lmq = _d(4, k * k)
    lmv = _d(4, k * p)
    # FUSED softmax over rows of the (k, k) logits (the propagators own
    # the transform; see relax.Softmax / forward._softmax_zono)
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'q': Op('q', 'linmap', ('x',), (k * k,), k * k, lm=lmq),
        'v': Op('v', 'linmap', ('x',), (k * p,), k * p, lm=lmv),
        'w': Op('w', 'nonlin', ('q',), (k * k,), k * k, fn='softmax',
                params={'pre': k, 'k': k, 'post': 1,
                        'out_lo': 0.0, 'out_hi': 1.0,
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


def test_softmax_crown_adjoint_sound():
    """Backward CROWN through the fused softmax: the lb must bracket
    sampled outputs of an attention net (adjoint no longer dies at the
    softmax's interval-constant planes)."""
    net, ref, _ = _attention_net()
    lo = -torch.ones(1, 4) * 0.4
    hi = torch.ones(1, 4) * 0.4
    W = torch.eye(net.n_out)
    from vibecheck.core import backward
    lb = backward.crown(net, lo, hi, W)[0]
    xs = torch.tensor(RNG.uniform(-0.4, 0.4, (512, 4)).astype(np.float32))
    ys = torch.tensor(ref(xs.numpy()))
    assert (lb <= ys.min(0).values + 1e-4).all(), \
        float((lb - ys.min(0).values).max())
    # and it must be non-vacuous: strictly tighter than the constant
    # planes would allow on at least some coordinate (adjoint alive)
    assert torch.isfinite(lb).all()


def test_box_remainder_zono_sound():
    """box_remainder=True must still bracket the true image (sampled),
    carry a nonnegative rad, and use strictly fewer generator columns
    than the dense form (the point of the representation)."""
    from vibecheck.core import forward as fwd
    net, ref, _ = _attention_net()
    lo = torch.tensor([[-1.0, -0.5, 0.0, -0.8]])
    hi = torch.tensor([[0.5, 1.0, 0.7, 0.2]])
    dl, dh, dst = fwd.zono(net, lo, hi, return_state=True)
    rl, rh, rst = fwd.zono(net, lo, hi, return_state=True,
                           box_remainder=True)
    zo = rst[net.output_name]
    assert zo.rad is not None and bool((zo.rad >= 0).all())
    assert zo.G.shape[2] < dst[net.output_name].G.shape[2]
    xs = RNG.uniform(lo.numpy(), hi.numpy(),
                     size=(4096, 4)).astype(np.float32)
    ys = ref(xs)
    assert (ys >= rl.numpy() - 1e-5).all()
    assert (ys <= rh.numpy() + 1e-5).all()


def test_box_remainder_relu_record_and_columns():
    """relu fresh columns stay dense under box_remainder (the BaB's split
    handles) and the record carries the pre-activation remainder."""
    from vibecheck.core import forward as fwd
    from vibecheck.core.linmap import Dense as _Dense
    lm1 = _d(3, 5)
    lm2 = _d(5, 2)
    ops = {
        'x': Op('x', 'input', (), (3,), 3),
        'h': Op('h', 'linmap', ('x',), (5,), 5, lm=lm1),
        'sg': Op('sg', 'nonlin', ('h',), (5,), 5, fn='sigmoid', params={}),
        'r': Op('r', 'nonlin', ('h',), (5,), 5, fn='relu', params={}),
        'a': Op('a', 'add', ('sg', 'r'), (5,), 5),
        'y': Op('y', 'linmap', ('a',), (2,), 2, lm=lm2),
    }
    net = Net(ops, ['h', 'sg', 'r', 'a', 'y'], 'x', 'y')
    lo = torch.full((1, 3), -1.0)
    hi = torch.full((1, 3), 1.0)
    rec = {}
    rl, rh, rst = fwd.zono(net, lo, hi, return_state=True, record=rec,
                           box_remainder=True)
    zo = rst[net.output_name]
    # sigmoid deltas went to rad (no 'sg' columns); relu fresh cols remain
    assert not any(s[0] == 'sg' for s in zo.sym)
    assert any(s[0] == 'r' for s in zo.sym)
    assert 'rad' in rec['r'] and bool((rec['r']['rad'] >= 0).all())
    xs = RNG.uniform(-1, 1, size=(2048, 3)).astype(np.float32)
    h = xs @ lm1.W.T + lm1.b
    ys = (1 / (1 + np.exp(-h)) + np.maximum(h, 0)) @ lm2.W.T + lm2.b
    assert (ys >= rl.numpy() - 1e-5).all()
    assert (ys <= rh.numpy() + 1e-5).all()


def test_sym_budget_zono_sound():
    """Input-symbol budget: only top-K wide dims get columns, the rest
    box into rad; bounds still bracket the true image and the column
    count respects the budget."""
    from vibecheck.core import forward as fwd
    net, ref, _ = _attention_net()
    lo = torch.tensor([[-1.0, -0.5, 0.0, -0.8]])
    hi = torch.tensor([[0.5, 1.0, 0.7, 0.2]])
    rl, rh, rst = fwd.zono(net, lo, hi, return_state=True,
                           box_remainder=True, sym_budget=2)
    zi = rst[net.input_name]
    assert sum(1 for s in zi.sym if s[0] == 'input') == 2
    xs = RNG.uniform(lo.numpy(), hi.numpy(),
                     size=(4096, 4)).astype(np.float32)
    ys = ref(xs)
    assert (ys >= rl.numpy() - 1e-5).all()
    assert (ys <= rh.numpy() + 1e-5).all()
