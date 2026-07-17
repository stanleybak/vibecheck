"""Tier-0 unit tests for the vibecheck core (design 4.2).

Run: PYTHONPATH=src <venv>/bin/python -m pytest tests2/ -q
Soundness invariants only need numpy/torch; no benchmark files.
"""
import numpy as np
import pytest
import torch

from vibecheck.core import linmap as lm
from vibecheck.core import memory
from vibecheck.core.relax import REL

torch.manual_seed(0)
_rng = np.random.default_rng(0)


def _linmaps():
    """One instance of every LinMap layout, small random params."""
    W = _rng.normal(size=(7, 5)).astype(np.float32)
    b = _rng.normal(size=7).astype(np.float32)
    k = _rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    kb = _rng.normal(size=4).astype(np.float32)
    kt = _rng.normal(size=(3, 4, 2, 2)).astype(np.float32)
    yield 'dense', lm.Dense(W, b)
    yield 'dense_nobias', lm.Dense(W, None)
    yield 'conv', lm.Conv2d(k, kb, (3, 6, 6), (4, 4, 4), (1, 1), (0, 0))
    yield 'conv_pad_stride', lm.Conv2d(k, kb, (3, 6, 6), (4, 3, 3), (2, 2), (1, 1))
    yield 'convT', lm.ConvT2d(kt, kb, (3, 5, 5), (4, 11, 11), (2, 2), (0, 0),
                              output_padding=(1, 1))
    yield 'avgpool', lm.AvgPool((3, 6, 6), (3, 3, 3), (2, 2), (2, 2), (0, 0))
    yield 'select', lm.Select(_rng.integers(0, 12, size=9), 12)
    yield 'scale_shift', lm.ScaleShift(_rng.normal(size=11).astype(np.float32),
                                       _rng.normal(size=11).astype(np.float32), 11)
    yield 'sum_axis', lm.SumAxis(3, 4, 5)
    yield 'mean_axis', lm.SumAxis(3, 4, 5, mean=True)


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_adjoint_identity(name, m):
    """<lin(x), y> == <x, lin_t(y)> for random x, y (exact adjoint)."""
    X = torch.randn(6, m.n_in)
    Y = torch.randn(6, m.n_out)
    lhs = (m.lin(X) * Y).sum(dim=1)
    rhs = (X * m.lin_t(Y)).sum(dim=1)
    assert torch.allclose(lhs, rhs, atol=1e-4), (name, (lhs - rhs).abs().max())


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_abs_dominates(name, m):
    """lin_abs on a nonnegative vector bounds |lin| on any sign pattern:
    |lin(s*r)| <= lin_abs(r) for r >= 0 and any s in {-1,1}^n."""
    r = torch.rand(4, m.n_in)
    bound = m.lin_abs(r)
    for _ in range(8):
        s = torch.where(torch.rand(4, m.n_in) < 0.5, -1.0, 1.0)
        val = m.lin(s * r)
        assert (val.abs() <= bound + 1e-4).all(), name


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_point_is_lin_plus_bias(name, m):
    X = torch.randn(3, m.n_in)
    assert torch.allclose(m.point(X), m.lin(X) + m.bias_vec(X), atol=1e-5)


def test_relu_planes_bracket():
    """Sampling VALIDATES the planes (never defines them): dense adversarial
    sampling incl. endpoints must satisfy al*x+bl <= relu(x) <= au*x+bu."""
    lo = torch.tensor([[-3.0, -1e-6, 0.0, 0.5, -2.0]])
    hi = torch.tensor([[2.0, 1e-6, 0.0, 1.5, -0.5]])
    al, bl, au, bu = REL['relu'].planes(lo, hi)
    for t in torch.linspace(0, 1, 101):
        x = lo + t * (hi - lo)
        y = torch.relu(x)
        assert (al * x + bl <= y + 1e-6).all()
        assert (y <= au * x + bu + 1e-6).all()


def test_memory_chunked_matches_unchunked():
    X = torch.randn(37, 4)
    fn = lambda b: b * 2 + 1                                  # noqa: E731
    out = memory.chunked(fn, X, bytes_per_item=1)             # forces chunks
    assert torch.equal(out, fn(X))
    out1 = memory.chunked(fn, X, bytes_per_item=1e12)         # single item/chunk
    assert torch.equal(out1, fn(X))


# --------------------------------------------------------------------------- #
# tiny synthetic net: forward mode soundness on a DAG with a residual merge
# --------------------------------------------------------------------------- #

def _tiny_residual_net(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W1 = numpy_helper.from_array(_rng.normal(size=(4, 4)).astype(np.float32).T, 'W1')
    W2 = numpy_helper.from_array(_rng.normal(size=(4, 4)).astype(np.float32).T, 'W2')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W1'], ['h1']),
         helper.make_node('Relu', ['h1'], ['r1']),
         helper.make_node('MatMul', ['r1', 'W2'], ['h2']),
         helper.make_node('Add', ['h2', 'X'], ['s']),        # residual merge
         helper.make_node('Relu', ['s'], ['Y'])],
        'g', [X], [Y], [W1, W2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'res.onnx')
    onnx.save(m, p)
    return p


def test_forward_modes_sound_on_residual_dag(tmp_path):
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(_tiny_residual_net(tmp_path))
    lo = torch.full((1, 4), -0.5)
    hi = torch.full((1, 4), 0.7)
    ilo, ihi = fwd.interval(net, lo, hi)
    zlo, zhi = fwd.zono(net, lo, hi)
    # both contain many exact point evaluations (validation sampling)
    xs = torch.rand(256, 4) * (hi - lo) + lo
    ys = fwd.point(net, xs)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all()
    assert (ys >= ilo - 1e-4).all() and (ys <= ihi + 1e-4).all()


def test_forward_batched_boxes(tmp_path):
    """Batched boxes bound their own samples (per-domain isolation)."""
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(_tiny_residual_net(tmp_path))
    lo = torch.tensor([[-1.0] * 4, [0.1] * 4, [-0.2] * 4])
    hi = torch.tensor([[-0.5] * 4, [0.9] * 4, [0.3] * 4])
    zlo, zhi = fwd.zono(net, lo, hi)
    for b in range(3):
        xs = torch.rand(128, 4) * (hi[b] - lo[b]) + lo[b]
        ys = fwd.point(net, xs)
        assert (ys >= zlo[b] - 1e-4).all() and (ys <= zhi[b] + 1e-4).all(), b


# --------------------------------------------------------------------------- #
# alpha_zono: relu-slope optimization over the forward zonotope
# --------------------------------------------------------------------------- #

def _tiny_mixed_net(tmp_path):
    """matmul -> relu -> matmul -> sin -> matmul (mixed nonlinearities)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W1 = numpy_helper.from_array(
        _rng.normal(size=(3, 4)).astype(np.float32), 'W1')
    W2 = numpy_helper.from_array(
        _rng.normal(size=(4, 4)).astype(np.float32), 'W2')
    W3 = numpy_helper.from_array(
        _rng.normal(size=(4, 2)).astype(np.float32), 'W3')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W1'], ['h1']),
         helper.make_node('Relu', ['h1'], ['r1']),
         helper.make_node('MatMul', ['r1', 'W2'], ['h2']),
         helper.make_node('Sin', ['h2'], ['s1']),
         helper.make_node('MatMul', ['s1', 'W3'], ['Y'])],
        'g', [X], [Y], [W1, W2, W3])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'mixed.onnx')
    onnx.save(m, p)
    return p


def test_alpha_zono_sound_and_never_worse(tmp_path):
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(_tiny_mixed_net(tmp_path))
    lo = torch.full((1, 3), -0.8)
    hi = torch.full((1, 3), 0.6)
    W = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-1.5, 0.3]])
    lb = fwd.alpha_zono(net, lo, hi, W, iters=60)[0]
    # sound: below every exact sample margin (validation sampling)
    xs = torch.rand(512, 3) * (hi - lo) + lo
    marg = fwd.point(net, xs) @ W.T
    assert (lb <= marg.min(dim=0).values + 1e-4).all()
    # never worse than the default-band zonotope (iterate 0 is that bound)
    _, _, st = fwd.zono(net, lo, hi, return_state=True)
    z = st[net.output_name]
    plain = (z.c @ W.T - torch.matmul(W, z.G).abs().sum(-1))[0]
    assert (lb >= plain - 1e-5).all()


def test_alpha_zono_beats_default_band_on_relu(tmp_path):
    """relu(x) on [-1, 1], query w=+1: the default DeepZ band gives -0.5;
    slope 0 is exact (lb 0). The optimizer must recover (nearly) all of it."""
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
    g = helper.make_graph([helper.make_node('Relu', ['X'], ['Y'])],
                          'g', [X], [Y], [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'relu1.onnx')
    onnx.save(m, p)
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(p)
    lo, hi = torch.full((1, 1), -1.0), torch.full((1, 1), 1.0)
    W = torch.eye(1)
    lb = fwd.alpha_zono(net, lo, hi, W, iters=100)[0]
    assert float(lb) >= -1e-3          # slope 0 makes the band exact


# --------------------------------------------------------------------------- #
# zono mul: sound bilinear product of two correlated branches
# --------------------------------------------------------------------------- #

def _tiny_mul_net(tmp_path):
    """Two linear branches of the same input multiplied elementwise."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    A = numpy_helper.from_array(
        _rng.normal(size=(3, 4)).astype(np.float32), 'A')
    Bm = numpy_helper.from_array(
        _rng.normal(size=(3, 4)).astype(np.float32), 'B')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    nodes = [helper.make_node('MatMul', ['X', 'A'], ['a']),
             helper.make_node('MatMul', ['X', 'B'], ['b']),
             helper.make_node('Mul', ['a', 'b'], ['Y'])]
    g = helper.make_graph(nodes, 'g', [X], [Y], [A, Bm])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'mul.onnx')
    onnx.save(m, p)
    return p


def test_zono_mul_sound_on_branch_product(tmp_path):
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(_tiny_mul_net(tmp_path))
    lo = torch.full((1, 3), -0.9)
    hi = torch.full((1, 3), 0.8)
    zlo, zhi = fwd.zono(net, lo, hi)
    xs = torch.rand(512, 3) * (hi - lo) + lo
    ys = fwd.point(net, xs)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all()


# --------------------------------------------------------------------------- #
# band_alpha: slope-parametrized bands are sound for every alpha in [0, 1]
# --------------------------------------------------------------------------- #

def test_band_alpha_sound_across_ops():
    tr = torch.rand
    cases = [
        ('relu', None, -2 + 3 * tr(64), 0.1 + 2 * tr(64)),
        ('sigmoid', None, -4 + 6 * tr(64), 0.1 + 3 * tr(64)),
        ('tanh', None, -4 + 6 * tr(64), 0.1 + 3 * tr(64)),
        ('sin', None, -5 + 8 * tr(64), 0.1 + 4 * tr(64)),
        ('cos', None, -5 + 8 * tr(64), 0.1 + 4 * tr(64)),
        ('pow', {'exponent': 2}, -2 + 3 * tr(64), 0.1 + 2 * tr(64)),
        ('pow', {'exponent': 3}, -2 + 3 * tr(64), 0.1 + 2 * tr(64)),
        ('exp', None, -3 + 4 * tr(64), 0.1 + 2 * tr(64)),
        ('reciprocal', None, 0.2 + 2 * tr(64), 0.1 + 2 * tr(64)),
        ('reciprocal', None, -4 + 1.5 * tr(64), 0.1 + 2 * tr(64)),
    ]
    for fn, params, lo, w in cases:
        hi = lo + w
        rel = REL[fn]
        f = rel.point
        for a in (0.0, 0.17, 0.5, 0.83, 1.0):
            alpha = torch.full_like(lo, a)
            lam, mu, delta = rel.band_alpha(lo, hi, alpha, params)
            assert bool((delta >= -1e-6).all()), (fn, a)
            u = torch.rand(500, *lo.shape)
            xs = lo + (hi - lo) * u
            gap = (f(xs, params) - (lam * xs + mu)).abs()
            assert bool((gap <= delta + 1e-5).all()), \
                (fn, a, float((gap - delta).max()))


# --------------------------------------------------------------------------- #
# slp polish: lands exactly on a razor-thin conjunctive face
# --------------------------------------------------------------------------- #

def test_slp_polish_reaches_equality_face(tmp_path):
    """CE set = the line x0+x1 = 0.7071067 (measure zero, conjunctive):
    the trust-region LP must step from a +1.3e-2 plateau point onto it."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    A = numpy_helper.from_array(
        np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32), 'A')
    C = numpy_helper.from_array(
        np.array([-0.7071067, 0.7071067], dtype=np.float32), 'C')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'A'], ['h']),
         helper.make_node('Add', ['h', 'C'], ['Y'])],
        'g', [X], [Y], [A, C])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, str(tmp_path / 'thin.onnx'))
    spec_txt = '\n'.join(
        [f'(declare-const X_{i} Real)' for i in range(2)]
        + [f'(declare-const Y_{i} Real)' for i in range(2)]
        + ['(assert (<= X_0 1.0))', '(assert (>= X_0 -1.0))',
           '(assert (<= X_1 1.0))', '(assert (>= X_1 -1.0))',
           '(assert (and (<= Y_0 0.0) (<= Y_1 0.0)))'])
    (tmp_path / 'thin.vnnlib').write_text(spec_txt)
    from vibecheck.frontend.vnnlib_loader import load_vnnlib
    from vibecheck.core import attack
    from vibecheck.core import graph as g2
    net = g2.load(str(tmp_path / 'thin.onnx'))
    spec = load_vnnlib(str(tmp_path / 'thin.vnnlib'))
    x0 = np.array([0.36, 0.36 - 0.7071067], dtype=np.float64)  # margin 1.3e-2
    w = attack._slp_polish(net, spec, x0,
                           np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
                           device='cpu')
    assert w is not None
    from vibecheck.core import forward as fwd
    y = fwd.point(net, torch.tensor(w, dtype=torch.float32).unsqueeze(0))[0]
    assert float(y.max()) <= 1e-7


# --------------------------------------------------------------------------- #
# stabilize_intermediates: split-and-tighten envelopes stay sound
# --------------------------------------------------------------------------- #

def test_stabilize_intermediates_sound(tmp_path):
    from vibecheck.core import backward, forward as fwd
    from vibecheck.core import graph as g2
    from vibecheck.core.budget import Budget
    from vibecheck.core.search import stabilize_intermediates
    net = g2.load(_tiny_residual_net(tmp_path))
    lo = torch.full((1, 4), -1.2)
    hi = torch.full((1, 4), 1.0)
    W = torch.tensor([[1.0, -1.0, 0.5, 0.0], [0.0, 2.0, -1.0, 1.0]])
    inter = backward.intermediates(net, lo, hi)
    inter2 = stabilize_intermediates(net, W, lo, hi, inter, Budget(30.0),
                                     passes=2)
    xs = torch.rand(512, 4) * (hi - lo) + lo
    state = {net.input_name: xs}
    for name in net.order:                    # exact per-edge values
        op = net.ops[name]
        if op.kind == 'linmap':
            state[name] = op.lm.point(state[op.inputs[0]])
        elif op.kind == 'nonlin':
            state[name] = torch.relu(state[op.inputs[0]])
        elif op.kind == 'add':
            state[name] = state[op.inputs[0]] + state[op.inputs[1]]
    for nm, v in inter2.items():
        if nm not in state:
            continue
        assert (state[nm] >= v[0] - 1e-4).all(), nm
        assert (state[nm] <= v[1] + 1e-4).all(), nm
        # never looser than the base bounds
        assert (v[0] >= inter[nm][0] - 1e-6).all(), nm
        assert (v[1] <= inter[nm][1] + 1e-6).all(), nm


def test_alpha_planes_sound_exp_reciprocal():
    """Tangent-position alpha planes for the convex softmax ops: sound for
    every alpha (tangents under convex f; chord above), validated by
    sampling."""
    tr = torch.rand
    cases = [('exp', -3 + 4 * tr(48), 0.1 + 2 * tr(48)),
             ('reciprocal', 0.2 + 2 * tr(48), 0.1 + 2 * tr(48)),
             ('reciprocal', -4 + 1.5 * tr(48), 0.1 + 2 * tr(48))]
    for fn, lo, w in cases:
        hi = lo + w
        rel = REL[fn]
        for a in (0.0, 0.31, 0.5, 0.77, 1.0):
            alpha = torch.full((2, lo.shape[0]), a)
            al, bl, au, bu = rel.alpha_planes(lo, hi, alpha)
            xs = lo + (hi - lo) * torch.rand(400, lo.shape[0])
            y = rel.point(xs)
            assert bool((al * xs + bl <= y + 1e-5).all()), (fn, a)
            assert bool((y <= au * xs + bu + 1e-5).all()), (fn, a)


def test_softmax_decomposition_exact_and_sound(tmp_path):
    """The softmax decomposition must be EXACT pointwise and sound under
    zono, with the output landing in [0, 1] even for wide (scale-8)
    logits. (A graph-level max-shift variant with an exact relu-max tree
    was built and MEASURED WORSE on vit -- tree relaxation slack + the
    collapsed final mul; see the vit memory note. The k x k difference
    form stays.)"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    A = numpy_helper.from_array(
        (8 * _rng.normal(size=(4, 10))).astype(np.float32), 'A')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'A'], ['h']),
         helper.make_node('Softmax', ['h'], ['Y'], axis=-1)],
        'g', [X], [Y], [A])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, str(tmp_path / 'sm.onnx'))
    from vibecheck.core import forward as fwd
    from vibecheck.core import graph as g2
    net = g2.load(str(tmp_path / 'sm.onnx'))
    xs = torch.tensor(_rng.uniform(-2, 2, size=(64, 4)).astype(np.float32))
    y = fwd.point(net, xs)
    ref = torch.softmax(xs @ torch.tensor(
        np.array(8 * _rng.normal(size=(4, 10)), dtype=np.float32)), dim=1)
    # regenerate A deterministically is fragile; compare against torch on
    # the SAME loaded weights instead: recover A from the net's first linmap
    import numpy as _np
    Aop = next(net.ops[nm] for nm in net.order
               if net.ops[nm].kind == 'linmap')
    W = torch.tensor(_np.asarray(Aop.lm.point(torch.eye(4))))
    ref = torch.softmax(xs @ W, dim=1)
    assert float((y - ref).abs().max()) < 1e-5
    # wide logits (scale 8): the old difference form's denominators blow
    # up; the shifted form must stay sound with bounded internals
    lo = torch.full((1, 4), -1.5)
    hi = torch.full((1, 4), 1.5)
    zlo, zhi = fwd.zono(net, lo, hi)
    pts = lo + (hi - lo) * torch.rand(512, 4)
    ys = fwd.point(net, pts)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all()
    assert float(zhi.max()) <= 1.0 + 1e-5   # softmax output range holds
