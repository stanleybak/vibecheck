"""Op x path coverage matrix with soundness checks (fast, CPU, no files).

Every op the graph loader can emit is exercised through every bound path:

    point       exact forward vs an independent reference
    interval    IBP bounds bracket sampled exact outputs
    zono        forward zonotope bounds bracket
    crown       backward CROWN (identity rows) brackets
    alpha       alpha-CROWN (3 iters) brackets
    inter       intermediates_crown feeding crown brackets
    dual        the LP state builds and its root bound is <= sampled min

A cell must be either SUPPORTED (and then it must be sound: bounds
bracket 128 sampled exact evaluations -- sampling validates, never
defines) or a KNOWN gap listed in GOLDEN. Silently losing support fails;
gaining support fails until GOLDEN is updated (a deliberate act).

Everything is 4-12 neurons wide; the whole module runs in seconds.
"""
import numpy as np
import pytest
import torch

from vibecheck2.core import backward, dual_lp
from vibecheck2.core import forward as fwd
from vibecheck2.core.graph import Net, Op
from vibecheck2.core.linmap import Dense

RNG = np.random.default_rng(7)
PATHS = ('point', 'interval', 'zono', 'crown', 'alpha', 'inter', 'dual')

# The coverage contract. True = supported (checked sound), False = known
# gap (must raise NotImplementedError, never return numbers).
GOLDEN = {}


def _dense(n_in, n_out, positive_bias=0.0):
    W = RNG.standard_normal((n_out, n_in)).astype(np.float32)
    b = RNG.standard_normal(n_out).astype(np.float32) + positive_bias
    return Dense(W, b)


def _sandwich(fn, params=None, positive=False):
    """input(4) -> dense(6) -> nonlin -> dense(3). `positive` biases the
    pre-activation range to be sign-definite (reciprocal, pow)."""
    lm1 = _dense(4, 6, positive_bias=8.0 if positive else 0.0)
    lm2 = _dense(6, 3)
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'h': Op('h', 'linmap', ('x',), (6,), 6, lm=lm1),
        'a': Op('a', 'nonlin', ('h',), (6,), 6, fn=fn,
                params=params or {}),
        'y': Op('y', 'linmap', ('a',), (3,), 3, lm=lm2),
    }
    net = Net(ops, ['h', 'a', 'y'], 'x', 'y')

    def ref(x):
        h = x @ lm1.W.T + lm1.b
        a = _REF_FN[fn](h, params or {})
        return a @ lm2.W.T + lm2.b

    lo = -np.ones(4, dtype=np.float32) * 0.7
    hi = np.ones(4, dtype=np.float32) * 0.7
    return net, ref, lo, hi


_REF_FN = {
    'relu': lambda z, p: np.maximum(z, 0),
    'leaky_relu': lambda z, p: np.where(z > 0, z, p.get('alpha', 0.01) * z),
    'sigmoid': lambda z, p: 1 / (1 + np.exp(-z)),
    'tanh': lambda z, p: np.tanh(z),
    'exp': lambda z, p: np.exp(z),
    'sin': lambda z, p: np.sin(z),
    'cos': lambda z, p: np.cos(z),
    'sign': lambda z, p: np.sign(z),
    'floor': lambda z, p: np.floor(z),
    'reciprocal': lambda z, p: 1 / z,
    'pow': lambda z, p: z ** p['exponent'],
}


def _structural(kind):
    """input(4) -> two dense branches -> kind -> dense(3)."""
    lm1 = _dense(4, 6)
    lm2 = _dense(4, 6)
    lm3 = _dense(6, 3)
    if kind == 'add':
        mk = Op('m', 'add', ('u', 'v'), (6,), 6)
        rf = lambda u, v: u + v                              # noqa: E731
    elif kind == 'mul':
        mk = Op('m', 'mul', ('u', 'v'), (6,), 6)
        rf = lambda u, v: u * v                              # noqa: E731
    elif kind == 'bmm':
        mk = Op('m', 'bmm', ('u', 'v'), (4,), 4,
                params={'a_shape': (2, 3), 'b_shape': (3, 2)})
        lm3 = _dense(4, 3)
        rf = lambda u, v: (u.reshape(-1, 2, 3)               # noqa: E731
                           @ v.reshape(-1, 3, 2)).reshape(-1, 4)
    else:
        raise AssertionError(kind)
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'u': Op('u', 'linmap', ('x',), (6,), 6, lm=lm1),
        'v': Op('v', 'linmap', ('x',), (6,), 6, lm=lm2),
        'm': mk,
        'y': Op('y', 'linmap', ('m',), (3,), 3, lm=lm3),
    }
    net = Net(ops, ['u', 'v', 'm', 'y'], 'x', 'y')

    def ref(x):
        u = x @ lm1.W.T + lm1.b
        v = x @ lm2.W.T + lm2.b
        return rf(u, v) @ lm3.W.T + lm3.b

    lo = -np.ones(4, dtype=np.float32) * 0.7
    hi = np.ones(4, dtype=np.float32) * 0.7
    return net, ref, lo, hi


def _concat_case():
    lm1 = _dense(4, 3)
    lm2 = _dense(4, 2)
    lm3 = _dense(6, 3)
    base = np.array([9., 9., 9., 9., 9., 9.], dtype=np.float32)
    pos1, pos2 = [0, 2, 4], [1, 5]                # slot 3 keeps base
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'u': Op('u', 'linmap', ('x',), (3,), 3, lm=lm1),
        'v': Op('v', 'linmap', ('x',), (2,), 2, lm=lm2),
        'm': Op('m', 'concat', ('u', 'v'), (6,), 6,
                params={'base': base, 'positions': [pos1, pos2],
                        'n_out': 6}),
        'y': Op('y', 'linmap', ('m',), (3,), 3, lm=lm3),
    }
    net = Net(ops, ['u', 'v', 'm', 'y'], 'x', 'y')

    def ref(x):
        u = x @ lm1.W.T + lm1.b
        v = x @ lm2.W.T + lm2.b
        m = np.tile(base, (x.shape[0], 1))
        m[:, pos1] = u
        m[:, pos2] = v
        return m @ lm3.W.T + lm3.b

    lo = -np.ones(4, dtype=np.float32) * 0.7
    hi = np.ones(4, dtype=np.float32) * 0.7
    return net, ref, lo, hi


def _maxpool_case():
    lm1 = _dense(4, 16)
    lm3 = _dense(4, 3)
    ops = {
        'x': Op('x', 'input', (), (4,), 4),
        'u': Op('u', 'linmap', ('x',), (16,), 16, lm=lm1),
        'm': Op('m', 'maxpool', ('u',), (4,), 4,
                params={'in_shape': (1, 4, 4), 'kernel_shape': (2, 2),
                        'stride': (2, 2), 'padding': (0, 0)}),
        'y': Op('y', 'linmap', ('m',), (3,), 3, lm=lm3),
    }
    from vibecheck2.core.graph import decompose_maxpool
    net = decompose_maxpool(Net(ops, ['u', 'm', 'y'], 'x', 'y'))

    def ref(x):
        u = (x @ lm1.W.T + lm1.b).reshape(-1, 1, 4, 4)
        m = np.stack([u[:, :, i:i + 2, j:j + 2].max(axis=(2, 3))
                      for i in (0, 2) for j in (0, 2)],
                     axis=2).reshape(-1, 4)
        return m @ lm3.W.T + lm3.b

    lo = -np.ones(4, dtype=np.float32) * 0.7
    hi = np.ones(4, dtype=np.float32) * 0.7
    return net, ref, lo, hi


CASES = {}
for _fn in _REF_FN:
    _params = ({'exponent': 2.0} if _fn == 'pow'
               else {'alpha': 0.1} if _fn == 'leaky_relu' else None)
    CASES[_fn] = lambda fn=_fn, p=_params: _sandwich(
        fn, p, positive=(fn == 'reciprocal'))
for _k in ('add', 'mul', 'bmm'):
    CASES[_k] = lambda k=_k: _structural(k)
CASES['concat'] = _concat_case
CASES['maxpool'] = _maxpool_case


def _samples(lo, hi, n=128):
    x = RNG.uniform(lo, hi, size=(n, lo.size)).astype(np.float32)
    x[0], x[1] = lo, hi
    return x


def _run_path(path, net, ref, lo, hi):
    """Returns None if sound, or raises. NotImplementedError passes up."""
    t_lo = torch.tensor(lo).reshape(1, -1)
    t_hi = torch.tensor(hi).reshape(1, -1)
    xs = _samples(lo, hi)
    ys = ref(xs)                                     # (n, n_out) exact
    tol = 1e-3 + 1e-3 * np.abs(ys).max()

    if path == 'point':
        got = fwd.point(net, torch.tensor(xs)).numpy()
        assert np.abs(got - ys).max() < tol, 'point mismatch vs reference'
        return

    if path == 'interval':
        l, h = fwd.interval(net, t_lo, t_hi)
        l, h = l.numpy()[0], h.numpy()[0]
    elif path == 'zono':
        l, h = fwd.zono(net, t_lo, t_hi)
        l, h = l.numpy()[0], h.numpy()[0]
    elif path in ('crown', 'alpha', 'inter'):
        W = torch.eye(net.n_out)
        Wb = torch.cat([W, -W])                      # both signs
        if path == 'inter':
            inter = backward.intermediates_crown(net, t_lo, t_hi)
            lb = backward.crown(net, t_lo, t_hi, Wb, inter)
        elif path == 'alpha':
            lb = backward.alpha_crown(net, t_lo, t_hi, Wb, iters=3)
        else:
            lb = backward.crown(net, t_lo, t_hi, Wb)
        lb = lb.detach().numpy()[0]
        l, h = lb[:net.n_out], -lb[net.n_out:]
    elif path == 'dual':
        from vibecheck.fast_dual_ascent import parse_problem
        inter = backward.intermediates_crown(net, t_lo, t_hi)
        slopes = {nm: torch.full((net.ops[nm].n,), 0.5)
                  for nm in net.order if net.ops[nm].kind == 'nonlin'}
        state, keys = dual_lp._state_for(net, t_lo, t_hi, inter, slopes,
                                         'cpu')
        w = RNG.standard_normal(net.n_out).astype(np.float64)
        prob = parse_problem(state, w, 0.0, keys)
        assert prob.root_bound <= (ys @ w).min() + tol, \
            'dual root bound above a sampled point'
        return
    else:
        raise AssertionError(path)

    assert (l <= ys.min(axis=0) + tol).all(), \
        f'{path}: lower bound above a sampled output'
    assert (h >= ys.max(axis=0) - tol).all(), \
        f'{path}: upper bound below a sampled output'


# observed support: filled by the golden dict below. A True cell runs the
# soundness check; a False cell must raise NotImplementedError.
GOLDEN = {
    # every nonlin sandwich supports every path
    **{fn: dict.fromkeys(PATHS, True) for fn in _REF_FN},
    'add': dict.fromkeys(PATHS, True),
    'concat': dict.fromkeys(PATHS, True),
    'mul': dict.fromkeys(PATHS, True),
    # maxpool decomposes to relu at load (max(a,b) = a + relu(b - a)),
    # so every path, split scoring, and alpha ride the relu machinery
    'maxpool': dict.fromkeys(PATHS, True),
    # bmm crown-side: the general McCormick bilinear adjoint (mul and bmm
    # are shape instances of one engine)
    'bmm': dict.fromkeys(PATHS, True),
}


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('path', PATHS)
def test_op_path(case, path):
    net, ref, lo, hi = CASES[case]()
    expected = GOLDEN[case][path]
    try:
        _run_path(path, net, ref, lo, hi)
    except NotImplementedError as e:
        if expected:
            pytest.fail(f'{case} x {path}: support LOST ({e})')
        return                                     # known gap, still a gap
    if not expected:
        pytest.fail(f'{case} x {path}: support GAINED -- update GOLDEN '
                    f'(and celebrate)')
