"""Pipeline routes beyond the basics --
the smooth-op (sin) relaxation route, the net-cache, the network-pair
route through the flat runner, load-failure disposition, and the standard
CLI's --serialise-assignments output."""
import os

import numpy as np
import pytest
import torch

import vibecheck.pipeline as vp
from vibecheck import Spec



def _sin_net(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('Sin', ['X'], ['s']),
         helper.make_node('MatMul', ['s', 'W'], ['Y'])],
        'sin',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.ones((2, 1), np.float32), 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    p = str(tmp_path / 'sin.onnx')
    onnx.save(m, p)
    return p


def _spec_path(tmp_path, name, spec):
    p = str(tmp_path / f'{name}.vnnlib')
    with open(p, 'w') as f:
        f.write(spec.to_vnnlib())
    return p


def test_smooth_sin_route_unsat_and_sat(tmp_path):
    """y = sin(x0)+sin(x1) on [-1,1]^2: max = 2 sin(1) ~ 1.683. The
    closed-form sin planes must prove 'y >= 2.1' unreachable at the root,
    and the attack must falsify 'y >= 1.5'."""
    net = _sin_net(tmp_path)
    unsat = _spec_path(tmp_path, 'safe',
                       Spec(x_lo=[-1, -1], x_hi=[1, 1])
                       .forbid([[1.0]], [-2.1]))       # unsafe: y >= 2.1
    v, _ = vp.verify(net, unsat, 20.0, 'cpu')
    assert v == 'unsat'
    sat = _spec_path(tmp_path, 'unsafe',
                     Spec(x_lo=[-1, -1], x_hi=[1, 1])
                     .forbid([[1.0]], [-1.5]))         # unsafe: y >= 1.5
    v, d = vp.verify(net, sat, 20.0, 'cpu')
    assert v == 'sat' and d.get('witness') is not None
    x = np.asarray(d['witness'], np.float64).ravel()
    assert np.sin(x).sum() >= 1.5                      # true violation


def test_net_cache_roundtrip(tmp_path):
    """--net-cache: first run converts and saves, second run loads the
    cache; both produce the same verdict."""
    net = _sin_net(tmp_path)
    spec = _spec_path(tmp_path, 'safe',
                      Spec(x_lo=[-1, -1], x_hi=[1, 1])
                      .forbid([[1.0]], [-2.1]))
    cache = str(tmp_path / 'net.pt')
    rf1 = str(tmp_path / 'r1.txt')
    rf2 = str(tmp_path / 'r2.txt')
    base = ['--net', net, '--spec', spec, '--timeout', '20',
            '--device', 'cpu', '--net-cache', cache]
    assert vp._legacy_main(base + ['--results-file', rf1]) == 0
    assert os.path.isfile(cache)
    assert vp._legacy_main(base + ['--results-file', rf2]) == 0
    assert open(rf1).read().splitlines()[0] == 'unsat'
    assert open(rf1).read() == open(rf2).read()


def _linear_pair(tmp_path, g_bias):
    """f = identity, g = identity + g_bias: Y_f[0] - Y_g[0] = -g_bias
    everywhere, so the pair atom (>= Y_f[0] Y_g[0]) is deterministic."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def mk(name, bias):
        g = helper.make_graph(
            [helper.make_node('MatMul', ['X', 'W'], ['h']),
             helper.make_node('Add', ['h', 'B'], ['Y'])],
            name,
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.eye(2, dtype=np.float32), 'W'),
             numpy_helper.from_array(np.full(2, bias, np.float32), 'B')])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
        m.ir_version = 7
        p = str(tmp_path / f'{name}.onnx')
        onnx.save(m, p)
        return p

    f = mk('f', 0.0)
    g = mk('g', g_bias)
    spec = str(tmp_path / 'pair.vnnlib')
    with open(spec, 'w') as fh:
        fh.write('(declare-network f (isomorphic-to g))\n'
                 '(declare-network g)\n'
                 '(assert (and (<= X_f[0] 1.0) (>= X_f[0] 0.0)))\n'
                 '(assert (and (<= X_f[1] 1.0) (>= X_f[1] 0.0)))\n'
                 '(assert (or (>= Y_f[0] Y_g[0])))\n')
    return repr([('f', f), ('g', g)]), spec


def test_pair_route_sat_and_unsat(tmp_path):
    """The flat runner's network-pair branch end to end: g = f - 1 makes
    the unsafe atom Y_f >= Y_g hold EVERYWHERE (sat); g = f + 1 makes it
    empty (unsat)."""
    field, spec = _linear_pair(tmp_path, g_bias=-1.0)
    rf = str(tmp_path / 'sat.txt')
    vp._legacy_main(['--net', field, '--spec', spec, '--timeout', '30',
                     '--device', 'cpu', '--results-file', rf])
    assert open(rf).read().splitlines()[0] == 'sat'
    field, spec = _linear_pair(tmp_path, g_bias=1.0)
    rf = str(tmp_path / 'unsat.txt')
    vp._legacy_main(['--net', field, '--spec', spec, '--timeout', '30',
                     '--device', 'cpu', '--results-file', rf])
    assert open(rf).read().splitlines()[0] == 'unsat'


def test_missing_net_is_a_clean_error(tmp_path):
    spec = _spec_path(tmp_path, 'p', Spec(x_lo=[0], x_hi=[1])
                      .forbid([[1.0]], [-2.0]))
    rf = str(tmp_path / 'r.txt')
    code = vp._legacy_main(['--net', str(tmp_path / 'nope.onnx'),
                            '--spec', spec, '--timeout', '5',
                            '--device', 'cpu', '--results-file', rf])
    first = open(rf).read().splitlines()[0]
    assert first in ('error', 'unknown')     # never a verdict, never a crash
    assert code in (1, 2)


def test_cli_serialise_assignments(tmp_path, capsys):
    """`vibecheck verify --serialise-assignments DIR` on a sat instance:
    strict stdout (verdict only) + one TensorProto per assigned variable."""
    from onnx import numpy_helper
    from vibecheck.cli_standard import run_verify
    net = _sin_net(tmp_path)
    spec = _spec_path(tmp_path, 'unsafe',
                      Spec(x_lo=[-1, -1], x_hi=[1, 1])
                      .forbid([[1.0]], [-1.5]))
    out = str(tmp_path / 'assign')
    code = run_verify([spec, '--network', f'N={net}', '--timeout', '30',
                       '--device', 'cpu',
                       '--serialise-assignments', out])
    assert code == 1
    assert capsys.readouterr().out.strip() == 'sat'   # strict stdout

    def _tensor(fn):
        import onnx
        tp = onnx.TensorProto()
        with open(os.path.join(out, fn), 'rb') as f:
            tp.ParseFromString(f.read())
        return numpy_helper.to_array(tp)

    assert sorted(os.listdir(out)) == ['X.pb', 'Y.pb']
    x = _tensor('X.pb')
    y = _tensor('Y.pb')
    assert np.isclose(np.sin(x.ravel()).sum(), y.ravel().sum(), atol=1e-5)
    assert y.ravel().sum() >= 1.5
