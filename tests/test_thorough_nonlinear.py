"""The nonlinear-v2 route (handlers/nonlinear_augment) end
to end -- polynomial atoms transpiled into an augmented net + linear spec
(adaptive_cruise shape), and the HC4 input-region prefilter."""
import numpy as np
import pytest

import vibecheck.pipeline as vp
from vibecheck.handlers.nonlinear_augment import is_nonlinear_v2_spec


_HEAD = """(vnnlib-version <2.0>)
(declare-network f
    (declare-input X real [1,2])
    (declare-output Y real [1,1])
)
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))
(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))
"""


def _sum_net(tmp_path):
    """y = x0 + x1."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['Y'])],
        'sum',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.ones((2, 1), np.float32), 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    p = str(tmp_path / 'sum.onnx')
    onnx.save(m, p)
    return p


def _spec(tmp_path, name, body):
    p = str(tmp_path / f'{name}.vnnlib')
    with open(p, 'w') as f:
        f.write(_HEAD + body)
    return p


def test_gate_detects_polynomial_specs(tmp_path):
    poly = _HEAD + '(assert (>= Y[0,0] (* X[0,0] X[0,0])))\n'
    linear = _HEAD + '(assert (>= Y[0,0] 3.0))\n'
    assert is_nonlinear_v2_spec(poly)
    assert not is_nonlinear_v2_spec(linear)     # linear v2: normal adapter


def test_poly_sat_witness_revalidated_on_original_atoms(tmp_path):
    """Unsafe iff y >= x0^2 (reachable: y = x0+x1): the augmented route
    must find a CE and the emitted witness must satisfy the ORIGINAL
    polynomial atom strictly."""
    net = _sum_net(tmp_path)
    spec = _spec(tmp_path, 'poly_sat',
                 '(assert (>= Y[0,0] (* X[0,0] X[0,0])))\n')
    v, d = vp.verify(net, spec, 25.0, 'cpu')
    assert v == 'sat' and d.get('witness') is not None
    x = np.asarray(d['witness'], np.float64).ravel()
    assert (x >= 0).all() and (x <= 1).all()
    assert x.sum() >= x[0] ** 2                 # the original atom holds


def test_poly_unsat_proved_through_augmented_net(tmp_path):
    """Unsafe iff y^2 <= -0.5 (empty): crown on the augmented net (with
    the mul-feature relaxation) must refute it."""
    net = _sum_net(tmp_path)
    spec = _spec(tmp_path, 'poly_unsat',
                 '(assert (<= (* Y[0,0] Y[0,0]) -0.5))\n')
    v, _ = vp.verify(net, spec, 25.0, 'cpu')
    assert v == 'unsat'


def test_empty_input_region_prefilter(tmp_path):
    """x0 >= 0.9 AND x0^2 <= 0.5 is infeasible: the HC4 contractor must
    prove the input region empty -> vacuously unsat, no verification."""
    net = _sum_net(tmp_path)
    spec = _spec(tmp_path, 'empty',
                 '(assert (and (>= X[0,0] 0.9) (<= (* X[0,0] X[0,0]) 0.5)))\n'
                 '(assert (>= Y[0,0] -100.0))\n')
    v, d = vp.verify(net, spec, 25.0, 'cpu')
    assert v == 'unsat'
    assert d.get('witness') is None
