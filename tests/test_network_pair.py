"""Network-pair merge (frontend/network_pair): a synthetic two-network
instance in the isomorphic_acasxu spec shape. build_merged_instance must
produce a merged onnx whose ORT forward matches running f and g separately
(its own 120-sample oracle also gates this at ORACLE_TOL) plus a v1 spec
that parses. The pair --net field parser must reject malformed input."""
import os

import numpy as np
import pytest

from vibecheck.frontend import network_pair as npair

_rng = np.random.default_rng(11)


def _linear_onnx(tmp_path, name, W, b):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['h']),
         helper.make_node('Add', ['h', 'B'], ['Y'])],
        name,
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(W.T.astype(np.float32).copy(), 'W'),
         numpy_helper.from_array(b.astype(np.float32), 'B')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / f'{name}.onnx')
    onnx.save(m, p)
    return p


_PAIR_SPEC = """; synthetic isomorphic pair
(declare-network f (isomorphic-to g))
(declare-network g)
(assert (and (<= X_f[0] 1.0) (>= X_f[0] 0.0)))
(assert (and (<= X_f[1] 0.5) (>= X_f[1] -0.5)))
(assert (or (>= Y_f[0] Y_g[0])))
"""


def _instance(tmp_path):
    f = _linear_onnx(tmp_path, 'f', _rng.normal(size=(2, 2)),
                     _rng.normal(size=2))
    g = _linear_onnx(tmp_path, 'g', _rng.normal(size=(2, 2)),
                     _rng.normal(size=2))
    spec = str(tmp_path / 'pair.vnnlib')
    with open(spec, 'w') as fh:
        fh.write(_PAIR_SPEC)
    return f, g, spec


def test_pair_field_parse_and_detect(tmp_path):
    f, g, spec = _instance(tmp_path)
    field = repr([('f', f), ('g', g)])
    assert npair.is_network_pair_net_field(field)
    assert not npair.is_network_pair_net_field(f)
    assert npair.parse_network_field(field) == [('f', f), ('g', g)]
    assert npair.parse_network_field('plain.onnx') is None
    with pytest.raises(Exception):
        npair.parse_network_field('[not a list')
    text = open(spec).read()
    assert npair.detect_kind(text) == 'iso'
    assert npair.declared_networks(text) == ['f', 'g']


def test_build_merged_instance_oracle_and_emit(tmp_path):
    """The full merge: an ORT-oracle-gated merged onnx + an emitted v1 spec
    that round-trips through the standard loaders."""
    import onnxruntime as ort
    f, g, spec = _instance(tmp_path)
    merged_onnx, merged_vnnlib = npair.build_merged_instance(
        repr([('f', f), ('g', g)]), spec)   # run_oracle=True gates the merge
    assert os.path.isfile(merged_onnx) and os.path.isfile(merged_vnnlib)

    from vibecheck.core import graph
    net = graph.load(merged_onnx)           # merged net loads into the core IR

    from vibecheck.frontend.vnnlib_loader import load_vnnlib
    mspec = load_vnnlib(merged_vnnlib)      # emitted v1 spec parses
    assert len(mspec.x_lo) == net.ops[net.input_name].n

    # independent replay: iso semantics feed BOTH nets the same base input,
    # and the merged output is the (sign-normalized) atom value
    # Y_f[0] - Y_g[0]. The 120-sample build oracle already gated exact
    # semantics; this pins the wiring end-to-end from outside.
    assert len(mspec.x_lo) == 2                      # no relational coord

    def run(p, x):
        s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        i = s.get_inputs()[0]
        return s.run(None, {i.name: x.reshape(1, -1).astype(np.float32)})[0]

    x = _rng.uniform([0, -0.5], [1, 0.5]).astype(np.float32)
    ym = run(merged_onnx, x).ravel()
    atom = run(f, x).ravel()[0] - run(g, x).ravel()[0]
    assert ym.shape == (1,)                          # one output atom
    assert np.isclose(abs(ym[0]), abs(atom), atol=1e-5), (ym, atom)
