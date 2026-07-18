"""THOROUGH tier: VNNLIB parsing edges (frontend/vnnlib_loader) and the
witness chokepoint's zero-tolerance policy (frontend/witness)."""
import gzip

import numpy as np
import pytest

from vibecheck.frontend.vnnlib_loader import load_vnnlib, parse_vnnlib_text
from vibecheck.frontend.witness import (_clamp_witness_to_box,
                                        _validate_sat_witness,
                                        _vnnlib_version)

pytestmark = pytest.mark.thorough


def test_or_blocks_union_input_box():
    """Per-block X bounds must UNION into the global box (the acasxu
    prop_6 / nn4sys lindex fix: successive blocks must not overwrite)."""
    spec = parse_vnnlib_text("""
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.0) (<= X_0 0.25) (>= Y_0 1.0))
  (and (>= X_0 0.75) (<= X_0 1.0) (>= Y_0 2.0))))
""")
    assert np.isclose(spec.x_lo[0], 0.0) and np.isclose(spec.x_hi[0], 1.0)
    assert len(spec.disjuncts) == 2
    rows = spec.as_linear_queries(1)
    assert sorted(b for _, _, b in rows) == [1.0, 2.0]


def test_top_level_y_constraint_ands_into_every_disjunct():
    """A top-level (assert (<= Y_1 v)) beside the OR must AND into every
    disjunct (the lsnc level-set band; dropping it is a false-SAT trap)."""
    spec = parse_vnnlib_text("""
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (or (and (>= Y_0 0.5)) (and (>= Y_0 2.5))))
(assert (<= Y_1 0.4))
""")
    assert len(spec.disjuncts) == 2
    for conj in spec.disjuncts:
        assert len(conj.constraints) == 2       # its own row + the band row


def test_strict_atoms_recorded():
    """Strict `>` must be distinguished from `>=` (the CE check rejects
    the boundary on strict atoms; zero-tolerance policy)."""
    spec = parse_vnnlib_text("""
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (or (and (> Y_0 0.5))))
""")
    (conj,) = spec.disjuncts
    (c,) = conj.constraints
    assert c.strict is True
    spec2 = parse_vnnlib_text("""
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (or (and (>= Y_0 0.5))))
""")
    assert spec2.disjuncts[0].constraints[0].strict is False


def test_gz_spec_loads(tmp_path):
    text = ('(declare-const X_0 Real)\n(declare-const Y_0 Real)\n'
            '(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n'
            '(assert (>= Y_0 2.0))\n')
    p = str(tmp_path / 'prop.vnnlib.gz')
    with gzip.open(p, 'wt') as f:
        f.write(text)
    spec = load_vnnlib(p)
    assert np.isclose(spec.x_hi[0], 1.0)


def test_vnnlib_version_detection(tmp_path):
    v1 = str(tmp_path / 'v1.vnnlib')
    with open(v1, 'w') as f:
        f.write('(declare-const X_0 Real)\n(assert (>= X_0 0.0))\n')
    v2 = str(tmp_path / 'v2.vnnlib')
    with open(v2, 'w') as f:
        f.write('; VNNLIB 2.0\n(declare-network f\n'
                '  (declare-input X Real [1,2])\n'
                '  (declare-output Y Real [1,1]))\n'
                '(assert (>= X[0,0] 0.0))\n')
    assert _vnnlib_version(v1) == '1.0'
    assert _vnnlib_version(v2) == '2.0'


def _identity_onnx(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['Y'])],
        'id',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.eye(2, dtype=np.float32), 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'id.onnx')
    onnx.save(m, p)
    return p


def _box_spec_text(atom):
    return f"""
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (or (and {atom})))
"""


def test_witness_boundary_strict_vs_nonstrict(tmp_path):
    """The chokepoint's boundary rule on the identity net: for a STRICT
    atom (< Y_0 0.5) a witness with y EXACTLY 0.5 must be REJECTED (the
    boundary is not a strict violation); for the NON-strict (<= Y_0 0.5)
    the boundary satisfies the atom and is scorer-faithful to accept.
    Interior points pass either way."""
    net = _identity_onnx(tmp_path)
    strict = parse_vnnlib_text(_box_spec_text('(< Y_0 0.5)'))
    ok, info = _validate_sat_witness(net, strict,
                                     np.array([0.4, 0.0], np.float64))
    assert ok, info
    ok_b, _ = _validate_sat_witness(net, strict,
                                    np.array([0.5, 0.0], np.float64))
    assert not ok_b                              # boundary is NOT a strict CE
    nonstrict = parse_vnnlib_text(_box_spec_text('(<= Y_0 0.5)'))
    ok_c, _ = _validate_sat_witness(net, nonstrict,
                                    np.array([0.5, 0.0], np.float64))
    assert ok_c                                  # closure atom: boundary OK


def test_witness_out_of_box_rejected_but_clamp_repairs(tmp_path):
    """A raw candidate slightly outside the box may be clamped IN and
    re-validated (repair, not acceptance); far outside stays rejected."""
    net = _identity_onnx(tmp_path)
    spec = parse_vnnlib_text("""
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (or (and (<= Y_0 0.5))))
""")
    # 1e-6 outside the lower box edge: clamp repair -> valid interior CE
    ok, info = _validate_sat_witness(net, spec,
                                     np.array([-1e-6, 0.0], np.float64))
    assert ok, info
    w = np.asarray(info['witness_inbox'])
    assert (w >= 0).all() and (w <= 1).all()     # emitted point is IN box
    # 0.2 outside: no repair
    ok2, _ = _validate_sat_witness(net, spec,
                                   np.array([-0.2, 0.0], np.float64))
    assert not ok2


def test_clamp_witness_to_box_is_strictly_inside():
    lo = np.zeros(3)
    hi = np.ones(3)
    w = np.array([-0.5, 0.5, 1.5])
    c = _clamp_witness_to_box(w, lo, hi)
    assert (c >= lo).all() and (c <= hi).all()
    assert np.isclose(c[1], 0.5)
