"""Quantized-net handler plumbing (handlers/quant_surrogate) and the
discrete-enumeration guard (handlers/cctsdb, discrete_enum): the ORT
oracle probe, quantized-op detection, and the loud NotImplementedError
fallback contract for non-applicable instances."""
import numpy as np
import pytest
import torch

from vibecheck.handlers.quant_surrogate import (
    detect_quant_oracle, has_quantized_ops, resolve_saturation)


def _plain_float_onnx(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W = numpy_helper.from_array(np.eye(2, dtype=np.float32), 'W')
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['h']),
         helper.make_node('Relu', ['h'], ['Y'])],
        'g',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'plain.onnx')
    onnx.save(m, p)
    return p


def _qdq_onnx(tmp_path):
    """A minimal QDQ-format model (QuantizeLinear -> DequantizeLinear),
    the format has_quantized_ops detects (smart_turn-style nets)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('QuantizeLinear', ['X', 's', 'z'], ['q']),
         helper.make_node('DequantizeLinear', ['q', 's', 'z'], ['Y'])],
        'qdq',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(np.float32(0.1), 's'),
         numpy_helper.from_array(np.uint8(128), 'z')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 19)])
    m.ir_version = 9
    p = str(tmp_path / 'quant.onnx')
    onnx.save(m, p)
    return p


def test_oracle_probe_reports_a_regime():
    """The ~3ms QLinearMatMul probe characterizes THIS machine's ORT: the
    answer is one of the two known kernels, and it is stable."""
    r1 = detect_quant_oracle()
    assert r1 in ('exact', 'saturating')
    assert detect_quant_oracle() == r1


def test_resolve_saturation_forced_and_auto():
    logs = []
    assert resolve_saturation('on', log=logs.append) is True
    assert resolve_saturation('off', log=logs.append) is False
    auto = resolve_saturation('auto', log=logs.append)
    assert auto == (detect_quant_oracle() == 'saturating')
    assert any('saturation' in m for m in logs)


def test_has_quantized_ops(tmp_path):
    assert has_quantized_ops(_plain_float_onnx(tmp_path)) is False
    assert has_quantized_ops(_qdq_onnx(tmp_path)) is True


def _spec_file(tmp_path, x_lo, x_hi, W, b):
    from vibecheck import Spec
    p = str(tmp_path / 'prop.vnnlib')
    with open(p, 'w') as f:
        f.write(Spec(x_lo=x_lo, x_hi=x_hi).forbid(W, b).to_vnnlib())
    return p


def test_discrete_enum_rejects_non_integer_box(tmp_path):
    """The fallback contract: a fractional-bounds box is NOT a discrete
    patch enumeration; the handler must raise (so the caller re-raises the
    original load error), never enumerate a continuous range."""
    from vibecheck.handlers.discrete_enum import try_discrete_enum
    vnnlib = _spec_file(tmp_path, [0, 0], [0.7, 1], [[1.0, 0.0]], [-2.0])
    with pytest.raises(NotImplementedError):
        try_discrete_enum(_plain_float_onnx(tmp_path), vnnlib, timeout=5,
                          log=lambda m: None)


def test_cctsdb_structure_detector(tmp_path):
    from vibecheck.handlers.cctsdb import has_cctsdb_structure
    frac = _spec_file(tmp_path, [0, 0], [0.7, 1], [[1.0, 0.0]], [-2.0])
    assert not has_cctsdb_structure(_plain_float_onnx(tmp_path), frac)
    grid = _spec_file(tmp_path, [0, 0], [2, 2], [[1.0, 0.0]], [-2.0])
    assert has_cctsdb_structure(_plain_float_onnx(tmp_path), grid)


def _sum_net_onnx(tmp_path):
    """y = x0 + x1 (1 output), for hand-checkable grid enumeration."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W = numpy_helper.from_array(np.ones((2, 1), np.float32), 'W')
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['Y'])],
        'sum',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'sum.onnx')
    onnx.save(m, p)
    return p


def test_discrete_enum_grid_verdicts(tmp_path):
    """Exhaustive integer-grid enumeration on y = x0+x1 over the EXCLUSIVE-hi
    grid {0,1}^2 (the ABC convention): max y = 2, so 'y >= 3.9' is unsat by
    complete enumeration and 'y >= 1.5' is sat at placement (1,1)."""
    from vibecheck.handlers.discrete_enum import try_discrete_enum
    net = _sum_net_onnx(tmp_path)
    safe = _spec_file(tmp_path, [0, 0], [2, 2], [[1.0]], [-3.9])
    verdict, details = try_discrete_enum(net, safe, timeout=20,
                                         log=lambda m: None)
    assert verdict == 'unsat'
    unsafe = _spec_file(tmp_path, [0, 0], [2, 2], [[1.0]], [-1.5])
    verdict, details = try_discrete_enum(net, unsafe, timeout=20,
                                         log=lambda m: None)
    assert verdict == 'sat'
    assert np.allclose(details['witness'], [1.0, 1.0])
