"""The quantized-surrogate attack handler end to end
(handlers/quant_surrogate) on a synthetic QDQ net.

Net: X[1,4] float -> QDQ(X) -> Gemm(W, b) -> Y[1,2], the smart_turn
format. The handler must build the float and fake-quant surrogates, run
the PGD attack on the torch conversion, and only ever emit a sat whose
witness survives the unified ORT gate on the ORIGINAL onnx."""
import numpy as np
import pytest
import torch


_W = np.array([[1.0, -1.0, 0.5, 0.25],
               [-0.5, 1.0, -0.25, 0.5]], np.float32)
_B = np.array([0.1, -0.1], np.float32)


def _qdq_gemm_onnx(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('QuantizeLinear', ['X', 's', 'z'], ['q']),
         helper.make_node('DequantizeLinear', ['q', 's', 'z'], ['dq']),
         helper.make_node('Gemm', ['dq', 'W', 'B'], ['Y'], transB=1)],
        'qdq_gemm',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.float32(0.02), 's'),
         numpy_helper.from_array(np.uint8(128), 'z'),
         numpy_helper.from_array(_W, 'W'),
         numpy_helper.from_array(_B, 'B')])
    # opset 13: QDQ exists and onnx2torch has converters for every node
    # the float surrogate emits (Identity has no opset-19 converter)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    p = str(tmp_path / 'qdq_gemm.onnx')
    onnx.save(m, p)
    return p


def _spec_file(tmp_path, theta):
    """Unsafe iff Y_0 > theta over x in [-1, 1]^4 (the STRICT v1 atom form
    the surrogate spec parser accepts -- the smart_turn shape)."""
    p = str(tmp_path / f'unsafe_gt_{theta}.vnnlib')
    L = [f'(declare-const X_{i} Real)' for i in range(4)]
    L += [f'(declare-const Y_{i} Real)' for i in range(2)]
    for i in range(4):
        L.append(f'(assert (>= X_{i} -1.0))')
        L.append(f'(assert (<= X_{i} 1.0))')
    L.append(f'(assert (> Y_0 {theta}))')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return p


def test_float_surrogate_matches_ort_up_to_quant_step(tmp_path):
    """The float surrogate strips QDQ: it must agree with the original
    ORT run within the quantization step propagated through |W|."""
    import onnxruntime as ort
    from vibecheck.handlers.quant_surrogate import build_float_surrogate
    src = _qdq_gemm_onnx(tmp_path)
    dst = str(tmp_path / 'float_surrogate.onnx')
    build_float_surrogate(src, dst)
    s0 = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s1 = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    tol = 0.02 * float(np.abs(_W).sum(1).max())      # one quant step thru W
    for _ in range(8):
        x = np.random.default_rng(_ or 0).uniform(
            -1, 1, size=(1, 4)).astype(np.float32)
        y0 = s0.run(None, {'X': x})[0]
        y1 = s1.run(None, {'X': x})[0]
        assert np.abs(y0 - y1).max() <= tol + 1e-6


def test_fakequant_surrogate_and_torch_conversion(tmp_path):
    """The fake-quant surrogate reproduces the ORIGINAL quantized forward
    exactly (round-trip through onnx2torch included)."""
    import onnxruntime as ort
    from vibecheck.handlers.quant_surrogate import (
        build_fakequant_surrogate, convert_onnx_to_torch)
    src = _qdq_gemm_onnx(tmp_path)
    dst = str(tmp_path / 'fakequant.onnx')
    build_fakequant_surrogate(src, dst)
    s0 = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    mod = convert_onnx_to_torch(dst)
    x = np.random.default_rng(1).uniform(-1, 1, size=(1, 4)) \
        .astype(np.float32)
    y_ort = s0.run(None, {'X': x})[0]
    with torch.no_grad():
        y_t = mod(torch.tensor(x)).numpy()
    assert np.allclose(y_ort, y_t, atol=1e-5), (y_ort, y_t)


def test_parse_box_and_output_and_margin(tmp_path):
    from vibecheck.handlers.quant_surrogate import (
        dnf_margin_np, parse_box_and_output)
    src = _qdq_gemm_onnx(tmp_path)          # noqa: F841 (input shapes side)
    spec = parse_box_and_output(_spec_file(tmp_path, 1.0))
    assert len(spec.inputs) == 1
    _, shape, lo, hi = spec.inputs[0]
    assert int(np.prod(shape)) == 4
    assert np.allclose(lo, -1.0) and np.allclose(hi, 1.0)
    # margin > 0 iff the unsafe region (Y_0 > 1.0) is HIT
    assert dnf_margin_np(spec.out_dnf, np.array([1.5, 0.0])) > 0
    assert dnf_margin_np(spec.out_dnf, np.array([0.2, 0.0])) < 0


def test_try_quant_surrogate_finds_validated_ce(tmp_path, monkeypatch):
    """Reachable unsafe region (max Y_0 ~ 2.85 in the box): the handler
    must return sat with a CE that replays through ORT on the original.
    TMPDIR is isolated: the surrogate cache is existence-only and keyed by
    onnx BASENAME, so a shared /tmp dir would reuse another test's (or
    another net's) surrogate."""
    from vibecheck.handlers.quant_surrogate import try_quant_surrogate
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    src = _qdq_gemm_onnx(tmp_path)
    verdict, details = try_quant_surrogate(
        src, _spec_file(tmp_path, 1.0), timeout=30, device='cpu',
        log=lambda m: None)
    assert verdict == 'sat', (verdict, details)
    assert 'ce_sexpr' in details and 'Y' in details['ce_sexpr']
    import onnxruntime as ort
    x = np.concatenate([np.asarray(w).ravel()
                        for w in details['witness_multi']])
    s = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    y = s.run(None, {'X': x.reshape(1, 4).astype(np.float32)})[0].ravel()
    assert y[0] >= 1.0                       # genuinely in the unsafe region
    assert (np.abs(x) <= 1 + 1e-9).all()


def test_try_quant_surrogate_unreachable_is_never_sat(tmp_path,
                                                      monkeypatch):
    """Y_0 > 10 is unreachable (|W| row sum + |b| < 3): attack-only
    handler must come back timeout/unknown, never sat, never unsat."""
    from vibecheck.handlers.quant_surrogate import try_quant_surrogate
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    src = _qdq_gemm_onnx(tmp_path)
    verdict, _ = try_quant_surrogate(
        src, _spec_file(tmp_path, 10.0), timeout=8, device='cpu',
        log=lambda m: None)
    assert verdict in ('timeout', 'unknown')


def test_surrogate_attack_forced_saturating_regime(tmp_path, monkeypatch):
    """saturation='on' (forced, regardless of this machine's ORT regime):
    the fake-quant surrogate gets the saturating twin kernels grafted in
    and the attack still only emits ORT-validated counterexamples."""
    from vibecheck.handlers.quant_surrogate import (parse_box_and_output,
                                                    surrogate_attack)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    src = _qdq_gemm_onnx(tmp_path)
    spec_file = _spec_file(tmp_path, 1.0)
    verdict, witness = surrogate_attack(
        src, spec_file, timeout=25, device='cpu', saturation='on',
        log=lambda m: None, spec=parse_box_and_output(spec_file))
    assert verdict == 'sat' and witness is not None
    import onnxruntime as ort
    x = np.concatenate([np.asarray(w).ravel() for w in witness])
    s = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    y = s.run(None, {'X': x.reshape(1, 4).astype(np.float32)})[0].ravel()
    assert y[0] > 1.0                            # strict, on the ORIGINAL


def test_try_quant_surrogate_rejects_unquantized(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from vibecheck.handlers.quant_surrogate import try_quant_surrogate
    g = helper.make_graph(
        [helper.make_node('Gemm', ['X', 'W', 'B'], ['Y'], transB=1)],
        'plain',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(_W, 'W'), numpy_helper.from_array(_B, 'B')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    m.ir_version = 8
    p = str(tmp_path / 'plain.onnx')
    onnx.save(m, p)
    with pytest.raises(NotImplementedError):
        try_quant_surrogate(p, _spec_file(tmp_path, 1.0), timeout=5,
                            log=lambda m: None)
