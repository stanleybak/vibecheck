"""End-to-end pipeline routes (verify.verify) on the seed-9 2-6-6-1 net
from test_bab_relu_split: unlike the trivially-crown-closable api tests,
this drives the real ladder -- PGD, interval/zono, CROWN, alpha-CROWN,
refine probe, input-split BaB -- and the witness emission + results-file
authority on the sat side."""
import numpy as np
import pytest
import torch

from test_bab_relu_split import THETA_SAT, THETA_UNSAT, _onnx_path
from vibecheck import Spec, verify


def _spec(theta):
    return Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[-1.0]], [theta])


def test_pipeline_unsat_beyond_root(tmp_path):
    """THETA_UNSAT is provable only past the root bound (premise pinned in
    test_bab_relu_split): the full pipeline must still land unsat."""
    r = verify(_onnx_path(tmp_path), _spec(THETA_UNSAT), timeout=50,
               device='cpu')
    assert r.verdict == 'unsat'
    assert r.exit_code == 0 and r.counterexample is None


def test_pipeline_sat_with_replayed_witness(tmp_path):
    onnx_path = _onnx_path(tmp_path)
    rf = str(tmp_path / 'results.txt')
    r = verify(onnx_path, _spec(THETA_SAT), timeout=50, device='cpu',
               results_file=rf)
    assert r.verdict == 'sat'
    x = r.counterexample['X'].ravel()
    y = r.counterexample['Y'].ravel()
    assert (x >= 0).all() and (x <= 1).all()
    assert y[0] < THETA_SAT                     # strict violation
    # Y in the result is the ORT replay of X on the original onnx
    import onnxruntime as ort
    s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    y_ort = s.run(None, {'X': x.reshape(1, 2).astype(np.float32)})[0]
    assert np.allclose(y_ort.ravel().astype(np.float64), y, atol=1e-6)
    # and the results file (the verdict authority) says the same thing
    with open(rf) as f:
        lines = f.read().splitlines()
    assert lines[0] == 'sat' and len(lines) > 1


def test_pipeline_timeout_budget_respected(tmp_path):
    """A near-zero budget must come back timeout/unknown -- never a verdict
    invented under pressure."""
    r = verify(_onnx_path(tmp_path), _spec(THETA_UNSAT), timeout=0.01,
               device='cpu', extra_args=['--no-attack'])
    assert r.verdict in ('timeout', 'unknown')


def test_pipeline_no_attack_never_sat(tmp_path):
    """--no-attack (soundness-sweep mode): no CE can be produced, so a sat
    instance must degrade to unknown/timeout, never 'sat' and never a
    false 'unsat'."""
    r = verify(_onnx_path(tmp_path), _spec(THETA_SAT), timeout=2,
               device='cpu', extra_args=['--no-attack'])
    assert r.verdict in ('unknown', 'timeout')
