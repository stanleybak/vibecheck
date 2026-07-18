"""Programmatic API: in-memory nets (torch module / ModelProto / bytes),
in-memory specs (VNNLIB text / Spec builder), and the VerifyResult surface.
Everything runs a tiny 2-relu net on CPU in well under a second."""
import os

import numpy as np
import pytest
import torch

from vibecheck import Spec, VerifyResult, verify
from vibecheck.frontend.vnnlib_loader import parse_vnnlib_text


class _TinyNet(torch.nn.Module):
    """y = relu(x) @ I: y_i = max(x_i, 0), so over x in [0,1]^2 the exact
    output range is [0,1]^2 -- verdicts below are ground truth by hand."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(2))
            self.lin.bias.zero_()

    def forward(self, x):
        return self.lin(torch.relu(x))


def _box_spec():
    """x in [0,1]^2, unsafe iff y_0 >= 2: unreachable -> unsat."""
    return Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[1.0, 0.0]], [-2.0])


def _sat_spec():
    """x in [0,1]^2, unsafe iff y_0 >= 0.5: reachable -> sat."""
    return Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[1.0, 0.0]], [-0.5])


def test_torch_module_unsat():
    r = verify(_TinyNet(), _box_spec(), timeout=20, device='cpu',
               example_input=torch.zeros(1, 2))
    assert isinstance(r, VerifyResult)
    assert r.verdict == 'unsat'
    assert r.counterexample is None
    assert r.exit_code == 0


def test_torch_module_sat_ce_replayed():
    r = verify(_TinyNet(), _sat_spec(), timeout=20, device='cpu',
               example_input=torch.zeros(1, 2))
    assert r.verdict == 'sat'
    x = r.counterexample['X'].ravel()
    y = r.counterexample['Y'].ravel()
    assert (x >= 0).all() and (x <= 1).all()      # witness inside the box
    assert y[0] >= 0.5                            # strict violation
    assert np.allclose(np.maximum(x, 0), y, atol=1e-6)


def test_torch_module_requires_example_input():
    with pytest.raises(ValueError, match='example_input'):
        verify(_TinyNet(), _box_spec(), timeout=5, device='cpu')


def test_modelproto_and_bytes(tmp_path):
    import onnx
    path = str(tmp_path / 'tiny.onnx')
    torch.onnx.export(_TinyNet().eval(), torch.zeros(1, 2), path,
                      dynamo=False)
    proto = onnx.load(path)
    assert verify(proto, _box_spec(), timeout=20,
                  device='cpu').verdict == 'unsat'
    assert verify(proto.SerializeToString(), _sat_spec(), timeout=20,
                  device='cpu').verdict == 'sat'


def test_vnnlib_text_spec():
    text = _sat_spec().to_vnnlib()
    assert text.lstrip().startswith('(')          # routed as text, not path
    r = verify(_TinyNet(), text, timeout=20, device='cpu',
               example_input=torch.zeros(1, 2))
    assert r.verdict == 'sat'


def test_spec_builder_roundtrips_through_parser():
    spec = parse_vnnlib_text(_box_spec().to_vnnlib())
    assert np.allclose(spec.x_lo, [0, 0]) and np.allclose(spec.x_hi, [1, 1])
    rows = spec.as_linear_queries(2)
    assert len(rows) == 1
    _, w, b = rows[0]
    # unsafe Y_0 >= 2 -> query row w=-e_0, bias=2 (safe iff w.y + b > 0)
    assert np.allclose(w, [-1.0, 0.0]) and np.isclose(b, 2.0)


def test_spec_builder_pairwise_and_unexpressible():
    # c*(y_0 - y_1) >= 0 maps to the pairwise atom (>= Y_0 Y_1)
    text = (Spec(x_lo=[0, 0], x_hi=[1, 1])
            .forbid([[2.0, -2.0]], [0.0]).to_vnnlib())
    assert '(>= Y_0 Y_1)' in text
    # a biased difference is outside the parser grammar: loud, at forbid()
    with pytest.raises(NotImplementedError, match='not expressible'):
        Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[1.0, -1.0]], [0.5])


def test_spec_builder_validation():
    with pytest.raises(ValueError, match='x_lo <= x_hi'):
        Spec(x_lo=[1], x_hi=[0])
    with pytest.raises(ValueError, match='vacuous'):
        Spec(x_lo=[0], x_hi=[1]).to_vnnlib()
    with pytest.raises(ValueError, match='biases'):
        Spec(x_lo=[0], x_hi=[1]).forbid([[1.0, 0.0]], [-2.0, 0.0])


def test_missing_spec_path_is_loud():
    with pytest.raises(FileNotFoundError, match='no_such_spec'):
        verify(_TinyNet(), 'no_such_spec.vnnlib', timeout=5, device='cpu',
               example_input=torch.zeros(1, 2))


def test_bad_net_type_is_loud():
    with pytest.raises(TypeError, match='unsupported net type'):
        verify(12345, _box_spec(), timeout=5, device='cpu')
