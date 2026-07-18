"""ReLU-phase splitting BaB (core/search.relu_split_bab) and the dual-LP
escalation (core/dual_lp.certify_queries) on a net where the root bound
provably cannot close the query.

The net is a fixed seeded 2-6-6-1 double-relu MLP chosen (seed sweep) so
that over x in [0,1]^2:
  * a 1001x1001 grid gives min y = 0.0000, and the inf-norm Lipschitz
    bound L = |w3||W2||W1| < 17 caps the true min at
    grid_min - L*h*sqrt(2)/2 > -0.02  (h = 1e-3), while
  * the root alpha-CROWN bound is ~ -0.52.
So the query "prove y > -0.15" is TRUE with margin (referee: grid +
Lipschitz, independent of the code under test) yet undecidable at the
root: any unsat verdict had to come from actual branching.
"""
import time

import numpy as np
import pytest
import torch

from vibecheck import Spec
from vibecheck.core import backward, forward
from vibecheck.core.dual_lp import certify_queries
from vibecheck.core.graph import Net, Op
from vibecheck.core.linmap import Dense
from vibecheck.core.search import relu_split_bab
from vibecheck.frontend.vnnlib_loader import parse_vnnlib_text

_H = 6
_R = np.random.default_rng(9)
_W1 = _R.normal(size=(_H, 2)).astype(np.float32)
_B1 = (_R.normal(size=_H) * 0.3).astype(np.float32)
_W2 = (_R.normal(size=(_H, _H)) / np.sqrt(_H)).astype(np.float32)
_B2 = (_R.normal(size=_H) * 0.3).astype(np.float32)
_W3 = _R.normal(size=(1, _H)).astype(np.float32)

THETA_UNSAT = -0.15      # true: min y >= -0.02 (grid + Lipschitz referee)
THETA_DUAL = -0.3        # true, and within the dual-LP state's reach
THETA_SAT = 0.1          # false: ~61% of the box has y < 0.1


def _net():
    return Net({'x': Op('x', 'input', (), (2,), 2),
                'h1': Op('h1', 'linmap', ('x',), (_H,), _H,
                         lm=Dense(_W1, _B1)),
                'r1': Op('r1', 'nonlin', ('h1',), (_H,), _H, fn='relu'),
                'h2': Op('h2', 'linmap', ('r1',), (_H,), _H,
                         lm=Dense(_W2, _B2)),
                'r2': Op('r2', 'nonlin', ('h2',), (_H,), _H, fn='relu'),
                'y': Op('y', 'linmap', ('r2',), (1,), 1,
                        lm=Dense(_W3, np.zeros(1, np.float32)))},
               ['h1', 'r1', 'h2', 'r2', 'y'], 'x', 'y')


def _onnx_path(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W1'], ['a0']),
         helper.make_node('Add', ['a0', 'B1'], ['a']),
         helper.make_node('Relu', ['a'], ['ra']),
         helper.make_node('MatMul', ['ra', 'W2'], ['b0']),
         helper.make_node('Add', ['b0', 'B2'], ['b']),
         helper.make_node('Relu', ['b'], ['rb']),
         helper.make_node('MatMul', ['rb', 'W3'], ['Y'])],
        'g',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(_W1.T.copy(), 'W1'),
         numpy_helper.from_array(_B1, 'B1'),
         numpy_helper.from_array(_W2.T.copy(), 'W2'),
         numpy_helper.from_array(_B2, 'B2'),
         numpy_helper.from_array(_W3.T.copy(), 'W3')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'bab.onnx')
    import onnx as _onnx
    _onnx.save(m, p)
    return p


def _query(theta):
    """Spec 'unsafe iff y <= theta' + its query row (refute = prove y > theta)."""
    spec = parse_vnnlib_text(
        Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[-1.0]], [theta]).to_vnnlib())
    rows = spec.as_linear_queries(1)
    W = torch.tensor(np.stack([w for _, w, _ in rows]), dtype=torch.float32)
    bias = torch.tensor([b for _, _, b in rows], dtype=torch.float32)
    disj = torch.tensor([d for d, _, _ in rows])
    return spec, W, bias, disj


def test_premise_referee_and_root_gap():
    """Pin the file's premise: the grid+Lipschitz referee proves the unsat
    query true with margin, and root alpha-CROWN cannot close it."""
    net = _net()
    g = torch.linspace(0, 1, 1001)
    xs = torch.cartesian_prod(g, g)
    gmin = min(forward.point(net, xs[i:i + 250000]).min().item()
               for i in range(0, len(xs), 250000))
    L = float((np.abs(_W3) @ np.abs(_W2) @ np.abs(_W1)).sum())
    true_min_floor = gmin - L * 1e-3 * np.sqrt(2) / 2
    assert true_min_floor > THETA_UNSAT + 0.05
    lo1, hi1 = torch.zeros(1, 2), torch.ones(1, 2)
    inter = backward.intermediates(net, lo1, hi1)
    lba = float(backward.alpha_crown(net, lo1, hi1,
                                     torch.tensor([[1.0]]), inter).min())
    assert lba < THETA_DUAL - 0.05      # root alpha-CROWN fails both true
    assert lba < THETA_UNSAT - 0.05     # queries: verdicts require search
    assert gmin < THETA_SAT             # and the sat query is really violated


def test_relu_split_bab_unsat_needs_splits(tmp_path):
    net = _net()
    spec, W, bias, disj = _query(THETA_UNSAT)
    verdict, info = relu_split_bab(
        net, spec, W, bias, disj, torch.zeros(2), torch.ones(2),
        deadline=time.time() + 60, device='cpu',
        onnx_path=_onnx_path(tmp_path))
    assert verdict == 'unsat', (verdict, info)
    assert info['bounded'] > 1          # root could not close it: it branched


def test_relu_split_bab_finds_validated_ce(tmp_path):
    """attack_every=1 means attack EVERY round (regression: the round gate
    was `rounds % attack_every == 1`, never true at 1)."""
    net = _net()
    spec, W, bias, disj = _query(THETA_SAT)
    verdict, info = relu_split_bab(
        net, spec, W, bias, disj, torch.zeros(2), torch.ones(2),
        deadline=time.time() + 60, device='cpu',
        onnx_path=_onnx_path(tmp_path), attack_every=1)
    assert verdict == 'sat', (verdict, info)
    x = torch.tensor(np.asarray(info['witness'], np.float32)).reshape(1, 2)
    assert (x >= -1e-6).all() and (x <= 1 + 1e-6).all()
    assert forward.point(net, x).item() < THETA_SAT   # strict violation


def test_relu_split_bab_falsifies_at_split_exhaustion(tmp_path):
    """With the attack cadence effectively off (attack_every huge, wall
    time << the 8s time gate), a sat instance exhausts its splits; the
    exhaustion path must then try to falsify once and return the validated
    CE (regression: it returned 'unknown' without the attempt)."""
    net = _net()
    spec, W, bias, disj = _query(THETA_SAT)
    verdict, info = relu_split_bab(
        net, spec, W, bias, disj, torch.zeros(2), torch.ones(2),
        deadline=time.time() + 60, device='cpu',
        onnx_path=_onnx_path(tmp_path), attack_every=10_000)
    assert verdict == 'sat', (verdict, info)
    x = torch.tensor(np.asarray(info['witness'], np.float32)).reshape(1, 2)
    assert forward.point(net, x).item() < THETA_SAT


def test_relu_split_bab_timeout_disposition(tmp_path):
    """An already-expired deadline must return timeout, never a verdict."""
    net = _net()
    spec, W, bias, disj = _query(THETA_UNSAT)
    verdict, _ = relu_split_bab(
        net, spec, W, bias, disj, torch.zeros(2), torch.ones(2),
        deadline=time.time() - 1, device='cpu',
        onnx_path=_onnx_path(tmp_path))
    assert verdict == 'timeout'


def test_dual_certify_queries_refutes():
    """The dual-ascent LP escalation refutes a root-open query (positive
    dual bound certifies; soundness is structural). THETA_DUAL sits inside
    the zonotope-LP state's reach on this net (THETA_UNSAT is beyond it:
    the dual honestly exhausts splits there and reports nothing)."""
    net = _net()
    spec, W, bias, disj = _query(THETA_DUAL)
    lo1, hi1 = torch.zeros(1, 2), torch.ones(1, 2)
    inter = backward.intermediates(net, lo1, hi1)
    refuted = certify_queries(net, spec, W, bias, disj, lo1, hi1, inter,
                              open_d={0}, deadline=time.time() + 60,
                              device='cpu')
    assert refuted == {0}


def test_dual_certify_queries_cannot_refute_false_property():
    """theta above the true min: the disjunct must stay open (a refutation
    here would be a soundness bug)."""
    net = _net()
    spec, W, bias, disj = _query(THETA_SAT)
    lo1, hi1 = torch.zeros(1, 2), torch.ones(1, 2)
    inter = backward.intermediates(net, lo1, hi1)
    refuted = certify_queries(net, spec, W, bias, disj, lo1, hi1, inter,
                              open_d={0}, deadline=time.time() + 20,
                              device='cpu')
    assert 0 not in refuted
