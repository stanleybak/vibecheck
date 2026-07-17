"""VNN-LIB standard CLI (vc2 port of v1 cli_standard): dispatch, supports,
network-flag resolution, verdict spelling, assignment parse/serialise, and
an end-to-end verify on a tiny generated instance."""
import gzip
import os
import sys

import numpy as np
import onnx
import pytest
import torch

from vibecheck import cli_standard as cs
from vibecheck.verify import _verdict_str, main as vmain


def test_name_version(capsys):
    assert cs.dispatch(['--name']) == 0
    assert capsys.readouterr().out.strip() == 'vibecheck'
    assert cs.dispatch(['--version']) == 0
    assert capsys.readouterr().out.strip() != ''


def test_supports_known_and_unknown(capsys):
    assert cs.dispatch(['supports', '--vnnlib-versions']) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ['1.0', '2.0']
    assert cs.run_supports(['--no-such-capability']) == 2
    assert 'unknown capability' in capsys.readouterr().err
    assert cs.run_supports([]) == 2


def test_supports_onnx_operators(capsys):
    assert cs.run_supports(['--onnx-operators']) == 0
    ops = capsys.readouterr().out.split()
    assert 'Relu' in ops and 'Conv' in ops


def test_parse_network_args_malformed():
    assert cs._parse_network_args(['f=a.onnx']) == [('f', 'a.onnx')]
    for bad in ['noequals', '=path', 'name=']:
        with pytest.raises(SystemExit) as e:
            cs._parse_network_args([bad])
        assert e.value.code == 2


def test_resolve_net_field_v1_spec(tmp_path):
    q = tmp_path / 'q.vnnlib'
    q.write_text('(declare-const X_0 Real)\n(assert (>= X_0 0))\n')
    assert cs._resolve_net_field(str(q), ['f=m.onnx']) == 'm.onnx'
    with pytest.raises(SystemExit):
        cs._resolve_net_field(str(q), ['f=a.onnx', 'g=b.onnx'])


def test_resolve_net_field_v2_pair_and_equal_to(tmp_path):
    q = tmp_path / 'q2.vnnlib'
    q.write_text('(declare-network f (declare-input X real [1]))\n'
                 '(declare-network g (equal-to f))\n')
    field = cs._resolve_net_field(str(q), ['f=a.onnx'])
    assert field == repr([('f', 'a.onnx'), ('g', 'a.onnx')])
    with pytest.raises(SystemExit):    # g needs no flag
        cs._resolve_net_field(str(q), ['f=a.onnx', 'g=b.onnx'])


def test_resolve_net_field_gzip_head(tmp_path):
    q = tmp_path / 'q3.vnnlib'
    with gzip.open(str(q) + '.gz', 'wt') as f:
        f.write('(declare-network h (declare-input X real [2]))\n')
    assert cs._resolve_net_field(str(q), ['h=n.onnx']) == 'n.onnx'


def test_verdict_str_styles():
    assert _verdict_str('timeout', 'vnncomp') == 'timeout'
    assert _verdict_str('timeout', 'standard') == 'timed-out'
    for tok in ('sat', 'unsat', 'unknown', 'error'):
        assert _verdict_str(tok, 'standard') == tok


def test_parse_assignment_v1_sexpr():
    text = '((X_0 0.5)\n (X_1 -1.0)\n (Y_0 2.0))'
    t = cs._parse_assignment(text)
    assert [(n, len(v)) for n, _, _, v in t] == [('X', 2), ('Y', 1)]


def test_parse_assignment_v2_blocks_and_serialise(tmp_path):
    text = 'X real [2]\n0.5\n-1.0\nY float32 [1]\n2.0'
    t = cs._parse_assignment(text)
    assert t[0][2] == (2,) and t[1][1] == 'float32'
    cs._serialise_assignment(text, str(tmp_path / 'out'))
    from onnx import numpy_helper
    with open(tmp_path / 'out' / 'X.pb', 'rb') as f:
        arr = numpy_helper.to_array(onnx.load_tensor_from_string(f.read()))
    assert arr.dtype == np.float64 and list(arr) == [0.5, -1.0]
    with pytest.raises(ValueError):
        cs._parse_assignment('garbage no header')


def _tiny_instance(tmp_path):
    """1-layer identity net + a trivially-unsat query."""
    import onnx.helper as oh
    W = np.eye(2, dtype=np.float32)
    node = oh.make_node('MatMul', ['X', 'W'], ['Y'])
    g = oh.make_graph(
        [node], 'tiny',
        [oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 2])],
        [oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(W, 'W')])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid('', 13)])
    net = tmp_path / 'tiny.onnx'
    onnx.save(m, str(net))
    q = tmp_path / 'tiny.vnnlib'
    q.write_text('(declare-const X_0 Real)\n(declare-const X_1 Real)\n'
                 '(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n'
                 '(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n'
                 '(assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n'
                 '(assert (>= Y_0 5.0))\n')     # unreachable: Y_0 <= 1
    return str(net), str(q)


def test_verify_end_to_end_unsat(tmp_path, capsys):
    net, q = _tiny_instance(tmp_path)
    code = vmain(['verify', q, '--network', f'f={net}', '--timeout', '30'])
    out = capsys.readouterr()
    assert out.out.splitlines()[0] == 'unsat'
    assert code == 0


def test_verify_implicit_and_results_file(tmp_path, capsys):
    net, q = _tiny_instance(tmp_path)
    rf = tmp_path / 'res.txt'
    code = vmain([q, '--network', f'f={net}', '--timeout', '30',
                  '--results-file', str(rf)])
    assert capsys.readouterr().out.splitlines()[0] == 'unsat'
    assert rf.read_text().splitlines()[0] == 'unsat'
    assert code == 0


def test_legacy_cli_untouched(tmp_path, capsys):
    net, q = _tiny_instance(tmp_path)
    rf = tmp_path / 'res2.txt'
    code = vmain(['--net', net, '--spec', q, '--timeout', '30',
                  '--results-file', str(rf)])
    assert rf.read_text().splitlines()[0] == 'unsat'
    assert code == 0


def test_nonlinear_augment_empty_region(tmp_path, capsys):
    """A nonlinear-v2 spec whose X-constraints are jointly infeasible is
    vacuously unsat via the input-region prefilter -- no net evaluation
    (the onnx path can even be bogus)."""
    q = tmp_path / 'empty.vnnlib'
    q.write_text(
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X real [1,2])'
        ' (declare-output Y real [1,1]))\n'
        '(assert (and (>= X[0,0] 30.0) (<= X[0,0] 40.0)))\n'
        '(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))\n'
        # the quadratic empties the region: X1^2 <= 1 but 200*X0 >= 6000
        '(assert (>= (* X[0,1] X[0,1]) (* X[0,0] 200.0)))\n'
        '(assert (> Y[0,0] 0.0))\n')
    net = tmp_path / 'missing.onnx'   # never touched
    code = vmain(['verify', str(q), '--network', f'f={net}',
                  '--timeout', '30'])
    assert capsys.readouterr().out.splitlines()[0] == 'unsat'
    assert code == 0
