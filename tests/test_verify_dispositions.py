"""Flat-runner dispositions (verify._run_flat): the results file is the
verdict authority, and crash/OOM outcomes map to the documented verdicts
(OOM -> honest 'unknown', crash -> 'error', never silently swallowed)."""
import numpy as np
import pytest
import torch

import vibecheck.pipeline as vmod


def _run(monkeypatch, tmp_path, outcome, extra=()):
    """Drive _legacy_main with the pipeline replaced by `outcome` (a result
    tuple or an exception instance to raise)."""
    def fake_verify(net, spec, timeout, device, **kw):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(vmod, 'verify', fake_verify)
    rf = str(tmp_path / 'out.txt')
    code = vmod._legacy_main(['--net', 'dummy.onnx', '--spec', 'dummy.vnnlib',
                              '--timeout', '5', '--device', 'cpu',
                              '--results-file', rf, *extra])
    with open(rf) as f:
        return code, f.read().splitlines()


def test_unsat_writes_results_file_and_exit_0(monkeypatch, tmp_path):
    code, lines = _run(monkeypatch, tmp_path, ('unsat', {'time': 0.1}))
    assert code == 0 and lines[0] == 'unsat'


def test_unknown_exit_1(monkeypatch, tmp_path):
    code, lines = _run(monkeypatch, tmp_path, ('unknown', {'time': 0.1}))
    assert code == 1 and lines[0] == 'unknown'


def test_timeout_spelling_vnncomp_vs_standard(monkeypatch, tmp_path):
    _, lines = _run(monkeypatch, tmp_path, ('timeout', {'time': 0.1}))
    assert lines[0] == 'timeout'                       # competition spelling
    _, lines = _run(monkeypatch, tmp_path, ('timeout', {'time': 0.1}),
                    extra=['--verdict-style', 'standard'])
    assert lines[0] == 'timed-out'                     # VNN-LIB standard


def test_crash_maps_to_error_with_cause(monkeypatch, tmp_path):
    code, lines = _run(monkeypatch, tmp_path, ValueError('boom'))
    assert code == 2
    assert lines[0] == 'error' and 'ValueError: boom' in lines[1]


def test_oom_message_maps_to_unknown_not_error(monkeypatch, tmp_path):
    code, lines = _run(monkeypatch, tmp_path,
                       RuntimeError('CUDA out of memory. Tried to allocate'))
    assert code == 1 and lines[0] == 'unknown'


def test_cuda_oom_type_maps_to_unknown(monkeypatch, tmp_path):
    code, lines = _run(monkeypatch, tmp_path,
                       torch.cuda.OutOfMemoryError('CUDA error'))
    assert code == 1 and lines[0] == 'unknown'


def test_sat_writes_formatted_counterexample(monkeypatch, tmp_path, ):
    """A sat with a handler-preformatted CE writes it verbatim after the
    verdict line (the scorer splits it off into .counterexample)."""
    ce = '(X_0 0.5)\n(Y_0 -1.0)'
    code, lines = _run(monkeypatch, tmp_path,
                       ('sat', {'time': 0.1, 'ce_sexpr': ce}))
    assert code == 1
    assert lines[0] == 'sat' and lines[1] == '(X_0 0.5)'
