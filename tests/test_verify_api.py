"""Programmatic API: `vibecheck.verify()` -> `VerifyResult`.

Exercises the in-process entry point on the bundled ACAS-Xu example (the same
files `vibecheck --examples-dir` ships), covering an unsat case, a sat case with
the numpy counterexample dict, results-file writing, and that it runs the full
production pipeline (auto-config) without arming the CLI's process-kill watchdog.
"""
import os

import numpy as np
import pytest

from vibecheck import verify, VerifyResult
from vibecheck.cli_standard import _examples_dir

_EX = _examples_dir()
_NET = os.path.join(_EX, 'ACASXU_run2a_2_2_batch_2000.onnx')
_UNSAT = os.path.join(_EX, 'prop_1.vnnlib')     # holds -> unsat
_SAT = os.path.join(_EX, 'prop_2.vnnlib')       # violated -> sat


def test_verify_unsat():
    r = verify(net=_NET, spec=_UNSAT, timeout=60)
    assert isinstance(r, VerifyResult)
    assert r.verdict == 'unsat'
    assert r.exit_code == 0
    assert r.counterexample is None
    assert r.details is not None            # the verifier's verbose object
    assert r.elapsed > 0.0


def test_verify_sat_counterexample():
    r = verify(net=_NET, spec=_SAT, timeout=60)
    assert r.verdict == 'sat'
    assert r.exit_code == 1
    ce = r.counterexample
    assert set(ce) == {'X', 'Y'}
    assert isinstance(ce['X'], np.ndarray) and isinstance(ce['Y'], np.ndarray)
    assert ce['X'].shape == (5,) and ce['Y'].shape == (5,)


def test_verify_writes_results_file(tmp_path):
    rf = str(tmp_path / 'out.txt')
    r = verify(net=_NET, spec=_SAT, timeout=60, results_file=rf)
    assert r.verdict == 'sat'
    assert os.path.isfile(rf)
    with open(rf) as f:
        assert f.readline().strip() == 'sat'      # authoritative verdict on line 1


def test_verify_temp_results_file_cleaned(tmp_path, monkeypatch):
    # No results_file -> a temp file is used internally and removed afterward.
    created = []
    import tempfile as _t
    real_mkstemp = _t.mkstemp

    def _spy(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        created.append(path)
        return fd, path

    monkeypatch.setattr(_t, 'mkstemp', _spy)
    r = verify(net=_NET, spec=_UNSAT, timeout=60)
    assert r.verdict == 'unsat'
    assert created and not any(os.path.exists(p) for p in created)


def test_verify_explicit_config():
    # An explicit --config path is honored (skips auto-config); acasxu config verifies.
    from vibecheck.config_loader import config_path
    r = verify(net=_NET, spec=_UNSAT, timeout=60,
               config=config_path('acasxu_2023.yaml'))
    assert r.verdict == 'unsat'
