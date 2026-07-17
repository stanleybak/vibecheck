"""Programmatic API: `vibecheck.verify()` -> `VerifyResult`.

A thin wrapper over the SAME flat pipeline the CLI drives
(`verify._run_flat`): structural route selection, network-pair merging,
and every SAT-witness soundness gate, so the verdict is identical to
``vibecheck verify ...``. The verdict and counterexample are read back
from the results file -- the verdict authority -- never from stdout or
the exit code. Progress output is redirected to stderr so the caller's
stdout stays clean.
"""
import contextlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VerifyResult:
    """The outcome of a verification run (returned by `verify()`).

    verdict: 'unsat' (proved safe) | 'sat' (counterexample found) |
             'unknown' | 'timeout' | 'error'.
    counterexample: for 'sat' only, a dict of numpy arrays keyed by the
             spec's declared I/O names (single network -> {'X': ndarray,
             'Y': ndarray}, Y replayed on the ORIGINAL network by the
             same ORT-CPU forward the VNNCOMP scorer uses). None for any
             other verdict.
    details: the verifier's details dict (timings, bounds, route) --
             includes 'error' with the exception text for crash verdicts.
    exit_code: process exit code the CLI uses (0 unsat, 1 sat/unknown/
             timeout, 2 error).
    elapsed: wall-clock seconds.
    """
    verdict: str
    counterexample: Optional[dict] = None
    details: Any = None
    exit_code: int = 0
    elapsed: float = 0.0


def _parse_counterexample(assignment_text):
    """The results-file assignment block -> {name: ndarray}. Reuses the
    standard CLI's parser, so both emitted formats (VNN-LIB 2.0 per-tensor
    and 1.0 flat s-expression) round-trip identically."""
    import numpy as np
    from .cli_standard import _SER_DTYPES, _parse_assignment
    out = {}
    for name, dt, shape, values in _parse_assignment(assignment_text):
        arr = np.asarray(values, dtype=_SER_DTYPES.get(dt, 'float64'))
        out[name] = arr.reshape(shape or (-1,))
    return out


def verify(net, spec, *, timeout=60, device=None, results_file=None,
           extra_args=None):
    """Verify an ONNX network against a VNNLIB spec; return a `VerifyResult`.

    Parameters
    ----------
    net : str
        Path to the ONNX network (or a network-pair ``--net`` field string).
    spec : str
        Path to the VNNLIB specification.
    timeout : float
        Cooperative time budget in seconds.
    device : str, optional
        'cpu' or 'cuda'; when omitted, cuda is auto-selected if available.
        No other configuration exists: the pipeline picks its verification
        route from the structure of the network and spec.
    results_file : str, optional
        Also write the VNNCOMP results line (+ counterexample) to this
        path. When omitted a temp file is used and removed afterward.
    extra_args : list, optional
        Extra raw flat-CLI flags, e.g. ``['--no-attack']``.
    """
    from .verify import _make_flat_parser, _run_flat
    argv = ['--net', str(net), '--spec', str(spec), '--timeout', str(timeout)]
    if device is not None:
        argv += ['--device', str(device)]
    _tmp = None
    if results_file is not None:
        argv += ['--results-file', str(results_file)]
    else:
        fd, _tmp = tempfile.mkstemp(suffix='.vibecheck-result')
        os.close(fd)
        argv += ['--results-file', _tmp]
    if extra_args:
        argv += [str(a) for a in extra_args]
    a = _make_flat_parser().parse_args(argv)
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            _, details, exit_code = _run_flat(a)
        from .cli_standard import _read_results
        verdict, assignment = _read_results(a.results_file)
        ce = None
        if verdict == 'sat' and assignment:
            ce = _parse_counterexample(assignment)
        return VerifyResult(verdict=verdict, counterexample=ce,
                            details=details, exit_code=exit_code,
                            elapsed=time.time() - t0)
    finally:
        if _tmp is not None:
            try:
                os.remove(_tmp)
            except OSError:
                pass
