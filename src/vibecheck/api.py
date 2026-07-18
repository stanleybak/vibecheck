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


class Spec:
    """A linear reachability property, built programmatically.

    Sugar over VNNLIB, never a second semantics: `to_vnnlib()` emits
    standard text and verification parses it with the same parser every
    file-based spec goes through.

    An instance is an input box plus one or more UNSAFE disjuncts. Each
    `forbid(W, b)` call adds one disjunct: the region where
    ``W @ y + b >= 0`` holds ROW-WISE (a conjunction). The property is
    violated (verdict 'sat') iff some input in the box produces an output
    inside ANY forbidden disjunct; 'unsat' proves the network never
    reaches them.

    Example -- prove y0 < 3.99 over the unit box:
        s = Spec(x_lo=[0]*5, x_hi=[1]*5).forbid([[1, 0, 0, 0, 0]], [-3.99])
    """

    def __init__(self, x_lo, x_hi):
        import numpy as np
        self.x_lo = np.asarray(x_lo, dtype=np.float64).ravel()
        self.x_hi = np.asarray(x_hi, dtype=np.float64).ravel()
        if self.x_lo.shape != self.x_hi.shape:
            raise ValueError('x_lo and x_hi must have the same length')
        if not (self.x_lo <= self.x_hi).all():
            raise ValueError('need x_lo <= x_hi elementwise')
        self._disjuncts = []

    @staticmethod
    def _atom(w, bi):
        """One row ``w . y + bi >= 0`` as a VNNLIB atom, in the grammar the
        parser (and every competition spec) uses: a Y threshold or a Y
        pairwise comparison. Rows outside that grammar raise loudly."""
        nz = [(j, c) for j, c in enumerate(w) if c != 0.0]
        if len(nz) == 1:
            j, c = nz[0]
            rhs = -bi / c
            return (f'(>= Y_{j} {rhs!r})' if c > 0
                    else f'(<= Y_{j} {rhs!r})')
        if len(nz) == 2 and bi == 0.0 and nz[0][1] == -nz[1][1]:
            (i, ci), (j, _) = nz
            return f'(>= Y_{i} Y_{j})' if ci > 0 else f'(>= Y_{j} Y_{i})'
        raise NotImplementedError(
            f'row {w} + {bi} is not expressible in the VNNLIB threshold/'
            f'pairwise grammar (single-output thresholds and zero-bias '
            f'differences); write the property as VNNLIB text instead')

    def forbid(self, W, b):
        """Add one unsafe disjunct: all rows of ``W @ y + b >= 0`` hold.

        Each row must be a single-output threshold (one nonzero) or a
        zero-bias difference (c*y_i - c*y_j) -- the grammar the parser
        accepts. Returns self for chaining."""
        import numpy as np
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        b = np.asarray(b, dtype=np.float64).ravel()
        if W.shape[0] != b.shape[0]:
            raise ValueError(f'{W.shape[0]} rows but {b.shape[0]} biases')
        if self._disjuncts and W.shape[1] != self._disjuncts[0][0].shape[1]:
            raise ValueError('all forbid() matrices need the same #columns')
        for r in range(W.shape[0]):            # fail fast, not at to_vnnlib
            self._atom(W[r].tolist(), float(b[r]))
        self._disjuncts.append((W, b))
        return self

    def to_vnnlib(self):
        """Serialize as VNNLIB text (declarations, box asserts, one
        `(assert (or (and ...) ...))` for the unsafe disjuncts)."""
        if not self._disjuncts:
            raise ValueError('no forbid() disjuncts: the property is vacuous')
        n_out = self._disjuncts[0][0].shape[1]
        L = [f'(declare-const X_{i} Real)' for i in range(len(self.x_lo))]
        L += [f'(declare-const Y_{i} Real)' for i in range(n_out)]
        for i, (lo, hi) in enumerate(zip(self.x_lo.tolist(),
                                         self.x_hi.tolist())):
            L.append(f'(assert (>= X_{i} {lo!r}))')
            L.append(f'(assert (<= X_{i} {hi!r}))')
        ands = ['(and ' + ' '.join(self._atom(W[r].tolist(), float(b[r]))
                                   for r in range(W.shape[0])) + ')'
                for W, b in self._disjuncts]
        L.append('(assert (or ' + ' '.join(ands) + '))')
        return '\n'.join(L) + '\n'


def _canon_net(net, example_input, workdir):
    """Canonicalize `net` to an ONNX file path inside `workdir`.

    Accepted forms: a path (str/PathLike, .gz and network-pair list-strings
    included), an `onnx.ModelProto`, serialized ONNX `bytes`, or a
    `torch.nn.Module` (exported with `torch.onnx.export`; needs
    `example_input`). The returned bytes-on-disk are the verification
    contract: bounds, attack, and the ORT counterexample replay all run
    against exactly this artifact."""
    import torch
    if isinstance(net, torch.nn.Module):
        if example_input is None:
            raise ValueError('a torch.nn.Module needs example_input= '
                             '(a dummy input tensor for torch.onnx.export)')
        path = os.path.join(workdir, 'exported.onnx')
        try:
            # the TorchScript exporter: self-contained, no onnxscript dep
            torch.onnx.export(net.eval(), example_input, path, dynamo=False)
        except TypeError:      # older torch without the dynamo kwarg
            torch.onnx.export(net.eval(), example_input, path)
        return path
    if isinstance(net, (bytes, bytearray)):
        path = os.path.join(workdir, 'model.onnx')
        with open(path, 'wb') as f:
            f.write(net)
        return path
    try:
        import onnx
        if isinstance(net, onnx.ModelProto):
            path = os.path.join(workdir, 'model.onnx')
            onnx.save(net, path)
            return path
    except ImportError:                    # onnx is a hard dep; be explicit
        raise
    if isinstance(net, (str, os.PathLike)):
        return str(net)
    raise TypeError(f'unsupported net type {type(net).__name__}: expected a '
                    f'path, onnx.ModelProto, bytes, or torch.nn.Module')


def _canon_spec(spec, workdir):
    """Canonicalize `spec` to a VNNLIB file path inside `workdir`.

    Accepted forms: a path (str/PathLike), raw VNNLIB text (a str with a
    newline or starting with '('), or a `Spec` builder. A path-like str
    that does not exist raises FileNotFoundError loudly rather than being
    guessed at as text."""
    if isinstance(spec, Spec):
        text = spec.to_vnnlib()
    elif isinstance(spec, str) and ('\n' in spec
                                    or spec.lstrip().startswith('(')):
        text = spec
    elif isinstance(spec, (str, os.PathLike)):
        p = str(spec)
        if not (os.path.isfile(p) or os.path.isfile(p + '.gz')):
            raise FileNotFoundError(f'spec file not found: {p!r} (pass '
                                    f'VNNLIB text or a vibecheck.Spec to '
                                    f'skip files entirely)')
        return p
    else:
        raise TypeError(f'unsupported spec type {type(spec).__name__}: '
                        f'expected a path, VNNLIB text, or vibecheck.Spec')
    path = os.path.join(workdir, 'property.vnnlib')
    with open(path, 'w') as f:
        f.write(text)
    return path


def verify(net, spec, *, timeout=60, device=None, results_file=None,
           example_input=None, extra_args=None):
    """Verify a network against a specification; return a `VerifyResult`.

    Parameters
    ----------
    net : str | os.PathLike | onnx.ModelProto | bytes | torch.nn.Module
        The network: an ONNX file path (or a network-pair ``--net`` field
        string), an in-memory ONNX model / its serialized bytes, or a
        torch module (exported via ``torch.onnx.export``; the verdict is
        then a statement about the exported graph).
    spec : str | os.PathLike | vibecheck.Spec
        The property: a VNNLIB file path, raw VNNLIB text, or a `Spec`
        builder (input box + linear unsafe disjuncts).
    timeout : float
        Cooperative time budget in seconds.
    device : str, optional
        'cpu' or 'cuda'; when omitted, cuda is auto-selected if available.
        No other configuration exists: the pipeline picks its verification
        route from the structure of the network and spec.
    results_file : str, optional
        Also write the VNNCOMP results line (+ counterexample) to this
        path. When omitted a temp file is used and removed afterward.
    example_input : torch.Tensor, optional
        Dummy input for the ONNX export; required iff `net` is a
        torch.nn.Module.
    extra_args : list, optional
        Extra raw flat-CLI flags, e.g. ``['--no-attack']``.
    """
    import shutil
    workdir = tempfile.mkdtemp(prefix='vibecheck_api_')
    try:
        spec_path = _canon_spec(spec, workdir)     # cheap; fails before any
        net_path = _canon_net(net, example_input, workdir)  # onnx export
        return _verify_paths(net_path, spec_path,
                             timeout=timeout, device=device,
                             results_file=results_file,
                             extra_args=extra_args)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _verify_paths(net, spec, *, timeout, device, results_file, extra_args):
    """File-path core of `verify()` (both arguments already canonical)."""
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
