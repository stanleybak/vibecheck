"""VNN-LIB standard solver CLI (standard Ch. 5) -- vc2 port of v1
cli_standard (main 35f6931), same surface:

    vibecheck --name                      # tool identifier
    vibecheck --version                   # tool version
    vibecheck verify <query.vnnlib> --network NAME=model.onnx [--timeout N]
                      [--serialise-assignments DIR]
    vibecheck supports <capability>       # e.g. --onnx-operators

`verify` keeps stdout STRICT per the standard: line 1 is the verdict
(`sat|unsat|unknown|timed-out`), followed only by the satisfying assignment
for `sat`; all progress goes to stderr. Verification delegates to the flat
legacy CLI (`verify._legacy_main`) via a mapped argv, so the route
selection (vc2 needs no per-benchmark config -- the pipeline picks its
route structurally and logs the choice) and every soundness gate behave
identically in both CLIs. `verify.main` dispatches here; a bare positional
first argument is an implicit `verify`.

The capabilities table is vc2's OWN (honest deltas from v1: MNC
comparisons are not ported; multi-input models run only through the
attack-only quantized surrogate handler).
"""

import argparse
import contextlib
import gzip
import os
import re
import sys
import tempfile

TOOL_NAME = 'vibecheck'
# PyPI distribution name (import package + console script stay `vibecheck`;
# the name `vibecheck` was already taken on PyPI, so we distribute as
# `vibecheck-nn`).
DIST_NAME = 'vibecheck-nn'


def dispatch(argv):
    """Route a standard-CLI invocation (see `verify.main` for the rule)."""
    if argv[0] == '--name':
        print(TOOL_NAME)
        return 0
    if argv[0] == '--version':
        print(_version())
        return 0
    if argv[0] == '--examples-dir':
        print(_examples_dir())
        return 0
    if argv[0] == 'supports':
        return run_supports(argv[1:])
    # 'verify', explicit or implicit (bare query filepath as the first arg).
    return run_verify(argv[1:] if argv[0] == 'verify' else argv)


def _examples_dir():
    """Absolute path to the bundled examples/ directory (a small ACAS-Xu
    ONNX + vnnlib specs), so a pip-installed user can run the README
    quickstart without a source checkout. Resolves inside the package
    whether run from source or an installed wheel."""
    from importlib.resources import files
    return str(files('vibecheck') / 'examples')


def _version():
    """Tool version from package metadata, falling back to the repo
    pyproject.toml for a PYTHONPATH=src dev checkout, then a dev tag."""
    import importlib.metadata
    try:
        return importlib.metadata.version(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        pyproject = os.path.join(os.path.dirname(__file__), '..', '..',
                                 'pyproject.toml')
        try:
            with open(pyproject) as f:
                m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
            if m:
                return m.group(1)
        except OSError:
            pass
        return '0.0.0-dev'


# ------------------------------------------------------------------- supports

# `supports <capability>` answer table: one identifier per line, first token
# machine-parseable; `IDENT * note` marks PARTIAL support with a short
# reason. Opset range mirrors v1 (same reused ONNX front end, measured over
# the local VNNCOMP 2025+2026 clones).
_NOTE_REAL = 'bounds computed in real arithmetic, not IEEE-754-faithful'
_SUPPORTS_TABLE = {
    '--vnnlib-versions': ['1.0', '2.0'],
    '--onnx-opset-versions': ['8', '20'],
    '--onnx-element-types': ['real',
                             f'float32 * {_NOTE_REAL}',
                             f'float64 * {_NOTE_REAL}'],
    '--hidden-node-theories': ['NH'],
    '--multiple-input-output-theories': [
        'SIO * multi-input models run only through the attack-only '
        'quantized-surrogate handler (sat/timeout, never unsat)'],
    '--multiple-network-theories': ['SNET', 'MENET',
                                    'MINET * two-network pairs only'],
    '--multiple-node-comparison-theories': ['SNC'],
    '--arithmetic-complexity-theories': [
        'BND', 'OUTC', 'LIN',
        'POLY * polynomial constraints transpiled via nonlinear-augment'],
    '--optimised-disjunctive-reasoning': ['true'],
    '--serialise-assignments': ['true'],
}


_KNOWN_CAPS = ' '.join(sorted(set(_SUPPORTS_TABLE) | {'--onnx-operators'}))


def run_supports(argv):
    """`vibecheck supports <capability>...` -- print supported identifiers,
    one per line (`* note` marks partial). Unknown/missing -> stderr + 2."""
    if not argv:
        print(f'Error: supports requires a capability argument, one of: '
              f'{_KNOWN_CAPS}', file=sys.stderr)
        return 2
    for cap in argv:
        if cap == '--onnx-operators':
            # every op the (reused v1) front end has a GraphNode for; an
            # empty per-op type list means "all reported element types"
            from .frontend.network import OP_REGISTRY
            lines = sorted(OP_REGISTRY)
        elif cap in _SUPPORTS_TABLE:
            lines = _SUPPORTS_TABLE[cap]
        else:
            print(f'Error: unknown capability {cap!r}; known: {_KNOWN_CAPS}',
                  file=sys.stderr)
            return 2
        print('\n'.join(lines))
    return 0


# --------------------------------------------------------------------- verify

def _fail(msg):
    """Standard-CLI usage/argument error: message on stderr, exit code 2."""
    print(f'Error: {msg}', file=sys.stderr)
    sys.exit(2)


def _parse_network_args(network_args):
    """Repeated `--network NAME=PATH` -> [(name, path), ...]; malformed -> 2."""
    pairs = []
    for s in network_args:
        name, sep, path = s.partition('=')
        if not sep or not name or not path:
            _fail(f'malformed --network value {s!r} (expected NAME=PATH)')
        pairs.append((name, path))
    return pairs


def _spec_head(spec_path):
    """Head of the query file (gzip-aware), enough for every
    `declare-network` header; unreadable -> '' (the verification load
    reports the real error through the legacy clean-exit path)."""
    p = spec_path
    if not os.path.exists(p) and os.path.exists(str(p) + '.gz'):
        p = str(p) + '.gz'
    opener = gzip.open if str(p).endswith('.gz') else open
    try:
        with opener(p, 'rt') as fh:
            return fh.read(16384)
    except (OSError, ValueError):
        return ''


def _declared_networks(head):
    """[(name, equal_to_source_or_None), ...] from `(declare-network ...)`
    headers; `equal-to` networks reuse the source ONNX (standard rule)."""
    out = []
    segments = re.split(r'\(declare-network\s+', head)[1:]
    for seg in segments:
        name = re.match(r'([^\s()]+)', seg).group(1)
        eq = re.search(r'\(equal-to\s+([^\s()]+)\s*\)', seg)
        out.append((name, eq.group(1) if eq else None))
    return out


def _resolve_net_field(query, network_args):
    """Map `--network NAME=PATH` flags onto the legacy `--net` field: a
    single ONNX path, or the two-network pair list-string. Validated
    against the spec's declarations (exit 2 on mismatch)."""
    provided = _parse_network_args(network_args)
    decls = _declared_networks(_spec_head(query))
    if not decls:
        if len(provided) != 1:
            _fail(f'query declares a single (implicit) network; got '
                  f'{len(provided)} --network flags (expected exactly 1)')
        return provided[0][1]
    need = [n for n, src in decls if src is None]
    given = [n for n, _ in provided]
    if sorted(given) != sorted(need):
        _fail(f'--network names {given} do not match the networks declared '
              f'by the query that need a file mapping {need} (equal-to '
              f'networks reuse their source network and take no --network '
              f'flag)')
    paths = dict(provided)
    for name, src in decls:
        if src is not None:
            if src not in paths:
                _fail(f'network {name!r} is declared equal-to {src!r}, '
                      f'which has no resolved file mapping')
            paths[name] = paths[src]
    if len(decls) == 1:
        return paths[decls[0][0]]
    if len(decls) == 2:
        return repr([(n, paths[n]) for n, _ in decls])
    _fail(f'{len(decls)} declared networks are not supported '
          f'(MINET: two-network pairs only)')


def run_verify(argv):
    """`vibecheck verify <query> [--network NAME=PATH]... [--timeout N]` --
    verify through the legacy pipeline with strict standard stdout."""
    parser = argparse.ArgumentParser(
        prog='vibecheck verify',
        description='Verify a VNN-LIB query (VNN-LIB standard CLI, Ch. 5).')
    parser.add_argument('query', help='Path to the VNN-LIB query file')
    parser.add_argument('--network', action='append', default=[],
                        metavar='NAME=PATH',
                        help='Map a declared network to an ONNX file '
                             '(repeatable; one per declared network that is '
                             'not equal-to another)')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Verification timeout in seconds')
    parser.add_argument('--serialise-assignments', dest='serialise_dir',
                        default=None, metavar='DIR',
                        help='For sat: write each assigned variable as an '
                             'ONNX TensorProto DIR/<name>.pb instead of '
                             'printing the values to stdout')
    parser.add_argument('--results-file', default=None,
                        help='Also write the verdict (+assignment) to this '
                             'file (non-standard extra, harness style)')
    parser.add_argument('--verdict-style', dest='verdict_style',
                        default='standard', choices=['standard', 'vnncomp'],
                        help="Verdict spelling: 'standard' (default: "
                             "'timed-out') or 'vnncomp' ('timeout')")
    parser.add_argument('--device', default=None, choices=['cpu', 'gpu'],
                        help='Compute device (default: cpu; the standard '
                             "spells the GPU 'gpu' -- mapped to cuda)")
    a = parser.parse_args(argv)

    net_field = _resolve_net_field(a.query, a.network)

    # the verdict authority is the results file (never stdout/exit code);
    # route through one even when the caller didn't ask for it
    results_file = a.results_file
    tmp_rf = None
    if results_file is None:
        fd, tmp_rf = tempfile.mkstemp(prefix='vibecheck_results_',
                                      suffix='.txt')
        os.close(fd)
        results_file = tmp_rf

    legacy = ['--net', net_field, '--spec', a.query,
              '--results-file', results_file,
              '--verdict-style', a.verdict_style]
    if a.timeout is not None:
        legacy += ['--timeout', str(a.timeout)]
    if a.device is not None:
        legacy += ['--device', 'cuda' if a.device == 'gpu' else 'cpu']

    from .pipeline import _legacy_main
    stdout = sys.stdout
    code = 0
    try:
        # STRICT standard stdout: every pipeline print goes to stderr for
        # the whole run; only the verdict+assignment hit the real stdout
        with contextlib.redirect_stdout(sys.stderr):
            code = _legacy_main(legacy)
        verdict, assignment = _read_results(results_file)
        if verdict == 'error':
            print(f'Error: verification failed'
                  + (f': {assignment}' if assignment else ''),
                  file=sys.stderr)
            return 2
        print(verdict, file=stdout)
        if assignment:
            if a.serialise_dir is not None:
                _serialise_assignment(assignment, a.serialise_dir)
            else:
                print(assignment, file=stdout)
        return code
    finally:
        if tmp_rf is not None:
            os.remove(tmp_rf)


def _read_results(results_file):
    """(verdict, assignment_text) from the results file: line 1 = verdict,
    the rest (sat assignment, or the crash cause for 'error') verbatim."""
    with open(results_file) as f:
        content = f.read()
    lines = content.splitlines()
    verdict = lines[0] if lines else 'unknown'
    assignment = '\n'.join(lines[1:]).strip('\n')
    return verdict, assignment


# spec-declared dtype -> numpy dtype for serialised assignment tensors;
# 'real' is serialised at the highest available precision
_SER_DTYPES = {'real': 'float64', 'float64': 'float64',
               'float32': 'float32', 'float16': 'float16'}


def _parse_assignment(text):
    """Parse an assignment block into
    [(name, dtype_str, shape_tuple, [float values]), ...]. Handles both
    emitted formats: the VNN-LIB 2.0 per-tensor form (`NAME dtype
    [d0,d1,...]` header + one value per line) and the 1.0 flat s-expression
    `((X_0 v) ... (Y_0 v))` (grouped into 1-D 'real' tensors X and Y)."""
    header_re = re.compile(r'^(\S+)\s+(\S+)\s+\[([\d,\s]*)\]$')
    if text.lstrip().startswith('('):
        xs, ys = [], []
        for name, val in re.findall(r'\((X|Y)_\d+\s+([^\s()]+)\)', text):
            (xs if name == 'X' else ys).append(float(val))
        if not xs and not ys:
            raise ValueError(f'unparseable v1 assignment: {text[:200]!r}')
        return [(n, 'real', (len(v),), v)
                for n, v in (('X', xs), ('Y', ys)) if v]
    tensors = []
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = header_re.match(line)
        if m:
            shape = tuple(int(s) for s in m.group(3).split(',') if s.strip())
            cur = (m.group(1), m.group(2), shape, [])
            tensors.append(cur)
        elif cur is not None:
            cur[3].append(float(line))
        else:
            raise ValueError(f'unparseable assignment line: {line!r}')
    if not tensors:
        raise ValueError(f'unparseable v2 assignment: {text[:200]!r}')
    return tensors


def _serialise_assignment(text, out_dir):
    """`--serialise-assignments DIR`: one ONNX TensorProto `DIR/<name>.pb`
    per assigned variable, in the spec-declared dtype ('real' -> float64)."""
    import numpy as np
    from onnx import numpy_helper
    os.makedirs(out_dir, exist_ok=True)
    for name, dt, shape, values in _parse_assignment(text):
        if dt not in _SER_DTYPES:
            _fail(f'cannot serialise variable {name!r}: '
                  f'unsupported declared dtype {dt!r}')
        arr = np.asarray(values, dtype=_SER_DTYPES[dt]).reshape(shape or (-1,))
        with open(os.path.join(out_dir, f'{name}.pb'), 'wb') as f:
            f.write(numpy_helper.from_array(arr, name).SerializeToString())
