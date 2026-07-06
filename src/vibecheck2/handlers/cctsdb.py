"""Discrete integer-grid enumeration for cctsdb_yolo (standalone vc2).

Ported from v1 vibecheck.cctsdb_yolo + the two helpers it used from
surrogate_pgd/sign_attack; the v1 settings object collapses to the two
values it actually read (atol, max_positions). Complete: 'unsat' means
every placement was evaluated safe on ORT-CPU.
"""
import gzip
import itertools
import os
import time

import numpy as np
import onnx
import onnxruntime as ort



def _load_onnx_model(path):
    # Benchmarks ship gzipped and instances.csv names the UN-gz file (`foo.onnx`)
    # while only `foo.onnx.gz` exists; resolve the .gz sibling so the quantized-op
    # probe (the first thing prepare does) doesn't crash on every gzipped net.
    if not os.path.isfile(path) and os.path.isfile(path + '.gz'):
        path = path + '.gz'
    if path.endswith('.gz'):
        with gzip.open(path) as fh:
            return onnx.load_model_from_string(fh.read())
    return onnx.load(path)


def _model_input_shapes(onnx_path):
    """Free-input (non-initializer) shapes of the ONNX, in graph order — the authoritative
    tensor shapes for feeding the model (the spec only carries a flat per-index box)."""
    m = _load_onnx_model(onnx_path)
    init = {i.name for i in m.graph.initializer}
    return [[d.dim_value if d.dim_value > 0 else 1 for d in i.type.tensor_type.shape.dim]
            for i in m.graph.input if i.name not in init]


# ---------------------------------------------------------------------- fold surrogate


def _worst_margin_np(y, disjuncts):
    """Numpy `worst_margin` for the ORT validation: <0 clear CE, in [0,atol] within-tol."""
    conj = []
    for c in disjuncts:
        cm = []
        for k in c.constraints:
            if hasattr(k, 'pred'):
                cm.append(float(y[k.pred] - y[k.comp]))
            elif k.op == '>=':
                cm.append(float(k.value - y[k.index]))
            else:
                cm.append(float(y[k.index] - k.value))
        conj.append(max(cm))
    return min(conj)


def has_cctsdb_structure(onnx_path, vnnlib_path):
    """True if this looks like a discrete-patch instance: a single-input net whose vnnlib leaves
    only a FEW integer-valued input dims free (the patch positions), the rest fixed."""
    from ..frontend.io_util import ensure_decompressed
    from ..frontend.vnnlib_loader import load_vnnlib
    if len(_model_input_shapes(onnx_path)) != 1:
        return False
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    lo = np.asarray(spec.x_lo, np.float64); hi = np.asarray(spec.x_hi, np.float64)
    free = [d for d in range(lo.size) if hi[d] - lo[d] > 1e-6]
    if not free or len(free) > 4:
        return False
    return all(abs(lo[d] - round(lo[d])) < 1e-6 and abs(hi[d] - round(hi[d])) < 1e-6
               for d in free)


def cctsdb_yolo_verify(onnx_path, vnnlib_path, timeout, log=print,
                       atol=1e-4, max_positions=1_000_000):
    """Enumerate the integer patch-position grid through the ORIGINAL net on ORT-CPU. Returns
    (verdict, witness): verdict in {'unsat','sat','timeout'}; witness is [input np.ndarray] for
    sat (the violating placement), else None. Complete: 'unsat' = every placement is safe."""
    import onnxruntime as ort
    from ..frontend.io_util import ensure_decompressed
    from ..frontend.vnnlib_loader import load_vnnlib

    t0 = time.time()
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    lo = np.asarray(spec.x_lo, np.float64); hi = np.asarray(spec.x_hi, np.float64)
    max_pos = int(max_positions)
    free = [d for d in range(lo.size) if hi[d] - lo[d] > 1e-6]
    for d in free:
        if abs(lo[d] - round(lo[d])) > 1e-6 or abs(hi[d] - round(hi[d])) > 1e-6:
            raise NotImplementedError(
                f'cctsdb_yolo: free input dim {d} range [{lo[d]},{hi[d]}] is not integer-valued '
                f'— this is not a discrete-patch instance')
    ranges = [range(int(round(lo[d])), int(round(hi[d]))) for d in free]   # exclusive hi (= ABC)
    total = int(np.prod([len(r) for r in ranges])) if ranges else 0
    if total <= 0 or total > max_pos:
        raise NotImplementedError(
            f'cctsdb_yolo: {total} positions to enumerate over free dims {free} '
            f'(cap {max_pos}) — not a discrete-patch instance?')

    in_shape = _model_input_shapes(onnx_path)[0]
    sess = ort.InferenceSession(ensure_decompressed(onnx_path),
                                providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    base = lo.copy()
    log(f'[cctsdb] enumerating {total} integer patch positions over free dims {free}')

    n = 0
    for combo in itertools.product(*ranges):
        if time.time() - t0 > timeout:
            log(f'[cctsdb] timeout after {n}/{total} positions')
            return 'timeout', None
        x = base.copy()
        for d, v in zip(free, combo):
            x[d] = v
        feed = x.reshape(in_shape).astype(np.float32)
        y = np.asarray(sess.run([oname], {iname: feed})[0]).ravel()
        n += 1
        m = _worst_margin_np(y, spec.disjuncts)
        # m <= 0 is an output violation under the spec's `<=`/`>=` comparison at zero
        # tolerance (boundary inclusive) — a counterexample. m > 0 does NOT violate
        # (no within-output-tolerance fallback under the 2026 rule).
        if m <= 0.0:
            log(f'[cctsdb] CLEAR SAT at position {tuple(combo)} (worst_margin={m:.3e})')
            return 'sat', [feed]
    # Complete enumeration finished with no violating position -> unsat.
    log(f'[cctsdb] all {n} positions safe -> unsat (complete) (t={time.time()-t0:.1f}s)')
    return 'unsat', None
