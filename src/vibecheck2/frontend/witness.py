"""Witness validation and counterexample emission (standalone vc2).

Ported from v1 (vibecheck.verify_graph: the ORT-CPU chokepoint
_validate_sat_witness and its helpers; vibecheck.main: the counterexample
s-expression formatting), extracted at function granularity -- the
surrounding 7k-line v1 modules are NOT copied. Every 'sat' must survive
_validate_sat_witness (input box within atol, output STRICTLY violating)
before it is emitted; the formatting recomputes Y with the same ORT
forward the scorer replays.
"""
import gzip
import os
import re

import numpy as np



def _ort_session_for(onnx_path):
    """Cached CPU InferenceSession for `onnx_path` (handles `.onnx.gz`)."""
    sess = _ORT_VALIDATE_SESSIONS.get(onnx_path)
    if sess is None:
        import onnxruntime as ort
        if onnx_path.endswith('.gz'):
            import gzip
            with gzip.open(onnx_path, 'rb') as _f:
                sess = ort.InferenceSession(_f.read(),
                                            providers=['CPUExecutionProvider'])
        else:
            sess = ort.InferenceSession(onnx_path,
                                        providers=['CPUExecutionProvider'])
        _ORT_VALIDATE_SESSIONS[onnx_path] = sess
    return sess


def _clamp_witness_to_box(witness, x_lo, x_hi, slack=0.0):
    """Clamp a counterexample witness into the input box `[x_lo, x_hi]` (widened
    by `slack` on each side) so it stays inside even after the float32 cast ORT /
    the VNNCOMP scorer applies.

    A box edge (e.g. `x >= 9.2`) is not generally representable in float32, so
    a witness sitting exactly on it can round to the *outside* of the box when
    cast (`float32(9.2) < 9.2`), failing the scorer's `x < lb - tol` test. Here
    we clamp into the box in float64, then pull any component whose float32 cast
    landed outside back toward the interior by one float32 ULP. The result is a
    float64 array that is provably within `[x_lo-slack, x_hi+slack]` both as
    float64 and as float32 (the coarser grid → also safe for float64 models).

    `slack` must stay 0.0 under the zero-input-tolerance policy (emitted
    witnesses are strictly in-box); the parameter remains only so the pure
    helper is testable against widened targets.

    Pure function: uses `np.nextafter`/`np.clip` only — it does NOT touch any FP
    rounding mode, so verification arithmetic elsewhere is unaffected (the clamp
    runs only at witness validation / output time).
    """
    w = np.asarray(witness, np.float64).flatten()
    lo = np.asarray(x_lo, np.float64).flatten() - slack
    hi = np.asarray(x_hi, np.float64).flatten() + slack
    w = np.minimum(np.maximum(w, lo), hi)            # into [lo, hi] in float64
    w32 = w.astype(np.float32)
    cast = w32.astype(np.float64)                    # value the scorer sees
    below = cast < lo                                # rounded under the floor
    above = cast > hi                                # rounded over the ceiling
    w32 = np.where(below, np.nextafter(w32, np.float32(np.inf)), w32)
    w32 = np.where(above, np.nextafter(w32, np.float32(-np.inf)), w32)
    return w32.astype(np.float64)


_ORT_VALIDATE_SESSIONS = {}


def _validate_witness_ort(onnx_path, witnesses, boxes, output_violated,
                          atol=1e-4):
    """Unified ORT-CPU witness validator for EVERY path (graph + surrogate/attack).

    This is the single authoritative gate: load the ORIGINAL ONNX, replay the
    witness on CPU onnxruntime, and check it actually violates the spec — catching
    spurious counterexamples from PGD/MILP/graph-builder bugs. It is multi-input
    (the surrogate/attack paths produce multi-tensor witnesses) and the
    output-violation rule is supplied by the caller (`output_violated`), so each
    spec representation keeps its own correct semantics (the graph path's inclusive
    `spec.check`/disjunctive `check_witness`; the surrogate's strict `>`/`<` margin
    with `sat_strict_buffer`).

    witnesses: list of per-input flat float arrays (single-input -> 1-element list).
    boxes:     list of (lo, hi) flat arrays per input, or None to skip the box check.
    output_violated(inbox_list, y_flat) -> (violated: bool, info_updates: dict).

    Returns (proceed, info). proceed=True means "emit this sat" — the output
    genuinely violates, OR validation was skipped (no onnx_path / onnxruntime
    missing, so we don't reject what we cannot check). proceed=False means
    reject/downgrade (out-of-box, ORT failure, or output does not violate). info
    carries 'out', 'witness_inbox'/'witnesses_inbox', 'spec_check', and any
    `output_violated` updates (e.g. 'worst_margin').
    """
    info = {'ok': False, 'reason': None}
    if onnx_path is None:
        info['reason'] = 'no onnx_path stashed on graph; skipping validation'
        return True, info
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        info['reason'] = 'onnxruntime not installed; skipping validation'
        return True, info
    # POLICY (zero input/output tolerance): the emitted witness is ALWAYS
    # the float32-safe clamp STRICTLY inside [lo, hi], and its replayed
    # output must strictly violate. `atol` is a candidate-REPAIR band only:
    # a raw PGD point up to `atol` outside the box is clamped in and
    # re-validated (the clamped point either violates strictly or the
    # candidate is rejected); it never loosens what is emitted. (An
    # `emit_slack` path that could emit just-outside witnesses --
    # v1's box-expansion, scorer CORRECT_WITH_TOLERANCE -- was removed
    # deliberately.)
    raw, boxes_lh = [], []
    for k, w in enumerate(witnesses):
        w = np.asarray(w, np.float64).ravel()
        box = boxes[k] if boxes is not None else None
        if box is not None:
            lo = np.asarray(box[0], np.float64).ravel()
            hi = np.asarray(box[1], np.float64).ravel()
            if w.shape != lo.shape:
                info['reason'] = (f'witness[{k}] shape {w.shape} != box shape '
                                  f'{lo.shape}')
                return False, info
            if np.any(w < lo - atol) or np.any(w > hi + atol):
                info['reason'] = f'witness[{k}] outside input box (atol={atol})'
                info['out_of_box'] = (float((lo - w).max()), float((w - hi).max()))
                return False, info
            box = (lo, hi)
        raw.append(w)
        boxes_lh.append(box)

    def _eval_at(slack):
        """Clamp each witness into [lo-slack, hi+slack], ORT-replay the ORIGINAL
        model. Returns (violated, inbox, y, updates); violated is None on ORT
        error (then updates carries the 'reason')."""
        inbox = [(_clamp_witness_to_box(w, b[0], b[1], slack=slack)
                  if b is not None else w)
                 for w, b in zip(raw, boxes_lh)]
        try:
            sess = _ort_session_for(onnx_path)
            feeds = {}
            for k, im in enumerate(sess.get_inputs()):
                shp = [d if isinstance(d, int) and d > 0 else 1 for d in im.shape]
                feeds[im.name] = inbox[k].reshape(shp).astype(np.float32)
            out = sess.run(None, feeds)[0]
            y = np.asarray(out).flatten().astype(np.float64)
        except (RuntimeError, OSError, ValueError, IndexError, KeyError) as e:
            return None, inbox, None, {
                'reason': f'ORT forward failed: {type(e).__name__}: {e}'}
        violated, updates = output_violated(inbox, y)
        return violated, inbox, y, updates

    violated, inbox, y, updates = _eval_at(0.0)
    info['witnesses_inbox'] = inbox
    info['witness_inbox'] = inbox[0] if len(inbox) == 1 else None
    if violated is None:                            # ORT failure -> reject
        info['reason'] = updates.get('reason', 'ORT forward failed')
        return False, info
    info['out'] = y
    if updates:
        info.update(updates)
    info['spec_check'] = 'unknown' if violated else 'verified'
    if violated:
        info['ok'] = True
        return True, info
    if not info.get('reason'):
        info['reason'] = 'ORT output does not violate spec'
    return False, info


def _validate_sat_witness(onnx_path, spec, witness, atol=1e-4, out_atol=0.0):
    """Run a SAT witness through ONNXRuntime + check it actually violates
    the spec. Catches spurious counterexamples from PGD/MILP bugs OR from
    graph-builder bugs (vibecheck's internal forward might compute a
    different value than the original ONNX). Mirrors VNNCOMP scoring's
    counterexample-validation step.

    VNN-COMP 2026 ruling (evaluation chairs): a witness is CORRECT iff its
    input satisfies the VNN-LIB input constraints AND the *replayed* ORT
    output satisfies the output constraints. The 1e-4 absolute tolerance
    applies ONLY to the INPUT box (a witness up to `atol` outside the box is
    CORRECT_WITH_TOLERANCE — no penalty, but not SAT ground truth). The
    OUTPUT must violate the spec with NO tolerance. Hence `atol` gates the
    input box and `out_atol` (default 0.0 = strict) gates the output. Output
    tolerance is NOT scorer-accepted under the 2026 rule — keep `out_atol=0`.

    Returns (ok, info_dict). `ok=True` iff witness is in the input box
    (within `atol`) AND its ORT output satisfies the unsafe condition
    (i.e., `spec.check(out, out)` returns 'unknown', i.e. worst margin <= 0
    within `out_atol` on constraint margins — inclusive of the boundary,
    matching the official checker's `<=`/`>=` comparison at zero tolerance).
    """
    # Single-input VNNSpec wrapper over the unified `_validate_witness_ort`. The
    # output-violation rule is the graph path's: a per-disjunct X-subrange spec
    # (nn4sys lindex, acasxu prop_6) uses `check_witness(x, y)` (else `spec.check`
    # ignores the X constraints and could report a false SAT); otherwise the
    # Y-only `spec.check` with the output band `out_atol` (default 0.0 = strict,
    # boundary inclusive per the official `<=`/`>=` comparison).
    w = np.asarray(witness).flatten().astype(np.float64)

    def _output_violated(inbox, y):
        # STRICT CE-check: a `<`/`>` constraint at the boundary (margin == 0) is
        # NOT a counterexample (`is_strict_ce` honors per-constraint strictness;
        # for non-strict specs it reduces to the closure `<=`/`>=`). The verifier
        # bound keeps the closure (sound for UNSAT) — this is only the CE side.
        _x = inbox[0].astype(np.float64)
        is_ce = spec.is_strict_ce(_x, y, out_atol)
        # worst_margin (closure, for diagnostics / the SAT disposition).
        _, check_info = spec.check(y - out_atol, y + out_atol)
        upd = {'worst_margin': check_info.get('worst_margin')}
        if not is_ce:
            upd['reason'] = (
                f'ORT output does not violate spec '
                f'(worst_margin={check_info.get("worst_margin"):.4g}, '
                f'out_atol={out_atol})')
        return is_ce, upd

    return _validate_witness_ort(onnx_path, [w], [(spec.x_lo, spec.x_hi)],
                                 _output_violated, atol)


def _onnx_io_meta(onnx_path):
    """(inputs, outputs) — each a list of (name, dtype_str, shape, size) for the ONNX's free
    inputs and outputs in graph order: the per-tensor structure a v2 counterexample needs."""
    import numpy as np
    from . import surrogate_pgd as sp
    m = sp._load_onnx_model(onnx_path)
    init = {i.name for i in m.graph.initializer}

    def meta(vi):
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        return (vi.name, _ONNX_DT.get(vi.type.tensor_type.elem_type, 'float32'),
                shape, int(np.prod(shape)) if shape else 1)

    init_names = init
    ins = [meta(i) for i in m.graph.input if i.name not in init_names]
    outs = [meta(o) for o in m.graph.output]
    return ins, outs


def _cex_sexpr(x_flat, y_flat, fmt='.17g'):
    """Build the VNNLIB 1.0 counterexample s-expression `((X_0 v) ... (Y_0 v) ...)`
    from flattened input/output arrays. `fmt` is the per-value precision from the
    `counterexample_precision` setting (default '.17g', round-trips float64 losslessly)."""
    atoms = [f'(X_{i} {v:{fmt}})' for i, v in enumerate(x_flat)]
    atoms += [f'(Y_{j} {v:{fmt}})' for j, v in enumerate(y_flat)]
    return '(' + '\n'.join(atoms) + ')'


# TensorProto elem_type -> the dtype string a v2 counterexample writes (FLOAT/DOUBLE/FLOAT16).
_ONNX_DT = {1: 'float32', 11: 'float64', 10: 'float16'}


def _cex_v2(ins_meta, outs_meta, x_flat, y_flat, fmt, order=None):
    """Build the VNNLIB 2.0 counterexample: per-tensor `NAME dtype [d0,d1,...]` header then
    the tensor's C-order values (one per line). Tensors are emitted in the spec's DECLARATION
    order (`order` = list of ('in'|'out', index)) — for a single network that's inputs then
    output, but for a network PAIR the spec interleaves per-network (X_f, Y_f, X_g, Y_g), and
    the v2 validator reads variables BY ORDER, so grouping all inputs first is rejected
    (malformed_ce: expected Y_f, found X_g). `order=None` falls back to inputs-then-outputs."""
    if order is None:
        order = ([('in', i) for i in range(len(ins_meta))]
                 + [('out', j) for j in range(len(outs_meta))])
    in_off, out_off = [0], [0]
    for _, _, _, sz in ins_meta:
        in_off.append(in_off[-1] + sz)
    for _, _, _, sz in outs_meta:
        out_off.append(out_off[-1] + sz)
    lines = []
    for kind, idx in order:
        if kind == 'in':
            name, dt, shape, size = ins_meta[idx]
            flat, off = x_flat, in_off[idx]
        else:
            name, dt, shape, size = outs_meta[idx]
            flat, off = y_flat, out_off[idx]
        lines.append(f"{name} {dt} [{','.join(str(d) for d in shape)}]")
        lines.extend(f'{v:{fmt}}' for v in flat[off:off + size])
    return '\n'.join(lines)


def _format_cex(version, onnx_path, x_flat, y_flat, fmt, io_meta=None):
    """Dispatch the counterexample to the v1 (flat X_i/Y_i s-expr) or v2 (per-tensor) format
    per the resolved spec version. For v2 the per-tensor headers MUST MIRROR the spec's
    `declare-input`/`declare-output` — name, dtype (e.g. `real`/`float32`, echoed verbatim),
    and shape — so the v2 validator accepts them (`io_meta` = the spec-declared tensors).
    Only if the spec didn't declare them (`io_meta is None`) do we fall back to the ONNX node
    metadata. The values under each header are plain numbers regardless. Logs the source."""
    if version == '2.0':
        meta = io_meta if io_meta is not None else _onnx_io_meta(onnx_path)
        ins, outs = meta[0], meta[1]
        order = meta[2] if len(meta) > 2 else None   # spec declaration order (pairs interleave)
        _src = 'spec-declared tensors' if io_meta is not None else 'ONNX node metadata'
        print(f'  [counterexample] format=v2.0 (per-tensor: "NAME dtype [shape]" + '
              f'C-order values in spec-declaration order; using {_src})', flush=True)
        return _cex_v2(ins, outs, x_flat, y_flat, fmt, order)
    print('  [counterexample] format=v1.0 (flat s-expr: ((X_i <v>) ... (Y_j <v>)))',
          flush=True)
    return _cex_sexpr(x_flat, y_flat, fmt)


def _vnnlib_version(spec_path):
    """Detect a VNNLIB spec's version ('2.0' vs '1.0') from its head (handles .gz, and the
    instances.csv-style plain name when only the .gz is on disk)."""
    import gzip
    if not os.path.exists(spec_path) and os.path.exists(spec_path + '.gz'):
        spec_path = spec_path + '.gz'
    opener = gzip.open if spec_path.endswith('.gz') else open
    with opener(spec_path, 'rt') as fh:
        txt = fh.read(4096)
    return '2.0' if ('vnnlib-version' in txt or 'declare-network' in txt
                     or 'declare-input' in txt) else '1.0'


def _resolve_cex_io_meta(spec_path):
    """The SPEC-declared I/O for a v2 counterexample, as
    ``((name, dtype, shape, size) inputs..., (...) outputs...)`` — so EVERY emit path
    (standard / augmented / surrogate-multi-input) writes the vnnlib's variable names
    (X / X1,X2 / Y) AND mirrors the spec's declared dtype + shape, instead of the ONNX node
    metadata. The cex header MUST match the spec's `declare-input`/`declare-output` (the v2
    validator compares them, so `real`/`float32` is echoed verbatim); the values underneath
    are plain numbers regardless. Parsed CHEAPLY from just the `declare-network` header so a
    spec with millions of input-bound asserts is never fully read. Returns ``None`` for v1 /
    on a read error (the cex then keeps the ONNX node names). Resolved ONCE in ``main`` from
    the ORIGINAL spec (before any pair/augment rewrite of ``args.spec``)."""
    import gzip
    import re
    p = spec_path
    if not os.path.exists(p) and os.path.exists(p + '.gz'):
        p = p + '.gz'
    try:
        opener = gzip.open if p.endswith('.gz') else open
        with opener(p, 'rt') as fh:
            head = fh.read(16384)
    except OSError:
        return None
    if 'declare-network' not in head:
        return None
    ins, outs, order = [], [], []
    for kind, name, dt, shp in re.findall(
            r'\(declare-(input|output)\s+(\S+)\s+(\S+)\s+\[([^\]]*)\]', head):
        shape = tuple(int(s.strip()) for s in shp.split(',') if s.strip())
        size = 1
        for d in shape:
            size *= d
        if kind == 'input':
            order.append(('in', len(ins)))
            ins.append((name, dt, shape, size))
        else:
            order.append(('out', len(outs)))
            outs.append((name, dt, shape, size))
    # order = the DECLARATION order (a network PAIR interleaves X_f,Y_f,X_g,Y_g; the v2
    # validator reads variables by order, so the cex must follow it, not group inputs first).
    return (tuple(ins), tuple(outs), tuple(order)) if (ins or outs) else None


def _counterexample_sexpr(onnx_path, spec, witness, cex_fmt='.17g', version='1.0',
                          io_meta=None):
    """Build the counterexample for a SAT witness in the v1 (flat) or v2 (per-tensor)
    format. Returns the cex string or None if the ONNX output can't be computed (e.g.
    onnxruntime missing). Y is obtained from the same ORT forward the soundness validator
    runs, so it matches the scoring harness's recomputed output within tolerance.
    """
    import numpy as np
    # _validate_sat_witness is local to this module (extracted alongside)
    x = np.asarray(witness).flatten().astype(np.float64)
    # _validate_sat_witness runs ORT and stashes the output in info['out'].
    _, info = _validate_sat_witness(onnx_path, spec, witness)
    y = info.get('out')
    if y is None:
        return None
    # Write the float32-safe in-box witness as X (same point that produced Y
    # via ORT), so the scorer's box check passes despite FP edge rounding.
    if info.get('witness_inbox') is not None:
        x = np.asarray(info['witness_inbox']).flatten().astype(np.float64)
    y = np.asarray(y).flatten().astype(np.float64)
    # v2: emit with the SPEC's declared I/O names (io_meta), not the ONNX node names.
    return _format_cex(version, onnx_path, x, y, cex_fmt, io_meta=io_meta)
