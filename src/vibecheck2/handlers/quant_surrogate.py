"""Surrogate-PGD attack for INT8-quantized ONNX (standalone vc2 port of v1
vibecheck.surrogate_pgd, validated 50/50 on smart_turn_multimodal_2026 on
both CPU regimes).

The graph loader rejects QDQ nets (DequantizeLinear kernels), so this
handler replaces the pipeline (attack-only, never unsat):

  1. fold Q/DQ into a continuous FLOAT surrogate ONNX (weight DQ -> baked
     float constant, activation Q/DQ -> Identity), loaded via onnx2torch.
  2. PGD maximizing the output-spec violation over the L-inf box; the
     surrogate supplies only the GRADIENT direction.
  3. platform detection: ORT's fused u8xs8 GEMM computes a DIFFERENT
     function on non-VNNI CPUs (MLAS sums adjacent product pairs into
     int16 WITH SATURATION; AMD Zen2) than on VNNI (exact int32; Intel).
     A ~3ms QLinearMatMul probe characterizes THIS machine's oracle; on a
     saturating host the surrogate gets the saturating twins grafted in
     (handlers.saturating_quant) so its gradient steers toward CEs that
     actually flip the local scorer.
  4. every candidate is replayed on the ORIGINAL quantized ONNX with
     ORT-CPU (the scoring engine): in-box within atol, output STRICTLY
     violating (a boundary point, e.g. quantization-pinned Y == 0.5 for
     `Y > 0.5`, is NOT a counterexample). The verdict is decided ONLY by
     the original model, so a mismatched surrogate can never emit a
     false sat.
"""
import gzip
import os
import re
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# --------------------------------------------------------------------- io

def _load_onnx_model(path):
    """gz-aware onnx load (benchmarks ship `foo.onnx.gz` while instances.csv
    names `foo.onnx`)."""
    if not os.path.isfile(path) and os.path.isfile(path + '.gz'):
        path = path + '.gz'
    if path.endswith('.gz'):
        with gzip.open(path) as fh:
            return onnx.load_model_from_string(fh.read())
    return onnx.load(path)


def _atomic_onnx_save(model, out_path):
    """Temp + os.replace so an interrupted build never leaves a corrupt
    surrogate a later timed run would fail to load."""
    tmp = out_path + '.tmp'
    onnx.save(model, tmp)
    os.replace(tmp, out_path)


def _decompressed(path):
    if path.endswith('.gz') or (not os.path.isfile(path)
                                and os.path.isfile(path + '.gz')):
        return _load_onnx_model(path).SerializeToString()
    return path


def has_quantized_ops(onnx_path):
    """True if the ONNX uses DequantizeLinear/QuantizeLinear."""
    m = _load_onnx_model(onnx_path)
    return any(n.op_type in ('DequantizeLinear', 'QuantizeLinear')
               for n in m.graph.node)


# ------------------------------------------------------------ oracle probe

_QUANT_ORACLE_PROBE = None


def detect_quant_oracle():
    """Which u8xs8 quantized-GEMM regime the LOCAL onnxruntime uses: replay a
    tiny known-saturating QLinearMatMul and read the answer. 'exact' =
    VNNI/int32 (y ~ 250); 'saturating' = non-VNNI int16-pair saturation
    (y ~ 126). Characterizes what ORT on THIS machine actually computes
    (the validation oracle), not a CPUID heuristic; ~3ms."""
    global _QUANT_ORACLE_PROBE
    import onnxruntime as ort
    if _QUANT_ORACLE_PROBE is None:
        K = 64
        scale = np.float32(K * 255 * 127 * 0.1 * 0.1 / 250.0)
        g = helper.make_graph(
            [helper.make_node(
                'QLinearMatMul',
                ['a', 'a_s', 'a_z', 'B', 'b_s', 'b_z', 'y_s', 'y_z'],
                ['y'])],
            'quant_oracle_probe',
            [helper.make_tensor_value_info('a', TensorProto.UINT8, [1, K])],
            [helper.make_tensor_value_info('y', TensorProto.UINT8, [1, 1])],
            [numpy_helper.from_array(np.float32(0.1), 'a_s'),
             numpy_helper.from_array(np.uint8(0), 'a_z'),
             numpy_helper.from_array(np.full((K, 1), 127, np.int8), 'B'),
             numpy_helper.from_array(np.float32(0.1), 'b_s'),
             numpy_helper.from_array(np.int8(0), 'b_z'),
             numpy_helper.from_array(scale, 'y_s'),
             numpy_helper.from_array(np.uint8(0), 'y_z')])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 19)])
        m.ir_version = 9
        _QUANT_ORACLE_PROBE = m.SerializeToString()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(_QUANT_ORACLE_PROBE, so,
                                providers=['CPUExecutionProvider'])
    y = int(sess.run(None, {'a': np.full((1, 64), 255, np.uint8)})[0]
            .ravel()[0])
    return 'exact' if y >= 200 else 'saturating'


def resolve_saturation(mode='auto', log=print):
    """'auto' probes the local ORT so the surrogate matches the validation
    oracle on this box; 'on'/'off' force."""
    mode = str(mode).lower()
    if mode in ('on', 'true'):
        log('[vc2/surrogate] saturation: ON (forced)')
        return True
    if mode in ('off', 'false'):
        log('[vc2/surrogate] saturation: OFF (forced)')
        return False
    oracle = detect_quant_oracle()
    sat = (oracle == 'saturating')
    log(f'[vc2/surrogate] saturation: {"ON" if sat else "OFF"} '
        f'(auto; local ORT oracle is {oracle!r})')
    return sat


# ------------------------------------------------------------- conversion

def convert_onnx_to_torch(onnx_path):
    """onnx2torch.convert with an opset upgrade so old-opset nets load. The
    module is only a GRADIENT ORACLE -- every CE is re-validated on the
    ORIGINAL model via ORT-CPU, so conversion drift can only cost a found
    CE, never produce a false sat."""
    from onnx import version_converter
    from onnx2torch import convert
    m = _load_onnx_model(onnx_path)
    opset = max((oi.version for oi in m.opset_import
                 if oi.domain in ('', 'ai.onnx')), default=0)
    if 0 < opset < 13:
        m = version_converter.convert_version(m, 13)
    return convert(m)


def _model_input_shapes(onnx_path):
    """Free-input shapes in graph order (the spec only carries flat boxes)."""
    m = _load_onnx_model(onnx_path)
    init = {i.name for i in m.graph.initializer}
    return [[d.dim_value if d.dim_value > 0 else 1
             for d in i.type.tensor_type.shape.dim]
            for i in m.graph.input if i.name not in init]


# ------------------------------------------------------- fold surrogates

def build_float_surrogate(onnx_path, out_path):
    """Fold Q/DQ into a continuous float ONNX (the STE surrogate): weight
    DequantizeLinear -> baked float constant (incl. per-axis); activation
    Q/DQ -> Identity (drops the rounding => differentiable)."""
    m = _load_onnx_model(onnx_path)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    new_nodes, add_init = [], []
    for n in g.node:
        if n.op_type == 'QuantizeLinear':
            new_nodes.append(
                helper.make_node('Identity', [n.input[0]], [n.output[0]]))
            continue
        if n.op_type == 'DequantizeLinear':
            x = n.input[0]
            if x in init:
                w = init[x].astype(np.float64)
                s = init[n.input[1]].astype(np.float64)
                z = (init[n.input[2]].astype(np.float64)
                     if len(n.input) > 2 and n.input[2] in init else 0.0)
                axis = next((a.i for a in n.attribute if a.name == 'axis'), 1)
                if np.ndim(s) > 0:
                    shp = [1] * w.ndim
                    shp[axis % w.ndim] = s.shape[0]
                    s = s.reshape(shp)
                    z = np.reshape(z, shp) if np.ndim(z) > 0 else z
                add_init.append(numpy_helper.from_array(
                    ((w - z) * s).astype(np.float32), n.output[0]))
            else:
                new_nodes.append(
                    helper.make_node('Identity', [x], [n.output[0]]))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(add_init)
    keep = [o for o in m.opset_import
            if (o.domain or 'ai.onnx') in ('ai.onnx', '')]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    m.ir_version = 8
    _atomic_onnx_save(m, out_path)
    return out_path


def build_fakequant_surrogate(onnx_path, out_path):
    """Fold Q/DQ into float ops that REPRODUCE the INT8 rounding: activation
    QuantizeLinear -> Clip(Round(x/s) + z, qmin, qmax), DequantizeLinear ->
    (q - z) * s; weight DQ baked as in build_float_surrogate. Round grad is
    0 a.e. -- an eval/base model, made differentiable by grafting STE
    twins (saturating_quant.make_fakequant_differentiable)."""
    m = _load_onnx_model(onnx_path)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    init_dtype = {i.name: i.data_type for i in g.initializer}
    new_nodes, add_init = [], []
    uid = [0]

    def _const(arr, base):
        uid[0] += 1
        nm = f'_fq_{base}_{uid[0]}'
        add_init.append(numpy_helper.from_array(np.asarray(arr, np.float32),
                                                nm))
        return nm

    def _tmp(base):
        uid[0] += 1
        return f'_fq_{base}_{uid[0]}'

    for n in g.node:
        if n.op_type == 'QuantizeLinear':
            x = n.input[0]
            if x in init:
                raise NotImplementedError(
                    'fake-quant: QuantizeLinear with initializer input')
            s = init[n.input[1]].astype(np.float64)
            z = (init[n.input[2]].astype(np.float64)
                 if len(n.input) > 2 and n.input[2] in init else 0.0)
            if np.size(s) > 1:
                raise NotImplementedError(
                    'fake-quant: per-axis activation QuantizeLinear')
            zdt = (init_dtype.get(n.input[2], TensorProto.UINT8)
                   if len(n.input) > 2 else TensorProto.UINT8)
            qmin, qmax = ((0.0, 255.0) if zdt == TensorProto.UINT8
                          else (-128.0, 127.0))
            s_nm, z_nm = _const(s, 'qs'), _const(z, 'qz')
            lo_nm, hi_nm = _const(qmin, 'qlo'), _const(qmax, 'qhi')
            t_div, t_rnd, t_add = _tmp('qdiv'), _tmp('qrnd'), _tmp('qadd')
            new_nodes.append(helper.make_node('Div', [x, s_nm], [t_div]))
            new_nodes.append(helper.make_node('Round', [t_div], [t_rnd]))
            new_nodes.append(helper.make_node('Add', [t_rnd, z_nm], [t_add]))
            new_nodes.append(helper.make_node('Clip', [t_add, lo_nm, hi_nm],
                                              [n.output[0]]))
            continue
        if n.op_type == 'DequantizeLinear':
            x = n.input[0]
            s = init[n.input[1]].astype(np.float64)
            z = (init[n.input[2]].astype(np.float64)
                 if len(n.input) > 2 and n.input[2] in init else 0.0)
            axis = next((a.i for a in n.attribute if a.name == 'axis'), 1)
            if x in init:
                w = init[x].astype(np.float64)
                if np.ndim(s) > 0:
                    shp = [1] * w.ndim
                    shp[axis % w.ndim] = s.shape[0]
                    s = s.reshape(shp)
                    z = np.reshape(z, shp) if np.ndim(z) > 0 else z
                add_init.append(numpy_helper.from_array(
                    ((w - z) * s).astype(np.float32), n.output[0]))
            else:
                if np.size(s) > 1:
                    raise NotImplementedError(
                        'fake-quant: per-axis activation DequantizeLinear')
                s_nm, z_nm = _const(s, 'ds'), _const(z, 'dz')
                t_sub = _tmp('dsub')
                new_nodes.append(helper.make_node('Sub', [x, z_nm], [t_sub]))
                new_nodes.append(helper.make_node('Mul', [t_sub, s_nm],
                                                  [n.output[0]]))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(add_init)
    keep = [o for o in m.opset_import
            if (o.domain or 'ai.onnx') in ('ai.onnx', '')]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    m.ir_version = 8
    _atomic_onnx_save(m, out_path)
    return out_path


# ------------------------------------------------------------ spec parse

class SurrogateSpec:
    """Per-input L-inf box + output DNF.

    inputs:  list of (name, shape, lo_flat, hi_flat) in ONNX input order.
    out_dnf: list of clauses; clause = [(out_index, 'gt'|'lt', rhs), ...];
             a violation = SOME clause fully satisfied."""

    def __init__(self, inputs, out_dnf):
        self.inputs = inputs
        self.out_dnf = out_dnf


def parse_box_and_output(vnnlib_path):
    """Parse a v1 OR v2 box-robustness spec into a SurrogateSpec (the
    L-inf/classification case the surrogate mode targets)."""
    from ..frontend.io_util import ensure_decompressed
    vnnlib_path = ensure_decompressed(vnnlib_path)
    if vnnlib_path.endswith('.gz'):
        with gzip.open(vnnlib_path, 'rt') as fh:
            txt = fh.read()
    else:
        txt = open(vnnlib_path).read()
    is_v2 = ('vnnlib-version' in txt or 'declare-network' in txt
             or 'declare-input' in txt)
    return _parse_v2(txt) if is_v2 else _parse_v1(txt)


def _c_strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def _parse_v2(txt):
    inputs = []
    for m in re.finditer(
            r'\(declare-input\s+(\w+)\s+\w+\s*\[([\d,\s]+)\]\)', txt):
        name = m.group(1)
        shape = tuple(int(x) for x in m.group(2).split(','))
        n = int(np.prod(shape))
        lo = np.full(n, -np.inf, np.float64)
        hi = np.full(n, np.inf, np.float64)
        # VECTORIZED scatter: per-match python parsing of a ~1.2M-bound
        # spec took ~7s/case; batch regex + stride-dot + scatter is <1s.
        strides = np.asarray(_c_strides(shape), dtype=np.int64)
        for op, arr in (('>=', lo), ('<=', hi)):
            pairs = re.findall(
                rf'\({op}\s*{name}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt)
            if not pairs:
                continue
            idx_strs, val_strs = zip(*pairs)
            flat = np.array([s.split(',') for s in idx_strs],
                            dtype=np.int64) @ strides
            arr[flat] = np.asarray(val_strs, dtype=np.float64)
        inputs.append((name, shape, lo, hi))
    om = re.search(r'\(declare-output\s+(\w+)\s', txt)
    yname = om.group(1) if om else 'Y'
    out_dnf = []
    for mm in re.finditer(rf'\(>\s*{yname}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1).split(',')[-1]), 'gt',
                         float(mm.group(2)))])
    for mm in re.finditer(rf'\(<\s*{yname}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1).split(',')[-1]), 'lt',
                         float(mm.group(2)))])
    if not inputs or not out_dnf:
        raise NotImplementedError(
            f'surrogate spec parse: unsupported v2 structure '
            f'(inputs={len(inputs)}, out_dnf={len(out_dnf)})')
    return SurrogateSpec(inputs, out_dnf)


def _parse_v1(txt):
    n = len(re.findall(r'\(declare-const\s+X_\d+\s+Real\)', txt))
    lo = np.full(n, -np.inf, np.float64)
    hi = np.full(n, np.inf, np.float64)
    for op, arr in (('>=', lo), ('<=', hi)):
        pairs = re.findall(rf'\({op}\s*X_(\d+)\s*([-\d.eE]+)\)', txt)
        if pairs:
            idx, val = zip(*pairs)
            arr[np.asarray(idx, dtype=np.int64)] = np.asarray(
                val, dtype=np.float64)
    out_dnf = []
    for mm in re.finditer(r'\(>\s*Y_(\d+)\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1)), 'gt', float(mm.group(2)))])
    for mm in re.finditer(r'\(<\s*Y_(\d+)\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1)), 'lt', float(mm.group(2)))])
    if n == 0 or not out_dnf:
        raise NotImplementedError(
            'surrogate spec parse: unsupported v1 structure')
    return SurrogateSpec([('X', (n,), lo, hi)], out_dnf)


# ---------------------------------------------------------- ORT validate

_ORT_SESSION_CACHE = {}


def _ort_eval(onnx_path, feed):
    """Replay on the ORIGINAL quantized ONNX with CPU onnxruntime (the
    scoring engine). Session cached per model: building it loads the 1.1GB
    smart_turn onnx (~0.5s) and this runs once PER PGD STEP."""
    import onnxruntime as ort
    sess = _ORT_SESSION_CACHE.get(onnx_path)
    if sess is None:
        sess = ort.InferenceSession(_decompressed(onnx_path),
                                    providers=['CPUExecutionProvider'])
        _ORT_SESSION_CACHE[onnx_path] = sess
    names = [i.name for i in sess.get_inputs()]
    return np.asarray(sess.run(
        None, {names[k]: feed[k].astype(np.float32)
               for k in range(len(names))})[0]).ravel()


# ------------------------------------------------------------------- PGD

def surrogate_attack(onnx_path, vnnlib_path, timeout, device='cpu',
                     log=print, spec=None, restarts=1, steps=50,
                     saturation='auto', atol=1e-4, strict_buffer=1e-9,
                     surrogate_dir=None, seed=0):
    """Surrogate-PGD (v1-validated flow). Returns (verdict, witness):
    verdict in {'sat','timeout'}, witness a list of per-input float64
    arrays (None unless sat). Candidates: box CENTER + PGD steps; each is
    ORT-CPU-confirmed on the ORIGINAL model with the STRICT output rule
    (a boundary point is not a CE)."""
    import torch

    # the verifier pins single-threaded BLAS for sound bounding; the
    # surrogate attack is an approximate gradient search (ORT decides),
    # and its forward -- especially the saturating GEMM's [M,K/2,N]
    # materialization -- is the bottleneck (~65s/step single-threaded,
    # ~15s at 12 threads)
    torch.set_num_threads(min(12, os.cpu_count() or 1))

    t0 = time.time()
    if spec is None:
        spec = parse_box_and_output(vnnlib_path)
    mshapes = _model_input_shapes(onnx_path)
    if len(mshapes) != len(spec.inputs):
        raise NotImplementedError(
            f'surrogate spec inputs ({len(spec.inputs)}) != model inputs '
            f'({len(mshapes)})')
    saturate = resolve_saturation(saturation, log=log)

    if surrogate_dir is None:
        surrogate_dir = os.path.join(
            os.environ.get('TMPDIR', '/tmp'), 'vc2_surrogates')
    os.makedirs(surrogate_dir, exist_ok=True)
    base = os.path.basename(onnx_path).replace('.gz', '').replace(
        '.onnx', '')
    surrogate_path = os.path.join(surrogate_dir, base + '_float.onnx')
    fq_path = os.path.join(surrogate_dir, base + '_fq.onnx')

    from onnx2torch import convert
    dev = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) \
        else 'cpu'
    if dev == 'cuda':
        # cap the process so the saturating surrogate (gradient-
        # checkpointed, ~3GB) can't grab the whole card
        _tot = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(0.95, 6.0 / _tot), 0)
    if saturate:
        # non-VNNI scorer: the float/fakequant surrogates track the VNNI
        # output, so their gradient finds nothing here (measured). Build
        # the SATURATING surrogate: differentiable fakequant (STE
        # round/clip) + int16-pair saturation grafted into matmuls and
        # audio Conv1d (video Conv3d skipped: saturating im2col OOMs and
        # they don't affect the flip). Partially faithful => bad RANKER;
        # its GRADIENT steers, ORT validates every stepped point.
        from .saturating_quant import (make_fakequant_differentiable,
                                       graft_saturating_matmuls,
                                       graft_saturating_convs)
        if not os.path.exists(fq_path):
            build_fakequant_surrogate(onnx_path, fq_path)
            log(f'[vc2/surrogate] built fakequant surrogate -> {fq_path}')
        model = convert(fq_path).eval().to(dev)
        make_fakequant_differentiable(model, log=log)
        graft_saturating_matmuls(model, onnx_path, log=log)
        graft_saturating_convs(model, onnx_path, log=log,
                               only_types=('Conv1d',))
        eval_model = None
    else:
        if not os.path.exists(surrogate_path):
            build_float_surrogate(onnx_path, surrogate_path)
            log(f'[vc2/surrogate] built float surrogate -> {surrogate_path}')
        model = convert(surrogate_path).eval().to(dev)
        eval_model = None
        if not os.path.exists(fq_path):
            build_fakequant_surrogate(onnx_path, fq_path)
        eval_model = convert(fq_path).eval().to(dev)
    log(f'[vc2/surrogate] loaded on {dev} in {time.time() - t0:.1f}s; '
        f'inputs={[(n, s) for n, s, _, _ in spec.inputs]} '
        f'restarts={restarts} steps={steps} '
        f'saturation={"on" if saturate else "off"}')

    def to_t(a, shp):
        return torch.tensor(a.astype(np.float32).reshape(tuple(shp)),
                            device=dev)

    los = [to_t(lo, mshapes[k])
           for k, (_, _, lo, _) in enumerate(spec.inputs)]
    his = [to_t(hi, mshapes[k])
           for k, (_, _, _, hi) in enumerate(spec.inputs)]
    cens = [(l + h) / 2 for l, h in zip(los, his)]

    def viol_loss(y):
        clause_vals = []
        for clause in spec.out_dnf:
            margins = [(y[i] - rhs) if op == 'gt' else (rhs - y[i])
                       for i, op, rhs in clause]
            clause_vals.append(torch.stack(margins).min())
        return torch.stack(clause_vals).max()

    def margin_np(y):
        # float64 per element: a float32 y minus a python rhs collapses to
        # float32 under NEP-50 and would hide a sub-float32 strict buffer
        best = -np.inf
        for clause in spec.out_dnf:
            m = min((float(y[i]) - rhs) if op == 'gt' else (rhs - float(y[i]))
                    for i, op, rhs in clause)
            best = max(best, m)
        return float(best)

    def fq_margin(pts):
        if eval_model is None:
            return None
        with torch.no_grad():
            y = eval_model(*pts)
            y = (y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
        return margin_np(y.detach().cpu().numpy())

    _n_steps = [0]
    _t_val = [0.0]
    _n_val = [0]

    def ort_consider(pts, tag):
        feed = [p.detach().cpu().numpy().reshape(mshapes[k])
                for k, p in enumerate(pts)]
        for f, (_, _, lo, hi) in zip(feed, spec.inputs):
            ff = f.ravel()
            assert (ff >= lo - atol).all() and (ff <= hi + atol).all(), \
                'surrogate produced an out-of-box witness'
        _v0 = time.time()
        y = _ort_eval(onnx_path, feed)
        _t_val[0] += time.time() - _v0
        _n_val[0] += 1
        m = margin_np(y)
        if m >= strict_buffer:
            log(f'[vc2/surrogate] CLEAR SAT at {tag} (ORT margin={m:.3e})')
            return feed
        return None

    alphas = [0.05, 0.1, 0.2, 0.02]
    if saturate:
        # slow forward (~10-20x): must crack in FEW steps; cycle alpha PER
        # STEP spanning large->small (inst 10 needs ~0.75 at step 4,
        # inst 24/28/39 need ~0.15 -- a big step overshoots their cell)
        alphas = [0.5, 0.2, 0.75, 0.15, 0.35, 0.1, 0.6, 0.9]

    res = ort_consider(cens, 'center')
    if res is not None:
        return 'sat', res

    rng = torch.Generator(device='cpu')
    for r in range(restarts):
        if time.time() - t0 > timeout:
            break
        base_alpha = alphas[r % len(alphas)]
        if r == 0:
            pts = [c.clone() for c in cens]
        else:
            rng.manual_seed(seed + r)
            pts = [l + (h - l) * torch.rand(l.shape, generator=rng).to(dev)
                   for l, h in zip(los, his)]
        best_loss = float('-inf')
        best_pts = [p.detach().clone() for p in pts]
        best_fq = float('-inf')
        best_fq_pts = None
        for it in range(steps):
            if time.time() - t0 > timeout:
                break
            alpha = alphas[it % len(alphas)] if saturate else base_alpha
            for p in pts:
                p.requires_grad_(True)
            y = model(*pts)
            y = (y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
            loss = viol_loss(y)
            _lv = float(loss.detach())
            if _lv > best_loss:
                best_loss = _lv
                snap = [p.detach().clone() for p in pts]
                best_pts = snap
                fqm_s = fq_margin(snap)
                if fqm_s is not None and fqm_s > best_fq:
                    best_fq, best_fq_pts = fqm_s, snap
                if fqm_s is not None and fqm_s >= strict_buffer:
                    res = ort_consider(
                        snap, f'restart{r} step{it} (a={alpha},'
                              f'fq={fqm_s:.3e})')
                    if res is not None:
                        return 'sat', res
            grads = torch.autograd.grad(loss, pts)
            with torch.no_grad():
                pts = [torch.minimum(
                    torch.maximum(p + alpha * (h - l) * g.sign(), l), h)
                    for p, g, l, h in zip(pts, grads, los, his)]
            _n_steps[0] += 1
            if saturate:
                # the partial saturating surrogate UNDER-predicts the real
                # flip (reads 0.5 where ORT already cracked) -- validate
                # every stepped point on the authoritative ORT directly
                res = ort_consider([p.detach() for p in pts],
                                   f'restart{r} step{it} (sat,a={alpha})')
                if res is not None:
                    return 'sat', res
        cand = best_fq_pts if best_fq_pts is not None else best_pts
        fqm = best_fq if best_fq_pts is not None else fq_margin(best_pts)
        if fqm is None or fqm >= -atol:
            res = ort_consider(
                cand, f'restart{r}(a={base_alpha}'
                      + (f',fq={fqm:.3e})' if fqm is not None else ')'))
            if res is not None:
                return 'sat', res

    log(f'[vc2/surrogate] no CE (t={time.time() - t0:.1f}s; '
        f'steps={_n_steps[0]} validate={_t_val[0]:.1f}s/{_n_val[0]})')
    return 'timeout', None


# ----------------------------------------------------------------- entry

def try_quant_surrogate(onnx_path, vnnlib_path, timeout, device='cpu',
                        log=print):
    """Handler entry: returns (verdict, details). Raises NotImplementedError
    when the net has no quantized ops (caller re-raises the original load
    error). On sat, details carries 'ce_sexpr' (multi-input v2 CE text) --
    the witness re-passes the unified ORT gate (frontend.witness.
    _validate_witness_ort) before emission, so this path keeps the same
    chokepoint discipline as the graph route."""
    if not has_quantized_ops(onnx_path):
        raise NotImplementedError('quant_surrogate: no Q/DQ ops')
    t0 = time.time()
    spec = parse_box_and_output(vnnlib_path)
    log(f'[vc2/surrogate] quantized net; spec parsed in '
        f'{time.time() - t0:.1f}s '
        f'({sum(int(np.prod(s)) for _, s, _, _ in spec.inputs)} input dims)')
    verdict, witness = surrogate_attack(
        onnx_path, vnnlib_path, timeout - (time.time() - t0) - 3.0,
        device=device, log=log, spec=spec)
    details = {'time': time.time() - t0, 'handler': 'quant_surrogate'}
    if verdict != 'sat' or witness is None:
        return ('timeout' if verdict == 'timeout' else verdict), details
    from ..frontend.witness import (_validate_witness_ort, _format_cex,
                                    _resolve_cex_io_meta, _vnnlib_version)
    strict_buffer = 1e-9
    wits = [np.asarray(w).ravel().astype(np.float64) for w in witness]
    boxes = [(lo, hi) for _, _, lo, hi in spec.inputs]

    def _violated(inbox, yv):
        m = max(min((float(yv[i]) - rhs) if op == 'gt'
                    else (rhs - float(yv[i]))
                    for i, op, rhs in clause)
                for clause in spec.out_dnf)
        return (m >= strict_buffer), {'worst_margin': m}

    ok, vinfo = _validate_witness_ort(onnx_path, wits, boxes, _violated,
                                      1e-4)
    if not ok:
        log(f'[vc2/surrogate] witness failed the unified ORT gate '
            f'(margin={vinfo.get("worst_margin")}); NOT emitting sat')
        return 'timeout', details
    inbox = vinfo.get('witnesses_inbox') or witness
    x = np.concatenate([np.asarray(w).ravel()
                        for w in inbox]).astype(np.float64)
    y = np.asarray(vinfo.get('out')
                   if vinfo.get('out') is not None
                   else _ort_eval(onnx_path,
                                  [np.asarray(w) for w in witness])
                   ).ravel().astype(np.float64)
    details['ce_sexpr'] = _format_cex(
        _vnnlib_version(vnnlib_path), onnx_path, x, y, '.17g',
        io_meta=_resolve_cex_io_meta(vnnlib_path))
    details['witness_multi'] = inbox
    return 'sat', details
