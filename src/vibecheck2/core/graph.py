"""Flat DAG IR and the converter from the v1 ONNX front end.

Design (docs/clean_slate_design.md 3.1/3.2): every edge is a FLAT vector
(C-order flatten, batch dim dropped). All shape reasoning happens ONCE, at
conversion time, as index math; the propagators never reshape. The complete
op vocabulary the propagators see:

  kind='input'                       the network input
  kind='linmap'   op.lm: LinMap      any affine op (Gemm/Conv/AvgPool/
                                     Slice/Transpose/Gather/Split/const-
                                     arith/scale-shift), y = lin(x)+b
  kind='nonlin'   op.fn: str         elementwise nonlinearity (relu,
                                     sigmoid, tanh, sin, cos, exp, ...)
  kind='add'      two live inputs    exact merge (residual)
  kind='concat'   live parts + const base, scatter by precomputed indices
  kind='maxpool'  window max         (own relaxation; point eval for now)

Reshape-like ops (Flatten/Reshape/Squeeze/Unsqueeze/Dropout/Identity) are
identity on flat C-order data and are ELIDED at conversion via aliasing.
Anything unrecognized raises NotImplementedError (never silently skipped).

Conversion reuses the mature v1 loader (constant folding, BN folding,
optimizer rewrites) and translates its `ComputeGraph` into this IR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import linmap as lm

# v1 nodes that are identity on flat C-order data
_ELIDE = {'Flatten', 'Reshape', 'Squeeze', 'Unsqueeze', 'Dropout', 'Identity'}

# elementwise nonlinearities passed through to RelaxLib by name
_NONLIN = {'Relu': 'relu', 'Sigmoid': 'sigmoid', 'Tanh': 'tanh',
           'Sin': 'sin', 'Cos': 'cos', 'Exp': 'exp', 'Sign': 'sign',
           'Floor': 'floor'}


@dataclass
class Op:
    name: str
    kind: str                      # input | linmap | nonlin | add | concat | maxpool
    inputs: tuple = ()
    shape: tuple = ()              # logical ND shape (batch dropped), doc only
    n: int = 0                     # flat output size
    lm: object = None              # LinMap for kind='linmap'
    fn: str = ''                   # nonlinearity name for kind='nonlin'
    params: dict = field(default_factory=dict)


class Net:
    """Topologically ordered flat DAG. ops[name] produce edge `name`."""

    def __init__(self, ops, order, input_name, output_name, onnx_path=None):
        self.ops = ops
        self.order = order            # topo order, excludes 'input'
        self.input_name = input_name
        self.output_name = output_name
        self.onnx_path = onnx_path
        self.n_in = ops[input_name].n
        self.n_out = ops[output_name].n

    def __repr__(self):
        kinds = {}
        for o in self.ops.values():
            kinds[o.kind if o.kind != 'nonlin' else o.fn] = \
                kinds.get(o.kind if o.kind != 'nonlin' else o.fn, 0) + 1
        return (f'Net({len(self.order)} ops, in={self.n_in}, out={self.n_out}, '
                + ', '.join(f'{k}:{v}' for k, v in sorted(kinds.items())) + ')')

    def consumers(self):
        """name -> [ops consuming it] (for backward sweeps)."""
        cons = {}
        for name in self.order:
            for i in self.ops[name].inputs:
                cons.setdefault(i, []).append(name)
        return cons


def tag_simplex_bmm(net):
    """Mark bmm ops whose LEFT factor is a softmax output over the
    contraction axis: each output row is then a CONVEX COMBINATION of the
    right factor's rows (weights >= 0, summing to 1), so the output is
    bounded by [min_j V_j, max_j V_j] coordinatewise -- the standard
    attention tightening. Without it, interval treats the weights as an
    independent [0,1] box and attn@V blows up by a factor of the token
    count (vit block 1: diffs +-712 instead of +-40)."""
    for nm in net.order:
        op = net.ops[nm]
        if op.kind != 'bmm':
            continue
        left = net.ops[op.inputs[0]]
        if (left.kind == 'nonlin' and left.fn == 'reciprocal'
                and left.params.get('softmax_axis_len') ==
                op.params['a_shape'][-1]
                and left.params.get('softmax_post') == 1):
            op.params['simplex_left'] = True
    return net


def decompose_maxpool(net):
    """Replace native maxpool ops with the exact pairwise relu tree:
    max(a, b) = a + relu(b - a), levels until one column remains. Window
    slots that fall in the padding REPEAT the window's first valid element
    (max(a, a) = a), so the -inf padding semantics need no special value
    and PADDED pools decompose exactly too (the ONNX-level rewrite refuses
    those). The introduced relus ride every bound path, the split scoring,
    and the alpha optimizer like any other relu.
    """
    from .linmap import Scale, Select
    if not any(net.ops[nm].kind == 'maxpool' for nm in net.order):
        return net
    ops = dict(net.ops)
    order = []
    for name in net.order:
        op = net.ops[name]
        if op.kind != 'maxpool':
            order.append(name)
            continue
        C, H, W = op.params['in_shape']
        kh, kw = op.params['kernel_shape']
        sh, sw = op.params['stride']
        ph, pw = op.params['padding']
        OH = (H + 2 * ph - kh) // sh + 1
        OW = (W + 2 * pw - kw) // sw + 1
        n_out = C * OH * OW
        src = op.inputs[0]
        n_src = net.ops[src].n
        # (n_out, m) window index matrix, invalid slots repeating the
        # window's first valid element
        win = np.empty((n_out, kh * kw), dtype=np.int64)
        o = 0
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    idxs = []
                    for i in range(kh):
                        for j in range(kw):
                            r, s = oh * sh - ph + i, ow * sw - pw + j
                            if 0 <= r < H and 0 <= s < W:
                                idxs.append(c * H * W + r * W + s)
                    assert idxs, 'all-padding window'
                    while len(idxs) < kh * kw:
                        idxs.append(idxs[0])
                    win[o] = idxs
                    o += 1

        def emit(nm, kind, inputs, n, **kw):
            ops[nm] = Op(nm, kind, tuple(inputs), (n,), n, **kw)
            order.append(nm)

        cols = kh * kw
        cur = f'{name}/w'
        emit(cur, 'linmap', [src], n_out * cols,
             lm=Select(win.T.reshape(-1), n_src))
        lvl = 0
        while cols > 1:
            P, odd = cols // 2, cols % 2
            npairs = P * n_out
            ev = np.concatenate([np.arange(n_out) + 2 * p * n_out
                                 for p in range(P)])
            od = ev + n_out
            u, v = f'{name}/u{lvl}', f'{name}/v{lvl}'
            emit(u, 'linmap', [cur], npairs,
                 lm=Select(ev, n_out * cols))
            emit(v, 'linmap', [cur], npairs,
                 lm=Select(od, n_out * cols))
            nu = f'{name}/n{lvl}'
            emit(nu, 'linmap', [u], npairs, lm=Scale(-1.0, npairs))
            dd = f'{name}/d{lvl}'
            emit(dd, 'add', [v, nu], npairs)
            rr = f'{name}/r{lvl}'
            emit(rr, 'nonlin', [dd], npairs, fn='relu')
            done = (P == 1 and not odd)
            mx = name if done else f'{name}/m{lvl}'
            emit(mx, 'add', [u, rr], npairs)
            if odd:
                tail = f'{name}/t{lvl}'
                emit(tail, 'linmap', [cur], n_out,
                     lm=Select(np.arange(n_out) + (cols - 1) * n_out,
                               n_out * cols))
                cols = P + 1
                done = cols == 1
                cat = name if done else f'{name}/c{lvl}'
                emit(cat, 'concat', [mx, tail], (P + 1) * n_out,
                     params={'base': np.zeros((P + 1) * n_out,
                                              dtype=np.float32),
                             'positions': [list(range(P * n_out)),
                                           list(range(P * n_out,
                                                      (P + 1) * n_out))],
                             'n_out': (P + 1) * n_out})
                cur = cat
            else:
                cols = P
                cur = mx
            lvl += 1
        if kh * kw == 1:                      # degenerate 1x1 pool
            emit(name, 'linmap', [cur], n_out,
                 lm=Select(np.arange(n_out), n_out))
    return Net(ops, order, net.input_name, net.output_name,
               onnx_path=net.onnx_path)


def _flat(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _drop_batch(shape):
    """v1 shapes carry a leading batch dim of 1 (mostly); normalize it away."""
    s = tuple(int(d) for d in shape)
    return s[1:] if len(s) > 1 and s[0] == 1 else s


def _broadcast_flat(const, shape, dtype):
    """Resolve numpy broadcasting of `const` against ND `shape` -> flat vec.

    The const may carry extra leading batch-like 1-dims (ONNX initializers
    often do); broadcast to the common shape and demand the flat size match.
    """
    arr = np.asarray(const, dtype=dtype)
    full = np.broadcast_shapes(arr.shape, shape) if arr.ndim else tuple(shape)
    out = np.ascontiguousarray(np.broadcast_to(arr, full)).reshape(-1)
    if out.size != _flat(shape):
        raise NotImplementedError(
            f'const of shape {arr.shape} does not broadcast onto {shape}')
    return out


def _onnx_true_shapes(onnx_path):
    """name -> ND shape from onnx.shape_inference: the AUTHORITATIVE shape
    oracle for edges that still carry their ONNX tensor names. v1's recorded
    shapes are runtime-dynamic hints and go stale around broadcasts; ONNX
    static inference does not. Best-effort: {} when inference fails (the
    converter then relies on v1 hints + its own tracking, all flat-checked)."""
    import gzip
    import onnx
    from onnx import shape_inference
    try:
        if str(onnx_path).endswith('.gz'):
            with gzip.open(onnx_path, 'rb') as f:
                m = onnx.load_model_from_string(f.read())
        else:
            m = onnx.load(onnx_path)
        mi = shape_inference.infer_shapes(m)
    except Exception as e:                      # noqa: BLE001 - oracle is optional
        print(f'[vc2] onnx shape inference unavailable ({type(e).__name__}); '
              f'using v1 shape hints only')
        return {}
    out = {}
    for v in (list(mi.graph.value_info) + list(mi.graph.output)
              + list(mi.graph.input)):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if dims and all(isinstance(d, int) and d > 0 for d in dims):
            out[v.name] = tuple(dims)
    return out


def from_compute_graph(cg, true_shapes=None) -> Net:
    """Translate a v1 ComputeGraph (post folding/optimization) into a Net."""
    dtype = cg.dtype
    true_shapes = true_shapes or {}
    alias = {}                      # v1 name -> IR name (for elided ops)
    shapes = {cg.input_name: _drop_batch(cg.input_shape)}
    ops = {}
    order = []

    in_shape = _drop_batch(cg.input_shape)
    ops[cg.input_name] = Op(cg.input_name, 'input', (), in_shape, _flat(in_shape))

    def src(v1name):
        return alias.get(v1name, v1name)

    def in_shape_of(node):
        nm = node.inputs[0]
        return shapes[nm] if nm in shapes else _drop_batch(
            cg.nodes[nm].output_shape if nm in cg.nodes else cg.input_shape)

    # ND shape per IR edge, OWNED BY THIS CONVERTER. v1's recorded shapes are
    # only hints: its runtime broadcasts dynamically, so after any broadcast
    # its downstream metadata can be stale. All index math must use nd[].
    nd = {cg.input_name: tuple(int(d) for d in cg.input_shape)}

    def v1shape(nm):
        """ND shape of edge `nm` for index math. Priority: ONNX-inferred
        (authoritative) > converter-tracked > v1 declared. Loud on flat
        mismatch with the actual data size."""
        ir = src(nm)
        actual = ops[ir].n if ir in ops else None
        ts = true_shapes.get(nm)
        if ts is not None and (actual is None or _flat(ts) == actual):
            return ts
        if nm == ir and nm in nd:
            return nd[ir]
        raw = tuple(int(d) for d in (
            cg.nodes[nm].output_shape if nm in cg.nodes else cg.input_shape))
        if actual is not None and _flat(raw) != actual:
            raise NotImplementedError(
                f'{nm}: declared shape {raw} (flat {_flat(raw)}) does '
                f'not match actual edge size {actual}')
        return raw

    def emit(name, kind, inputs, shape, nd_shape=None, **kw):
        op = Op(name, kind, tuple(inputs), tuple(shape), _flat(shape), **kw)
        ops[name] = op
        order.append(name)
        # track the converter-owned ND shape of this edge (see nd above):
        # explicit where the emitter computed one; inherited from the input
        # for size-preserving ops; v1's declared shape when its flat size
        # agrees; a flat vector otherwise.
        if nd_shape is not None:
            nd[name] = tuple(int(d) for d in nd_shape)
        elif (inputs and inputs[0] in nd and ops[inputs[0]].n == op.n
              and (kind == 'nonlin'
                   or (kind == 'linmap' and isinstance(op.lm, lm.ScaleShift)))):
            # layout-preserving elementwise op: same ND view as its input
            nd[name] = nd[inputs[0]]
        else:
            raw = tuple(int(d) for d in cg.nodes[name].output_shape) \
                if name in cg.nodes else tuple(shape)
            nd[name] = raw if _flat(raw) == op.n else (op.n,)

    def broadcast_in(node, target_shape):
        """Input edge for a const-arith op, expanded up to `target_shape` via
        a Select when ONNX broadcasting grows the input (e.g. (24,1) - c(24,54)).
        Returns (ir_name, n) of the (possibly expanded) input edge."""
        nm = node.inputs[0]
        n_t = _flat(target_shape)
        ish = v1shape(nm)
        if _flat(ish) == n_t:
            return src(nm), n_t
        grid = np.arange(_flat(ish)).reshape(ish)
        full = np.broadcast_shapes(ish, tuple(target_shape))
        if _flat(full) != n_t:
            raise NotImplementedError(
                f'{node.name}: input {ish} does not broadcast to {target_shape}')
        idx = np.ascontiguousarray(np.broadcast_to(grid, full)).reshape(-1)
        bname = node.name + '/bcast'
        emit(bname, 'linmap', [src(nm)], full, lm=lm.Select(idx, _flat(ish)))
        return bname, n_t

    def broadcast_pair(node, suffix=''):
        """Two live inputs joined elementwise: compute the TRUE numpy
        broadcast shape from the raw v1 input shapes (v1 metadata for these
        nodes is unreliable) and expand either side via Select as needed.
        Returns (name_a, name_b, full_shape)."""
        sa, sb = v1shape(node.inputs[0]), v1shape(node.inputs[1])
        full = np.broadcast_shapes(sa, sb)
        names = []
        for nm, sh in ((node.inputs[0], sa), (node.inputs[1], sb)):
            if _flat(sh) == _flat(full):
                names.append(src(nm))
            else:
                grid = np.arange(_flat(sh)).reshape(sh)
                idx = np.ascontiguousarray(
                    np.broadcast_to(grid, full)).reshape(-1)
                bname = f'{node.name}/bcast{suffix}{len(names)}'
                emit(bname, 'linmap', [src(nm)], full,
                     lm=lm.Select(idx, _flat(sh)))
                names.append(bname)
        return names[0], names[1], _drop_batch(full)

    for name in cg.topo_order:
        node = cg.nodes[name]
        t = node.op_type
        out_shape = _drop_batch(node.output_shape)
        shapes[name] = out_shape
        n_out = _flat(out_shape)

        if t in _ELIDE:
            alias[name] = src(node.inputs[0])
            continue

        if t == 'Cast':
            # value-preserving for float targets (all propagation is fp32/64);
            # an int cast truncates and needs a real handler -> loud
            to = node.params.get('to')
            if to in (None, 1, 10, 11):        # unset/float32/float16/float64
                alias[name] = src(node.inputs[0])
                continue
            raise NotImplementedError(
                f'{name}: Cast to ONNX dtype {to} (non-float) unsupported')

        if t in _NONLIN:
            emit(name, 'nonlin', [src(node.inputs[0])], out_shape, fn=_NONLIN[t])

        elif t == 'LeakyRelu':
            emit(name, 'nonlin', [src(node.inputs[0])], out_shape,
                 fn='leaky_relu', params={'alpha': node.params.get('alpha', 0.01)})

        elif t in ('Gemm', 'MatMul') and 'W' not in node.params:
            # variable-weight matmul (attention QK^T / AV): batched over the
            # leading dims of both operands. Point/attack support now; the
            # bilinear McCormick relaxation is the M6 core work.
            sa, sb = v1shape(node.inputs[0]), v1shape(node.inputs[1])
            if len(sa) < 2 or len(sb) < 2 or sa[-1] != sb[-2]:
                raise NotImplementedError(
                    f'{name}: bmm shapes {sa} @ {sb}')
            emit(name, 'bmm', [src(node.inputs[0]), src(node.inputs[1])],
                 out_shape, params={'a_shape': sa, 'b_shape': sb})

        elif t == 'Softmax':
            # decompose into existing RelaxLib ops: softmax(x) =
            # softmax_i = 1 / sum_j exp(x_j - x_i): shift-invariant BY
            # CONSTRUCTION (differences cancel the common scale, so fp32
            # never sees exp of a raw logit), and the sum contains the
            # j = i term exp(0) = 1, so the reciprocal's input is >= 1
            # -- no zero-range refusals, output lands in (0, 1] for
            # free. CROWN also bounds logit DIFFERENCES far tighter than
            # raw logits (common terms cancel through the linear map).
            # Costs k x k intermediate width; huge axes keep the plain
            # exp/sum/reciprocal/mul form.
            ish = v1shape(node.inputs[0])
            axis = node.params.get('axis', -1)
            a = axis if axis >= 0 else len(ish) + axis
            pre, k, post = _flat(ish[:a]), ish[a], _flat(ish[a + 1:])
            n_sm = pre * k * post
            n_d = pre * k * k * post
            if n_d <= 4_000_000:
                pi = np.arange(pre)[:, None, None, None]
                ii = np.arange(k)[None, :, None, None]
                jj = np.arange(k)[None, None, :, None]
                qq = np.arange(post)[None, None, None, :]
                idx_j = np.ascontiguousarray(np.broadcast_to(
                    (pi * k + jj) * post + qq,
                    (pre, k, k, post))).reshape(-1)
                idx_i = np.ascontiguousarray(np.broadcast_to(
                    (pi * k + ii) * post + qq,
                    (pre, k, k, post))).reshape(-1)
                x0 = src(node.inputs[0])
                emit(name + '/sj', 'linmap', [x0], (n_d,),
                     nd_shape=(pre, k, k, post), lm=lm.Select(idx_j, n_sm))
                emit(name + '/si', 'linmap', [x0], (n_d,),
                     nd_shape=(pre, k, k, post), lm=lm.Select(idx_i, n_sm))
                emit(name + '/ni', 'linmap', [name + '/si'], (n_d,),
                     nd_shape=(pre, k, k, post), lm=lm.Scale(-1.0, n_d))
                emit(name + '/d', 'add', [name + '/sj', name + '/ni'],
                     (n_d,), nd_shape=(pre, k, k, post))
                emit(name + '/e', 'nonlin', [name + '/d'], (n_d,),
                     nd_shape=(pre, k, k, post), fn='exp')
                emit(name + '/s', 'linmap', [name + '/e'], (n_sm,),
                     nd_shape=ish, lm=lm.SumAxis(pre * k, k, post))
                emit(name, 'nonlin', [name + '/s'], out_shape,
                     nd_shape=ish, fn='reciprocal',
                     params={'out_lo': 0.0, 'out_hi': 1.0,
                             'softmax_axis_len': k, 'softmax_post': post})
            else:
                e = name + '/exp'
                emit(e, 'nonlin', [x0 := src(node.inputs[0])],
                     _drop_batch(ish), nd_shape=ish, fn='exp')
                sm = name + '/sum'
                emit(sm, 'linmap', [e], (pre, post),
                     nd_shape=(pre, 1, post), lm=lm.SumAxis(pre, k, post))
                rc = name + '/recip'
                emit(rc, 'nonlin', [sm], (pre, post),
                     nd_shape=(pre, 1, post), fn='reciprocal')
                bc = name + '/bcast'
                grid = np.arange(pre * post).reshape(pre, 1, post)
                idx = np.ascontiguousarray(
                    np.broadcast_to(grid, (pre, k, post))).reshape(-1)
                emit(bc, 'linmap', [rc], _drop_batch(ish), nd_shape=ish,
                     lm=lm.Select(idx, pre * post))
                emit(name, 'mul', [e, bc], out_shape, nd_shape=ish)

        elif t in ('Gemm', 'MatMul'):
            W, b = node.params['W'], node.params.get('b')
            if W.ndim == 1:
                # (..., K) @ (K,): per-row dot along the last axis
                ishd = v1shape(node.inputs[0])
                if not ishd or ishd[-1] != W.shape[0]:
                    raise NotImplementedError(
                        f'{name}: 1-D MatMul W{W.shape} vs input {ishd}')
                rows = _flat(ishd) // W.shape[0]
                W = np.kron(np.eye(rows, dtype=W.dtype), W[np.newaxis, :])
                if b is not None:
                    b = np.asarray(b).reshape(-1)
                    if b.size == rows:
                        pass
                    elif not b.any():          # stray zero bias (LUT rewrite)
                        b = None
                    else:
                        raise NotImplementedError(
                            f'{name}: 1-D MatMul bias size {b.size} != '
                            f'{rows} rows and nonzero')
            if W.ndim != 2:
                raise NotImplementedError(
                    f'{name}: Gemm with W.ndim={W.ndim} not yet in IR')
            ish = in_shape_of(node)
            if _flat(ish) == W.shape[1]:
                emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                     lm=lm.Dense(W, b))
            elif ish and ish[-1] == W.shape[1]:
                # ND matmul (..., K) @ (K, M): block-diagonal dense over rows.
                rows = _flat(ish) // ish[-1]
                Wb = np.zeros((rows * W.shape[0], rows * W.shape[1]), dtype=W.dtype)
                for r in range(rows):
                    Wb[r*W.shape[0]:(r+1)*W.shape[0],
                       r*W.shape[1]:(r+1)*W.shape[1]] = W
                bb = None if b is None else np.tile(np.asarray(b).reshape(-1), rows)
                emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                     lm=lm.Dense(Wb, bb))
            else:
                raise NotImplementedError(
                    f'{name}: MatMul input shape {ish} vs W {W.shape}')

        elif t == 'Conv':
            kernel = node.params['kernel']
            stride = tuple(node.params['stride'])
            padding = tuple(node.params['padding'])
            groups = int(node.params.get('groups', 1) or 1)
            ish = in_shape_of(node)
            osh = out_shape
            if kernel.ndim == 3:                       # 1D conv -> lift to 2D
                kernel = kernel[:, :, np.newaxis, :]
                stride, padding = (1, stride[0]), (0, padding[0])
                ish = (ish[-2], 1, ish[-1]) if len(ish) >= 2 else (1, 1, ish[-1])
                osh3 = (_drop_batch(node.output_shape))
                osh = (osh3[-2], 1, osh3[-1])
            elif len(ish) >= 3:
                ish = ish[-3:]
                osh = osh[-3:]
            else:
                # a flattening passthrough erased the spatial shape upstream
                # (collins fc-as-conv); recover it from kernel + output shape
                osh = osh[-3:] if len(osh) >= 3 else (out_shape[-1], 1, 1)
                C_in = kernel.shape[1] * groups
                H_in = (osh[1] - 1) * stride[0] + kernel.shape[2] - 2 * padding[0]
                W_in = (osh[2] - 1) * stride[1] + kernel.shape[3] - 2 * padding[1]
                if C_in * H_in * W_in != _flat(ish):
                    raise NotImplementedError(
                        f'{name}: cannot recover conv input shape from '
                        f'{ish} with kernel {kernel.shape}')
                ish = (C_in, H_in, W_in)
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.Conv2d(kernel, node.params.get('bias'), ish, osh,
                              stride, padding, groups))

        elif t == 'ConvTranspose':
            kernel = node.params['kernel']
            ish = in_shape_of(node)[-3:]
            osh = out_shape[-3:]
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.ConvT2d(kernel, node.params.get('bias'), ish, osh,
                               node.params['stride'], node.params['padding'],
                               node.params.get('output_padding', (0, 0)),
                               int(node.params.get('groups', 1) or 1)))

        elif t == 'BatchNormalization':
            # inference BN is per-channel affine: a=(scale/sqrt(var+eps)),
            # b=(bias - a*mean), broadcast over the trailing spatial dims
            p = node.params
            a_c = (np.asarray(p['scale'], dtype=dtype)
                   / np.sqrt(np.asarray(p['var'], dtype=np.float64)
                             + float(p.get('epsilon', 1e-5))).astype(dtype))
            b_c = np.asarray(p['bias'], dtype=dtype) - a_c * np.asarray(
                p['mean'], dtype=dtype)
            ish = in_shape_of(node)
            C = a_c.shape[0]
            ax = next((i for i, d in enumerate(ish) if d == C), 0)
            bshape = tuple(C if i == ax else 1 for i in range(len(ish)))
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.ScaleShift(_broadcast_flat(a_c.reshape(bshape), ish, dtype),
                                  _broadcast_flat(b_c.reshape(bshape), ish, dtype),
                                  n_out))

        elif t == 'Pad':
            pads = node.params.get('pads')
            if pads is None:
                raise NotImplementedError(f'{name}: dynamic pads')
            val = float(node.params.get('constant_value', 0.0) or 0.0)
            ish = v1shape(node.inputs[0])
            k = len(pads) // 2
            if k != len(ish):
                raise NotImplementedError(
                    f'{name}: pads rank {k} vs input rank {len(ish)}')
            osh_full = tuple(int(d) + int(pads[i]) + int(pads[i + k])
                             for i, d in enumerate(ish))
            grid = np.arange(int(np.prod(osh_full))).reshape(osh_full)
            sl = tuple(slice(int(pads[i]), int(pads[i]) + int(d))
                       for i, d in enumerate(ish))
            pos = grid[sl].reshape(-1)
            base = np.full(int(np.prod(osh_full)), val, dtype=dtype)
            emit(name, 'concat', [src(node.inputs[0])], out_shape,
                 nd_shape=osh_full,
                 params={'positions': [pos], 'base': base,
                         'n_out': int(np.prod(osh_full))})

        elif t == 'AveragePool':
            ish = in_shape_of(node)[-3:]
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.AvgPool(ish, out_shape[-3:], node.params['kernel_shape'],
                               node.params['stride'], node.params['padding']))

        elif t == 'MaxPool':
            ish = in_shape_of(node)[-3:]
            emit(name, 'maxpool', [src(node.inputs[0])], out_shape,
                 params={'in_shape': ish,
                         'kernel_shape': tuple(node.params['kernel_shape']),
                         'stride': tuple(node.params['stride']),
                         'padding': tuple(node.params['padding'])})

        elif t == 'Add':
            two_live = (len(node.inputs) == 2
                        and (node.inputs[1] in cg.nodes
                             or node.inputs[1] == cg.input_name))
            if two_live:
                na, nb, full = broadcast_pair(node)
                emit(name, 'add', [na, nb], full)
            else:
                b = _broadcast_flat(node.params.get('bias', 0), out_shape, dtype)
                inp, _n = broadcast_in(node, out_shape)
                emit(name, 'linmap', [inp], out_shape,
                     lm=lm.ScaleShift(None, b, n_out))

        elif t == 'Sub':
            if len(node.inputs) == 2 and node.inputs[1] in cg.nodes:
                # a - b = a + (-1 * b): broadcast both, negate b, add
                na, nb_name, full = broadcast_pair(node)
                nb = _flat(full)
                neg = name + '/neg'
                emit(neg, 'linmap', [nb_name], full,
                     lm=lm.ScaleShift(-np.ones(nb, dtype=dtype), None, nb))
                emit(name, 'add', [na, neg], full)
            elif node.params.get('negate'):
                a = -np.ones(n_out, dtype=dtype)
                b = _broadcast_flat(node.params.get('bias', 0), out_shape, dtype)
                inp, _n = broadcast_in(node, out_shape)
                emit(name, 'linmap', [inp], out_shape,
                     lm=lm.ScaleShift(a, b, n_out))
            else:
                b = -_broadcast_flat(node.params.get('sub_val', 0), out_shape, dtype)
                inp, _n = broadcast_in(node, out_shape)
                emit(name, 'linmap', [inp], out_shape,
                     lm=lm.ScaleShift(None, b, n_out))

        elif t in ('Min', 'Max'):
            # EXACT Min/Max -> ReLU+affine (only Sub/Relu/Add, all sound):
            #   max(x,c) = c + ReLU(x - c)     min(x,c) = c - ReLU(c - x)
            #   max(a,b) = a + ReLU(b - a)     min(a,b) = a - ReLU(a - b)
            # Needed by the monotonic_acasxu clamp Min(Max(v,LO),HI). Native
            # here so the IR owns it (no v1 onnx_optimizer.min_max_to_relu).
            is_max = t == 'Max'
            neg = lambda k: lm.ScaleShift(-np.ones(k, dtype=dtype), None, k)
            sub_nm, rel_nm = name + '/mm_sub', name + '/mm_relu'
            ckey = next((k for k in node.params
                         if k.startswith('const_')), None)
            if ckey is not None:
                c = _broadcast_flat(node.params[ckey], out_shape, dtype)
                inp, _n = broadcast_in(node, out_shape)
                if is_max:                          # c + ReLU(x - c)
                    emit(sub_nm, 'linmap', [inp], out_shape,
                         lm=lm.ScaleShift(None, -c, n_out))
                    emit(rel_nm, 'nonlin', [sub_nm], out_shape, fn='relu')
                    emit(name, 'linmap', [rel_nm], out_shape,
                         lm=lm.ScaleShift(None, c, n_out))
                else:                               # c - ReLU(c - x)
                    emit(sub_nm, 'linmap', [inp], out_shape,
                         lm=lm.ScaleShift(-np.ones(n_out, dtype=dtype), c,
                                          n_out))
                    emit(rel_nm, 'nonlin', [sub_nm], out_shape, fn='relu')
                    emit(name, 'linmap', [rel_nm], out_shape,
                         lm=lm.ScaleShift(-np.ones(n_out, dtype=dtype), c,
                                          n_out))
            else:
                na, nb, full = broadcast_pair(node)
                nf = _flat(full)
                ng = name + '/mm_neg'
                if is_max:                          # a + ReLU(b - a)
                    emit(ng, 'linmap', [na], full, lm=neg(nf))
                    emit(sub_nm, 'add', [nb, ng], full)
                    emit(rel_nm, 'nonlin', [sub_nm], full, fn='relu')
                    emit(name, 'add', [na, rel_nm], full)
                else:                               # a - ReLU(a - b)
                    emit(ng, 'linmap', [nb], full, lm=neg(nf))
                    emit(sub_nm, 'add', [na, ng], full)
                    emit(rel_nm, 'nonlin', [sub_nm], full, fn='relu')
                    ngr = name + '/mm_negr'
                    emit(ngr, 'linmap', [rel_nm], full, lm=neg(nf))
                    emit(name, 'add', [na, ngr], full)

        elif t in ('Mul', 'Div'):
            if 'scale' in node.params:      # Div's scale is pre-inverted by v1
                a = _broadcast_flat(node.params['scale'], out_shape, dtype)
                inp, _n = broadcast_in(node, out_shape)
                emit(name, 'linmap', [inp], out_shape,
                     lm=lm.ScaleShift(a, None, n_out))
            elif len(node.inputs) == 2:
                # bilinear: Mul is native; Div lowers to mul(a, recip(b))
                if t == 'Div':
                    bshape = v1shape(node.inputs[1])
                    rname = name + '/recip'
                    emit(rname, 'nonlin', [src(node.inputs[1])],
                         _drop_batch(bshape), nd_shape=bshape,
                         fn='reciprocal')
                    import types
                    node = types.SimpleNamespace(
                        name=node.name, inputs=[node.inputs[0], rname])
                na, nb, full = broadcast_pair(node)
                emit(name, 'mul', [na, nb], full)
            else:
                raise NotImplementedError(
                    f'{name}: bilinear {t} not yet in IR (M5+)')

        elif t == 'Neg':
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.ScaleShift(-np.ones(n_out, dtype=dtype), None, n_out))

        elif t == 'Transpose':
            ish = v1shape(node.inputs[0])
            perm = node.params.get('perm') or list(range(len(ish) - 1, -1, -1))
            if len(perm) != len(ish):
                # v1 dropped/added a batch dim relative to the ONNX rank
                if len(perm) == len(ish) + 1 and perm[0] == 0:
                    perm = [p - 1 for p in perm[1:]]
                elif len(perm) == len(ish) - 1:
                    ish = ish[1:] if ish[0] == 1 else ish
                if len(perm) != len(ish):
                    raise NotImplementedError(
                        f'{name}: perm {perm} vs shape {ish}')
            tgrid = np.transpose(
                np.arange(_flat(ish)).reshape(ish), perm)
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 nd_shape=tgrid.shape, lm=lm.Select(tgrid.reshape(-1), _flat(ish)))

        elif t == 'Slice':
            ish = v1shape(node.inputs[0])
            grid = np.arange(_flat(ish)).reshape(ish)
            sl = [slice(None)] * len(ish)
            axes = node.params.get('axes', [0])
            starts = node.params.get('starts', [0])
            ends = node.params.get('ends', [None])
            steps = node.params.get('steps', [1] * len(axes))
            for ax, s, e, st in zip(axes, starts, ends, steps):
                a = ax if ax >= 0 else len(ish) + ax
                if a >= len(ish):
                    raise NotImplementedError(
                        f'{name}: slice axis {ax} out of rank {len(ish)}')
                dim = ish[a]
                s = dim + s if s is not None and s < 0 else (s or 0)
                e = dim if e is None or e > dim else (dim + e if e < 0 else e)
                sl[a] = slice(int(s), int(e), int(st or 1))
            sgrid = grid[tuple(sl)]
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 nd_shape=sgrid.shape,
                 lm=lm.Select(sgrid.reshape(-1), _flat(ish)))

        elif t == 'Concat':
            _emit_concat(cg, node, name, out_shape, v1shape, src, emit, dtype)

        elif t == 'Split' or t == 'SplitOutput':
            _emit_split(cg, node, name, t, out_shape, v1shape, src, emit, alias)

        elif t == 'Gather':
            ish = v1shape(node.inputs[0])
            axis = node.params.get('axis', 0)
            if node.params.get('indices') is None:
                raise NotImplementedError(f'{name}: dynamic Gather indices')
            indices = np.asarray(node.params['indices'])
            if not np.issubdtype(indices.dtype, np.integer):
                # some producers store gather indices as float consts
                if not np.all(indices == np.round(indices)):
                    raise NotImplementedError(
                        f'{name}: non-integral Gather indices')
                indices = indices.astype(np.int64)
            grid = np.arange(_flat(ish)).reshape(ish)
            ggrid = np.take(grid, indices, axis=axis if axis >= 0
                            else len(ish) + axis)
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 nd_shape=ggrid.shape,
                 lm=lm.Select(ggrid.reshape(-1), _flat(ish)))

        elif t in ('Resize', 'Upsample'):
            # nearest-neighbor resize: pure index map (each output element
            # reads floor(coord/scale) of the input). YOLO upsamples 2x.
            scales = node.params.get('scales')
            if scales is None:
                raise NotImplementedError(f'{name}: Resize without scales')
            ish = v1shape(node.inputs[0])
            if len(scales) != len(ish):
                raise NotImplementedError(
                    f'{name}: scales rank {len(scales)} vs shape {ish}')
            osh_full = tuple(int(d * s) for d, s in zip(ish, scales))
            grids = np.meshgrid(*[np.minimum((np.arange(o) / s).astype(np.int64),
                                             d - 1)
                                  for o, s, d in zip(osh_full, scales, ish)],
                                indexing='ij')
            flat_in = np.ravel_multi_index(tuple(grids), ish).reshape(-1)
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 nd_shape=osh_full, lm=lm.Select(flat_in, _flat(ish)))

        elif t in ('ReduceSum', 'ReduceMean'):
            ish = v1shape(node.inputs[0])
            axes = node.params.get('axes')
            if axes is None:
                pre, k, post = 1, _flat(ish), 1        # reduce all
            elif len(axes) == 1:
                a = axes[0] if axes[0] >= 0 else len(ish) + axes[0]
                pre = _flat(ish[:a])
                k = ish[a]
                post = _flat(ish[a + 1:])
            else:
                raise NotImplementedError(f'{name}: multi-axis {t} {axes}')
            emit(name, 'linmap', [src(node.inputs[0])], out_shape,
                 lm=lm.SumAxis(pre, k, post, mean=(t == 'ReduceMean')))

        elif t == 'Pow':
            exp = node.params.get('exponent')
            if exp is None:
                raise NotImplementedError(f'{name}: variable-exponent Pow')
            emit(name, 'nonlin', [src(node.inputs[0])], out_shape,
                 fn='pow', params={'exponent': float(exp)})

        else:
            raise NotImplementedError(
                f"op '{t}' (node '{name}') not yet in the vibecheck2 IR")

    out = src(cg.output_name)
    net = Net(ops, order, cg.input_name, out,
              onnx_path=getattr(cg, 'onnx_path', None))
    _validate_sizes(net)
    return net


def _validate_sizes(net):
    """Static whole-net size check: every edge consumed at the size its
    producer emits. Catches v1-metadata inconsistencies at load, loudly,
    instead of as silent index corruption at propagation time."""
    for name in net.order:
        op = net.ops[name]
        ins = [net.ops[i].n for i in op.inputs]
        if op.kind == 'linmap':
            if ins[0] != op.lm.n_in or op.lm.n_out != op.n:
                raise ValueError(
                    f'{name}: linmap sizes {op.lm.n_in}->{op.lm.n_out} vs '
                    f'edge sizes in={ins[0]} out={op.n}')
        elif op.kind in ('add', 'mul'):
            if ins[0] != ins[1] or ins[0] != op.n:
                raise ValueError(f'{name}: {op.kind} sizes {ins} -> {op.n}')
        elif op.kind == 'bmm':
            fa = _flat(op.params['a_shape'])
            fb = _flat(op.params['b_shape'])
            if ins[0] != fa or ins[1] != fb:
                raise ValueError(f'{name}: bmm sizes {ins} vs {fa},{fb}')
        elif op.kind == 'concat':
            for src_n, pos in zip(ins, op.params['positions']):
                if src_n != len(pos):
                    raise ValueError(
                        f'{name}: concat part size {src_n} vs {len(pos)} slots')
            if op.params['n_out'] != op.n:
                raise ValueError(
                    f'{name}: concat n_out {op.params["n_out"]} vs edge {op.n}')


def _emit_concat(cg, node, name, out_shape, v1shape, src, emit, dtype):
    """Concat -> const base vector + per-live-part scatter indices.

    All index math on the RAW v1 shapes so the ONNX axis param lines up;
    const parts are padded with leading 1-dims to the live rank.
    """
    ish = [v1shape(i) for i in node.inputs]
    rank = len(ish[0]) if ish else 1
    axis = node.params.get('axis', 0)
    consts = {int(p): np.asarray(a, dtype=dtype)
              for p, a in (node.params.get('const_inputs') or [])}
    a = axis if axis >= 0 else rank + axis
    if not (0 <= a < rank) or any(len(s) != rank for s in ish):
        raise NotImplementedError(
            f'{name}: concat axis {axis} over shapes {ish}')
    n_positions = len(node.inputs) + len(consts)
    live_iter = iter(zip(node.inputs, ish))
    chunks = []
    for p in range(n_positions):
        if p in consts:
            arr = consts[p]
            csh = (1,) * (rank - arr.ndim) + tuple(arr.shape)
            chunks.append(('const', csh, arr.reshape(csh)))
        else:
            nm, sh = next(live_iter)
            chunks.append(('live', sh, nm))
    o = list(chunks[0][1])
    o[a] = sum(sh[a] for _, sh, _ in chunks)
    n_out = _flat(o)
    grid = np.arange(n_out).reshape(o)
    base = np.zeros(n_out, dtype=dtype)
    live_inputs, live_pos = [], []
    off = 0
    for kind, sh, payload in chunks:
        sl = [slice(None)] * len(o)
        sl[a] = slice(off, off + sh[a])
        pos = grid[tuple(sl)].reshape(-1)
        if kind == 'const':
            base[pos] = payload.reshape(-1)
        else:
            live_inputs.append(src(payload))
            live_pos.append(pos)
        off += sh[a]
    # the COMPUTED layout `o` is the op's shape: v1's declared concat shape
    # sums the (sometimes stale) declared part dims and can be plain wrong
    # (ml4acopf 194: declared 160, ORT says 186)
    emit(name, 'concat', live_inputs, o, nd_shape=o,
         params={'positions': live_pos, 'base': base, 'n_out': n_out})


def _emit_split(cg, node, name, t, out_shape, v1shape, src, emit, alias):
    """Split/SplitOutput -> Select gathers on the split input."""
    if t == 'SplitOutput':
        split_node = cg.nodes[node.inputs[0]]
        index = node.params['index']
        parent_in = split_node.inputs[0]
    else:
        split_node, index, parent_in = node, 0, node.inputs[0]
    ish = v1shape(parent_in)
    sizes = split_node.params.get('split')
    axis = split_node.params.get('axis', 0)
    a = axis if axis >= 0 else len(ish) + axis
    if sizes is None:
        alias[name] = src(parent_in)
        return
    if not (0 <= a < len(ish)):
        raise NotImplementedError(f'{name}: split axis {axis} on {ish}')
    grid = np.arange(_flat(ish)).reshape(ish)
    off = sum(sizes[:index])
    sl = [slice(None)] * len(ish)
    sl[a] = slice(off, off + sizes[index])
    sgrid = grid[tuple(sl)]
    emit(name, 'linmap', [src(parent_in)], out_shape, nd_shape=sgrid.shape,
         lm=lm.Select(sgrid.reshape(-1), _flat(ish)))


def load(onnx_path, dtype=np.float32) -> Net:
    """ONNX file -> Net via the v1 front end (folding + optimizer reuse),
    with onnx.shape_inference as the ND-shape oracle. The v1 EXACT rewrites
    run first (identity-pad drop, MaxPool -> ReLU decomposition, Min/Max ->
    ReLU+affine); all are semantics-preserving, so the parity gates against
    ORT still bind."""
    from ..frontend.network import ComputeGraph
    cg = ComputeGraph.from_onnx(onnx_path, dtype=dtype)
    # graph-opt is now fully vc2-native: from_compute_graph handles Pad (any
    # constant pad), Min/Max, and MaxPool directly on the IR, and the passes
    # below fold split-relus + decompose maxpool. No v1 onnx_optimizer.
    net = from_compute_graph(cg, true_shapes=_onnx_true_shapes(onnx_path))
    from .graph_opt import fold_split_relu
    net = fold_split_relu(net)          # undo relu-split hardening (exact)
    # MaxPool is emitted as a native maxpool op (from_compute_graph) and
    # decomposed here into the exact pairwise relu tree -- one vc2-native path
    # for padded AND unpadded pools (no ONNX-level maxpool_to_relu needed).
    net = decompose_maxpool(net)
    net = tag_simplex_bmm(net)
    net.onnx_path = onnx_path
    return net
