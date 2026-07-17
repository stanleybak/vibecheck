"""Semantics-preserving IR rewrites, applied after ONNX -> Net conversion.

These are vc2-native passes on the flat Net IR (no dependency on the v1 front
end's graph optimizer). Each is EXACT -- it changes the op graph, never the
function it computes -- so it can run unconditionally at load and stays sound.

- `fold_split_relu`: undoes the relu-split hardening (relusplitter benchmark).
  A splitter wraps a neuron z -> ReLU(z) in an expanded pair
  `Gemm(C -> C+S) -> ReLU -> Gemm(C+S -> C) -> ReLU`, where the merge Gemm is a
  +-1 selector recombining paired rows (w, -w)/(b, -b) via ReLU(u)-ReLU(-u)=u.
  We detect the pattern on the IR and fold it back to a single
  `linmap -> ReLU`, recovering the original (small) net -- so the inflated
  unstable-relu count that made BaB explode simply disappears. No-op on any
  net that was not split.
"""
from __future__ import annotations

import numpy as np
import torch

from .linmap import Dense


def _dense_of(lm, n_in):
    """(W, b) float64 of a small LinMap via identity probes (any layout)."""
    eye = torch.eye(n_in, dtype=torch.float64)
    W = np.asarray(lm.lin(eye).T.numpy(), dtype=np.float64)
    b = np.asarray(lm.bias_vec(torch.zeros(1, n_in, dtype=torch.float64))
                   .numpy(), dtype=np.float64).reshape(-1)
    return W, b


def fuse_affine(net, max_n=4096, max_elems=4_000_000):
    """Compose single-consumer linmap->linmap chains into ONE Dense map.

    Per-pass op-dispatch cost dominates tiny wide-route nets: lsnc's 51-op
    quadrotor graph pays ~51 Python ops x 4-5 passes per BaB round, and
    the decomposed BaB runs 640k domains/s where abcrown (fused affine
    layers) does 1.2M/s on the IDENTICAL 13.4M-domain tree. Composition
    happens in float64 and only when both maps materialize small dense
    matrices -- conv-scale nets are untouched (densifying a conv is a
    memory disaster). Exact up to fp reassociation (the same class of
    deviation as the front end's existing folds; every sat still
    validates against the real onnx through the ORT chokepoint)."""
    def _small(nm_in, *ops_):
        n0 = net.ops[nm_in].n
        if n0 > max_n:
            return False
        for o in ops_:
            if o.n > max_n or n0 * o.n > max_elems:
                return False
        return True

    def _sole_linmaps_of_same_source(op, consumers):
        """All inputs single-consumer linmaps of ONE shared source (and
        none of them the output edge) -> that source, else None."""
        srcs = set()
        for e in op.inputs:
            u = net.ops.get(e)
            if (u is None or u.kind != 'linmap'
                    or consumers.get(e, 0) != 1 or e == net.output_name):
                return None
            srcs.add(u.inputs[0])
        return srcs.pop() if len(srcs) == 1 else None

    changed = True
    while changed:
        changed = False
        consumers = {}
        for nm2 in net.order:
            for e in net.ops[nm2].inputs:
                consumers[e] = consumers.get(e, 0) + 1
        for nm2 in list(net.order):
            op = net.ops.get(nm2)
            if op is None:
                continue
            if op.kind == 'linmap':
                up = net.ops.get(op.inputs[0])
                if (up is None or up.kind != 'linmap'
                        or consumers.get(up.name, 0) != 1
                        or up.name == net.output_name
                        or not _small(up.inputs[0], up, op)):
                    continue
                n0 = net.ops[up.inputs[0]].n
                W1, b1 = _dense_of(up.lm, n0)
                W2, b2 = _dense_of(op.lm, up.n)
                op.lm = Dense((W2 @ W1).astype(np.float32),
                              (W2 @ b1 + b2).astype(np.float32))
                op.inputs = (up.inputs[0],)
                del net.ops[up.name]
                net.order.remove(up.name)
                changed = True
            elif op.kind in ('add', 'concat'):
                # add/concat of single-consumer linmaps sharing ONE source
                # becomes a single linmap (sum of weights / stacked rows);
                # this is what breaks lsnc's chains apart (the fused net
                # then collapses further under the linmap-linmap rule)
                src = _sole_linmaps_of_same_source(op, consumers)
                if src is None:
                    continue
                ups = [net.ops[e] for e in op.inputs]
                if not _small(src, op, *ups):
                    continue
                n0 = net.ops[src].n
                mats = [_dense_of(u.lm, n0) for u in ups]
                if op.kind == 'add':
                    Wf = sum(m[0] for m in mats)
                    bf = sum(m[1] for m in mats)
                else:
                    base = np.asarray(op.params['base'],
                                      dtype=np.float64).reshape(-1)
                    Wf = np.zeros((op.n, n0))
                    bf = base.copy()
                    for u, (Wu, bu), pos in zip(
                            ups, mats, op.params['positions']):
                        p = np.asarray(pos).reshape(-1)
                        Wf[p] = Wu
                        bf[p] = bu
                for u in ups:
                    del net.ops[u.name]
                    net.order.remove(u.name)
                net.ops[nm2] = type(op)(
                    op.name, 'linmap', (src,), op.shape, op.n,
                    lm=Dense(Wf.astype(np.float32), bf.astype(np.float32)))
                changed = True
    return net


def _wb(dense):
    """(W, b) of a Dense as float64 arrays; b defaults to zeros."""
    W = np.asarray(dense.W, dtype=np.float64)
    b = (np.zeros(W.shape[0], dtype=np.float64) if dense.b is None
         else np.asarray(dense.b, dtype=np.float64))
    return W, b


def _fold_pair(dense1, dense2, tol=1e-5):
    """Recover (W_orig, b_orig) for `dense1 -> ReLU -> dense2 -> ReLU` if
    `dense2` is a sound +-1 split-selector over `dense1`'s rows, else None.

    Each output row j of W2 is either a PASSTHROUGH (one +1, zero merge bias
    -> otherwise ReLU(ReLU(z)+b) != ReLU(z+b)) selecting an unsplit neuron, or
    a SPLIT PAIR (+1 / -1 over rows (w,-w) with biases (b,-b)) recombining to
    z via ReLU(u)-ReLU(-u)=u; the merge bias then folds into the ReLU bias."""
    W1, b1 = _wb(dense1)
    W2, b2 = _wb(dense2)
    C, M = W2.shape
    if W1.shape[0] != M or M < C:
        return None
    W_orig = np.zeros((C, W1.shape[1]), dtype=np.float64)
    b_orig = np.zeros(C, dtype=np.float64)
    for j in range(C):
        nz = np.where(np.abs(W2[j]) > tol)[0]
        if len(nz) == 1:
            i = nz[0]
            if not (abs(W2[j, i] - 1.0) < tol and abs(b2[j]) < tol):
                return None
            W_orig[j] = W1[i]
            b_orig[j] = b1[i]
        elif len(nz) == 2:
            ip, ineg = nz
            vp, vn = W2[j, ip], W2[j, ineg]
            if not (abs(abs(vp) - 1.0) < tol and abs(abs(vn) - 1.0) < tol
                    and vp * vn < 0):
                return None
            if not np.allclose(W1[ip], -W1[ineg], atol=tol):
                return None
            if abs(b1[ip] + b1[ineg]) >= tol:
                return None
            keep = ip if vp > 0 else ineg
            W_orig[j] = W1[keep]
            b_orig[j] = b1[keep] + b2[j]
        else:
            return None
    return W_orig, b_orig


def fold_split_relu(net):
    """Fold every `linmap -> ReLU -> linmap -> ReLU` split pattern (Dense
    linmaps, each intermediate single-consumer) back to `linmap -> ReLU`.
    Mutates and returns `net`. Iterates to a fixed point (a fold can expose
    another)."""
    while _fold_once(net):
        pass
    return net


def _fold_once(net):
    cons = net.consumers()

    def only(edge):
        c = cons.get(edge, ())
        return net.ops[c[0]] if len(c) == 1 else None

    for name in net.order:
        op1 = net.ops[name]
        if op1.kind != 'linmap' or not isinstance(op1.lm, Dense):
            continue
        op2 = only(name)
        if op2 is None or op2.kind != 'nonlin' or op2.fn != 'relu':
            continue
        op3 = only(op2.name)
        if op3 is None or op3.kind != 'linmap' \
                or not isinstance(op3.lm, Dense):
            continue
        op4 = only(op3.name)
        if op4 is None or op4.kind != 'nonlin' or op4.fn != 'relu':
            continue
        folded = _fold_pair(op1.lm, op3.lm)
        if folded is None:
            continue
        W_orig, b_orig = folded
        dt = np.asarray(op1.lm.W).dtype
        # fold: op1 becomes the recovered linmap, op2 its ReLU; op3/op4 vanish
        op1.lm = Dense(W_orig.astype(dt), b_orig.astype(dt))
        op1.n = op2.n = int(W_orig.shape[0])
        op1.shape = op2.shape = op3.shape
        # op4's edge is now produced by op2 (op1's ReLU): rewire consumers
        for other in net.ops.values():
            if op4.name in other.inputs:
                other.inputs = tuple(op2.name if i == op4.name else i
                                     for i in other.inputs)
        if net.output_name == op4.name:
            net.output_name = op2.name
        del net.ops[op3.name]
        del net.ops[op4.name]
        net.order = [o for o in net.order if o not in (op3.name, op4.name)]
        return True
    return False
