"""Fast equivalence tests for the vc2-native IR rewrites (graph_opt).

All synthetic, forward-only, sub-second: build a tiny net, apply the pass,
assert the rewritten net computes the SAME function (point-eval) -- the
soundness bar for an "exact" rewrite -- and that no-op nets pass through.
No benchmark files, no GPU.
"""
import numpy as np
import torch

from vibecheck.core import forward as fwd
from vibecheck.core.graph import Net, Op
from vibecheck.core.graph_opt import fold_split_relu
from vibecheck.core.linmap import Dense

RNG = np.random.default_rng(0)


def _dense(op_out, op_in):
    return Dense(RNG.standard_normal((op_out, op_in)).astype(np.float32),
                 RNG.standard_normal(op_out).astype(np.float32))


def _mlp(widths):
    """linmap->relu stack over widths[0]->...->widths[-1] (no final relu)."""
    ops = {'x': Op('x', 'input', (), (widths[0],), widths[0])}
    order = []
    prev = 'x'
    for k in range(1, len(widths)):
        lm = _dense(widths[k], widths[k - 1])
        h = f'h{k}'
        ops[h] = Op(h, 'linmap', (prev,), (widths[k],), widths[k], lm=lm)
        order.append(h)
        prev = h
        if k < len(widths) - 1:
            r = f'r{k}'
            ops[r] = Op(r, 'nonlin', (h,), (widths[k],), widths[k], fn='relu')
            order.append(r)
            prev = r
    return Net(ops, order, 'x', prev), ops


def _split_layer(W, b):
    """Return (W1,b1,W2,b2): the relu-split expansion of one ReLU layer with
    pre-activation Wx+b, splitting rows into (+row, -row) pairs recombined by
    a +-1 selector. Exact: ReLU folds to ReLU(Wx+b) either way."""
    C = W.shape[0]
    split = C // 2                      # split the first `split` neurons
    W1_rows, b1_rows, sel = [], [], []
    col = 0
    for j in range(C):
        if j < split:                   # paired (+w,-w)/(+b,-b)
            W1_rows += [W[j], -W[j]]
            b1_rows += [b[j], -b[j]]
            sel.append((col, +1.0, col + 1, -1.0))
            col += 2
        else:                           # passthrough
            W1_rows.append(W[j])
            b1_rows.append(b[j])
            sel.append((col, +1.0, None, None))
            col += 1
    M = col
    W1 = np.stack(W1_rows).astype(np.float32)
    b1 = np.array(b1_rows, dtype=np.float32)
    W2 = np.zeros((C, M), dtype=np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    for j, (cp, vp, cn, vn) in enumerate(sel):
        W2[j, cp] = vp
        if cn is not None:
            W2[j, cn] = vn
    return W1, b1, W2, b2


def test_fold_split_relu_is_exact():
    # original 4->6->6->3 MLP
    orig, ops = _mlp([4, 6, 6, 3])
    # build a SPLIT version: expand the FIRST hidden layer (h1)
    W, b = ops['h1'].lm.W, ops['h1'].lm.b
    W1, b1, W2, b2 = _split_layer(W, b)
    sops = {
        'x': Op('x', 'input', (), (4,), 4),
        'e': Op('e', 'linmap', ('x',), (W1.shape[0],), W1.shape[0],
                lm=Dense(W1, b1)),
        're': Op('re', 'nonlin', ('e',), (W1.shape[0],), W1.shape[0],
                 fn='relu'),
        'm': Op('m', 'linmap', ('re',), (6,), 6, lm=Dense(W2, b2)),
        'rm': Op('rm', 'nonlin', ('m',), (6,), 6, fn='relu'),
        'h2': Op('h2', 'linmap', ('rm',), (6,), 6, lm=ops['h2'].lm),
        'r2': Op('r2', 'nonlin', ('h2',), (6,), 6, fn='relu'),
        'h3': Op('h3', 'linmap', ('r2',), (3,), 3, lm=ops['h3'].lm),
    }
    split = Net(sops, ['e', 're', 'm', 'rm', 'h2', 'r2', 'h3'], 'x', 'h3')
    x = torch.tensor(RNG.uniform(-2, 2, (32, 4)), dtype=torch.float32)
    y_split = fwd.point(split, x)
    y_orig = fwd.point(orig, x)
    assert torch.allclose(y_split, y_orig, atol=1e-4), "split net != original"

    n_relu_before = sum(1 for o in split.ops.values()
                        if o.kind == 'nonlin' and o.fn == 'relu')
    folded = fold_split_relu(split)
    n_relu_after = sum(1 for o in folded.ops.values()
                       if o.kind == 'nonlin' and o.fn == 'relu')
    assert n_relu_after == n_relu_before - 1, "did not fold the split layer"
    y_folded = fwd.point(folded, x)
    assert torch.allclose(y_folded, y_orig, atol=1e-4), \
        "folded net != original function"


def test_fold_is_noop_on_plain_mlp():
    net, _ = _mlp([5, 7, 4])
    x = torch.tensor(RNG.uniform(-1, 1, (16, 5)), dtype=torch.float32)
    y0 = fwd.point(net, x)
    n0 = len(net.order)
    fold_split_relu(net)
    assert len(net.order) == n0, "folded a net with no split pattern"
    assert torch.allclose(fwd.point(net, x), y0, atol=1e-5)


def test_fold_does_not_touch_non_selector_pair():
    # two stacked real layers (the second is NOT a +-1 selector) must survive
    net, _ = _mlp([4, 5, 5, 2])
    x = torch.tensor(RNG.uniform(-1, 1, (16, 4)), dtype=torch.float32)
    y0 = fwd.point(net, x)
    n0 = len(net.order)
    fold_split_relu(net)
    assert len(net.order) == n0
    assert torch.allclose(fwd.point(net, x), y0, atol=1e-5)


def test_decompose_maxpool_is_exact():
    """maxpool op -> pairwise relu tree computes the identical max (exact)."""
    from vibecheck.core.graph import decompose_maxpool
    C, H, W = 2, 6, 6
    n_in = C * H * W
    ops = {
        'x': Op('x', 'input', (), (C, H, W), n_in),
        'p': Op('p', 'maxpool', ('x',), (C, 3, 3), C * 3 * 3,
                params={'in_shape': (C, H, W), 'kernel_shape': (2, 2),
                        'stride': (2, 2), 'padding': (0, 0)}),
    }
    net = Net(ops, ['p'], 'x', 'p')
    x = torch.tensor(RNG.uniform(-3, 3, (8, n_in)), dtype=torch.float32)
    y_pool = fwd.point(net, x)
    net2 = decompose_maxpool(net)
    assert not any(o.kind == 'maxpool' for o in net2.ops.values())
    y_relu = fwd.point(net2, x)
    assert torch.allclose(y_pool, y_relu, atol=1e-5), "maxpool decomp not exact"


def _load_from_onnx_nodes(nodes, inits, in_dims, out_name):
    """Build a tiny ONNX model in memory and load it through vc2 -> Net."""
    import tempfile, os, onnx
    from onnx import helper, TensorProto as TP
    from vibecheck.core.graph import load as load_net
    xi = helper.make_tensor_value_info('x', TP.FLOAT, [1, *in_dims])
    yo = helper.make_tensor_value_info(out_name, TP.FLOAT, None)
    g = helper.make_graph(nodes, 'g', [xi], [yo], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    fd, p = tempfile.mkstemp(suffix='.onnx')
    os.close(fd)
    onnx.save(m, p)
    try:
        return load_net(p)
    finally:
        os.remove(p)


def test_native_minmax_is_exact_clamp():
    from onnx import helper, TensorProto as TP
    inits = [helper.make_tensor('lo', TP.FLOAT, [1], [0.5]),
             helper.make_tensor('hi', TP.FLOAT, [1], [2.0])]
    nodes = [helper.make_node('Max', ['x', 'lo'], ['m']),
             helper.make_node('Min', ['m', 'hi'], ['y'])]
    net = _load_from_onnx_nodes(nodes, inits, [4], 'y')
    assert not any(o.kind == 'maxpool' for o in net.ops.values())
    x = torch.tensor(RNG.uniform(-3, 3, (16, 4)), dtype=torch.float32)
    y = fwd.point(net, x)
    assert torch.allclose(y, torch.clamp(x, 0.5, 2.0), atol=1e-5)


def test_native_pad_is_identity_on_zero_pad():
    from onnx import helper, TensorProto as TP
    pads = helper.make_tensor('pads', TP.INT64, [4], [0, 0, 0, 0])
    nodes = [helper.make_node('Pad', ['x', 'pads'], ['y'], mode='constant')]
    net = _load_from_onnx_nodes(nodes, [pads], [3], 'y')
    x = torch.tensor(RNG.uniform(-2, 2, (8, 3)), dtype=torch.float32)
    assert torch.allclose(fwd.point(net, x), x, atol=1e-6)
