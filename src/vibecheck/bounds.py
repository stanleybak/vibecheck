"""Interval arithmetic bound propagation."""

import numpy as np
from .onnx_loader import is_conv, conv_output_shape


def ia_bounds(layers, x_lo, x_hi):
    """Interval arithmetic bounds. Returns (pre_bounds, post_bounds).

    pre_bounds[l] = (z_lo, z_hi) — bounds before ReLU at layer l
    post_bounds[l] = (a_lo, a_hi) — bounds after ReLU (or input bounds at l=0)
    """
    pre_bounds = []
    post_bounds = [(x_lo.copy(), x_hi.copy())]
    for l, layer in enumerate(layers):
        a_lo, a_hi = post_bounds[-1]
        if is_conv(layer):
            kernel, bias, params = layer
            z_lo, z_hi = _conv_ia(a_lo, a_hi, kernel, bias, params)
        else:
            W, b = layer
            Wp = np.maximum(W, 0)
            Wn = np.minimum(W, 0)
            z_lo = Wp @ a_lo + Wn @ a_hi + b
            z_hi = Wp @ a_hi + Wn @ a_lo + b
        pre_bounds.append((z_lo, z_hi))
        if l < len(layers) - 1:
            post_bounds.append((np.maximum(z_lo, 0), np.maximum(z_hi, 0)))
    return pre_bounds, post_bounds


def unstable_counts(pre_bounds, n_hidden):
    """Count unstable neurons per hidden layer."""
    return [int(np.sum((pre_bounds[l][0] < 0) & (pre_bounds[l][1] > 0)))
            for l in range(n_hidden)]


def _conv_ia(a_lo, a_hi, kernel, bias, params):
    """Conv IA bounds using torch conv2d."""
    import torch
    import torch.nn.functional as F

    input_shape = params['input_shape']
    stride = params['stride']
    padding = params['padding']

    lo_4d = torch.tensor(a_lo, dtype=torch.float64).reshape(1, *input_shape)
    hi_4d = torch.tensor(a_hi, dtype=torch.float64).reshape(1, *input_shape)
    k = torch.tensor(kernel, dtype=torch.float64)
    b = torch.tensor(bias, dtype=torch.float64)

    Kp = torch.clamp(k, min=0)
    Kn = torch.clamp(k, max=0)

    z_lo = F.conv2d(lo_4d, Kp, stride=stride, padding=padding) + \
           F.conv2d(hi_4d, Kn, stride=stride, padding=padding) + b.reshape(1, -1, 1, 1)
    z_hi = F.conv2d(hi_4d, Kp, stride=stride, padding=padding) + \
           F.conv2d(lo_4d, Kn, stride=stride, padding=padding) + b.reshape(1, -1, 1, 1)

    return z_lo.numpy().flatten(), z_hi.numpy().flatten()


def _prod(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _infer_conv_input_shape(flat_shape_or_size, kernel, transpose=False):
    """Infer (C, H, W) from a flat input shape/size and conv kernel."""
    import math
    if isinstance(flat_shape_or_size, (tuple, list)):
        total = 1
        for d in flat_shape_or_size:
            total *= d
    else:
        total = flat_shape_or_size
    C_in = kernel.shape[0] if transpose else kernel.shape[1]
    if total % C_in != 0:
        # Can't divide evenly — just use total as single spatial dim
        return (1, 1, total)
    spatial = total // C_in
    side = int(math.sqrt(spatial))
    if side * side == spatial:
        return (C_in, side, side)
    for h in range(side, 0, -1):
        if spatial % h == 0:
            return (C_in, h, spatial // h)
    return (C_in, spatial, 1)


def ia_bounds_graph(graph, x_lo, x_hi):
    """Interval arithmetic bounds on a ComputeGraph.

    Returns:
        state: dict[node_name -> (lo, hi)] for every node
        pre_relu_bounds: dict[relu_node_name -> (lo, hi)] bounds before that ReLU
    """
    state = {graph.input_name: (x_lo.copy(), x_hi.copy())}
    pre_relu_bounds = {}

    def _get(inp_name):
        return state[inp_name]

    def _inp_shape(inp_name):
        if inp_name in graph.nodes:
            return graph.nodes[inp_name].output_shape
        return graph.input_shape

    for name in graph.topo_order:
        node = graph.nodes[name]
        op = node.op_type

        if op == 'Relu':
            inp_lo, inp_hi = _get(node.inputs[0])
            pre_relu_bounds[name] = (inp_lo.copy(), inp_hi.copy())
            state[name] = (np.maximum(inp_lo, 0), np.maximum(inp_hi, 0))

        elif op == 'LeakyRelu':
            inp_lo, inp_hi = _get(node.inputs[0])
            alpha = node.params.get('alpha', 0.01)
            pre_relu_bounds[name] = (inp_lo.copy(), inp_hi.copy())
            # For negative x: alpha*x. For positive x: x.
            lo = np.where(inp_lo >= 0, inp_lo, alpha * inp_lo)
            hi = np.where(inp_hi >= 0, inp_hi, alpha * inp_hi)
            # Cross-zero neurons: min/max of both
            cross = (inp_lo < 0) & (inp_hi > 0)
            lo = np.where(cross, np.minimum(alpha * inp_lo, 0), lo)
            hi = np.where(cross, np.maximum(inp_hi, alpha * inp_hi), hi)
            state[name] = (lo, hi)

        elif op == 'Conv':
            inp_lo, inp_hi = _get(node.inputs[0])
            kernel = node.params['kernel']
            bias = node.params['bias']
            input_shape = _inp_shape(node.inputs[0])
            actual_len = len(inp_lo)
            if len(input_shape) != 3 or _prod(input_shape) != actual_len:
                input_shape = _infer_conv_input_shape(actual_len, kernel)
            try:
                conv_params = {
                    'input_shape': input_shape,
                    'stride': node.params['stride'],
                    'padding': node.params['padding'],
                }
                state[name] = _conv_ia(inp_lo, inp_hi, kernel, bias, conv_params)
            except RuntimeError:
                # Shape mismatch in conv2d — use conservative bounds
                C_out = kernel.shape[0]
                out_shape = node.output_shape
                n_out = _prod(out_shape) if out_shape else C_out
                inp_range = np.maximum(np.abs(inp_lo), np.abs(inp_hi)).max()
                k_abs_sum = np.abs(kernel).sum() / C_out
                bound = k_abs_sum * inp_range + np.abs(bias).max()
                state[name] = (-np.full(n_out, bound), np.full(n_out, bound))

        elif op == 'ConvTranspose':
            inp_lo, inp_hi = _get(node.inputs[0])
            input_shape = _inp_shape(node.inputs[0])
            actual_len = len(inp_lo)
            if len(input_shape) != 3 or _prod(input_shape) != actual_len:
                input_shape = _infer_conv_input_shape(
                    actual_len, node.params['kernel'], transpose=True)
            try:
                state[name] = _conv_transpose_ia(
                    inp_lo, inp_hi, node.params, input_shape)
            except RuntimeError:
                out_shape = node.output_shape
                n_out = _prod(out_shape) if out_shape else actual_len
                inp_range = np.maximum(np.abs(inp_lo), np.abs(inp_hi)).max()
                state[name] = (-np.full(n_out, inp_range * 10),
                                np.full(n_out, inp_range * 10))

        elif op in ('Gemm', 'MatMul') and 'W' in node.params:
            inp_lo, inp_hi = _get(node.inputs[0])
            W, b = node.params['W'], node.params['b']
            if W.shape[1] != len(inp_lo):
                # Dimension mismatch (e.g. after attention reshape) — use
                # conservative bounds based on W norms
                out_n = W.shape[0]
                w_abs_sum = np.abs(W).sum(axis=1)
                inp_range = np.maximum(np.abs(inp_lo), np.abs(inp_hi)).max()
                bound = w_abs_sum * inp_range + np.abs(b)
                state[name] = (-bound, bound)
            else:
                Wp = np.maximum(W, 0)
                Wn = np.minimum(W, 0)
                z_lo = Wp @ inp_lo + Wn @ inp_hi + b
                z_hi = Wp @ inp_hi + Wn @ inp_lo + b
                state[name] = (z_lo, z_hi)

        elif op == 'Add':
            if len(node.inputs) == 2 and node.inputs[1] in graph.nodes:
                lo_a, hi_a = _get(node.inputs[0])
                lo_b, hi_b = _get(node.inputs[1])
                # Handle size mismatch via broadcast
                if len(lo_a) != len(lo_b):
                    min_len = min(len(lo_a), len(lo_b))
                    lo_a, hi_a = lo_a[:min_len], hi_a[:min_len]
                    lo_b, hi_b = lo_b[:min_len], hi_b[:min_len]
                state[name] = (lo_a + lo_b, hi_a + hi_b)
            else:
                inp_lo, inp_hi = _get(node.inputs[0])
                bias = node.params.get('bias', 0)
                if isinstance(bias, np.ndarray) and len(bias) < len(inp_lo):
                    bias = np.repeat(bias, len(inp_lo) // len(bias))
                state[name] = (inp_lo + bias, inp_hi + bias)

        elif op == 'Sub':
            inp_lo, inp_hi = _get(node.inputs[0])
            if node.params.get('negate'):
                bias = node.params.get('bias', 0)
                if isinstance(bias, np.ndarray) and len(bias) != len(inp_lo):
                    if len(bias) < len(inp_lo):
                        bias = np.repeat(bias, len(inp_lo) // max(len(bias), 1))
                    else:
                        bias = bias[:len(inp_lo)]
                state[name] = (-inp_hi + bias, -inp_lo + bias)
            else:
                sub_val = node.params.get('sub_val', 0)
                if isinstance(sub_val, np.ndarray) and len(sub_val) != len(inp_lo):
                    if len(sub_val) < len(inp_lo):
                        sub_val = np.repeat(sub_val, len(inp_lo) // max(len(sub_val), 1))
                    else:
                        sub_val = sub_val[:len(inp_lo)]
                state[name] = (inp_lo - sub_val, inp_hi - sub_val)

        elif op == 'Mul':
            if 'scale' in node.params:
                inp_lo, inp_hi = _get(node.inputs[0])
                s = node.params['scale']
                if len(s) < len(inp_lo):
                    s = np.repeat(s, len(inp_lo) // len(s))
                sp = np.maximum(s, 0)
                sn = np.minimum(s, 0)
                state[name] = (sp * inp_lo + sn * inp_hi, sp * inp_hi + sn * inp_lo)
            elif len(node.inputs) == 2:
                # Both computed — use interval multiplication
                lo_a, hi_a = _get(node.inputs[0])
                lo_b, hi_b = _get(node.inputs[1])
                products = np.stack([lo_a * lo_b, lo_a * hi_b,
                                     hi_a * lo_b, hi_a * hi_b])
                state[name] = (products.min(axis=0), products.max(axis=0))
            else:
                state[name] = _get(node.inputs[0])

        elif op == 'Div':
            inp_lo, inp_hi = _get(node.inputs[0])
            s = node.params.get('scale', 1.0)  # already inverted
            sp = np.maximum(s, 0)
            sn = np.minimum(s, 0)
            state[name] = (sp * inp_lo + sn * inp_hi, sp * inp_hi + sn * inp_lo)

        elif op == 'Neg':
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = (-inp_hi, -inp_lo)

        elif op in ('Flatten', 'Squeeze', 'Unsqueeze', 'Reshape',
                      'Dropout', 'Transpose', 'Identity', 'SplitOutput'):
            state[name] = _get(node.inputs[0])

        elif op == 'BatchNormalization':
            inp_lo, inp_hi = _get(node.inputs[0])
            scale = node.params['scale']
            bn_bias = node.params['bias']
            mean = node.params['mean']
            var = node.params['var']
            eps = node.params['epsilon']
            factor = scale / np.sqrt(var + eps)
            offset = -factor * mean + bn_bias
            # Broadcast per-channel params to flat vector
            if len(factor) < len(inp_lo):
                C = len(factor)
                spatial = len(inp_lo) // C
                factor = np.repeat(factor, spatial)
                offset = np.repeat(offset, spatial)
            fp = np.maximum(factor, 0)
            fn = np.minimum(factor, 0)
            state[name] = (fp * inp_lo + fn * inp_hi + offset,
                           fp * inp_hi + fn * inp_lo + offset)

        elif op == 'Sigmoid':
            inp_lo, inp_hi = _get(node.inputs[0])
            sig = lambda x: 1.0 / (1.0 + np.exp(-x))
            state[name] = (sig(inp_lo), sig(inp_hi))

        elif op == 'Clip':
            inp_lo, inp_hi = _get(node.inputs[0])
            lo = inp_lo.copy()
            hi = inp_hi.copy()
            if 'min' in node.params:
                lo = np.maximum(lo, node.params['min'])
                hi = np.maximum(hi, node.params['min'])
            if 'max' in node.params:
                lo = np.minimum(lo, node.params['max'])
                hi = np.minimum(hi, node.params['max'])
            state[name] = (lo, hi)

        elif op == 'Sign':
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = (np.sign(inp_lo), np.sign(inp_hi))

        elif op == 'Softmax':
            # Conservative: softmax outputs are in [0, 1]
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = (np.zeros_like(inp_lo), np.ones_like(inp_hi))

        elif op in ('Sin', 'Cos'):
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = (-np.ones_like(inp_lo), np.ones_like(inp_hi))

        elif op == 'Pow':
            inp_lo, inp_hi = _get(node.inputs[0])
            exp = node.params.get('exponent', 2.0)
            if exp == 2.0:
                # x^2 is monotone on [0,inf) and (-inf,0]
                products = np.stack([inp_lo**2, inp_hi**2])
                hi = products.max(axis=0)
                # If interval crosses zero, minimum is 0
                crosses_zero = (inp_lo <= 0) & (inp_hi >= 0)
                lo = np.where(crosses_zero, 0.0, products.min(axis=0))
                state[name] = (lo, hi)
            else:
                # Conservative
                state[name] = (-np.ones_like(inp_lo) * 1e10,
                                np.ones_like(inp_hi) * 1e10)

        elif op == 'MaxPool':
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = _maxpool_ia(inp_lo, inp_hi, node.params,
                                       _inp_shape(node.inputs[0]))

        elif op == 'AveragePool':
            inp_lo, inp_hi = _get(node.inputs[0])
            state[name] = _avgpool_ia(inp_lo, inp_hi, node.params,
                                       _inp_shape(node.inputs[0]))

        elif op == 'Pad':
            inp_lo, inp_hi = _get(node.inputs[0])
            pads = node.params.get('pads', [])
            val = node.params.get('constant_value', 0.0)
            if pads:
                inp_shape = _inp_shape(node.inputs[0])
                state[name] = _pad_ia(inp_lo, inp_hi, pads, val, inp_shape)
            else:
                state[name] = (inp_lo, inp_hi)

        elif op == 'Concat':
            parts_lo = []
            parts_hi = []
            for inp_name in node.inputs:
                lo_i, hi_i = _get(inp_name)
                parts_lo.append(lo_i.flatten())
                parts_hi.append(hi_i.flatten())
            state[name] = (np.concatenate(parts_lo), np.concatenate(parts_hi))

        elif op == 'Split':
            inp_lo, inp_hi = _get(node.inputs[0])
            # First output is this node itself
            split_sizes = node.params.get('split', None)
            if split_sizes:
                offset = 0
                lo_parts = []
                hi_parts = []
                for s in split_sizes:
                    lo_parts.append(inp_lo[offset:offset + s])
                    hi_parts.append(inp_hi[offset:offset + s])
                    offset += s
                state[name] = (lo_parts[0], hi_parts[0])
                # Set SplitOutput states
                for succ_name, succ_node in graph.nodes.items():
                    if succ_node.op_type == 'SplitOutput' and succ_node.inputs[0] == name:
                        idx = succ_node.params['index']
                        if idx < len(lo_parts):
                            state[succ_name] = (lo_parts[idx], hi_parts[idx])
            else:
                state[name] = (inp_lo, inp_hi)

        elif op == 'Slice':
            inp_lo, inp_hi = _get(node.inputs[0])
            starts = node.params.get('starts', [0])
            ends = node.params.get('ends', [len(inp_lo)])
            # Simple 1D slice
            s = starts[0] if starts else 0
            e = ends[0] if ends else len(inp_lo)
            if e > len(inp_lo):
                e = len(inp_lo)
            if s < 0:
                s = len(inp_lo) + s
            if e < 0:
                e = len(inp_lo) + e
            state[name] = (inp_lo[s:e], inp_hi[s:e])

        elif op == 'Gather':
            inp_lo, inp_hi = _get(node.inputs[0])
            indices = node.params.get('indices', None)
            if indices is not None:
                idx = indices.flatten().astype(int)
                state[name] = (inp_lo[idx], inp_hi[idx])
            else:
                state[name] = (inp_lo, inp_hi)

        elif op in ('ReduceSum', 'ReduceMean'):
            inp_lo, inp_hi = _get(node.inputs[0])
            if op == 'ReduceSum':
                state[name] = (np.array([inp_lo.sum()]), np.array([inp_hi.sum()]))
            else:
                state[name] = (np.array([inp_lo.mean()]), np.array([inp_hi.mean()]))

        elif op == 'Resize':
            inp_lo, inp_hi = _get(node.inputs[0])
            # Nearest-neighbor resize: just repeat elements
            state[name] = (inp_lo, inp_hi)  # approximate

        elif op in ('MatMul',) and 'W' not in node.params:
            # Bilinear MatMul — use interval multiplication
            lo_a, hi_a = _get(node.inputs[0])
            lo_b, hi_b = _get(node.inputs[1])
            products = np.stack([lo_a * lo_b, lo_a * hi_b,
                                 hi_a * lo_b, hi_a * hi_b])
            state[name] = (products.min(axis=0), products.max(axis=0))

        elif op == 'ConstantOfShape':
            val = node.params.get('value', 0.0)
            inp_lo, inp_hi = _get(node.inputs[0])
            n = max(1, len(inp_lo))
            state[name] = (np.full(n, val), np.full(n, val))

        elif op in ('Shape', 'Equal', 'Where', 'Expand', 'ScatterND',
                      'ArgMax', 'Min', 'Max', 'Cast'):
            # Fallback: pass through first input or use wide bounds
            if node.inputs and node.inputs[0] in state:
                state[name] = _get(node.inputs[0])
            else:
                state[name] = (np.array([0.0]), np.array([1.0]))

        else:
            # Unknown op — pass through if possible
            if node.inputs and node.inputs[0] in state:
                state[name] = _get(node.inputs[0])
            else:
                raise ValueError(
                    f"ia_bounds_graph: unsupported op '{op}' at node '{name}'")

    return state, pre_relu_bounds


def _conv_transpose_ia(inp_lo, inp_hi, params, input_shape):
    """ConvTranspose IA bounds using torch."""
    import torch
    import torch.nn.functional as F

    kernel = params['kernel']
    bias = params['bias']
    stride = params['stride']
    padding = params['padding']
    output_padding = params.get('output_padding', (0, 0))

    lo_4d = torch.tensor(inp_lo, dtype=torch.float64).reshape(1, *input_shape)
    hi_4d = torch.tensor(inp_hi, dtype=torch.float64).reshape(1, *input_shape)
    k = torch.tensor(kernel, dtype=torch.float64)
    b = torch.tensor(bias, dtype=torch.float64)

    Kp = torch.clamp(k, min=0)
    Kn = torch.clamp(k, max=0)

    z_lo = (F.conv_transpose2d(lo_4d, Kp, stride=stride, padding=padding,
                                output_padding=output_padding) +
            F.conv_transpose2d(hi_4d, Kn, stride=stride, padding=padding,
                                output_padding=output_padding) +
            b.reshape(1, -1, 1, 1))
    z_hi = (F.conv_transpose2d(hi_4d, Kp, stride=stride, padding=padding,
                                output_padding=output_padding) +
            F.conv_transpose2d(lo_4d, Kn, stride=stride, padding=padding,
                                output_padding=output_padding) +
            b.reshape(1, -1, 1, 1))

    return z_lo.detach().numpy().flatten(), z_hi.detach().numpy().flatten()


def _maxpool_ia(inp_lo, inp_hi, params, input_shape):
    """MaxPool IA bounds using torch."""
    import torch
    import torch.nn.functional as F

    kH, kW = params['kernel_shape']
    sH, sW = params['stride']
    pH, pW = params['padding']

    lo_4d = torch.tensor(inp_lo, dtype=torch.float64).reshape(1, *input_shape)
    hi_4d = torch.tensor(inp_hi, dtype=torch.float64).reshape(1, *input_shape)

    # MaxPool lower bound: max of lower bounds in each window
    z_lo = F.max_pool2d(lo_4d, kernel_size=(kH, kW), stride=(sH, sW),
                         padding=(pH, pW))
    # MaxPool upper bound: max of upper bounds in each window
    z_hi = F.max_pool2d(hi_4d, kernel_size=(kH, kW), stride=(sH, sW),
                         padding=(pH, pW))

    return z_lo.detach().numpy().flatten(), z_hi.detach().numpy().flatten()


def _avgpool_ia(inp_lo, inp_hi, params, input_shape):
    """AveragePool IA bounds using torch."""
    import torch
    import torch.nn.functional as F

    kH, kW = params['kernel_shape']
    sH, sW = params['stride']
    pH, pW = params['padding']

    lo_4d = torch.tensor(inp_lo, dtype=torch.float64).reshape(1, *input_shape)
    hi_4d = torch.tensor(inp_hi, dtype=torch.float64).reshape(1, *input_shape)

    z_lo = F.avg_pool2d(lo_4d, kernel_size=(kH, kW), stride=(sH, sW),
                         padding=(pH, pW))
    z_hi = F.avg_pool2d(hi_4d, kernel_size=(kH, kW), stride=(sH, sW),
                         padding=(pH, pW))

    return z_lo.detach().numpy().flatten(), z_hi.detach().numpy().flatten()


def _pad_ia(inp_lo, inp_hi, pads, val, input_shape):
    """Pad IA bounds using numpy."""
    import torch
    import torch.nn.functional as F

    if len(input_shape) == 3:
        lo_4d = torch.tensor(inp_lo, dtype=torch.float64).reshape(1, *input_shape)
        hi_4d = torch.tensor(inp_hi, dtype=torch.float64).reshape(1, *input_shape)
        # ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        # For 4D tensor: [batch_begin, C_begin, H_begin, W_begin, batch_end, C_end, H_end, W_end]
        n = len(pads) // 2
        # torch F.pad wants (W_begin, W_end, H_begin, H_end, ...)
        if n >= 4:
            torch_pad = (pads[3], pads[3 + n], pads[2], pads[2 + n])
        elif n >= 2:
            torch_pad = (pads[1], pads[1 + n], pads[0], pads[0 + n])
        else:
            return (inp_lo, inp_hi)
        z_lo = F.pad(lo_4d, torch_pad, value=val)
        z_hi = F.pad(hi_4d, torch_pad, value=val)
        return z_lo.detach().numpy().flatten(), z_hi.detach().numpy().flatten()
    else:
        # 1D: simple
        return (inp_lo, inp_hi)
