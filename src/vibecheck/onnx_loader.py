"""ONNX model loading."""

import numpy as np
import gzip


def is_conv(layer):
    """Check if layer is Conv (3-tuple) vs FC (2-tuple)."""
    return len(layer) == 3


def conv_output_shape(input_shape, kernel, params):
    """Compute output spatial shape for a Conv layer.

    Args:
        input_shape: (C_in, H_in, W_in)
        kernel: numpy array of shape (C_out, C_in, kH, kW)
        params: dict with 'stride' and 'padding'

    Returns: (C_out, H_out, W_out)
    """
    C_in, H_in, W_in = input_shape
    C_out = kernel.shape[0]
    kH, kW = kernel.shape[2], kernel.shape[3]
    sH, sW = params['stride']
    pH, pW = params['padding']
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    return (C_out, H_out, W_out)


def conv_connections(j, kernel, params):
    """For flat output neuron j in a Conv layer, return sparse input connections.

    Args:
        j: flat output neuron index
        kernel: (C_out, C_in, kH, kW) numpy array
        params: dict with 'input_shape', 'stride', 'padding'

    Returns:
        connections: list of (flat_input_idx, weight)
    """
    input_shape = params['input_shape']
    C_in, H_in, W_in = input_shape
    C_out, _, kH, kW = kernel.shape
    sH, sW = params['stride']
    pH, pW = params['padding']

    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1

    c_out = j // (H_out * W_out)
    rem = j % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    connections = []
    for c_in in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                h_in = h_out * sH - pH + kh
                w_in = w_out * sW - pW + kw
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    weight = kernel[c_out, c_in, kh, kw]
                    if weight != 0:
                        flat_in = c_in * H_in * W_in + h_in * W_in + w_in
                        connections.append((flat_in, float(weight)))

    return connections


def _compose_fc_conv(W_fc, b_fc, reshape_shape, kernel, bias_conv, conv_params):
    """Compose FC (Gemm) + Reshape + Conv into a single FC layer."""
    C_in, H_in, W_in = reshape_shape
    C_out = kernel.shape[0]
    kH, kW = kernel.shape[2], kernel.shape[3]
    sH, sW = conv_params['stride']
    pH, pW = conv_params['padding']
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    n_out = C_out * H_out * W_out
    n_input = W_fc.shape[1]

    W_composed = np.zeros((n_out, n_input), dtype=np.float64)
    b_composed = np.zeros(n_out, dtype=np.float64)

    for j in range(n_out):
        c_out = j // (H_out * W_out)
        rem = j % (H_out * W_out)
        h_out = rem // W_out
        w_out = rem % W_out

        b_composed[j] = bias_conv[c_out]
        for c_in in range(C_in):
            for kh in range(kH):
                for kw in range(kW):
                    h_in = h_out * sH - pH + kh
                    w_in = w_out * sW - pW + kw
                    if 0 <= h_in < H_in and 0 <= w_in < W_in:
                        w = kernel[c_out, c_in, kh, kw]
                        if w != 0:
                            flat_in = c_in * H_in * W_in + h_in * W_in + w_in
                            W_composed[j, :] += w * W_fc[flat_in, :]
                            b_composed[j] += w * b_fc[flat_in]

    return W_composed, b_composed


def load_onnx(onnx_path):
    """Load network from ONNX. Handles both FC and Conv networks.

    Returns:
        layers: list of (W, b) for FC layers or (kernel, bias, params) for Conv layers
        input_shape: tuple — (n_input,) for FC or (C, H, W) for Conv input
    """
    import onnx
    from onnx import numpy_helper
    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model = onnx.load_from_string(f.read())
    else:
        model = onnx.load(onnx_path)

    inits = {init.name: numpy_helper.to_array(init).astype(np.float64)
             for init in model.graph.initializer}

    graph_input = model.graph.input[0]
    input_dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]

    constants = {}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)

    relu_inputs = set()
    for node in model.graph.node:
        if node.op_type == 'Relu':
            relu_inputs.add(node.input[0])

    ops = []
    input_sub = None  # tracks Sub normalization to fold into first layer
    shape_map = {}
    shape_map[graph_input.name] = tuple(input_dims)

    for node in model.graph.node:
        if node.op_type == 'Conv':
            kernel = inits[node.input[1]]
            bias = inits[node.input[2]] if len(node.input) > 2 and node.input[2] in inits else np.zeros(kernel.shape[0], dtype=np.float64)
            attrs = {'stride': (1, 1), 'padding': (0, 0)}
            for attr in node.attribute:
                if attr.name == 'strides':
                    attrs['stride'] = (list(attr.ints)[0], list(attr.ints)[1])
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    attrs['padding'] = (pads[0], pads[1])

            has_relu = node.output[0] in relu_inputs
            ops.append(('Conv', node.output[0],
                        {'kernel': kernel, 'bias': bias, 'attrs': attrs},
                        has_relu))

        elif node.op_type == 'Gemm':
            W = inits[node.input[1]]
            b = inits[node.input[2]]
            transB = any(attr.i for attr in node.attribute if attr.name == 'transB')
            if not transB:
                W = W.T
            has_relu = node.output[0] in relu_inputs
            ops.append(('Gemm', node.output[0], {'W': W, 'b': b}, has_relu))

        elif node.op_type == 'Reshape':
            if node.input[1] in constants:
                target = tuple(int(x) for x in constants[node.input[1]])
                shape_map[node.output[0]] = target

        elif node.op_type in ('Flatten', 'Squeeze'):
            pass  # shape-only ops, handled implicitly

        elif node.op_type == 'Sub':
            if node.input[1] in inits:
                sub_val = inits[node.input[1]].flatten()
                # Store as an affine layer: identity W, bias = -sub_val
                # Will be folded into the next layer's bias
                input_sub = sub_val

        elif node.op_type == 'MatMul':
            W = inits[node.input[1]].T  # ONNX MatMul: x @ W, we need W.T @ x
            has_relu = node.output[0] in relu_inputs
            ops.append(('Gemm', node.output[0],
                        {'W': W, 'b': np.zeros(W.shape[0], dtype=np.float64)},
                        has_relu))

        elif node.op_type == 'Add':
            if node.input[1] in inits:
                b = inits[node.input[1]].flatten()
                if ops and ops[-1][0] == 'Gemm' and np.all(ops[-1][2]['b'] == 0):
                    ops[-1][2]['b'] = b

    layers = []
    current_shape = None

    has_conv = any(op[0] == 'Conv' for op in ops)
    if has_conv and len(input_dims) == 4:
        current_shape = tuple(input_dims[1:])

    reshape_targets = {}
    for node in model.graph.node:
        if node.op_type == 'Reshape' and node.input[1] in constants:
            target = tuple(int(x) for x in constants[node.input[1]])
            reshape_targets[node.input[0]] = target

    i = 0
    while i < len(ops):
        op_type, out_name, data, has_relu = ops[i]

        if op_type == 'Gemm' and not has_relu and i + 1 < len(ops) and ops[i + 1][0] == 'Conv':
            W_fc, b_fc = data['W'], data['b']
            reshape_shape = None
            if out_name in reshape_targets:
                reshape_shape = reshape_targets[out_name][1:]
            else:
                next_kernel = ops[i + 1][2]['kernel']
                C_in = next_kernel.shape[1]
                n_gemm_out = W_fc.shape[0]
                spatial = n_gemm_out // C_in
                import math
                side = int(math.sqrt(spatial))
                if side * side == spatial:
                    reshape_shape = (C_in, side, side)

            if reshape_shape is not None:
                next_data = ops[i + 1][2]
                conv_params = next_data['attrs'].copy()
                conv_params['input_shape'] = reshape_shape

                W_composed, b_composed = _compose_fc_conv(
                    W_fc, b_fc, reshape_shape,
                    next_data['kernel'], next_data['bias'], conv_params)

                layers.append((W_composed, b_composed))
                out_shape = conv_output_shape(reshape_shape, next_data['kernel'], conv_params)
                current_shape = out_shape
                i += 2
                continue

        if op_type == 'Conv':
            kernel, bias = data['kernel'], data['bias']
            attrs = data['attrs'].copy()
            attrs['input_shape'] = current_shape
            layers.append((kernel, bias, attrs))
            current_shape = conv_output_shape(current_shape, kernel, attrs)

        elif op_type == 'Gemm':
            W, b = data['W'], data['b']
            layers.append((W, b))
            current_shape = None

        i += 1

    # Fold input Sub normalization into first FC layer's bias
    if input_sub is not None and layers and not is_conv(layers[0]):
        W0, b0 = layers[0]
        layers[0] = (W0, b0 - W0 @ input_sub)

    if has_conv and len(input_dims) == 4:
        input_shape = tuple(input_dims[1:])
    else:
        input_shape = (input_dims[-1],)

    return layers, input_shape
