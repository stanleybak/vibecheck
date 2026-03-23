"""Verification orchestration — multi-zonotope intersection strategy."""

import numpy as np
from .bounds import ia_bounds, ia_bounds_graph, _infer_conv_input_shape, _prod
from .zonotope import DenseZonotope


def _interval_to_zonotope(lo, hi):
    """Convert interval bounds to a fresh zonotope (loses correlations)."""
    return DenseZonotope.from_input_bounds(lo, hi)


def zonotope_verify(layers, x_lo, x_hi, pred_label, competitors, relu_types=None):
    """Run zonotope analysis to verify a spec.

    Propagates through all layers using multi-zonotope (intersecting
    multiple ReLU relaxation variants), then checks if the output
    bounds prove the property.

    Args:
        layers: network layers from load_onnx
        x_lo, x_hi: input bounds
        pred_label: predicted class index
        competitors: list of competitor class indices
        relu_types: list of ReLU relaxation types to intersect

    Returns:
        result: 'verified' or 'unknown'
        details: dict with output_lo, output_hi, margins, worst_margin
    """
    if relu_types is None:
        relu_types = ['min_area', 'y_bloat', 'box']

    n_hidden = len(layers) - 1
    pre_bounds, _ = ia_bounds(layers, x_lo, x_hi)

    best_lo = pre_bounds[-1][0].copy()
    best_hi = pre_bounds[-1][1].copy()

    for relu_type in relu_types:
        zono = DenseZonotope.from_input_bounds(x_lo, x_hi)
        for l, layer in enumerate(layers):
            zono.propagate_linear(layer)
            z_lo, z_hi = zono.bounds()
            if l < n_hidden:
                pb_lo, pb_hi = pre_bounds[l]
                zono.apply_relu(
                    np.maximum(z_lo, pb_lo),
                    np.minimum(z_hi, pb_hi),
                    relu_type,
                )
        best_lo = np.maximum(best_lo, z_lo)
        best_hi = np.minimum(best_hi, z_hi)

    margins = {comp: float(best_lo[pred_label] - best_hi[comp]) for comp in competitors}
    worst_margin = min(margins.values())

    return ('verified' if worst_margin > 0 else 'unknown'), {
        'output_lo': best_lo,
        'output_hi': best_hi,
        'margins': margins,
        'worst_margin': float(worst_margin),
    }


def zonotope_verify_graph(graph, x_lo, x_hi, pred_label, competitors, relu_types=None):
    """Run zonotope analysis on a ComputeGraph.

    Same multi-zonotope intersection strategy as zonotope_verify, but
    traverses the graph in topological order with dict-based state.
    Handles fork/merge (skip connections) via DenseZonotope.add().

    Args:
        graph: ComputeGraph from graph.py
        x_lo, x_hi: input bounds
        pred_label: predicted class index
        competitors: list of competitor class indices
        relu_types: list of ReLU relaxation types to intersect

    Returns:
        result: 'verified' or 'unknown'
        details: dict with output_lo, output_hi, margins, worst_margin
    """
    from .onnx_loader import conv_output_shape

    if relu_types is None:
        relu_types = ['min_area', 'y_bloat', 'box']

    ia_state, pre_relu_bounds = ia_bounds_graph(graph, x_lo, x_hi)
    relu_nodes = graph.relu_nodes()
    forks = graph.fork_points()

    # Initialize best bounds from IA at output
    out_lo, out_hi = ia_state[graph.output_name]
    best_lo = out_lo.copy()
    best_hi = out_hi.copy()

    for relu_type in relu_types:
        zono_state = {}
        # Track generator count at fork points for merge
        gen_count = {}  # tensor_name -> n_generators when last written

        zono_state[graph.input_name] = DenseZonotope.from_input_bounds(x_lo, x_hi)
        gen_count[graph.input_name] = zono_state[graph.input_name].generators.shape[1]

        for name in graph.topo_order:
            node = graph.nodes[name]
            op = node.op_type

            # Get input zonotope(s), copying at fork points
            def _get_input(inp_name):
                if inp_name in forks:
                    return zono_state[inp_name].copy()
                return zono_state[inp_name]

            if op == 'Relu':
                z = _get_input(node.inputs[0])
                ia_lo, ia_hi = pre_relu_bounds[name]
                z_lo, z_hi = z.bounds()
                z.apply_relu(
                    np.maximum(z_lo, ia_lo),
                    np.minimum(z_hi, ia_hi),
                    relu_type,
                )
                zono_state[name] = z

            elif op == 'LeakyRelu':
                z = _get_input(node.inputs[0])
                ia_lo, ia_hi = pre_relu_bounds[name]
                z_lo, z_hi = z.bounds()
                # Fallback to interval for LeakyRelu
                lo, hi = ia_state[name]
                zono_state[name] = _interval_to_zonotope(lo, hi)

            elif op == 'Conv':
                z = _get_input(node.inputs[0])
                n_gens = z.generators.shape[1]
                n_elems = len(z.center)
                # If generator matrix is too large for conv, fall back
                if n_gens * n_elems > 5_000_000:
                    lo, hi = ia_state[name]
                    zono_state[name] = _interval_to_zonotope(lo, hi)
                else:
                    input_shape = graph.nodes[node.inputs[0]].output_shape \
                        if node.inputs[0] in graph.nodes \
                        else graph.input_shape
                    actual_len = n_elems
                    if len(input_shape) != 3 or _prod(input_shape) != actual_len:
                        input_shape = _infer_conv_input_shape(
                            actual_len, node.params['kernel'])
                    conv_params = {
                        'input_shape': input_shape,
                        'stride': node.params['stride'],
                        'padding': node.params['padding'],
                    }
                    layer = (node.params['kernel'], node.params['bias'], conv_params)
                    z.propagate_linear(layer)
                    zono_state[name] = z

            elif op == 'ConvTranspose':
                # Fallback to interval
                lo, hi = ia_state[name]
                zono_state[name] = _interval_to_zonotope(lo, hi)

            elif op in ('Gemm', 'MatMul') and 'W' in node.params:
                z = _get_input(node.inputs[0])
                W = node.params['W']
                if W.shape[1] != len(z.center):
                    # Dimension mismatch — fall back to interval
                    lo, hi = ia_state[name]
                    zono_state[name] = _interval_to_zonotope(lo, hi)
                else:
                    layer = (W, node.params['b'])
                    z.propagate_linear(layer)
                    zono_state[name] = z

            elif op == 'Add':
                if len(node.inputs) == 2 and node.inputs[1] in graph.nodes:
                    z_a = _get_input(node.inputs[0])
                    z_b = _get_input(node.inputs[1])
                    shared = _find_shared_gens(
                        node.inputs[0], node.inputs[1], graph, gen_count)
                    zono_state[name] = z_a.add(z_b, shared)
                else:
                    z = _get_input(node.inputs[0])
                    bias = node.params.get('bias', 0)
                    z.center = z.center + bias
                    zono_state[name] = z

            elif op == 'Sub':
                z = _get_input(node.inputs[0])
                if node.params.get('negate'):
                    bias = node.params.get('bias', 0)
                    z.center = -z.center + bias
                    z.generators = -z.generators
                else:
                    sub_val = node.params.get('sub_val', 0)
                    z.center = z.center - sub_val
                zono_state[name] = z

            elif op == 'Mul' and 'scale' in node.params:
                z = _get_input(node.inputs[0])
                s = node.params['scale']
                z.center = z.center * s
                z.generators = z.generators * s[:, None]
                zono_state[name] = z

            elif op == 'Div' and 'scale' in node.params:
                z = _get_input(node.inputs[0])
                s = node.params['scale']  # already inverted
                z.center = z.center * s
                z.generators = z.generators * s[:, None]
                zono_state[name] = z

            elif op == 'Neg':
                z = _get_input(node.inputs[0])
                z.center = -z.center
                z.generators = -z.generators
                zono_state[name] = z

            elif op in ('Flatten', 'Squeeze', 'Unsqueeze', 'Reshape',
                          'Dropout', 'Transpose', 'Identity', 'SplitOutput'):
                zono_state[name] = _get_input(node.inputs[0])

            elif op == 'Concat':
                # Concatenate zonotope states
                parts = [_get_input(inp) for inp in node.inputs]
                # Find max generator columns
                max_k = max(p.generators.shape[1] for p in parts)
                centers = []
                gens = []
                for p in parts:
                    centers.append(p.center)
                    k = p.generators.shape[1]
                    if k < max_k:
                        pad = np.zeros((p.generators.shape[0], max_k - k))
                        gens.append(np.hstack([p.generators, pad]))
                    else:
                        gens.append(p.generators)
                zono_state[name] = DenseZonotope(
                    np.concatenate(centers),
                    np.vstack(gens),
                )

            elif op == 'Slice':
                z = _get_input(node.inputs[0])
                starts = node.params.get('starts', [0])
                ends = node.params.get('ends', [len(z.center)])
                s = starts[0] if starts else 0
                e = ends[0] if ends else len(z.center)
                if e > len(z.center):
                    e = len(z.center)
                if s < 0:
                    s = len(z.center) + s
                if e < 0:
                    e = len(z.center) + e
                zono_state[name] = DenseZonotope(
                    z.center[s:e], z.generators[s:e, :])

            elif op == 'Gather':
                z = _get_input(node.inputs[0])
                indices = node.params.get('indices', None)
                if indices is not None:
                    idx = indices.flatten().astype(int)
                    zono_state[name] = DenseZonotope(
                        z.center[idx], z.generators[idx, :])
                else:
                    zono_state[name] = z

            elif op == 'Split':
                z = _get_input(node.inputs[0])
                split_sizes = node.params.get('split', None)
                if split_sizes:
                    offset = 0
                    s = split_sizes[0]
                    zono_state[name] = DenseZonotope(
                        z.center[offset:offset + s],
                        z.generators[offset:offset + s, :])
                    # Set SplitOutput states
                    cum = 0
                    for succ_name, succ_node in graph.nodes.items():
                        if (succ_node.op_type == 'SplitOutput'
                                and succ_node.inputs[0] == name):
                            idx = succ_node.params['index']
                            start = sum(split_sizes[:idx])
                            end = start + split_sizes[idx]
                            zono_state[succ_name] = DenseZonotope(
                                z.center[start:end],
                                z.generators[start:end, :])
                            gen_count[succ_name] = z.generators.shape[1]
                else:
                    zono_state[name] = z

            elif op in ('ReduceSum', 'ReduceMean'):
                z = _get_input(node.inputs[0])
                if op == 'ReduceSum':
                    zono_state[name] = DenseZonotope(
                        np.array([z.center.sum()]),
                        z.generators.sum(axis=0, keepdims=True))
                else:
                    n = len(z.center)
                    zono_state[name] = DenseZonotope(
                        np.array([z.center.mean()]),
                        z.generators.mean(axis=0, keepdims=True))

            else:
                # Interval fallback: drop correlations, re-wrap as zonotope
                if name in ia_state:
                    lo, hi = ia_state[name]
                    zono_state[name] = _interval_to_zonotope(lo, hi)
                elif node.inputs and node.inputs[0] in zono_state:
                    zono_state[name] = _get_input(node.inputs[0])
                else:
                    zono_state[name] = DenseZonotope(
                        np.array([0.0]), np.zeros((1, 1)))

            # Record generator count for fork tracking
            gen_count[name] = zono_state[name].generators.shape[1]

        # Extract output bounds
        z_out = zono_state[graph.output_name]
        z_lo, z_hi = z_out.bounds()
        best_lo = np.maximum(best_lo, z_lo)
        best_hi = np.minimum(best_hi, z_hi)

    margins = {comp: float(best_lo[pred_label] - best_hi[comp]) for comp in competitors}
    worst_margin = min(margins.values())

    return ('verified' if worst_margin > 0 else 'unknown'), {
        'output_lo': best_lo,
        'output_hi': best_hi,
        'margins': margins,
        'worst_margin': float(worst_margin),
    }


def _find_shared_gens(name_a, name_b, graph, gen_count):
    """Find the generator count at the fork point of two merging branches.

    Walk back from both inputs to find their latest common ancestor
    that is a fork point, then return the gen_count at that point.
    """
    forks = graph.fork_points()

    # Build ancestor sets by walking backwards
    def _ancestors(name):
        """Return ordered list of ancestors (including self) back to input."""
        visited = []
        stack = [name]
        seen = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            visited.append(n)
            if n in graph.nodes:
                for inp in graph.nodes[n].inputs:
                    stack.append(inp)
        return visited

    anc_a = _ancestors(name_a)
    anc_b_set = set(_ancestors(name_b))

    # Find first ancestor of a that is also ancestor of b and is a fork point
    for anc in anc_a:
        if anc in anc_b_set and anc in forks:
            return gen_count[anc]

    # Fallback: the graph input itself is the fork
    if graph.input_name in anc_b_set:
        return gen_count[graph.input_name]

    # Should not happen in well-formed graphs
    return 0
