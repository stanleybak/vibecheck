"""Compute graph representation for non-sequential ONNX networks."""

from dataclasses import dataclass, field
from collections import deque
import numpy as np
import gzip


@dataclass
class GraphNode:
    """A single operation in the compute graph."""
    name: str               # ONNX output tensor name (unique ID)
    op_type: str            # 'Conv', 'Gemm', 'Relu', 'Add', 'Flatten', ...
    inputs: list            # names of input tensor nodes
    params: dict = field(default_factory=dict)
    output_shape: tuple = None  # inferred shape (without batch dim)


# Op types that just pass through their input unchanged
_PASSTHROUGH_OPS = frozenset([
    'Flatten', 'Squeeze', 'Unsqueeze', 'Reshape', 'Dropout',
    'Shape', 'Cast', 'Transpose', 'Identity',
])

# Op types that are element-wise unary (shape preserved)
_ELEMENTWISE_UNARY_OPS = frozenset([
    'Relu', 'LeakyRelu', 'Sigmoid', 'Sign', 'Neg', 'Clip',
    'Sin', 'Cos', 'Softmax', 'Pow',
])

# Op types that are element-wise binary with a constant
_ELEMENTWISE_CONST_OPS = frozenset([
    'Mul', 'Div', 'Sub', 'Add', 'Min', 'Max', 'Where', 'Equal',
])


def _try_fold_constant(op, computed_inputs, params, const_fn):
    """Try to evaluate an op on constant inputs. Returns result or None."""
    try:
        vals = [const_fn(i) for i in computed_inputs]
        if op == 'Relu':
            return np.maximum(vals[0], 0)
        elif op == 'LeakyRelu':
            alpha = params.get('alpha', 0.01)
            return np.where(vals[0] >= 0, vals[0], alpha * vals[0])
        elif op == 'Neg':
            return -vals[0]
        elif op == 'Sigmoid':
            return 1.0 / (1.0 + np.exp(-vals[0]))
        elif op in ('Flatten', 'Squeeze', 'Unsqueeze', 'Reshape',
                      'Dropout', 'Identity', 'Cast'):
            return vals[0]
        elif op == 'Concat':
            return np.concatenate([v.flatten() for v in vals])
        elif op == 'Slice':
            starts = params.get('starts', [0])
            ends = params.get('ends', [len(vals[0])])
            return vals[0].flatten()[starts[0]:ends[0]]
        elif op == 'Gather':
            indices = params.get('indices')
            if indices is not None:
                return vals[0].flatten()[indices.flatten().astype(int)]
        elif op == 'Transpose':
            return vals[0]  # Keep flat
        elif op == 'Gemm' and 'W' in params:
            return params['W'] @ vals[0].flatten() + params['b']
        elif op == 'MatMul' and 'W' in params:
            return params['W'] @ vals[0].flatten() + params['b']
        elif op == 'Div' and 'scale' in params:
            return vals[0].flatten() * params['scale']
        elif op == 'Sign':
            return np.sign(vals[0])
        elif op == 'ReduceSum':
            return np.array([vals[0].sum()])
        elif op == 'ReduceMean':
            return np.array([vals[0].mean()])
    except Exception:
        pass
    return None


class ComputeGraph:
    """DAG of operations loaded from ONNX.

    Nodes are keyed by their output tensor name. Traversal order is
    topological (Kahn's algorithm), cached after construction.
    """

    def __init__(self):
        self.nodes = {}          # name -> GraphNode
        self.input_name = None   # graph input tensor name
        self.output_name = None  # graph output tensor name
        self.input_shape = None  # without batch dim
        self.topo_order = []     # list of node names

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(cls, onnx_path):
        """Load an ONNX model into a ComputeGraph."""
        import onnx
        from onnx import numpy_helper

        if onnx_path.endswith('.gz'):
            with gzip.open(onnx_path, 'rb') as f:
                model = onnx.load_from_string(f.read())
        else:
            model = onnx.load(onnx_path)

        graph = cls()
        inits = {init.name: numpy_helper.to_array(init).astype(np.float64)
                 for init in model.graph.initializer}

        # Identify graph input (skip initializers)
        init_names = set(inits.keys())
        for inp in model.graph.input:
            if inp.name not in init_names:
                dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                graph.input_name = inp.name
                graph._raw_input_dims = dims
                break

        # Graph output
        graph.output_name = model.graph.output[0].name

        # Parse constants
        constants = {}
        for node in model.graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        constants[node.output[0]] = numpy_helper.to_array(attr.t)

        # Helper: resolve a name to a constant numpy array if possible
        def _const(name):
            if name in inits:
                return inits[name]
            if name in constants:
                return constants[name]
            return None

        # Build nodes
        for node in model.graph.node:
            if node.op_type == 'Constant':
                continue

            out_name = node.output[0]
            computed_inputs = []
            params = {}
            op = node.op_type

            # Extract attributes into a dict for convenience
            attrs = {}
            for attr in node.attribute:
                if attr.ints:
                    attrs[attr.name] = list(attr.ints)
                elif attr.type == 2:  # INT
                    attrs[attr.name] = attr.i
                elif attr.type == 1:  # FLOAT
                    attrs[attr.name] = attr.f
                elif attr.type == 3:  # STRING
                    attrs[attr.name] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s

            if op == 'Conv':
                computed_inputs = [node.input[0]]
                params['kernel'] = inits[node.input[1]]
                if len(node.input) > 2 and node.input[2] in inits:
                    params['bias'] = inits[node.input[2]]
                else:
                    params['bias'] = np.zeros(inits[node.input[1]].shape[0], dtype=np.float64)
                params['stride'] = tuple(attrs.get('strides', [1, 1]))
                pads = attrs.get('pads', [0, 0, 0, 0])
                params['padding'] = (pads[0], pads[1])
                params['group'] = attrs.get('group', 1)

            elif op == 'ConvTranspose':
                computed_inputs = [node.input[0]]
                params['kernel'] = inits[node.input[1]]
                if len(node.input) > 2 and node.input[2] in inits:
                    params['bias'] = inits[node.input[2]]
                else:
                    params['bias'] = np.zeros(inits[node.input[1]].shape[1], dtype=np.float64)
                params['stride'] = tuple(attrs.get('strides', [1, 1]))
                pads = attrs.get('pads', [0, 0, 0, 0])
                params['padding'] = (pads[0], pads[1])
                params['output_padding'] = tuple(attrs.get('output_padding', [0, 0]))
                params['group'] = attrs.get('group', 1)

            elif op == 'Gemm':
                computed_inputs = [node.input[0]]
                W = inits[node.input[1]]
                b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] in inits else np.zeros(W.shape[0], dtype=np.float64)
                transB = attrs.get('transB', 0)
                if not transB:
                    W = W.T
                params['W'] = W
                params['b'] = b

            elif op == 'MatMul':
                c1 = _const(node.input[1])
                c0 = _const(node.input[0])
                if c0 is not None and c1 is not None:
                    # Both constant — fold to a constant
                    result = c0 @ c1
                    constants[out_name] = result
                    continue  # skip adding this node
                elif c1 is not None:
                    computed_inputs = [node.input[0]]
                    params['W'] = c1.T if c1.ndim == 2 else c1
                    out_dim = c1.shape[1] if c1.ndim == 2 else c1.shape[0]
                    params['b'] = np.zeros(out_dim, dtype=np.float64)
                elif c0 is not None:
                    computed_inputs = [node.input[1]]
                    params['W'] = c0
                    params['b'] = np.zeros(c0.shape[0], dtype=np.float64)
                else:
                    computed_inputs = [node.input[0], node.input[1]]

            elif op == 'Add':
                c0 = _const(node.input[0])
                c1 = _const(node.input[1])
                if c0 is not None and c1 is not None:
                    constants[out_name] = c0 + c1
                    continue
                elif c0 is not None:
                    computed_inputs = [node.input[1]]
                    params['bias'] = c0.flatten()
                elif c1 is not None:
                    computed_inputs = [node.input[0]]
                    params['bias'] = c1.flatten()
                else:
                    computed_inputs = [node.input[0], node.input[1]]

            elif op == 'Sub':
                c0_s = _const(node.input[0])
                c1_s = _const(node.input[1])
                if c0_s is not None and c1_s is not None:
                    constants[out_name] = c0_s - c1_s
                    continue
                if c0_s is not None:
                    # constant - x => Neg(x) + constant
                    computed_inputs = [node.input[1]]
                    params['negate'] = True
                    params['bias'] = c0_s.flatten()
                else:
                    computed_inputs = [node.input[0]]
                    c1 = _const(node.input[1])
                    if c1 is not None:
                        params['sub_val'] = c1.flatten()

            elif op == 'Mul':
                c0 = _const(node.input[0])
                c1 = _const(node.input[1])
                if c0 is not None and c1 is not None:
                    constants[out_name] = c0 * c1
                    continue
                elif c0 is not None:
                    computed_inputs = [node.input[1]]
                    params['scale'] = c0.flatten()
                elif c1 is not None:
                    computed_inputs = [node.input[0]]
                    params['scale'] = c1.flatten()
                else:
                    computed_inputs = [node.input[0], node.input[1]]

            elif op == 'Div':
                computed_inputs = [node.input[0]]
                c1 = _const(node.input[1])
                if c1 is not None:
                    params['scale'] = 1.0 / c1.flatten()

            elif op == 'Neg':
                computed_inputs = [node.input[0]]

            elif op in ('Relu', 'LeakyRelu', 'Sigmoid', 'Sign', 'Softmax'):
                computed_inputs = [node.input[0]]
                if 'alpha' in attrs:
                    params['alpha'] = attrs['alpha']
                if 'axis' in attrs:
                    params['axis'] = attrs['axis']

            elif op == 'Clip':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    c_min = _const(node.input[1])
                    if c_min is not None:
                        params['min'] = float(c_min)
                if len(node.input) > 2:
                    c_max = _const(node.input[2])
                    if c_max is not None:
                        params['max'] = float(c_max)

            elif op == 'BatchNormalization':
                computed_inputs = [node.input[0]]
                params['scale'] = inits[node.input[1]]
                params['bias'] = inits[node.input[2]]
                params['mean'] = inits[node.input[3]]
                params['var'] = inits[node.input[4]]
                params['epsilon'] = attrs.get('epsilon', 1e-5)

            elif op == 'MaxPool':
                computed_inputs = [node.input[0]]
                params['kernel_shape'] = tuple(attrs.get('kernel_shape', [2, 2]))
                params['stride'] = tuple(attrs.get('strides', [2, 2]))
                pads = attrs.get('pads', [0, 0, 0, 0])
                params['padding'] = (pads[0], pads[1])

            elif op == 'AveragePool':
                computed_inputs = [node.input[0]]
                params['kernel_shape'] = tuple(attrs.get('kernel_shape', [2, 2]))
                params['stride'] = tuple(attrs.get('strides', [2, 2]))
                pads = attrs.get('pads', [0, 0, 0, 0])
                params['padding'] = (pads[0], pads[1])

            elif op == 'Pad':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    c_pads = _const(node.input[1])
                    if c_pads is not None:
                        params['pads'] = c_pads.astype(int).tolist()
                if len(node.input) > 2:
                    c_val = _const(node.input[2])
                    if c_val is not None:
                        params['constant_value'] = float(c_val)

            elif op == 'Concat':
                computed_inputs = [i for i in node.input if _const(i) is None and i != '']
                # Some inputs may be constants
                const_inputs = [(idx, _const(i)) for idx, i in enumerate(node.input) if _const(i) is not None]
                if const_inputs:
                    params['const_inputs'] = const_inputs
                params['axis'] = attrs.get('axis', 0)

            elif op == 'Split':
                computed_inputs = [node.input[0]]
                params['axis'] = attrs.get('axis', 0)
                if len(node.input) > 1:
                    c_split = _const(node.input[1])
                    if c_split is not None:
                        params['split'] = c_split.astype(int).tolist()
                elif 'split' in attrs:
                    params['split'] = attrs['split']
                # Split produces multiple outputs — register all of them
                for i, out in enumerate(node.output):
                    if i == 0:
                        continue
                    graph.nodes[out] = GraphNode(
                        name=out,
                        op_type='SplitOutput',
                        inputs=[out_name],
                        params={'index': i},
                    )

            elif op == 'Slice':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    starts = _const(node.input[1])
                    if starts is not None:
                        params['starts'] = starts.astype(int).tolist()
                if len(node.input) > 2:
                    ends = _const(node.input[2])
                    if ends is not None:
                        params['ends'] = ends.astype(int).tolist()
                if len(node.input) > 3:
                    axes = _const(node.input[3])
                    if axes is not None:
                        params['axes'] = axes.astype(int).tolist()
                if len(node.input) > 4:
                    steps = _const(node.input[4])
                    if steps is not None:
                        params['steps'] = steps.astype(int).tolist()

            elif op == 'Gather':
                computed_inputs = [node.input[0]]
                c_indices = _const(node.input[1])
                if c_indices is not None:
                    params['indices'] = c_indices
                else:
                    # indices from computed tensor
                    computed_inputs.append(node.input[1])
                params['axis'] = attrs.get('axis', 0)

            elif op == 'ReduceSum':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    c_axes = _const(node.input[1])
                    if c_axes is not None:
                        params['axes'] = c_axes.astype(int).tolist()
                if 'axes' in attrs:
                    params['axes'] = attrs['axes']
                params['keepdims'] = attrs.get('keepdims', 1)

            elif op == 'ReduceMean':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    c_axes = _const(node.input[1])
                    if c_axes is not None:
                        params['axes'] = c_axes.astype(int).tolist()
                if 'axes' in attrs:
                    params['axes'] = attrs['axes']
                params['keepdims'] = attrs.get('keepdims', 1)

            elif op == 'Resize':
                computed_inputs = [node.input[0]]
                # Resize has (input, roi, scales, sizes)
                for idx, param_name in [(2, 'scales'), (3, 'sizes')]:
                    if len(node.input) > idx and node.input[idx] != '':
                        c = _const(node.input[idx])
                        if c is not None and c.size > 0:
                            params[param_name] = c

            elif op == 'Transpose':
                computed_inputs = [node.input[0]]
                if 'perm' in attrs:
                    params['perm'] = attrs['perm']

            elif op in ('Flatten', 'Squeeze', 'Unsqueeze'):
                computed_inputs = [node.input[0]]
                if 'axis' in attrs:
                    params['axis'] = attrs['axis']
                # Unsqueeze axes may come from input
                if op == 'Unsqueeze' and len(node.input) > 1:
                    c_axes = _const(node.input[1])
                    if c_axes is not None:
                        params['axes'] = c_axes.astype(int).tolist()

            elif op == 'Reshape':
                computed_inputs = [node.input[0]]
                if len(node.input) > 1:
                    c_shape = _const(node.input[1])
                    if c_shape is not None:
                        params['shape'] = tuple(int(x) for x in c_shape)

            elif op == 'Dropout':
                computed_inputs = [node.input[0]]

            elif op in ('Sin', 'Cos', 'Pow'):
                computed_inputs = [node.input[0]]
                if op == 'Pow' and len(node.input) > 1:
                    c_exp = _const(node.input[1])
                    if c_exp is not None:
                        params['exponent'] = float(c_exp)

            elif op in ('ConstantOfShape',):
                computed_inputs = [node.input[0]]
                if 'value' in attrs:
                    params['value'] = attrs['value']
                else:
                    params['value'] = 0.0

            elif op in ('Expand', 'Where', 'Equal', 'ScatterND', 'ArgMax',
                         'Min', 'Max'):
                # Grab all non-constant inputs
                computed_inputs = [i for i in node.input
                                   if _const(i) is None and i != '']
                # Store constant inputs
                for idx, i in enumerate(node.input):
                    c = _const(i)
                    if c is not None:
                        params[f'const_{idx}'] = c

            elif op == 'Shape':
                computed_inputs = [node.input[0]]

            elif op == 'Identity':
                computed_inputs = [node.input[0]]

            else:
                # Unknown op — best-effort
                computed_inputs = [i for i in node.input
                                   if _const(i) is None and i != ''
                                   and i not in init_names]

            # Constant folding: if all computed inputs are actually constants,
            # try to evaluate and store result as a constant
            if computed_inputs and all(_const(i) is not None for i in computed_inputs):
                folded = _try_fold_constant(op, computed_inputs, params, _const)
                if folded is not None:
                    constants[out_name] = folded
                    continue

            graph.nodes[out_name] = GraphNode(
                name=out_name,
                op_type=op,
                inputs=computed_inputs,
                params=params,
            )

        # Resolve input shape
        dims = graph._raw_input_dims
        has_conv = any(n.op_type in ('Conv', 'ConvTranspose')
                       for n in graph.nodes.values())
        if has_conv and len(dims) == 4:
            graph.input_shape = tuple(dims[1:])
        else:
            total = 1
            for d in dims:
                if d > 0:
                    total *= d
            graph.input_shape = (total,)

        graph._topological_sort()
        graph._infer_shapes()
        graph.fold_batchnorm()
        return graph

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def _topological_sort(self):
        """Kahn's algorithm."""
        in_degree = {name: 0 for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in self.nodes:
                    in_degree[node.name] += 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order = []

        successors = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in self.nodes:
                    successors[inp].append(node.name)

        while queue:
            name = queue.popleft()
            order.append(name)
            for succ in successors[name]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        assert len(order) == len(self.nodes), \
            f"Cycle detected: sorted {len(order)} of {len(self.nodes)} nodes"
        self.topo_order = order

    def fork_points(self):
        """Return set of node names whose output feeds multiple consumers."""
        ref_count = {}
        for node in self.nodes.values():
            for inp in node.inputs:
                ref_count[inp] = ref_count.get(inp, 0) + 1
        return {name for name, count in ref_count.items() if count > 1}

    def successors(self, name):
        """Return list of node names that consume this node's output."""
        return [n.name for n in self.nodes.values() if name in n.inputs]

    def relu_nodes(self):
        """Return set of Relu node names."""
        return {name for name, node in self.nodes.items() if node.op_type == 'Relu'}

    # ------------------------------------------------------------------
    # Optimizations
    # ------------------------------------------------------------------

    def fold_batchnorm(self):
        """Fold BatchNormalization into preceding Conv or Gemm."""
        to_remove = []
        for name in self.topo_order:
            node = self.nodes[name]
            if node.op_type != 'BatchNormalization':
                continue

            pred_name = node.inputs[0]
            if pred_name not in self.nodes:
                continue
            pred = self.nodes[pred_name]
            if pred.op_type not in ('Conv', 'Gemm'):
                continue

            scale = node.params['scale']
            bn_bias = node.params['bias']
            mean = node.params['mean']
            var = node.params['var']
            eps = node.params['epsilon']

            factor = scale / np.sqrt(var + eps)

            if pred.op_type == 'Conv':
                pred.params['kernel'] = pred.params['kernel'] * factor[:, None, None, None]
                pred.params['bias'] = factor * (pred.params['bias'] - mean) + bn_bias
            else:
                pred.params['W'] = pred.params['W'] * factor[:, None]
                pred.params['b'] = factor * (pred.params['b'] - mean) + bn_bias

            for other in self.nodes.values():
                other.inputs = [pred_name if inp == name else inp for inp in other.inputs]
            if self.output_name == name:
                self.output_name = pred_name

            to_remove.append(name)

        for name in to_remove:
            del self.nodes[name]

        if to_remove:
            self._topological_sort()

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def _infer_shapes(self):
        """Propagate shapes through the graph in topological order."""
        shapes = {self.input_name: self.input_shape}

        for name in self.topo_order:
            node = self.nodes[name]
            op = node.op_type

            if node.inputs and node.inputs[0] in shapes:
                inp_shape = shapes[node.inputs[0]]
            else:
                inp_shape = None

            if op == 'Conv':
                kernel = node.params['kernel']
                C_out = kernel.shape[0]
                kH, kW = kernel.shape[2], kernel.shape[3]
                sH, sW = node.params['stride']
                pH, pW = node.params['padding']
                if inp_shape is not None and len(inp_shape) == 3:
                    C_in, H_in, W_in = inp_shape
                elif inp_shape is not None:
                    # Input is flat — infer spatial dims from kernel
                    C_in = kernel.shape[1]
                    import math
                    spatial = inp_shape[0] // C_in if inp_shape[0] > 0 else 1
                    side = int(math.sqrt(spatial))
                    H_in = W_in = side
                    node.params['_inferred_input_shape'] = (C_in, H_in, W_in)
                else:
                    C_in = kernel.shape[1]
                    H_in = W_in = 1
                H_out = (H_in + 2 * pH - kH) // sH + 1
                W_out = (W_in + 2 * pW - kW) // sW + 1
                node.output_shape = (C_out, H_out, W_out)

            elif op == 'ConvTranspose':
                kernel = node.params['kernel']
                C_out = kernel.shape[1]  # ConvTranspose: (C_in, C_out, kH, kW)
                kH, kW = kernel.shape[2], kernel.shape[3]
                sH, sW = node.params['stride']
                pH, pW = node.params['padding']
                opH, opW = node.params.get('output_padding', (0, 0))
                if inp_shape is not None and len(inp_shape) == 3:
                    C_in, H_in, W_in = inp_shape
                else:
                    C_in = kernel.shape[0]
                    H_in = W_in = 1
                H_out = (H_in - 1) * sH - 2 * pH + kH + opH
                W_out = (W_in - 1) * sW - 2 * pW + kW + opW
                node.output_shape = (C_out, H_out, W_out)

            elif op == 'Gemm' or (op == 'MatMul' and 'W' in node.params):
                node.output_shape = (node.params['W'].shape[0],)

            elif op in ('Relu', 'LeakyRelu', 'Sigmoid', 'Sign', 'Neg',
                         'Clip', 'Sin', 'Cos', 'Softmax', 'Pow',
                         'BatchNormalization', 'Dropout', 'Identity'):
                node.output_shape = inp_shape

            elif op == 'Add':
                node.output_shape = inp_shape

            elif op in ('Sub', 'Mul', 'Div'):
                node.output_shape = inp_shape

            elif op in ('MaxPool', 'AveragePool'):
                kH, kW = node.params['kernel_shape']
                sH, sW = node.params['stride']
                pH, pW = node.params['padding']
                C, H_in, W_in = inp_shape
                H_out = (H_in + 2 * pH - kH) // sH + 1
                W_out = (W_in + 2 * pW - kW) // sW + 1
                node.output_shape = (C, H_out, W_out)

            elif op == 'Pad':
                # Best effort — flatten
                if inp_shape is not None:
                    total = 1
                    for d in inp_shape:
                        total *= d
                    node.output_shape = inp_shape  # approximate

            elif op == 'Concat':
                # Approximate: flatten
                if inp_shape is not None:
                    total = 0
                    for i_name in node.inputs:
                        if i_name in shapes and shapes[i_name] is not None:
                            s = 1
                            for d in shapes[i_name]:
                                s *= d
                            total += s
                    node.output_shape = (total,)

            elif op in ('Flatten', 'Squeeze', 'Unsqueeze', 'Reshape'):
                if inp_shape is not None:
                    total = 1
                    for d in inp_shape:
                        total *= d
                    node.output_shape = (total,)

            elif op == 'Transpose':
                # Just track as flattened
                if inp_shape is not None:
                    total = 1
                    for d in inp_shape:
                        total *= d
                    node.output_shape = (total,)

            elif op == 'Slice':
                # Approximate as same shape (proper tracking is hard)
                node.output_shape = inp_shape

            elif op == 'Gather':
                node.output_shape = inp_shape

            elif op in ('ReduceSum', 'ReduceMean'):
                if inp_shape is not None:
                    total = 1
                    for d in inp_shape:
                        total *= d
                    node.output_shape = (total,)

            elif op == 'Resize':
                if inp_shape is not None and 'scales' in node.params:
                    scales = node.params['scales']
                    # scales includes batch dim
                    if len(scales) == 4 and len(inp_shape) == 3:
                        C, H, W = inp_shape
                        node.output_shape = (C, int(H * scales[2]), int(W * scales[3]))
                    else:
                        node.output_shape = inp_shape
                else:
                    node.output_shape = inp_shape

            elif op == 'SplitOutput':
                node.output_shape = inp_shape

            elif op == 'Split':
                node.output_shape = inp_shape

            elif op == 'Shape':
                # Shape outputs a 1D tensor of dimension sizes
                if inp_shape is not None:
                    node.output_shape = (len(inp_shape) + 1,)  # includes batch
                else:
                    node.output_shape = (1,)

            else:
                # Best effort
                node.output_shape = inp_shape

            shapes[name] = node.output_shape

    def flat_size(self, name):
        """Return flat dimension for a tensor name."""
        if name == self.input_name:
            shape = self.input_shape
        else:
            shape = self.nodes[name].output_shape
        if shape is None:
            return 0
        total = 1
        for d in shape:
            total *= d
        return total
