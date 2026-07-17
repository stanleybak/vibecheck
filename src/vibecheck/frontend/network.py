# Ported from v1 vibecheck.network (standalone vc2), PRUNED to the
# loading/shape-inference surface: node dataclasses + infer_shape (the
# ND-shape oracle for every net family) + ComputeGraph.from_onnx /
# topological_sort. The v1 zonotope runtime (zonotope_propagate,
# gpu_layers, gpu_graph ~1.5k lines) and optimizer hooks are NOT
# copied -- vc2 has its own propagators (coverage-audited prune).
"""Network representation: ComputeGraph and GraphNode op subclasses."""

from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _prod(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _infer_conv_input_shape(flat_shape_or_size, kernel, transpose=False):
    """Infer (C, H, W) from a flat input size and conv kernel."""
    import math
    if isinstance(flat_shape_or_size, (tuple, list)):
        total = _prod(flat_shape_or_size)
    else:
        total = flat_shape_or_size
    C_in = kernel.shape[0] if transpose else kernel.shape[1]
    if total % C_in != 0:
        return (1, 1, total)
    spatial = total // C_in
    side = int(math.sqrt(spatial))
    if side * side == spatial:
        return (C_in, side, side)
    for h in range(side, 0, -1):  # h=1 always divides, so loop always returns
        if spatial % h == 0:
            return (C_in, h, spatial // h)














# ---------------------------------------------------------------------------
# GraphNode base
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Base class for all operations in the compute graph."""
    name: str
    op_type: str
    inputs: list
    params: dict = field(default_factory=dict)
    output_shape: tuple = None

    def infer_shape(self, input_shapes):
        """Default: same shape as first input."""
        if self.inputs and self.inputs[0] in input_shapes:
            self.output_shape = input_shapes[self.inputs[0]]



# ---------------------------------------------------------------------------
# Passthrough / shape-changing ops
# ---------------------------------------------------------------------------

class PassthroughNode(GraphNode):
    """Flatten, Dropout, Identity — data unchanged, shape flattened."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            if len(inp) > 2:
                self.output_shape = (inp[0], _prod(inp[1:]))
            else:
                self.output_shape = inp



class UnsqueezeNode(GraphNode):
    """Unsqueeze — insert size-1 dimensions. Data unchanged."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes', [])
            out = list(inp)
            for a in sorted(axes):
                if a < 0:
                    a = len(out) + 1 + a
                out.insert(a, 1)
            self.output_shape = tuple(out)



class SqueezeNode(GraphNode):
    """Squeeze — remove size-1 dimensions. Data unchanged."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes', None)
            if axes:
                out = [d for i, d in enumerate(inp) if i not in axes]
            else:
                out = [d for d in inp if d != 1]
            if not out:
                out = [1]
            self.output_shape = tuple(out)



class ReshapeNode(GraphNode):
    """Reshape — preserves data, changes shape metadata."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            target = self.params.get('shape')
            if target:
                total = _prod(inp)
                out = list(target)
                neg_idx = None
                known = 1
                for i, d in enumerate(out):
                    if d == -1:
                        neg_idx = i
                    elif d == 0:
                        if i < len(inp):
                            out[i] = inp[i]
                        known *= out[i]
                    else:
                        known *= d
                if neg_idx is not None and known > 0:
                    out[neg_idx] = total // known
                self.output_shape = tuple(out)
            else:
                self.output_shape = inp  # no target shape, keep as-is



class SplitOutputNode(PassthroughNode):
    """Placeholder for Split's secondary outputs."""
    def infer_shape(self, input_shapes):
        # Get shape from parent Split node's params
        parent_shape = input_shapes.get(self.inputs[0])
        if parent_shape is None:
            return
        # Find parent Split's split sizes
        # We need to look this up from the graph, but we only have input_shapes.
        # The parent's infer_shape set its output_shape to the first split.
        # For secondary outputs, we compute from the full input to Split.
        # Since we don't have the graph here, use the passthrough shape.
        # The correct shape will be set during zonotope propagation by SplitNode.
        if len(parent_shape) > 2:
            self.output_shape = (parent_shape[0], _prod(parent_shape[1:]))
        else:
            self.output_shape = parent_shape


class TransposeNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            perm = self.params.get('perm')
            if perm is None:
                perm = list(range(len(inp) - 1, -1, -1))  # reverse
            if len(perm) == len(inp):
                self.output_shape = tuple(inp[p] for p in perm)
            else:
                self.output_shape = inp



# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------

class ReluNode(GraphNode):
    pass


class LeakyReluNode(GraphNode):
    pass


class SigmoidNode(GraphNode):
    pass


class ClipNode(GraphNode):
    pass


class SignNode(GraphNode):
    pass


class SoftmaxNode(GraphNode):
    pass


class TanhNode(GraphNode):
    pass


class TrigNode(GraphNode):
    """Sin, Cos."""


class PWLNode(GraphNode):
    """Merged 1-D piecewise-linear lookup table:
        f(x) = bias + sum_i weights_i * ReLU(x - offsets_i),  applied elementwise.

    Created by ``onnx_optimizer.merge_relu_lookup_table`` from the expanded
    Unsqueeze->Sub->ReLU->MatMul->Add ReLU-sum encoding (sigmoid/sin/cos PWL
    approximations in the ml4acopf linear-surrogate nets). The basic-zono path
    only point-evaluates (like Sin/Cos); the tight forward-zonotope bound lives in
    ``verify_zono_bnb`` via ``nl_pwl.PWLRelax`` (sound affine band)."""



class PowNode(GraphNode):
    pass


class FloorNode(GraphNode):
    pass


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class NegNode(GraphNode):
    pass


class AddNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            bias = self.params.get('bias')
            if bias is not None and isinstance(bias, np.ndarray):
                try:
                    out = np.broadcast_shapes(inp, bias.shape)
                    self.output_shape = out
                    return
                except ValueError:
                    pass
            self.output_shape = inp



class SubNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            const = self.params.get('sub_val')
            if const is None:
                const = self.params.get('bias')
            if const is not None and isinstance(const, np.ndarray):
                try:
                    self.output_shape = np.broadcast_shapes(inp, const.shape)
                    return
                except ValueError:
                    pass
            self.output_shape = inp



class MulNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            scale = self.params.get('scale')
            if scale is not None and isinstance(scale, np.ndarray):
                try:
                    self.output_shape = np.broadcast_shapes(inp, scale.shape)
                    return
                except ValueError:
                    pass
            self.output_shape = inp



class DivNode(GraphNode):
    pass


# ---------------------------------------------------------------------------
# Linear ops: Conv, ConvTranspose, Gemm/MatMul
# ---------------------------------------------------------------------------

class ConvNode(GraphNode):
    def infer_shape(self, input_shapes):
        kernel = self.params['kernel']
        C_out = kernel.shape[0]
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None

        if kernel.ndim == 3:
            # 1D conv: kernel (C_out, C_in, kW)
            kW = kernel.shape[2]
            sW = self.params['stride'][0]
            pW = self.params['padding'][0]
            if inp_shape is not None and len(inp_shape) == 3:
                _, C_in, W_in = inp_shape
            elif inp_shape is not None:
                W_in = _prod(inp_shape) // kernel.shape[1]
            else:
                W_in = 1
            W_out = (W_in + 2 * pW - kW) // sW + 1
            self.output_shape = (1, C_out, W_out)
        else:
            # 2D conv: kernel (C_out, C_in, kH, kW)
            kH, kW = kernel.shape[2], kernel.shape[3]
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            if inp_shape is not None and len(inp_shape) == 4:
                _, C_in, H_in, W_in = inp_shape
            elif inp_shape is not None and len(inp_shape) == 3:
                C_in, H_in, W_in = inp_shape
            elif inp_shape is not None:
                C_in = kernel.shape[1]
                import math
                total = _prod(inp_shape)
                spatial = total // C_in if total > 0 else 1
                side = int(math.sqrt(spatial))
                H_in = W_in = side
            else:
                H_in = W_in = 1
            H_out = (H_in + 2 * pH - kH) // sH + 1
            W_out = (W_in + 2 * pW - kW) // sW + 1
            self.output_shape = (1, C_out, H_out, W_out)

    def precache_conv_layer(self, graph):
        """Pre-build and cache the conv layer tuple with torch tensors.

        Called during graph loading (after shape inference) so that
        zonotope_propagate pays no tensor-creation overhead.
        """
        inp_name = self.inputs[0]
        inp_shape = (graph.nodes[inp_name].output_shape
                     if inp_name in graph.nodes else graph.input_shape)
        n_elems = _prod(inp_shape)
        spatial = self._spatial_shape(graph, n_elems)
        kernel = self.params['kernel']
        stride = self.params['stride']
        padding = self.params['padding']
        if kernel.ndim == 3:
            kernel = kernel[:, :, np.newaxis, :]
            stride = (1, stride[0])
            padding = (0, padding[0])
        torch_dt = torch.float32 if graph.dtype == np.float32 else torch.float64
        cache_key = '_torch_kernel_f32' if torch_dt == torch.float32 else '_torch_kernel'
        bias_key = '_torch_bias_f32' if torch_dt == torch.float32 else '_torch_bias'
        self._conv_layer = (kernel, self.params['bias'], {
            'input_shape': spatial,
            'stride': stride,
            'padding': padding,
            cache_key: torch.tensor(kernel, dtype=torch_dt),
            bias_key: torch.tensor(self.params['bias'], dtype=torch_dt),
        })


    def _spatial_shape(self, graph, n_elems):
        """Get (C, H, W) for torch conv2d. For 1D conv, returns (C, 1, W)."""
        inp_name = self.inputs[0]
        inp_shape = (graph.nodes[inp_name].output_shape
                     if inp_name in graph.nodes else graph.input_shape)
        kernel = self.params['kernel']
        if len(inp_shape) == 4:
            return inp_shape[1:]  # (C, H, W)
        if len(inp_shape) == 3 and kernel.ndim == 3:
            # 1D: (1, C, W) -> unsqueeze to (C, 1, W) for conv2d
            return (inp_shape[1], 1, inp_shape[2])
        if len(inp_shape) == 3:
            return inp_shape
        return _infer_conv_input_shape(n_elems, kernel)


class ConvTransposeNode(GraphNode):
    def infer_shape(self, input_shapes):
        kernel = self.params['kernel']
        C_out = kernel.shape[1]
        kH, kW = kernel.shape[2], kernel.shape[3]
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        opH, opW = self.params.get('output_padding', (0, 0))
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape is not None and len(inp_shape) == 4:
            _, C_in, H_in, W_in = inp_shape
        elif inp_shape is not None and len(inp_shape) == 3:
            C_in, H_in, W_in = inp_shape
        else:
            C_in = kernel.shape[0]
            H_in = W_in = 1
        H_out = (H_in - 1) * sH - 2 * pH + kH + opH
        W_out = (W_in - 1) * sW - 2 * pW + kW + opW
        self.output_shape = (1, C_out, H_out, W_out)



class GemmNode(GraphNode):
    """Gemm and MatMul with constant weight matrix."""
    def infer_shape(self, input_shapes):
        W = self.params['W']
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if W.ndim == 1 and inp is not None:
            # (..., K) @ (K,) -> (...)
            self.output_shape = inp[:-1] if len(inp) > 1 else (1,)
        elif W.ndim == 2 and inp is not None and len(inp) > 2:
            # ND matmul: (..., K) @ (K, M) -> (..., M) where W stored as (M, K)
            if inp[-1] == W.shape[1]:
                self.output_shape = inp[:-1] + (W.shape[0],)
            else:
                self.output_shape = (1, W.shape[0])
        elif W.ndim == 2:
            self.output_shape = (1, W.shape[0])
        else:
            self.output_shape = (1, _prod(W.shape[:-1])) if W.ndim > 0 else (1,)



class MatMulBilinearNode(GraphNode):
    """MatMul with two computed inputs (no constant weight)."""
    def infer_shape(self, input_shapes):
        sa = input_shapes.get(self.inputs[0])
        sb = input_shapes.get(self.inputs[1])
        if sa is None or sb is None:
            return
        # (..., M, K) @ (..., K, N) → (..., M, N). Broadcast the
        # leading dims.
        if len(sa) < 2 or len(sb) < 2:
            return
        M, K_a = sa[-2], sa[-1]
        K_b, N = sb[-2], sb[-1]
        if K_a != K_b:
            # Inner dims don't match standard matmul rule. Leave shape
            # unset and let downstream ops/tests deal with it.
            return
        # Broadcast leading dims (simple case: equal or one is empty).
        lead_a = sa[:-2]; lead_b = sb[:-2]
        if lead_a == lead_b or not lead_a:
            lead = lead_b
        elif not lead_b:
            lead = lead_a
        else:
            # Conservative broadcast: match lengths.
            lead = tuple(max(a, b) for a, b in zip(lead_a, lead_b))
        self.output_shape = tuple(lead) + (M, N)



# ---------------------------------------------------------------------------
# BatchNorm (when not folded into preceding Conv/Gemm)
# ---------------------------------------------------------------------------

class BatchNormNode(GraphNode):
    pass


# ---------------------------------------------------------------------------
# Pooling / Pad (concrete execution only)
# ---------------------------------------------------------------------------

class MaxPoolNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape and len(inp_shape) >= 3:
            kH, kW = self.params['kernel_shape']
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            # Handle both (C,H,W) and (1,C,H,W)
            if len(inp_shape) == 4:
                _, C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)
            else:
                C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)



class AveragePoolNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape and len(inp_shape) >= 3:
            kH, kW = self.params['kernel_shape']
            sH, sW = self.params['stride']
            pH, pW = self.params['padding']
            if len(inp_shape) == 4:
                _, C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)
            else:
                C, H_in, W_in = inp_shape
                self.output_shape = (1, C, (H_in+2*pH-kH)//sH+1, (W_in+2*pW-kW)//sW+1)



class PadNode(GraphNode):
    pass


# ---------------------------------------------------------------------------
# Structure ops: Concat, Split, Slice, Gather
# ---------------------------------------------------------------------------

class ConcatNode(GraphNode):
    def infer_shape(self, input_shapes):
        # True N-D inference: out = live shape with dim[axis] summed over
        # ALL inputs (live + const). The old flat `(total,)` ignored const
        # inputs entirely and erased rank — downstream shape-sensitive ops
        # (Transpose on vit's CLS-token concat) then saw a bogus shape.
        live = [input_shapes.get(i) for i in self.inputs]
        axis = self.params.get('axis', 0)
        consts = self.params.get('const_inputs') or []
        if live and all(s is not None for s in live):
            base = live[0]
            a = axis if axis >= 0 else len(base) + axis
            if 0 <= a < len(base):
                total_ax = sum(s[a] for s in live)
                for _pos, arr in consts:
                    ash = np.asarray(arr).shape
                    if len(ash) == len(base):
                        total_ax += ash[a]
                out = list(base)
                out[a] = total_ax
                self.output_shape = tuple(out)
                return
        total = sum(_prod(s) for s in live if s is not None)
        for _pos, arr in consts:
            total += int(np.asarray(arr).size)
        if total > 0:
            self.output_shape = (total,)



class SplitNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            split_sizes = self.params.get('split')
            axis = self.params.get('axis', 0)
            if split_sizes and axis < len(inp):
                out = list(inp)
                out[axis] = split_sizes[0]
                self.output_shape = tuple(out)
            else:
                self.output_shape = inp



class SliceNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = self._sliced_shape(inp)


    def _sliced_shape(self, inp_shape):
        axes = self.params.get('axes', [0])
        starts = self.params.get('starts', [0])
        ends = self.params.get('ends', [None])
        out = list(inp_shape)
        for ax, s, e in zip(axes, starts, ends):
            a = ax if ax >= 0 else len(inp_shape) + ax
            if a >= len(out):
                continue
            dim = out[a]
            if s < 0: s = dim + s
            if e is None or e > dim: e = dim
            if e < 0: e = dim + e
            out[a] = e - s
        return tuple(out)


class GatherNode(GraphNode):
    def infer_shape(self, input_shapes):
        indices = self.params.get('indices', None)
        inp_shape = (input_shapes.get(self.inputs[0])
                     if self.inputs else None)
        if indices is None:
            if inp_shape is not None:
                self.output_shape = inp_shape
            return
        axis = int(self.params.get('axis', 0))
        if inp_shape is not None:
            # ONNX semantics: output = input.shape[:axis] +
            # indices.shape + input.shape[axis+1:]. 0-D indices drop
            # the gather axis.
            a = axis if axis >= 0 else len(inp_shape) + axis
            if 0 <= a < len(inp_shape):
                idx_shape = tuple(indices.shape) if indices.ndim > 0 else ()
                out = list(inp_shape[:a]) + list(idx_shape) + \
                    list(inp_shape[a + 1:])
                self.output_shape = tuple(out) if out else (1,)
                return
        # Fallback (input shape unknown): flat indices count.
        self.output_shape = (len(indices.flatten()),)



# ---------------------------------------------------------------------------
# Reduce ops
# ---------------------------------------------------------------------------

class ReduceNode(GraphNode):
    """ReduceSum and ReduceMean."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            axes = self.params.get('axes')
            keepdims = self.params.get('keepdims', 1)
            if axes:
                out = list(inp)
                for a in sorted(axes, reverse=True):
                    if a < 0:
                        a = len(out) + a
                    if keepdims:
                        out[a] = 1
                    else:
                        out.pop(a)
                self.output_shape = tuple(out) if out else (1,)
            else:
                # Reduce all axes
                if keepdims:
                    self.output_shape = tuple(1 for _ in inp)
                else:
                    self.output_shape = (1,)



# ---------------------------------------------------------------------------
# Other ops
# ---------------------------------------------------------------------------

class ResizeNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape is not None and 'scales' in self.params:
            scales = self.params['scales']
            if len(scales) == len(inp_shape):
                self.output_shape = tuple(
                    int(d * s) for d, s in zip(inp_shape, scales))
            else:
                self.output_shape = inp_shape
        elif inp_shape is not None:
            self.output_shape = inp_shape



class ConstantOfShapeNode(GraphNode):
    pass


class ShapeOpNode(GraphNode):
    """Shape op — outputs dimension sizes."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = (len(inp),)
        else:
            self.output_shape = (1,)



class MiscNode(GraphNode):
    """Registry placeholder for ops with no sound zonotope handler (Cast, Equal,
    Where, Expand, ScatterND, ArgMax, Min, Max). Graph build + shape inference work
    (so constant-folding can still eliminate them), but PROPAGATION raises: passing
    input[0] through ignores the operation entirely (e.g. Min/Max/Where), which is
    unsound. A loud NotImplementedError is more informative than a wrong bound — if
    one of these is actually needed, give it a real handler (Min/Max already have one
    via onnx_optimizer.min_max_to_relu)."""


# ---------------------------------------------------------------------------
# Op registry: ONNX op_type string -> GraphNode subclass
# ---------------------------------------------------------------------------

OP_REGISTRY = {
    # Passthrough
    'Flatten': PassthroughNode,
    'Squeeze': SqueezeNode,
    'Unsqueeze': UnsqueezeNode,
    'Reshape': ReshapeNode,
    'Dropout': PassthroughNode,
    'Identity': PassthroughNode,
    'SplitOutput': SplitOutputNode,
    # Transpose (actual permutation)
    'Transpose': TransposeNode,
    # Activations
    'Relu': ReluNode,
    'LeakyRelu': LeakyReluNode,
    'Sigmoid': SigmoidNode,
    'Clip': ClipNode,
    'Sign': SignNode,
    'Softmax': SoftmaxNode,
    'Tanh': TanhNode,
    'Sin': TrigNode,
    'Cos': TrigNode,
    'Pow': PowNode,
    'Floor': FloorNode,
    'PWLLookup': PWLNode,
    # Arithmetic
    'Neg': NegNode,
    'Add': AddNode,
    'Sub': SubNode,
    'Mul': MulNode,
    'Div': DivNode,
    # Linear
    'Conv': ConvNode,
    'ConvTranspose': ConvTransposeNode,
    'Gemm': GemmNode,
    'MatMul': GemmNode,  # overridden to MatMulBilinearNode when no weight
    # BatchNorm
    'BatchNormalization': BatchNormNode,
    # Pooling
    'MaxPool': MaxPoolNode,
    'AveragePool': AveragePoolNode,
    'Pad': PadNode,
    # Structure
    'Concat': ConcatNode,
    'Split': SplitNode,
    'Slice': SliceNode,
    'Gather': GatherNode,
    # Reduce
    'ReduceSum': ReduceNode,
    'ReduceMean': ReduceNode,
    # Other
    'Resize': ResizeNode,
    'Upsample': ResizeNode,
    'ConstantOfShape': ConstantOfShapeNode,
    'Shape': ShapeOpNode,
    'Cast': MiscNode,
    'Equal': MiscNode,
    'Where': MiscNode,
    'Expand': MiscNode,
    'ScatterND': MiscNode,
    'ArgMax': MiscNode,
    'Min': MiscNode,
    'Max': MiscNode,
}


# ---------------------------------------------------------------------------
# ComputeGraph
# ---------------------------------------------------------------------------

class ComputeGraph:
    """DAG of operations loaded from ONNX.

    Nodes are keyed by their output tensor name. Traversal order is
    topological (Kahn's algorithm), cached after construction.
    """

    def __init__(self, dtype=np.float32):
        self.nodes = {}          # name -> GraphNode
        self.input_name = None
        self.output_name = None
        self.input_shape = None  # without batch dim
        self.topo_order = []
        self.dtype = dtype       # numpy dtype for computation

    @classmethod
    def from_onnx(cls, onnx_path, dtype=np.float32):
        """Load an ONNX model into a ComputeGraph."""
        from .onnx_loader import load_onnx
        return load_onnx(onnx_path, dtype=dtype)


    def topological_sort(self):
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









