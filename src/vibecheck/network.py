"""Network representation: ComputeGraph and GraphNode op subclasses."""

from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .zonotope import DenseZonotope


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
    for h in range(side, 0, -1):
        if spatial % h == 0:
            return (C_in, h, spatial // h)
    return (C_in, spatial, 1)


def _find_shared_gens(name_a, name_b, graph, gen_count):
    """Find the generator count at the fork point of two merging branches."""
    forks = graph.fork_points()

    def _ancestors(name):
        visited, stack, seen = [], [name], set()
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
    for anc in anc_a:
        if anc in anc_b_set and anc in forks:
            return gen_count[anc]
    if graph.input_name in anc_b_set:
        return gen_count[graph.input_name]
    return 0


def _get_spatial_shape(node, graph, actual_len, kernel=None, transpose=False):
    """Resolve (C, H, W) input shape for torch spatial ops (no batch dim)."""
    inp_name = node.inputs[0]
    inp_shape = (graph.nodes[inp_name].output_shape
                 if inp_name in graph.nodes else graph.input_shape)
    # Strip batch if present: (1, C, H, W) -> (C, H, W)
    if len(inp_shape) == 4 and inp_shape[0] == 1:
        inp_shape = inp_shape[1:]
    if len(inp_shape) == 3 and _prod(inp_shape) == actual_len:
        return inp_shape
    if kernel is not None:
        return _infer_conv_input_shape(actual_len, kernel, transpose)
    return inp_shape


def _point_zono(center):
    """Create a zero-generator zonotope from a center value."""
    return DenseZonotope(center, np.zeros((len(center), 0)))


def _require_point(node, z):
    """Raise if zonotope has generators (op only supports concrete execution)."""
    if z.generators.shape[1] > 0:
        raise NotImplementedError(
            f"Zonotope propagation not implemented for op "
            f"'{node.op_type}' (node '{node.name}', "
            f"{z.generators.shape[1]} generators)")


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        """Default: raise for unknown ops."""
        raise NotImplementedError(
            f"Op '{self.op_type}' not supported (node '{self.name}')")


# ---------------------------------------------------------------------------
# Passthrough ops: Flatten, Squeeze, Unsqueeze, Reshape, Dropout, Identity
# ---------------------------------------------------------------------------

class PassthroughNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            # Flatten keeps batch dim: (1, C, H, W) -> (1, C*H*W)
            if len(inp) > 2:
                self.output_shape = (inp[0], _prod(inp[1:]))
            else:
                self.output_shape = inp

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        zono_state[self.name] = get_input(self.inputs[0])


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)
        perm = self.params.get('perm')
        if perm is None:
            perm = list(range(len(inp_shape) - 1, -1, -1))

        if len(perm) != len(inp_shape) or len(inp_shape) < 2:
            zono_state[self.name] = z
            return

        center = np.transpose(z.center.reshape(inp_shape), perm).flatten()
        n_gens = z.generators.shape[1]
        if n_gens > 0:
            g_nd = z.generators.reshape(*inp_shape, n_gens)
            g_nd = np.transpose(g_nd, list(perm) + [len(perm)])
            gens = g_nd.reshape(-1, n_gens)
        else:
            gens = np.zeros((len(center), 0))
        zono_state[self.name] = DenseZonotope(center, gens)


# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------

class ReluNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        z_lo, z_hi = z.bounds()
        z.apply_relu(z_lo, z_hi, relu_type)
        zono_state[self.name] = z


class LeakyReluNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        alpha = self.params.get('alpha', 0.01)
        center = np.where(z.center >= 0, z.center, alpha * z.center)
        zono_state[self.name] = _point_zono(center)


class SigmoidNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(
            1.0 / (1.0 + np.exp(-z.center)))


class ClipNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        center = z.center.copy()
        if 'min' in self.params:
            center = np.maximum(center, self.params['min'])
        if 'max' in self.params:
            center = np.minimum(center, self.params['max'])
        zono_state[self.name] = _point_zono(center)


class SignNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(np.sign(z.center))


class SoftmaxNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        e = np.exp(z.center - z.center.max())
        zono_state[self.name] = _point_zono(e / e.sum())


class TanhNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = _point_zono(np.tanh(z.center))


class TrigNode(GraphNode):
    """Sin, Cos."""
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        fn = np.sin if self.op_type == 'Sin' else np.cos
        zono_state[self.name] = _point_zono(fn(z.center))


class PowNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        exp = self.params.get('exponent', 2.0)
        zono_state[self.name] = _point_zono(z.center ** exp)


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class NegNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        z.center = -z.center
        z.generators = -z.generators
        zono_state[self.name] = z


class AddNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        if len(self.inputs) == 2 and self.inputs[1] in graph.nodes:
            z_a = get_input(self.inputs[0])
            z_b = get_input(self.inputs[1])
            shared = _find_shared_gens(
                self.inputs[0], self.inputs[1], graph, gen_count)
            zono_state[self.name] = z_a.add(z_b, shared)
        else:
            z = get_input(self.inputs[0])
            bias = self.params.get('bias', 0)
            z.center = z.center + bias
            zono_state[self.name] = z


class SubNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        if self.params.get('negate'):
            bias = self.params.get('bias', 0)
            z.center = -z.center + bias
            z.generators = -z.generators
        else:
            sub_val = self.params.get('sub_val', 0)
            z.center = z.center - sub_val
        zono_state[self.name] = z


class MulNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        if 'scale' in self.params:
            z = get_input(self.inputs[0])
            s = self.params['scale']
            z.center = z.center * s
            z.generators = z.generators * s[:, None]
            zono_state[self.name] = z
        else:
            # Bilinear mul — point only
            z = get_input(self.inputs[0])
            _require_point(self, z)
            z_b = get_input(self.inputs[1])
            _require_point(self, z_b)
            zono_state[self.name] = _point_zono(z.center * z_b.center)


class DivNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        if 'scale' in self.params:
            s = self.params['scale']  # already inverted
            z.center = z.center * s
            z.generators = z.generators * s[:, None]
            zono_state[self.name] = z
        else:
            # Both inputs computed — point only
            z_b = get_input(self.inputs[1])
            _require_point(self, z)
            _require_point(self, z_b)
            zono_state[self.name] = _point_zono(z.center / z_b.center)


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        n_gens, n_elems = z.generators.shape[1], len(z.center)
        if n_gens * n_elems > 5_000_000:
            raise NotImplementedError(
                f"Conv generator matrix too large ({n_gens} × {n_elems} > 5M) "
                f"at node '{self.name}'")
        # Extract (C, H, W) for torch conv2d
        spatial = self._spatial_shape(graph, n_elems)
        kernel = self.params['kernel']
        stride = self.params['stride']
        padding = self.params['padding']
        if kernel.ndim == 3:
            # 1D conv -> unsqueeze to 2D for conv2d
            kernel = kernel[:, :, np.newaxis, :]
            stride = (1, stride[0])
            padding = (0, padding[0])
        conv_params = {
            'input_shape': spatial,
            'stride': stride,
            'padding': padding,
        }
        layer = (kernel, self.params['bias'], conv_params)
        z.propagate_linear(layer)
        zono_state[self.name] = z

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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        import torch
        import torch.nn.functional as F
        z = get_input(self.inputs[0])
        _require_point(self, z)
        inp_shape = _get_spatial_shape(
            self, graph, len(z.center), self.params['kernel'], transpose=True)
        center_4d = torch.tensor(z.center, dtype=torch.float64).reshape(
            1, *inp_shape)
        k = torch.tensor(self.params['kernel'], dtype=torch.float64)
        b = torch.tensor(self.params['bias'], dtype=torch.float64)
        out = F.conv_transpose2d(
            center_4d, k, bias=b,
            stride=self.params['stride'],
            padding=self.params['padding'],
            output_padding=self.params.get('output_padding', (0, 0)))
        zono_state[self.name] = _point_zono(out.flatten().numpy())


class GemmNode(GraphNode):
    """Gemm and MatMul with constant weight matrix."""
    def infer_shape(self, input_shapes):
        self.output_shape = (1, self.params['W'].shape[0])

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        W = self.params['W']
        if W.shape[1] != len(z.center):
            raise NotImplementedError(
                f"Gemm/MatMul dimension mismatch: W is {W.shape} but "
                f"input has {len(z.center)} elements at node '{self.name}'")
        z.propagate_linear((W, self.params['b']))
        zono_state[self.name] = z


class MatMulBilinearNode(GraphNode):
    """MatMul with two computed inputs (no constant weight)."""
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z_a = get_input(self.inputs[0])
        z_b = get_input(self.inputs[1])
        _require_point(self, z_a)
        _require_point(self, z_b)
        zono_state[self.name] = _point_zono(z_a.center * z_b.center)


# ---------------------------------------------------------------------------
# BatchNorm (when not folded into preceding Conv/Gemm)
# ---------------------------------------------------------------------------

class BatchNormNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        scale = self.params['scale']
        bn_bias = self.params['bias']
        mean = self.params['mean']
        var = self.params['var']
        eps = self.params['epsilon']
        factor = scale / np.sqrt(var + eps)
        offset = -factor * mean + bn_bias
        # Broadcast per-channel to flat vector
        if len(factor) < len(z.center):
            C = len(factor)
            spatial = len(z.center) // C
            factor = np.repeat(factor, spatial)
            offset = np.repeat(offset, spatial)
        z.center = factor * z.center + offset
        z.generators = factor[:, None] * z.generators
        zono_state[self.name] = z


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        import torch
        import torch.nn.functional as F
        z = get_input(self.inputs[0])
        _require_point(self, z)
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        c4d = torch.tensor(z.center, dtype=torch.float64).reshape(1, *inp_shape)
        kH, kW = self.params['kernel_shape']
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        out = F.max_pool2d(c4d, kernel_size=(kH, kW), stride=(sH, sW),
                           padding=(pH, pW))
        zono_state[self.name] = _point_zono(out.flatten().numpy())


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        import torch
        import torch.nn.functional as F
        z = get_input(self.inputs[0])
        _require_point(self, z)
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        c4d = torch.tensor(z.center, dtype=torch.float64).reshape(1, *inp_shape)
        kH, kW = self.params['kernel_shape']
        sH, sW = self.params['stride']
        pH, pW = self.params['padding']
        out = F.avg_pool2d(c4d, kernel_size=(kH, kW), stride=(sH, sW),
                           padding=(pH, pW))
        zono_state[self.name] = _point_zono(out.flatten().numpy())


class PadNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        import torch
        import torch.nn.functional as F
        z = get_input(self.inputs[0])
        _require_point(self, z)
        pads = self.params.get('pads', [])
        val = self.params.get('constant_value', 0.0)
        inp_shape = _get_spatial_shape(self, graph, len(z.center))
        if pads and len(inp_shape) == 3:
            c4d = torch.tensor(z.center, dtype=torch.float64).reshape(
                1, *inp_shape)
            n = len(pads) // 2
            if n >= 4:
                torch_pad = (pads[3], pads[3 + n], pads[2], pads[2 + n])
            elif n >= 2:
                torch_pad = (pads[1], pads[1 + n], pads[0], pads[0 + n])
            else:
                zono_state[self.name] = z
                return
            out = F.pad(c4d, torch_pad, value=val)
            zono_state[self.name] = _point_zono(out.flatten().numpy())
        else:
            zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Structure ops: Concat, Split, Slice, Gather
# ---------------------------------------------------------------------------

class ConcatNode(GraphNode):
    def infer_shape(self, input_shapes):
        total = 0
        for i_name in self.inputs:
            if i_name in input_shapes and input_shapes[i_name] is not None:
                total += _prod(input_shapes[i_name])
        if total > 0:
            self.output_shape = (total,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        parts = [get_input(inp) for inp in self.inputs]
        max_k = max(p.generators.shape[1] for p in parts)
        centers, gens = [], []
        for p in parts:
            centers.append(p.center)
            k = p.generators.shape[1]
            if k < max_k:
                pad = np.zeros((p.generators.shape[0], max_k - k))
                gens.append(np.hstack([p.generators, pad]))
            else:
                gens.append(p.generators)
        zono_state[self.name] = DenseZonotope(
            np.concatenate(centers), np.vstack(gens))


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

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        split_sizes = self.params.get('split', None)
        if not split_sizes:
            zono_state[self.name] = z
            return

        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)
        axis = self.params.get('axis', 0)

        # Split along the axis in ND, then flatten each part
        parts_center = []
        parts_gens = []
        offset = 0
        for s in split_sizes:
            slices = [slice(None)] * len(inp_shape)
            slices[axis] = slice(offset, offset + s)
            slices = tuple(slices)
            c_part = z.center.reshape(inp_shape)[slices].flatten()
            parts_center.append(c_part)
            n_gens = z.generators.shape[1]
            if n_gens > 0:
                g_part = z.generators.reshape(*inp_shape, n_gens)[slices].reshape(-1, n_gens)
            else:
                g_part = np.zeros((len(c_part), 0))
            parts_gens.append(g_part)
            offset += s

        # First part is this node
        zono_state[self.name] = DenseZonotope(parts_center[0], parts_gens[0])

        # Set SplitOutput children
        for succ_name, succ_node in graph.nodes.items():
            if (succ_node.op_type == 'SplitOutput'
                    and succ_node.inputs[0] == self.name):
                idx = succ_node.params['index']
                if idx < len(parts_center):
                    zono_state[succ_name] = DenseZonotope(
                        parts_center[idx], parts_gens[idx])
                    gen_count[succ_name] = z.generators.shape[1]


class SliceNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = self._sliced_shape(inp)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        inp_shape = (graph.nodes[self.inputs[0]].output_shape
                     if self.inputs[0] in graph.nodes else graph.input_shape)

        if inp_shape is not None and len(inp_shape) > 1:
            # ND slice
            axes = self.params.get('axes', [0])
            starts = self.params.get('starts', [0])
            ends = self.params.get('ends', [None])
            slices = [slice(None)] * len(inp_shape)
            for ax, s, e in zip(axes, starts, ends):
                a = ax if ax >= 0 else len(inp_shape) + ax
                if a >= len(inp_shape):
                    continue
                dim = inp_shape[a]
                if s < 0:
                    s = dim + s
                if e is None or e > dim:
                    e = dim
                if e < 0:
                    e = dim + e
                slices[a] = slice(s, e)
            slices = tuple(slices)
            center = z.center.reshape(inp_shape)[slices].flatten()
            n_gens = z.generators.shape[1]
            if n_gens > 0:
                g_nd = z.generators.reshape(*inp_shape, n_gens)
                gens = g_nd[slices].reshape(-1, n_gens)
            else:
                gens = np.zeros((len(center), 0))
            zono_state[self.name] = DenseZonotope(center, gens)
        else:
            # 1D fallback
            n = len(z.center)
            s = self.params.get('starts', [0])[0]
            e = self.params.get('ends', [n])[0]
            if e > n: e = n
            if s < 0: s = n + s
            if e < 0: e = n + e
            zono_state[self.name] = DenseZonotope(
                z.center[s:e], z.generators[s:e, :])

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
        if indices is not None:
            self.output_shape = (len(indices.flatten()),)
        elif self.inputs and self.inputs[0] in input_shapes:
            self.output_shape = input_shapes[self.inputs[0]]

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        indices = self.params.get('indices', None)
        if indices is not None:
            idx = indices.flatten().astype(int)
            zono_state[self.name] = DenseZonotope(
                z.center[idx], z.generators[idx, :])
        else:
            zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Reduce ops
# ---------------------------------------------------------------------------

class ReduceNode(GraphNode):
    """ReduceSum and ReduceMean."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = (_prod(inp),)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        if self.op_type == 'ReduceSum':
            zono_state[self.name] = DenseZonotope(
                np.array([z.center.sum()]),
                z.generators.sum(axis=0, keepdims=True))
        else:
            zono_state[self.name] = DenseZonotope(
                np.array([z.center.mean()]),
                z.generators.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------------------
# Other ops
# ---------------------------------------------------------------------------

class ResizeNode(GraphNode):
    def infer_shape(self, input_shapes):
        inp_shape = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp_shape is not None and 'scales' in self.params:
            scales = self.params['scales']
            if len(scales) == 4 and len(inp_shape) == 3:
                C, H, W = inp_shape
                self.output_shape = (C, int(H * scales[2]), int(W * scales[3]))
            else:
                self.output_shape = inp_shape
        elif inp_shape is not None:
            self.output_shape = inp_shape

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = z  # approximate


class ConstantOfShapeNode(GraphNode):
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        val = self.params.get('value', 0.0)
        n = max(1, len(z.center))
        zono_state[self.name] = _point_zono(np.full(n, val))


class ShapeOpNode(GraphNode):
    """Shape op — outputs dimension sizes."""
    def infer_shape(self, input_shapes):
        inp = input_shapes.get(self.inputs[0]) if self.inputs else None
        if inp is not None:
            self.output_shape = (len(inp),)
        else:
            self.output_shape = (1,)

    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = z


class MiscNode(GraphNode):
    """Fallback for Cast, Equal, Where, Expand, ScatterND, ArgMax, Min, Max."""
    def zonotope_propagate(self, zono_state, gen_count, get_input,
                           relu_type, graph):
        z = get_input(self.inputs[0])
        _require_point(self, z)
        zono_state[self.name] = z


# ---------------------------------------------------------------------------
# Op registry: ONNX op_type string -> GraphNode subclass
# ---------------------------------------------------------------------------

OP_REGISTRY = {
    # Passthrough
    'Flatten': PassthroughNode,
    'Squeeze': PassthroughNode,
    'Unsqueeze': PassthroughNode,
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

    def __init__(self):
        self.nodes = {}          # name -> GraphNode
        self.input_name = None
        self.output_name = None
        self.input_shape = None  # without batch dim
        self.topo_order = []

    @classmethod
    def from_onnx(cls, onnx_path):
        """Load an ONNX model into a ComputeGraph."""
        from .onnx_loader import load_onnx
        return load_onnx(onnx_path)

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

    def fork_points(self):
        """Return set of node names whose output feeds multiple consumers."""
        ref_count = {}
        for node in self.nodes.values():
            for inp in node.inputs:
                ref_count[inp] = ref_count.get(inp, 0) + 1
        return {name for name, count in ref_count.items() if count > 1}

    def predecessors(self, name):
        if name in self.nodes:
            return list(self.nodes[name].inputs)
        return []

    def successors(self, name):
        return [n.name for n in self.nodes.values() if name in n.inputs]

    def relu_nodes(self):
        return {name for name, node in self.nodes.items()
                if node.op_type == 'Relu'}

    def flat_size(self, name):
        if name == self.input_name:
            shape = self.input_shape
        else:
            shape = self.nodes[name].output_shape
        if shape is None:
            return 0
        return _prod(shape)

    def __repr__(self):
        return (f"ComputeGraph(input={self.input_name}, output={self.output_name}, "
                f"nodes={len(self.nodes)}, input_shape={self.input_shape})")

    def __str__(self):
        forks = self.fork_points()
        topo_idx = {name: i for i, name in enumerate(self.topo_order)}
        succ_map = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in succ_map:
                    succ_map[inp].append(node.name)

        lines = []
        lines.append(f"ComputeGraph: {len(self.nodes)} ops, "
                      f"input={self.input_shape}")
        lines.append(f"  input: {self.input_name}  shape={self.input_shape}")
        lines.append("")

        idx_w = len(str(len(self.topo_order)))
        for i, name in enumerate(self.topo_order):
            node = self.nodes[name]
            shape_str = str(node.output_shape) if node.output_shape else '?'
            flat = _prod(node.output_shape) if node.output_shape else 0

            pred_indices = []
            for inp in node.inputs:
                if inp in topo_idx:
                    pred_indices.append(str(topo_idx[inp]))
                elif inp == self.input_name:
                    pred_indices.append('in')
            pred_str = ','.join(pred_indices) if pred_indices else 'in'

            succ_indices = [str(topo_idx[s]) for s in succ_map[name]
                            if s in topo_idx]
            succ_str = ','.join(succ_indices) if succ_indices else 'out'

            key_params = []
            if node.op_type == 'Conv':
                k = node.params.get('kernel')
                if k is not None:
                    key_params.append(f'kernel={k.shape}')
                key_params.append(f's={node.params.get("stride")}')
                key_params.append(f'p={node.params.get("padding")}')
            elif node.op_type == 'ConvTranspose':
                k = node.params.get('kernel')
                if k is not None:
                    key_params.append(f'kernel={k.shape}')
                key_params.append(f's={node.params.get("stride")}')
            elif node.op_type in ('Gemm', 'MatMul'):
                W = node.params.get('W')
                if W is not None:
                    key_params.append(f'W={W.shape}')
            elif node.op_type in ('MaxPool', 'AveragePool'):
                key_params.append(f'k={node.params.get("kernel_shape")}')
                key_params.append(f's={node.params.get("stride")}')
            elif node.op_type == 'LeakyRelu':
                key_params.append(f'alpha={node.params.get("alpha", 0.01)}')
            elif node.op_type == 'Transpose':
                key_params.append(f'perm={node.params.get("perm")}')
            param_str = f'  {" ".join(key_params)}' if key_params else ''

            fork_marker = ' *' if name in forks else ''
            lines.append(
                f"  [{i:>{idx_w}}] {node.op_type:20s} "
                f"{shape_str:>16s} ({flat:>6d})  "
                f"<-[{pred_str:>5s}]  ->[{succ_str:>5s}]"
                f"{fork_marker}{param_str}")

        lines.append("")
        lines.append(f"  output: {self.output_name}")
        if forks:
            fork_names = [f"[{topo_idx[f]}]" for f in forks if f in topo_idx]
            lines.append(f"  fork points: {', '.join(fork_names)}")
        return '\n'.join(lines)
