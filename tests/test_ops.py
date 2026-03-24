"""Unit tests for individual op node propagation."""

import numpy as np
import pytest
from vibecheck.zonotope import DenseZonotope
from vibecheck.network import (
    GraphNode, PassthroughNode, ReshapeNode, TransposeNode, SliceNode,
    GatherNode, ReluNode, ConvNode, GemmNode, AddNode, SubNode,
    MulNode, DivNode, NegNode, BatchNormNode, ConcatNode, SplitNode,
    SplitOutputNode, SigmoidNode, SoftmaxNode, SignNode, ClipNode,
    TanhNode, ReduceNode, ComputeGraph, _prod,
)


def _make_graph(nodes, input_name='input', input_shape=(1, 4),
                output_name=None):
    """Helper to build a minimal ComputeGraph from a list of nodes."""
    g = ComputeGraph()
    g.input_name = input_name
    g.input_shape = input_shape
    for n in nodes:
        g.nodes[n.name] = n
    g.output_name = output_name or nodes[-1].name
    g.topological_sort()
    # Infer shapes
    shapes = {input_name: input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


def _run_point(graph, center):
    """Run a point zonotope (0 generators) through the graph."""
    zono_state = {graph.input_name: DenseZonotope(
        center, np.zeros((len(center), 0)))}
    gen_count = {graph.input_name: 0}
    forks = graph.fork_points()
    def get(name):
        if name in forks:
            return zono_state[name].copy()
        return zono_state[name]
    for name in graph.topo_order:
        if name in zono_state:
            continue
        graph.nodes[name].zonotope_propagate(
            zono_state, gen_count, get, 'min_area', graph)
        gen_count[name] = zono_state[name].generators.shape[1]
    return zono_state[graph.output_name].center


# --- Reshape ---

def test_reshape_shape_inference():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (-1, 2, 3)})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 2, 3)


def test_reshape_keeps_batch():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 3, 2)})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 3, 2)


def test_reshape_propagation():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 2, 3)})
    g = _make_graph([node], input_shape=(1, 6))
    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, center)  # data unchanged


# --- Transpose ---

def test_transpose_nhwc_to_nchw():
    """Transpose (1,H,W,C) -> (1,C,H,W)."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 3, 1, 2]})
    g = _make_graph([node], input_shape=(1, 2, 3, 4))
    assert node.output_shape == (1, 4, 2, 3)

    # Data: fill with index to verify permutation
    center = np.arange(24, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 2, 3, 4).transpose(0, 3, 1, 2).flatten()
    np.testing.assert_array_equal(out, expected)


def test_transpose_reverse():
    """Default transpose (reverse dims)."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={})
    g = _make_graph([node], input_shape=(1, 2, 3))
    assert node.output_shape == (3, 2, 1)

    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 2, 3).transpose(2, 1, 0).flatten()
    np.testing.assert_array_equal(out, expected)


def test_transpose_with_generators():
    """Transpose permutes generator rows too."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 2, 1]})
    g = _make_graph([node], input_shape=(1, 2, 3))

    center = np.arange(6, dtype=float)
    gens = np.eye(6)  # one generator per element
    z_in = DenseZonotope(center, gens)

    zono_state = {'input': z_in}
    gen_count = {'input': 6}
    node.zonotope_propagate(zono_state, gen_count, lambda n: zono_state[n],
                            'min_area', g)
    z_out = zono_state['t']

    expected_center = center.reshape(1, 2, 3).transpose(0, 2, 1).flatten()
    np.testing.assert_array_equal(z_out.center, expected_center)
    # Generators should be permuted the same way
    assert z_out.generators.shape == (6, 6)


# --- Slice ---

def test_slice_axis1():
    """Slice along axis 1 of (1, 6, 8) -> (1, 1, 8) = 8 elements."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [2], 'ends': [3], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 6, 8))
    assert node.output_shape == (1, 1, 8)

    center = np.arange(48, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 6, 8)[:, 2:3, :].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_axis2():
    """Slice along last axis."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [1], 'ends': [4], 'axes': [2]})
    g = _make_graph([node], input_shape=(1, 3, 5))
    assert node.output_shape == (1, 3, 3)

    center = np.arange(15, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 3, 5)[:, :, 1:4].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_flat():
    """Slice on a 2D shape (1, N) — flat slice."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [2], 'ends': [5], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 10))
    assert node.output_shape == (1, 3)

    center = np.arange(10, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 10)[:, 2:5].flatten()
    np.testing.assert_array_equal(out, expected)


# --- Gather ---

def test_gather_indices():
    node = GatherNode(name='g', op_type='Gather', inputs=['input'],
                      params={'indices': np.array([0, 3, 1])})
    g = _make_graph([node], input_shape=(1, 5))
    assert node.output_shape == (3,)

    center = np.array([10, 20, 30, 40, 50], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [10, 40, 20])


# --- Relu ---

def test_relu_point():
    node = ReluNode(name='r', op_type='Relu', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-1, 0, 1, 2], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [0, 0, 1, 2])


# --- Sigmoid, Tanh, Sign, Clip, Softmax ---

def test_sigmoid_point():
    node = SigmoidNode(name='s', op_type='Sigmoid', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, 1, -1], dtype=float)
    out = _run_point(g, center)
    expected = 1 / (1 + np.exp(-center))
    np.testing.assert_allclose(out, expected)


def test_tanh_point():
    node = TanhNode(name='t', op_type='Tanh', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, 1, -1], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_allclose(out, np.tanh(center))


def test_sign_point():
    node = SignNode(name='s', op_type='Sign', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-2, 0, 0.5, 3], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [-1, 0, 1, 1])


def test_clip_point():
    node = ClipNode(name='c', op_type='Clip', inputs=['input'],
                    params={'min': -1.0, 'max': 1.0})
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-5, -0.5, 0.5, 5], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [-1, -0.5, 0.5, 1])


def test_softmax_point():
    node = SoftmaxNode(name='s', op_type='Softmax', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([1, 2, 3], dtype=float)
    out = _run_point(g, center)
    e = np.exp(center - center.max())
    np.testing.assert_allclose(out, e / e.sum())


# --- Neg, Add, Sub, Mul, Div ---

def test_neg_point():
    node = NegNode(name='n', op_type='Neg', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([1, -2, 3], dtype=float))
    np.testing.assert_array_equal(out, [-1, 2, -3])


def test_add_bias():
    node = AddNode(name='a', op_type='Add', inputs=['input'],
                   params={'bias': np.array([10, 20, 30])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([1, 2, 3], dtype=float))
    np.testing.assert_array_equal(out, [11, 22, 33])


def test_mul_scale():
    node = MulNode(name='m', op_type='Mul', inputs=['input'],
                   params={'scale': np.array([2, 0.5, -1])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([4, 6, 8], dtype=float))
    np.testing.assert_array_equal(out, [8, 3, -8])


# --- BatchNorm (unfused) ---

def test_batchnorm_point():
    node = BatchNormNode(name='bn', op_type='BatchNormalization',
                         inputs=['input'],
                         params={
                             'scale': np.array([2.0, 1.0]),
                             'bias': np.array([0.0, 1.0]),
                             'mean': np.array([0.5, 0.5]),
                             'var': np.array([1.0, 1.0]),
                             'epsilon': 0.0,
                         })
    g = _make_graph([node], input_shape=(1, 2))
    center = np.array([1.0, 2.0])
    out = _run_point(g, center)
    # factor = scale / sqrt(var + eps) = [2, 1]
    # offset = -factor * mean + bias = [-1, 0.5]
    # out = factor * center + offset = [2*1 - 1, 1*2 + 0.5] = [1, 2.5]
    np.testing.assert_allclose(out, [1.0, 2.5])


# --- Reduce ---

def test_reduce_sum():
    node = ReduceNode(name='r', op_type='ReduceSum', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    out = _run_point(g, np.array([1, 2, 3, 4], dtype=float))
    np.testing.assert_array_equal(out, [10])


def test_reduce_mean():
    node = ReduceNode(name='r', op_type='ReduceMean', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    out = _run_point(g, np.array([1, 2, 3, 4], dtype=float))
    np.testing.assert_array_equal(out, [2.5])
