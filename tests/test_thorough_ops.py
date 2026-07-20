"""ONNX op-conversion sweep (frontend/onnx_loader +
frontend/network + core/graph).

One tiny model per op family; each must load through the full front end
and agree with onnxruntime pointwise, and its zonotope forward bounds must
contain sampled exact outputs (the soundness invariant every conversion
must preserve)."""
import numpy as np
import pytest
import torch


_rng = np.random.default_rng(21)


def _f32(*shape):
    return _rng.normal(size=shape).astype(np.float32)


def _mk_model(nodes, in_shape, out_shape, inits, opset=17):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        nodes,
        'op_sweep',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', opset)])
    m.ir_version = 8
    return m


def _node(op, inputs, outputs, **attrs):
    from onnx import helper
    return helper.make_node(op, inputs, outputs, **attrs)


# every case: (name, nodes, in_shape, out_shape, inits)
def _cases():
    W = _f32(4, 3)
    yield ('matmul_add', [_node('MatMul', ['X', 'W'], ['h']),
                          _node('Add', ['h', 'B'], ['Y'])],
           [1, 4], [1, 3], {'W': W, 'B': _f32(3)})
    yield ('gemm_transB', [_node('Gemm', ['X', 'W', 'B'], ['Y'],
                                 alpha=1.0, beta=1.0, transB=1)],
           [1, 4], [1, 3], {'W': _f32(3, 4), 'B': _f32(3)})
    yield ('conv_pad_stride', [_node('Conv', ['X', 'K', 'Kb'], ['Y'],
                                     pads=[1, 1, 1, 1], strides=[2, 2])],
           [1, 2, 6, 6], [1, 3, 4, 4], {'K': _f32(3, 2, 3, 3), 'Kb': _f32(3)})
    yield ('convtranspose', [_node('ConvTranspose', ['X', 'K'], ['Y'],
                                   strides=[2, 2])],
           [1, 2, 3, 3], [1, 3, 7, 7], {'K': _f32(2, 3, 3, 3)})
    yield ('maxpool', [_node('MaxPool', ['X'], ['Y'],
                             kernel_shape=[2, 2], strides=[2, 2])],
           [1, 2, 4, 4], [1, 2, 2, 2], {})
    yield ('averagepool', [_node('AveragePool', ['X'], ['Y'],
                                 kernel_shape=[2, 2], strides=[2, 2])],
           [1, 2, 4, 4], [1, 2, 2, 2], {})
    yield ('batchnorm', [_node('BatchNormalization',
                               ['X', 's', 'b', 'm', 'v'], ['Y'])],
           [1, 3, 2, 2], [1, 3, 2, 2],
           {'s': _f32(3), 'b': _f32(3), 'm': _f32(3),
            'v': np.abs(_f32(3)) + 0.5})
    yield ('flatten_relu', [_node('Flatten', ['X'], ['f'], axis=1),
                            _node('Relu', ['f'], ['Y'])],
           [1, 2, 3, 2], [1, 12], {})
    yield ('reshape', [_node('Reshape', ['X', 'shp'], ['Y'])],
           [1, 2, 6], [1, 12], {'shp': np.array([1, 12], np.int64)})
    yield ('transpose_matmul', [_node('Transpose', ['X'], ['t'],
                                      perm=[0, 2, 1]),
                                _node('MatMul', ['t', 'W'], ['Y'])],
           [1, 4, 3], [1, 3, 2], {'W': _f32(4, 2)})
    yield ('concat_sub', [_node('Sub', ['X', 'C'], ['s']),
                          _node('Concat', ['X', 's'], ['Y'], axis=1)],
           [1, 3], [1, 6], {'C': _f32(3)})
    yield ('slice', [_node('Slice', ['X', 'st', 'en', 'ax'], ['Y'])],
           [1, 6], [1, 3],
           {'st': np.array([1], np.int64), 'en': np.array([4], np.int64),
            'ax': np.array([1], np.int64)})
    yield pytest.param(
        'pad_last_axis', [_node('Pad', ['X', 'p'], ['Y'],
                                mode='constant')],
        [1, 4], [1, 6],
        {'p': np.array([0, 1, 0, 1], np.int64)},
        marks=pytest.mark.xfail(
            reason='frontend shape propagation treats a synthetic '
                   'standalone Pad as identity (n stays at the input '
                   'size); the yolo benchmark Pads load via ONNX-declared '
                   'downstream shapes', strict=True))
    yield ('sigmoid', [_node('Sigmoid', ['X'], ['Y'])], [1, 4], [1, 4], {})
    yield ('tanh', [_node('Tanh', ['X'], ['Y'])], [1, 4], [1, 4], {})
    yield ('softmax', [_node('Softmax', ['X'], ['Y'], axis=-1)],
           [1, 4], [1, 4], {})
    yield ('sin_cos_add', [_node('Sin', ['X'], ['s']),
                           _node('Cos', ['X'], ['c']),
                           _node('Add', ['s', 'c'], ['Y'])],
           [1, 4], [1, 4], {})
    yield ('leakyrelu', [_node('LeakyRelu', ['X'], ['Y'], alpha=0.1)],
           [1, 4], [1, 4], {})
    # NOTE deliberately absent: Clip is in the frontend OP_REGISTRY (the
    # `supports --onnx-operators` answer) but NOT in the core IR -- the
    # capabilities table over-advertises it (surfaced by this sweep).
    yield ('neg_div', [_node('Neg', ['X'], ['n']),
                       _node('Div', ['n', 'D'], ['Y'])],
           [1, 4], [1, 4], {'D': np.abs(_f32(4)) + 1.0})
    yield ('mul_const', [_node('Mul', ['X', 'M'], ['Y'])],
           [1, 4], [1, 4], {'M': _f32(4)})
    yield ('reducemean', [_node('ReduceMean', ['X'], ['Y'],
                                axes=[1], keepdims=1)],
           [1, 6], [1, 1], {})
    yield ('reducesum', [_node('ReduceSum', ['X', 'ax'], ['Y'],
                               keepdims=1)],
           [1, 6], [1, 1], {'ax': np.array([1], np.int64)})
    yield ('squeeze_unsqueeze', [_node('Unsqueeze', ['X', 'ax1'], ['u']),
                                 _node('Squeeze', ['u', 'ax1'], ['Y'])],
           [1, 5], [1, 5], {'ax1': np.array([2], np.int64)})
    yield ('gather_cols', [_node('Gather', ['X', 'idx'], ['Y'], axis=1)],
           [1, 6], [1, 3], {'idx': np.array([4, 0, 2], np.int64)})
    yield ('split_relu_concat', [_node('Split', ['X', 'sp'], ['a', 'b'],
                                       axis=1),
                                 _node('Relu', ['a'], ['ra']),
                                 _node('Relu', ['b'], ['rb']),
                                 _node('Concat', ['ra', 'rb'], ['Y'],
                                       axis=1)],
           [1, 6], [1, 6], {'sp': np.array([3, 3], np.int64)})
    yield ('min_max_const', [_node('Min', ['X', 'mn'], ['m1']),
                             _node('Max', ['m1', 'mx'], ['Y'])],
           [1, 4], [1, 4],
           {'mn': np.full(4, 0.5, np.float32),
            'mx': np.full(4, -0.5, np.float32)})
    yield ('floor', [_node('Floor', ['X'], ['Y'])], [1, 4], [1, 4], {})
    yield ('sign', [_node('Sign', ['X'], ['Y'])], [1, 4], [1, 4], {})
    yield ('pow2', [_node('Pow', ['X', 'e'], ['Y'])],
           [1, 4], [1, 4], {'e': np.float32(2.0)})
    yield ('identity_dropout', [_node('Dropout', ['X'], ['d']),
                                _node('Identity', ['d'], ['Y'])],
           [1, 4], [1, 4], {})


@pytest.mark.parametrize('name,nodes,in_shape,out_shape,inits',
                         list(_cases()))
def test_op_ort_parity_and_zono_soundness(tmp_path, name, nodes, in_shape,
                                          out_shape, inits):
    import onnx
    import onnxruntime as ort
    from vibecheck.core import forward, graph
    m = _mk_model(nodes, in_shape, out_shape, inits)
    p = str(tmp_path / f'{name}.onnx')
    onnx.save(m, p)
    net = graph.load(p)
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    n_in = int(np.prod(in_shape))
    lo = torch.full((1, n_in), -1.0)
    hi = torch.full((1, n_in), 1.0)
    xs = torch.rand(64, n_in) * 2 - 1
    ys = forward.point(net, xs)
    for i in (0, 17, 63):
        y_ort = sess.run(None, {'X': xs[i].reshape(in_shape).numpy()})[0]
        # Floor/Sign step exactly at grid points: nudge the comparison tol
        assert np.allclose(ys[i].numpy(), y_ort.ravel(), atol=2e-5), name
    zlo, zhi = forward.zono(net, lo, hi)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all(), name
