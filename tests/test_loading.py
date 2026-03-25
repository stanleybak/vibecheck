"""Tests for onnx_loader.py and main.py — use small real ONNX files."""

import sys
import numpy as np
import pytest
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify import zonotope_verify


# ---- ONNX loading ----

def test_load_acasxu(vnncomp_benchmarks):
    """ACAS Xu: FC network, dynamic input dims [0,0,0,5]."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"))
    assert g.input_shape == (1, 1, 1, 5)
    assert len(g.relu_nodes()) == 6
    assert len(g.fork_points()) == 0
    # Verify all nodes have shapes
    for name in g.topo_order:
        assert g.nodes[name].output_shape is not None, f'{name} has no shape'


def test_load_cersyve_fork(vnncomp_benchmarks):
    """Cersyve: fork points, Add merge."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cersyve/onnx/point_mass_pretrain_con.onnx.gz"))
    assert g.input_shape == (1, 4)
    assert len(g.fork_points()) > 0
    assert g.nodes[g.output_name].op_type == 'Add'


def test_load_resnet_bn_fold(vnncomp_benchmarks):
    """cifar100 ResNet: BatchNorm should be folded out."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cifar100_2024/onnx/CIFAR100_resnet_medium.onnx.gz"))
    assert g.input_shape == (1, 3, 32, 32)
    bn_nodes = [n for n in g.nodes.values() if n.op_type == 'BatchNormalization']
    assert len(bn_nodes) == 0
    # Residual skip connections
    add_nodes = [n for n in g.nodes.values()
                 if n.op_type == 'Add' and len(n.inputs) == 2
                 and n.inputs[1] in g.nodes]
    assert len(add_nodes) == 8


def test_load_cgan_convtranspose(vnncomp_benchmarks):
    """cGAN: ConvTranspose + BatchNorm (BN not folded into ConvTranspose)."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cgan_2023/onnx/cGAN_imgSz32_nCh_1.onnx.gz"))
    assert any(n.op_type == 'ConvTranspose' for n in g.nodes.values())


def test_load_pensieve_split(vnncomp_benchmarks):
    """nn4sys pensieve: has Slice, Gather, Split, Reshape."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "nn4sys/onnx/pensieve_small_simple.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Slice' in ops
    assert 'Gather' in ops
    assert 'Reshape' in ops


def test_load_traffic_signs(vnncomp_benchmarks):
    """traffic_signs: NHWC layout with Transpose + Sign + Softmax."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "traffic_signs_recognition_2023/onnx/"
        "3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx.gz"))
    assert g.input_shape == (1, 30, 30, 3)
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Transpose' in ops
    assert 'Sign' in ops
    assert 'Softmax' in ops


def test_load_collins_conv1d(vnncomp_benchmarks):
    """collins: Conv with (C_in=1) channel."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "collins_rul_cnn_2022/onnx/NN_rul_small_window_20.onnx.gz"))
    assert g.input_shape == (1, 1, 20, 20)
    assert any(n.op_type == 'Conv' for n in g.nodes.values())


def test_load_sat_relu(vnncomp_benchmarks):
    """sat_relu: tiny 2-layer FC."""
    import glob
    nets = sorted(glob.glob(str(vnncomp_benchmarks / "sat_relu/onnx/*.onnx*")))
    g = ComputeGraph.from_onnx(nets[0])
    assert len(g.relu_nodes()) == 1


def test_load_yolo(vnncomp_benchmarks):
    """yolo: Conv+Relu+Add+AveragePool+Pad+Flatten."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "yolo_2023/onnx/TinyYOLO.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'AveragePool' in ops
    assert 'Pad' in ops
    assert 'Flatten' in ops
    assert len(g.fork_points()) > 0


def test_load_mscn(vnncomp_benchmarks):
    """nn4sys mscn: Slice+Split+Gather+ReduceSum+Div+Sigmoid+Concat."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "nn4sys/onnx/mscn_128d.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Split' in ops
    assert 'ReduceSum' in ops
    assert 'Div' in ops
    assert 'Sigmoid' in ops
    assert 'Concat' in ops


def test_load_dist_shift(vnncomp_benchmarks):
    """dist_shift: Gemm+Relu+Reshape+Sigmoid."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "dist_shift_2023/onnx/mnist_concat.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Sigmoid' in ops
    assert 'Reshape' in ops


def test_load_linearizenn(vnncomp_benchmarks):
    """linearizenn: Gemm+MatMul+Relu."""
    import glob
    nets = sorted(glob.glob(str(vnncomp_benchmarks / "linearizenn_2024/onnx/*.onnx*")))
    g = ComputeGraph.from_onnx(nets[0])
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Relu' in ops


def test_load_relusplitter_conv(vnncomp_benchmarks):
    """relusplitter: Conv+Flatten+Gemm (cifar variant)."""
    import glob
    nets = sorted(glob.glob(str(
        vnncomp_benchmarks / "relusplitter/onnx/cifar*")))
    if nets:
        g = ComputeGraph.from_onnx(nets[0])
        ops = {n.op_type for n in g.nodes.values()}
        assert 'Conv' in ops
        assert 'Flatten' in ops


def test_load_traffic_signs_maxpool(vnncomp_benchmarks):
    """traffic_signs larger: MaxPool + BN (unfused) + Mul."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "traffic_signs_recognition_2023/onnx/"
        "3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'MaxPool' in ops
    assert 'Mul' in ops  # from unfused BN scale


def test_load_cgan_upsample(vnncomp_benchmarks):
    """cGAN upsample variant: Upsample + Sigmoid + Tanh."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_upsample.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Upsample' in ops or 'Resize' in ops
    assert 'Sigmoid' in ops


def test_load_cgan_nonlinear(vnncomp_benchmarks):
    """cGAN nonlinear: Tanh + Sigmoid."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_nonlinear_activations.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Tanh' in ops


def test_load_tllverifybench(vnncomp_benchmarks):
    """tllverifybench: deep MatMul+Add+Relu."""
    import glob
    nets = sorted(glob.glob(str(
        vnncomp_benchmarks / "tllverifybench_2023/onnx/*.onnx*")))
    g = ComputeGraph.from_onnx(nets[0])
    assert len(g.relu_nodes()) >= 6


def test_load_ml4acopf(vnncomp_benchmarks):
    """ml4acopf: Sin, Cos, Pow, Neg, Sigmoid, Unsqueeze, Concat, Slice, Transpose."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "ml4acopf_2024/onnx/14_ieee_ml4acopf.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Sin' in ops
    assert 'Cos' in ops
    assert 'Pow' in ops
    assert 'Neg' in ops


def test_load_ml4acopf_linear(vnncomp_benchmarks):
    """ml4acopf linear-residual: Floor, Div, Mul, Unsqueeze."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "ml4acopf_2024/onnx/14_ieee_ml4acopf-linear-residual.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Floor' in ops
    assert 'Unsqueeze' in ops


def test_load_collins_aerospace(vnncomp_benchmarks):
    """collins_aerospace: LeakyRelu, Clip (via Relu6), MaxPool, Resize, Split."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "collins_aerospace_benchmark/onnx/yolov5nano_LRelu_640.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'LeakyRelu' in ops
    assert 'MaxPool' in ops
    assert 'Resize' in ops
    assert 'Split' in ops


def test_load_lsnc_relu(vnncomp_benchmarks):
    """lsnc_relu: ReduceSum, Neg, Div, Mul, Cast, Gather, Concat, Slice."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "lsnc_relu/onnx/relu_quadrotor2d_state.onnx.gz"))
    ops = {n.op_type for n in g.nodes.values()}
    assert 'Neg' in ops
    assert 'ReduceSum' in ops
    assert 'Cast' in ops


# ---- Constant folding ----

def test_try_fold_constant():
    """Test constant folding for various ops."""
    from vibecheck.onnx_loader import _try_fold_constant
    c = lambda name: np.array([1.0, -2.0, 3.0])

    assert _try_fold_constant('Relu', ['x'], {}, c) is not None
    np.testing.assert_array_equal(
        _try_fold_constant('Relu', ['x'], {}, c), [1, 0, 3])

    np.testing.assert_array_equal(
        _try_fold_constant('Neg', ['x'], {}, c), [-1, 2, -3])

    r = _try_fold_constant('Sigmoid', ['x'], {}, c)
    assert r is not None

    r = _try_fold_constant('LeakyRelu', ['x'], {'alpha': 0.1}, c)
    np.testing.assert_allclose(r, [1, -0.2, 3])

    np.testing.assert_array_equal(
        _try_fold_constant('Flatten', ['x'], {}, c), [1, -2, 3])

    np.testing.assert_array_equal(
        _try_fold_constant('Sign', ['x'], {}, c), [1, -1, 1])

    r = _try_fold_constant('Concat', ['x', 'x'], {}, c)
    assert len(r) == 6

    r = _try_fold_constant('Slice', ['x'], {'starts': [0], 'ends': [2]}, c)
    np.testing.assert_array_equal(r, [1, -2])

    r = _try_fold_constant('Gather', ['x'],
                           {'indices': np.array([2, 0])}, c)
    np.testing.assert_array_equal(r, [3, 1])

    r = _try_fold_constant('Transpose', ['x'], {}, c)
    assert r is not None

    W = np.eye(3)
    b = np.zeros(3)
    r = _try_fold_constant('Gemm', ['x'], {'W': W, 'b': b}, c)
    np.testing.assert_array_equal(r, [1, -2, 3])

    r = _try_fold_constant('Div', ['x'], {'scale': np.array([2, 2, 2])}, c)
    np.testing.assert_array_equal(r, [2, -4, 6])

    r = _try_fold_constant('ReduceSum', ['x'], {}, c)
    np.testing.assert_array_equal(r, [2])

    r = _try_fold_constant('ReduceMean', ['x'], {}, c)
    np.testing.assert_allclose(r, [2/3])

    # Unknown op returns None
    assert _try_fold_constant('UnknownOp', ['x'], {}, c) is None


def test_graph_str(vnncomp_benchmarks):
    """print(graph) should work, including fork points and key params."""
    # Sequential network
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"))
    s = str(g)
    assert 'ComputeGraph' in s
    assert 'Relu' in s

    # Network with forks + Conv + MaxPool
    g2 = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "traffic_signs_recognition_2023/onnx/"
        "3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx.gz"))
    s2 = str(g2)
    assert 'MaxPool' in s2
    assert 'Transpose' in s2

    # Network with LeakyRelu
    g3 = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "collins_aerospace_benchmark/onnx/yolov5nano_LRelu_640.onnx.gz"))
    s3 = str(g3)
    assert 'LeakyRelu' in s3
    assert 'fork points' in s3


def test_graph_methods(vnncomp_benchmarks):
    """Test ComputeGraph utility methods."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"))
    assert g.flat_size(g.input_name) == 5
    assert g.flat_size(g.output_name) == 5
    preds = g.predecessors(g.topo_order[1])
    assert len(preds) > 0
    succs = g.successors(g.topo_order[0])
    assert len(succs) > 0
    assert repr(g).startswith('ComputeGraph')


# ---- End-to-end verify ----

def test_verify_acasxu(vnncomp_benchmarks):
    """Full pipeline: load ACAS Xu + spec, verify."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"))
    spec = load_vnnlib(str(
        vnncomp_benchmarks / "acasxu_2023/vnnlib/prop_2.vnnlib.gz"))
    result, details = zonotope_verify(g, spec)
    assert result in ('verified', 'unknown')
    assert 'worst_margin' in details


def test_verify_with_fork(vnncomp_benchmarks):
    """Verify on cersyve (has fork points)."""
    g = ComputeGraph.from_onnx(str(
        vnncomp_benchmarks / "cersyve/onnx/point_mass_pretrain_con.onnx.gz"))
    spec = load_vnnlib(str(
        vnncomp_benchmarks / "cersyve/vnnlib/prop_point_mass.vnnlib.gz"))
    result, details = zonotope_verify(g, spec)
    assert result in ('verified', 'unknown')


# ---- main.py CLI ----

def test_main_cli(vnncomp_benchmarks, monkeypatch):
    """Test CLI entry point."""
    from vibecheck.main import main
    net = str(vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz")
    spec = str(vnncomp_benchmarks / "acasxu_2023/vnnlib/prop_2.vnnlib.gz")
    monkeypatch.setattr(sys, 'argv', ['vibecheck', '--net', net, '--spec', spec])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code in (0, 1)  # verified or unknown


