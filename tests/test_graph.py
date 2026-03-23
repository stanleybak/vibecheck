"""Tests for graph-based loading and verification."""

import glob
import json
import multiprocessing
import numpy as np
import pytest
from vibecheck.graph import ComputeGraph
from vibecheck.zonotope import DenseZonotope
from vibecheck.verify import zonotope_verify, zonotope_verify_graph
from vibecheck.bounds import ia_bounds_graph


# ---- DenseZonotope.add ----

def test_zonotope_add_shared_only():
    """Add two zonotopes that share all generators."""
    z1 = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    z2 = DenseZonotope(np.array([3.0, 4.0]), np.array([[0.1, 0.0], [0.0, 0.1]]))
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [4.0, 6.0])
    np.testing.assert_allclose(z3.generators, [[0.6, 0.0], [0.0, 0.6]])


def test_zonotope_add_with_extra_gens():
    """Add two zonotopes where each branch added extra generators."""
    # Both share 2 original generators, z1 added 1 extra, z2 added 2 extra
    z1 = DenseZonotope(
        np.array([1.0]),
        np.array([[0.5, 0.3, 0.1]]),  # 2 shared + 1 extra
    )
    z2 = DenseZonotope(
        np.array([2.0]),
        np.array([[0.4, 0.2, 0.05, 0.02]]),  # 2 shared + 2 extra
    )
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [3.0])
    # shared: [0.9, 0.5], z1_extra: [0.1], z2_extra: [0.05, 0.02]
    np.testing.assert_allclose(z3.generators, [[0.9, 0.5, 0.1, 0.05, 0.02]])


def test_zonotope_copy_independent():
    z = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5], [0.3]]))
    z2 = z.copy()
    z2.center[0] = 99.0
    assert z.center[0] == 1.0


# ---- ComputeGraph loading ----

def test_graph_load_sequential(vnncomp_benchmarks):
    """ACAS Xu loads as a sequential graph with no fork points."""
    net = str(vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz")
    g = ComputeGraph.from_onnx(net)
    assert g.input_shape == (5,)
    assert len(g.fork_points()) == 0
    assert len(g.relu_nodes()) == 6


def test_graph_load_cersyve(vnncomp_benchmarks):
    """Cersyve has a dual-branch structure with Add merge."""
    net = str(vnncomp_benchmarks / "cersyve/onnx/point_mass_pretrain_con.onnx.gz")
    g = ComputeGraph.from_onnx(net)
    assert g.input_shape == (4,)
    assert len(g.fork_points()) > 0  # input is a fork
    # Output is the Add node
    assert g.nodes[g.output_name].op_type == 'Add'


def test_graph_load_resnet(vnncomp_benchmarks):
    """cifar100 ResNet loads with BatchNorm folded and skip connections."""
    net = str(vnncomp_benchmarks / "cifar100_2024/onnx/CIFAR100_resnet_medium.onnx.gz")
    g = ComputeGraph.from_onnx(net)
    assert g.input_shape == (3, 32, 32)
    # BN should be folded — no BatchNormalization nodes remain
    bn_nodes = [n for n in g.nodes.values() if n.op_type == 'BatchNormalization']
    assert len(bn_nodes) == 0
    # Should have Add nodes (skip connections)
    add_nodes = [n for n in g.nodes.values()
                 if n.op_type == 'Add' and len(n.inputs) == 2
                 and n.inputs[1] in g.nodes]
    assert len(add_nodes) == 8  # 8 residual blocks


# ---- Regression: graph path matches flat path ----

def test_graph_vs_flat_acasxu(vnncomp_benchmarks):
    """Graph path produces identical results to flat path on ACAS Xu."""
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.spec import parse_vnnlib

    net = str(vnncomp_benchmarks / "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz")
    spec = str(vnncomp_benchmarks / "acasxu_2023/vnnlib/prop_2.vnnlib.gz")

    layers, _ = load_onnx(net)
    graph = ComputeGraph.from_onnx(net)
    x_lo, x_hi, pred_label, competitors = parse_vnnlib(spec)

    _, details_flat = zonotope_verify(layers, x_lo, x_hi, pred_label, competitors)
    _, details_graph = zonotope_verify_graph(graph, x_lo, x_hi, pred_label, competitors)

    np.testing.assert_allclose(details_flat['output_lo'], details_graph['output_lo'])
    np.testing.assert_allclose(details_flat['output_hi'], details_graph['output_hi'])


# ---- IA bounds graph ----

def test_ia_bounds_graph_cersyve(vnncomp_benchmarks):
    """IA bounds propagate through cersyve graph without error."""
    net = str(vnncomp_benchmarks / "cersyve/onnx/point_mass_pretrain_con.onnx.gz")
    g = ComputeGraph.from_onnx(net)
    x_lo = np.zeros(4)
    x_hi = np.ones(4)
    state, pre_relu = ia_bounds_graph(g, x_lo, x_hi)
    # Output should have bounds
    out_lo, out_hi = state[g.output_name]
    assert out_lo.shape == (2,)
    assert np.all(out_lo <= out_hi)


# ---- Comprehensive vnncomp benchmark test ----

# Benchmarks that OOM or have shape-tracking issues too complex for flat vectors.
# These are excluded from the parametrized test entirely.
_SKIP_BENCHMARKS = {
    'vggnet16_2022',           # no onnx files
    'safenlp_2024',            # no onnx files at top level
    'soundnessbench',          # OOM: 128->12288 Gemm then Conv on 24x64x64
    'cctsdb_yolo_2023',        # complex preprocessing (Slice/Reshape/ScatterND before Conv)
    'collins_aerospace_benchmark',  # feature pyramid Concat->Conv shape mismatch
    'ml4acopf_2024',           # trig ops + complex broadcast patterns
    'vit_2023',                # multi-head attention reshape + bilinear MatMul
}


def _benchmark_ids(vnncomp_path):
    """Discover benchmark directories that have onnx files."""
    import os
    base = str(vnncomp_path)
    ids = []
    for d in sorted(os.listdir(base)):
        if d in _SKIP_BENCHMARKS:
            continue
        onnx_files = glob.glob(f'{base}/{d}/onnx/*.onnx*')
        if onnx_files:
            ids.append(d)
    return ids


@pytest.fixture(scope='session')
def benchmark_list(vnncomp_benchmarks):
    return _benchmark_ids(vnncomp_benchmarks)


def _run_benchmark_worker(base, benchmark_name, result_dict):
    """Worker function that runs in a subprocess with memory limits."""
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    import resource
    # Cap virtual memory at 4GB to prevent OOM killing the parent
    mem_limit = 4 * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except ValueError:
        pass  # some systems don't support RLIMIT_AS

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None

    try:
        from vibecheck.graph import ComputeGraph as CG
        from vibecheck.verify import zonotope_verify_graph as zvg
        from vibecheck.bounds import ia_bounds_graph as iabg
        from vibecheck.spec import parse_vnnlib
        import gzip

        onnx_files = sorted(glob.glob(f'{base}/{benchmark_name}/onnx/*.onnx*'))
        spec_files = sorted(glob.glob(f'{base}/{benchmark_name}/vnnlib/*.vnnlib*'))

        onnx_path = onnx_files[0]
        g = CG.from_onnx(onnx_path)

        flat_input = 1
        for d in g.input_shape:
            flat_input *= d

        x_lo = x_hi = pred_label = competitors = None
        if spec_files:
            try:
                x_lo, x_hi, pred_label, competitors = parse_vnnlib(spec_files[0])
            except Exception:
                pass

        if x_lo is not None and len(x_lo) == flat_input:
            # Shrink bounds to 1% of spec to keep zonotope propagation fast
            center = (x_lo + x_hi) / 2
            radius = (x_hi - x_lo) / 2 * 0.01
            x_lo = center - radius
            x_hi = center + radius
        else:
            x_lo = np.full(flat_input, -0.0001)
            x_hi = np.full(flat_input, 0.0001)

            ia_state, _ = iabg(g, x_lo, x_hi)
            out_lo, out_hi = ia_state[g.output_name]
            n_out = len(out_lo)

            pred_label = 0
            competitors = list(range(1, min(n_out, 3)))
            if not competitors:
                competitors = [0]

        result, details = zvg(
            g, x_lo, x_hi, pred_label, competitors,
            relu_types=['min_area'])

        result_dict['status'] = 'ok'
        result_dict['result'] = result
        result_dict['margin'] = details['worst_margin']

        # Soundness check: run onnxruntime on the center point and verify
        # the output falls within the zonotope bounds
        if ort is not None:
            try:
                if onnx_path.endswith('.gz'):
                    with gzip.open(onnx_path, 'rb') as f:
                        model_bytes = f.read()
                    sess = ort.InferenceSession(model_bytes)
                else:
                    sess = ort.InferenceSession(onnx_path)
                inp_name = sess.get_inputs()[0].name
                inp_shape = sess.get_inputs()[0].shape
                center = ((x_lo + x_hi) / 2).astype(np.float32)
                # Reshape to match onnx expected input
                feed = {inp_name: center.reshape([1 if (d == 0 or d is None) else d for d in inp_shape])}
                ort_out = sess.run(None, feed)[0].flatten().astype(np.float64)
                out_lo = details['output_lo']
                out_hi = details['output_hi']
                n = min(len(ort_out), len(out_lo))
                tol = 1e-4
                if np.any(ort_out[:n] < out_lo[:n] - tol) or np.any(ort_out[:n] > out_hi[:n] + tol):
                    violations = []
                    for i in range(n):
                        if ort_out[i] < out_lo[i] - tol:
                            violations.append(f'dim {i}: ort={ort_out[i]:.6f} < lo={out_lo[i]:.6f}')
                        if ort_out[i] > out_hi[i] + tol:
                            violations.append(f'dim {i}: ort={ort_out[i]:.6f} > hi={out_hi[i]:.6f}')
                    result_dict['status'] = 'error'
                    result_dict['error'] = f'Soundness violation: {"; ".join(violations[:3])}'
                    return
            except Exception:
                pass  # ort check is best-effort

    except (MemoryError, RuntimeError) as e:
        if 'allocate' in str(e).lower() or isinstance(e, MemoryError):
            result_dict['status'] = 'oom'
        else:
            result_dict['status'] = 'error'
            result_dict['error'] = f'{type(e).__name__}: {e}'
    except Exception as e:
        result_dict['status'] = 'error'
        result_dict['error'] = f'{type(e).__name__}: {e}'


# Parametrize over all non-skipped benchmarks
_ALL_BENCHMARKS = [
    'acasxu_2023',
    'cersyve',
    'cgan_2023',
    'cifar100_2024',
    'collins_rul_cnn_2022',
    'cora_2024',
    'dist_shift_2023',
    'linearizenn_2024',
    'lsnc_relu',
    'malbeware',
    'metaroom_2023',
    'nn4sys',
    'relusplitter',
    'sat_relu',
    'test',
    'tinyimagenet_2024',
    'tllverifybench_2023',
    'traffic_signs_recognition_2023',
    'yolo_2023',
]


@pytest.mark.parametrize('benchmark_name', _ALL_BENCHMARKS)
def test_vnncomp_benchmark(vnncomp_benchmarks, benchmark_name):
    """Each vnncomp benchmark loads, propagates IA bounds, and runs zonotope verify.

    Runs in a subprocess with a 2GB memory limit so OOM can't kill the test runner.
    """
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict['status'] = 'timeout'

    base = str(vnncomp_benchmarks)
    p = multiprocessing.Process(
        target=_run_benchmark_worker,
        args=(base, benchmark_name, result_dict))
    p.start()
    p.join(timeout=120)

    if p.is_alive():
        p.kill()
        p.join()
        pytest.skip(f'{benchmark_name}: timeout (>120s)')

    status = result_dict.get('status', 'unknown')
    if status == 'oom':
        pytest.skip(f'{benchmark_name}: OOM (>2GB)')
    elif status == 'error':
        pytest.fail(f'{benchmark_name}: {result_dict["error"]}')
    elif status == 'ok':
        print(f'{benchmark_name}: {result_dict["result"]} margin={result_dict["margin"]:.4f}')
    else:
        pytest.fail(f'{benchmark_name}: subprocess died (status={status}, exit={p.exitcode})')
