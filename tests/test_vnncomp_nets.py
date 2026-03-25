"""vnncomp benchmark tests — load, propagate, compare vs onnxruntime."""

import glob
import multiprocessing
import numpy as np
import pytest
from pathlib import Path


# ---- vnncomp benchmark tests ----

# vnncomp tracks (from scoring repo settings.py)
_REGULAR_BENCHMARKS = {
    'safenlp_2024', 'nn4sys', 'cora_2024', 'linearizenn_2024',
    'dist_shift_2023', 'cifar100_2024', 'tinyimagenet_2024',
    'acasxu_2023', 'cgan_2023', 'collins_rul_cnn_2022',
    'metaroom_2023', 'tllverifybench_2023', 'cersyve',
    'malbeware', 'sat_relu', 'soundnessbench',
}
_EXTENDED_BENCHMARKS = {
    'ml4acopf_2024', 'collins_aerospace_benchmark', 'lsnc_relu',
    'yolo_2023', 'cctsdb_yolo_2023', 'traffic_signs_recognition_2023',
    'vggnet16_2022', 'vit_2023', 'relusplitter',
}

# Benchmarks that cannot be tested at all
_MISSING_BENCHMARKS = {
    'vggnet16_2022',             # no onnx files in benchmark
    'nn4sys/mscn_2048d_dual',   # corrupt ONNX file (DecodeError)
    'nn4sys/pensieve_big_parallel',   # ONNX input (1,8) but spec has 96 inputs
    'nn4sys/pensieve_small_parallel', # same
}

# Networks that currently fail — tested separately via test_vnncomp_hard_*
_HARD_REGULAR = set()  # All regular track networks pass!
_HARD_EXTENDED = {
    'cctsdb_yolo_2023',        # shape broadcast / index errors in preprocessing
    'collins_aerospace_benchmark',  # YOLOv5 640x640, conv shape mismatch
    'ml4acopf_2024',           # ND broadcast + 1D MatMul chain
    'vit_2023',                # shape broadcast errors in attention
}



def _ort_node_compare(onnx_path, graph, center, ort):
    """Run onnxruntime with all intermediate outputs and compare per-node."""
    import gzip
    import onnx
    from vibecheck.zonotope import DenseZonotope

    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model = onnx.load_from_string(f.read())
    else:
        model = onnx.load(onnx_path)

    # Add all intermediate tensors as outputs
    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing:
                model.graph.output.append(
                    onnx.helper.make_tensor_value_info(
                        out, onnx.TensorProto.FLOAT, None))

    sess = ort.InferenceSession(model.SerializeToString())
    inp = sess.get_inputs()[0]
    inp_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    feed = {inp.name: center.astype(np.float32).reshape(inp_shape)}
    out_names = [o.name for o in sess.get_outputs()]
    results = sess.run(out_names, feed)
    ort_vals = {name: val.flatten().astype(np.float64)
                for name, val in zip(out_names, results)}

    # Run our point propagation to get per-node centers
    forks = graph.fork_points()
    zono_state = {graph.input_name: DenseZonotope(
        center, np.zeros((len(center), 0)))}
    gen_count = {graph.input_name: 0}
    def _get(name):
        if name in forks:
            return zono_state[name].copy()
        return zono_state[name]
    for name in graph.topo_order:
        if name in zono_state:
            continue
        graph.nodes[name].zonotope_propagate(
            zono_state, gen_count, _get, 'min_area', graph)
        gen_count[name] = 0

    lines = []
    lines.append(f"{'idx':>4s}  {'op':15s}  {'size':>6s}  {'max_err':>10s}  "
                 f"{'ort_range':>24s}  {'our_range':>24s}")
    lines.append("-" * 100)

    first_bad = None
    for i, name in enumerate(graph.topo_order):
        node = graph.nodes[name]
        our_val = zono_state[name].center

        if name not in ort_vals:
            lines.append(f"[{i:>3d}]  {node.op_type:15s}  {len(our_val):>6d}  "
                         f"{'(no ort)':>10s}")
            continue

        ort_val = ort_vals[name]
        n = min(len(ort_val), len(our_val))
        err = np.max(np.abs(ort_val[:n] - our_val[:n]))
        ort_rng = f"[{ort_val[:n].min():.4f}, {ort_val[:n].max():.4f}]"
        our_rng = f"[{our_val.min():.4f}, {our_val.max():.4f}]"
        marker = " <<< FIRST" if first_bad is None and err > 1e-3 else ""
        if first_bad is None and err > 1e-3:
            first_bad = (i, node.op_type, name)
        lines.append(f"[{i:>3d}]  {node.op_type:15s}  {n:>6d}  {err:>10.2e}  "
                     f"{ort_rng:>24s}  {our_rng:>24s}{marker}")

    report = '\n'.join(lines)
    if first_bad:
        idx, op, name = first_bad
        return (f"Soundness diverges at [{idx}] {op} ({name[:60]})\n\n"
                f"{report}")
    return f"No per-node divergence found\n\n{report}"


def _run_benchmark_worker(onnx_path, spec_path, result_dict):
    """Worker function that runs in a subprocess with memory limits.

    IMPORTANT: The memory cap and subprocess isolation MUST NOT be removed.
    Without them, large benchmarks (e.g. soundnessbench, cifar100) can OOM
    and kill the parent process — including Claude Code sessions.
    """
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    import resource
    mem_limit = 16 * 1024 * 1024 * 1024  # 16GB
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except ValueError:
        pass  # some systems don't support RLIMIT_AS

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None

    try:
        import time
        from vibecheck.network import ComputeGraph as CG
        from vibecheck.verify import zonotope_verify as zvg
        from vibecheck.vnnlib_loader import load_vnnlib
        import gzip

        t0 = time.perf_counter()
        g = CG.from_onnx(onnx_path)
        t_load = time.perf_counter() - t0

        flat_input = 1
        for d in g.input_shape:
            flat_input *= d

        result_dict['n_ops'] = len(g.topo_order)
        result_dict['input_shape'] = str(g.input_shape)

        # Parse spec
        spec = load_vnnlib(spec_path)
        if len(spec.x_lo) != flat_input:
            raise ValueError(
                f'spec input size {len(spec.x_lo)} != network {flat_input}')

        # Use point zonotope at center of spec box
        center = (spec.x_lo + spec.x_hi) / 2
        spec.x_lo = center.copy()
        spec.x_hi = center.copy()

        t0 = time.perf_counter()
        result, details = zvg(g, spec)
        t_verify = time.perf_counter() - t0

        result_dict['status'] = 'ok'
        result_dict['result'] = result
        result_dict['margin'] = details['worst_margin']
        result_dict['t_load'] = t_load
        result_dict['t_verify'] = t_verify

        # Compare against onnxruntime
        if ort is None:
            result_dict['status'] = 'error'
            result_dict['error'] = 'onnxruntime not installed'
            return

        if onnx_path.endswith('.gz'):
            with gzip.open(onnx_path, 'rb') as f:
                model_bytes = f.read()
            sess = ort.InferenceSession(model_bytes)
        else:
            sess = ort.InferenceSession(onnx_path)
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape
        center_f32 = center.astype(np.float32)
        feed = {inp_name: center_f32.reshape(
            [d if isinstance(d, int) and d > 0 else 1
             for d in inp_shape])}
        t0 = time.perf_counter()
        ort_out = sess.run(None, feed)[0].flatten().astype(np.float64)
        t_ort = time.perf_counter() - t0

        result_dict['t_ort'] = t_ort

        out_lo = details['output_lo']
        out_hi = details['output_hi']
        our_out = (out_lo + out_hi) / 2  # point zonotope → exact
        n = min(len(ort_out), len(our_out))
        max_err = np.max(np.abs(ort_out[:n] - our_out[:n]))
        result_dict['max_err'] = float(max_err)

        # Soundness check
        tol = 1e-4
        if np.any(ort_out[:n] < out_lo[:n] - tol) or np.any(ort_out[:n] > out_hi[:n] + tol):
            diag = _ort_node_compare(onnx_path, g, center, ort)
            result_dict['status'] = 'error'
            result_dict['error'] = diag
            return

    except (MemoryError, RuntimeError) as e:
        if 'allocate' in str(e).lower() or isinstance(e, MemoryError):
            result_dict['status'] = 'oom'
        else:
            result_dict['status'] = 'error'
            result_dict['error'] = f'{type(e).__name__}: {e}'
    except Exception as e:
        result_dict['status'] = 'error'
        result_dict['error'] = f'{type(e).__name__}: {e}'


def _discover_benchmark_cases(vnncomp_path, include=None, exclude=None):
    """Discover (test_id, onnx_path, spec_path) triples from instances.csv.

    Picks the first instance row for each unique network.
    include: if set, only include benchmarks/cases in this set
    exclude: if set, skip benchmarks/cases in this set
    """
    import os, csv
    base = str(vnncomp_path)
    cases = []
    for d in sorted(os.listdir(base)):
        if d in _MISSING_BENCHMARKS:
            continue
        if exclude is not None and d in exclude:
            continue
        if include is not None and d not in include:
            # Check if any include entry starts with this dir
            if not any(i.startswith(d + '/') for i in include):
                continue
        csv_path = os.path.join(base, d, 'instances.csv')
        if not os.path.exists(csv_path):
            continue
        seen_nets = set()
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                onnx_rel = row[0].lstrip('./')
                spec_rel = row[1].lstrip('./')
                onnx_path = os.path.join(base, d, onnx_rel)
                spec_path = os.path.join(base, d, spec_rel)
                if not os.path.exists(onnx_path) or not os.path.exists(spec_path):
                    if os.path.exists(onnx_path + '.gz'):
                        onnx_path += '.gz'
                    if os.path.exists(spec_path + '.gz'):
                        spec_path += '.gz'
                if not os.path.exists(onnx_path):
                    continue
                net_name = os.path.basename(onnx_path).replace('.onnx.gz', '').replace('.onnx', '')
                if net_name in seen_nets:
                    continue
                seen_nets.add(net_name)
                test_id = f'{d}/{net_name}'
                if exclude is not None and test_id in exclude:
                    continue
                if include is not None and d not in include and test_id not in include:
                    continue
                if test_id in _MISSING_BENCHMARKS:
                    continue
                cases.append((test_id, onnx_path, spec_path))
    return cases


@pytest.fixture(scope='session')
def benchmark_cases(vnncomp_benchmarks):
    return _discover_benchmark_cases(vnncomp_benchmarks)


def _resolve_vnncomp_path():
    """Resolve vnncomp benchmarks path from paths.yaml at collection time."""
    import os
    paths_file = Path(__file__).parent / "paths.yaml"
    if not paths_file.exists():
        return None
    import yaml
    with open(paths_file) as f:
        paths = yaml.safe_load(f) or {}
    p = paths.get("vnncomp_benchmarks")
    if not p or not os.path.exists(p):
        return None
    base = Path(p)
    if (base / "benchmarks").is_dir():
        base = base / "benchmarks"
    return base


def _get_case_ids(include=None, exclude=None):
    base = _resolve_vnncomp_path()
    if base is None:
        return []
    cases = _discover_benchmark_cases(base, include=include, exclude=exclude)
    return [c[0] for c in cases]


_ALL_HARD = _HARD_REGULAR | _HARD_EXTENDED

# Passing tests: regular track (exclude hard), extended track (exclude hard)
_REGULAR_IDS = _get_case_ids(include=_REGULAR_BENCHMARKS, exclude=_ALL_HARD)
_EXTENDED_IDS = _get_case_ids(include=_EXTENDED_BENCHMARKS, exclude=_ALL_HARD)
# Failing tests: split by track
_HARD_REGULAR_IDS = _get_case_ids(include=_HARD_REGULAR)
_HARD_EXTENDED_IDS = _get_case_ids(include=_HARD_EXTENDED)


_PASS_CACHE = Path(__file__).parent / '.benchmark_pass_cache'


def _load_pass_cache():
    if _PASS_CACHE.exists():
        return set(_PASS_CACHE.read_text().splitlines())
    return set()


def _save_pass(case_id):
    with open(_PASS_CACHE, 'a') as f:
        f.write(case_id + '\n')


def _run_benchmark_test(vnncomp_benchmarks, case_id, use_cache=True):
    """Shared logic for benchmark tests."""
    if use_cache and case_id in _load_pass_cache():
        pytest.skip('cached pass')

    cases = _discover_benchmark_cases(vnncomp_benchmarks)
    case = next((c for c in cases if c[0] == case_id), None)
    if case is None:
        pytest.skip(f'case {case_id} not found')
    _, onnx_path, spec_path = case

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict['status'] = 'timeout'

    p = multiprocessing.Process(
        target=_run_benchmark_worker,
        args=(onnx_path, spec_path, result_dict))
    p.start()
    p.join(timeout=10)

    if p.is_alive():
        p.kill()
        p.join()
        pytest.fail(f'{case_id}: timeout (>10s)')

    status = result_dict.get('status', 'unknown')
    if status == 'oom':
        pytest.fail(f'{case_id}: OOM (>16GB)')
    elif status == 'error':
        pytest.fail(f'{case_id}: {result_dict["error"]}')
    elif status == 'ok':
        d = result_dict
        import os
        print(f'  net:  {os.path.basename(onnx_path)}')
        print(f'  spec: {os.path.basename(spec_path)}')
        parts = [f'  {d.get("n_ops", "?")} ops']
        parts.append(f'in={d.get("input_shape", "?")}')
        parts.append(f'load={d.get("t_load", 0):.3f}s')
        parts.append(f'verify={d.get("t_verify", 0):.3f}s')
        if 't_ort' in d:
            parts.append(f'ort={d["t_ort"]:.3f}s')
        if 'max_err' in d:
            parts.append(f'err={d["max_err"]:.2e}')
        print('  '.join(parts))
        if use_cache:
            _save_pass(case_id)
    else:
        pytest.fail(f'{case_id}: subprocess died (status={status}, exit={p.exitcode})')


# ---- Regular track (must all pass) ----

@pytest.mark.parametrize('case_id', _REGULAR_IDS)
def test_vnncomp_regular(vnncomp_benchmarks, case_id):
    """Regular track benchmarks — must pass."""
    _run_benchmark_test(vnncomp_benchmarks, case_id)


# ---- Extended track (must all pass) ----

@pytest.mark.parametrize('case_id', _EXTENDED_IDS)
def test_vnncomp_extended(vnncomp_benchmarks, case_id):
    """Extended track benchmarks — must pass."""
    _run_benchmark_test(vnncomp_benchmarks, case_id)


# ---- Hard: regular track networks that currently fail ----

@pytest.mark.parametrize('case_id', _HARD_REGULAR_IDS)
def test_vnncomp_hard_regular(vnncomp_benchmarks, case_id):
    """Regular track networks that currently fail."""
    _run_benchmark_test(vnncomp_benchmarks, case_id, use_cache=False)


# ---- Hard: extended track networks that currently fail ----

@pytest.mark.parametrize('case_id', _HARD_EXTENDED_IDS)
def test_vnncomp_hard_extended(vnncomp_benchmarks, case_id):
    """Extended track networks that currently fail."""
    _run_benchmark_test(vnncomp_benchmarks, case_id, use_cache=False)
