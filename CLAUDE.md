# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is VibeCheck?

A zonotope-based neural network verification tool. Given an ONNX network and a VNNLIB specification (input bounds + output property), it determines whether the property is provably satisfied ("verified") or "unknown" using abstract interpretation with zonotope domains.

## Development Commands

Always use the venv (`.venv/bin/python`) for all commands.

```bash
# Setup
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test
.venv/bin/python -m pytest tests/test_zonotope.py::test_from_input_bounds -v

# Run the verifier
.venv/bin/vibecheck --net model.onnx --spec property.vnnlib
```

## Architecture

The verification pipeline flows: **ONNX loading → graph construction → interval arithmetic bounds → zonotope propagation → margin check**.

- **`graph.py`** — Core graph representation. `ComputeGraph` loads ONNX models into a DAG of `GraphNode` objects keyed by tensor name. Supports non-sequential architectures (ResNets, skip connections, feature pyramids). Performs topological sort (Kahn's algorithm), BatchNorm folding into preceding Conv/Gemm, constant folding, and shape inference. Handles Conv, ConvTranspose, Gemm, MatMul, Add, Sub, Mul, Div, Relu, LeakyRelu, Sigmoid, Clip, MaxPool, AveragePool, Pad, Concat, Split, Slice, Gather, Reshape, and many more ops.

- **`onnx_loader.py`** — Legacy sequential ONNX loader. Returns a flat list of `(W, b)` or `(kernel, bias, params)` tuples. Still used by `zonotope_verify()` for backward compatibility. The `is_conv()` function distinguishes layer types by tuple length.

- **`spec.py`** — Parses VNNLIB specifications (supports `.gz`). Extracts input bounds (`x_lo`, `x_hi`) and output constraints identifying `pred_label` and `competitors`.

- **`bounds.py`** — Interval arithmetic (IA) bound propagation. `ia_bounds()` works on flat layer lists (legacy). `ia_bounds_graph()` works on `ComputeGraph`, returning `dict[node_name -> (lo, hi)]`. Conv/MaxPool/AveragePool/Pad IA uses PyTorch via numpy↔torch conversion. Falls back to conservative bounds on shape mismatches.

- **`zonotope.py`** — `DenseZonotope` represents sets as `{center + G @ e | ||e||_inf <= 1}`. Methods: `propagate_linear()` (FC/Conv), `apply_relu()` with three relaxation types (`min_area`, `y_bloat`, `box`), `copy()` for fork points, `add(other, shared_gens)` for skip connection merges. The add method splits generators into shared prefix (added element-wise) and branch-specific suffix (concatenated).

- **`verify.py`** — Verification orchestration. `zonotope_verify()` is the legacy flat-layer path. `zonotope_verify_graph()` traverses `ComputeGraph` in topological order, tracks generator counts at fork points for merge operations, and falls back to interval bounds for unsupported/expensive ops (ConvTranspose, LeakyRelu, bilinear MatMul, or when generator matrices exceed 5M elements).

- **`main.py`** — CLI entry point using the graph path. Exit code 0 = verified, 1 = unknown.

## Testing

- **`test_zonotope.py`** — Unit tests for `DenseZonotope` (bounds, FC propagation, ReLU cases). No external data needed.
- **`test_graph.py`** — Graph loading tests (sequential, cersyve dual-branch, ResNet with BN folding), zonotope add/copy tests, regression test (graph path matches flat path on ACAS Xu), and parametrized vnncomp benchmark tests. Benchmark tests run in subprocesses with 4GB memory caps and 120s timeouts. Each benchmark also checks soundness against onnxruntime (output of center point must fall within zonotope bounds).
- **`test_acasxu.py`** — Legacy integration tests against ACAS Xu using the flat loader path.
- External data paths configured in `tests/paths.yaml` (gitignored) with key `vnncomp_benchmarks` pointing to the benchmarks directory. Loaded via fixtures in `conftest.py`.

## Key Design Decisions

- **Graph-based pipeline**: `ComputeGraph` replaced the flat layer list as the primary representation. Non-sequential architectures (ResNets, parallel branches) are supported via topological traversal with fork/merge tracking.
- **Fork/merge zonotope semantics**: At fork points, zonotopes are copied. At Add merges, shared generator columns (from before the fork) are added element-wise while branch-specific columns are concatenated. The `shared_gens` count is found by walking ancestors to the common fork point.
- **Interval fallback**: Ops without zonotope relaxations (Sigmoid, LeakyRelu, ConvTranspose, bilinear MatMul) fall back to interval bounds re-wrapped as zonotopes. This loses correlations but stays sound.
- **BatchNorm folding**: BN nodes following Conv/Gemm are folded into the preceding layer's weights during graph construction (before verification).
- **Constant folding**: Chains of constant-only operations (e.g. MatMul(C,C) → Add(C,C) → Relu(C)) are evaluated at load time and stored as constants.
- Conv operations go through PyTorch (`torch.nn.functional.conv2d`) even though the rest is pure numpy.
- The multi-zonotope strategy intersects bounds from different ReLU relaxation types rather than picking one — this is the main precision mechanism.

## vnncomp Benchmark Support

19 of 25 benchmarks run end-to-end. Skipped benchmarks:
- `vggnet16_2022`, `safenlp_2024` — no ONNX files
- `soundnessbench` — OOM (128→12288 Gemm then Conv on 24×64×64)
- `cctsdb_yolo_2023` — complex preprocessing (Slice/Reshape/ScatterND before Conv)
- `collins_aerospace_benchmark` — feature pyramid Concat→Conv shape mismatch
- `ml4acopf_2024` — trig ops + complex broadcast patterns
- `vit_2023` — multi-head attention reshape + bilinear MatMul
