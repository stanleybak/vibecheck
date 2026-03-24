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

The codebase uses an **object-oriented dispatch** pattern: each ONNX op type is a `GraphNode` subclass (in `network.py`) that implements `infer_shape()`, `ia_bounds()`, and `zonotope_propagate()`. The main loops in `bounds.py` and `verify.py` simply iterate in topological order and call these methods.

- **`network.py`** — Core graph representation. `ComputeGraph` holds a DAG of `GraphNode` subclass instances keyed by tensor name. Each subclass (`ConvNode`, `ReluNode`, `AddNode`, etc.) implements its own shape inference, IA bounds, and zonotope propagation. Also contains: `OP_REGISTRY` (maps ONNX op strings to subclasses), `_find_shared_gens()` for fork/merge tracking, and torch-based IA helpers for spatial ops. Use `print(graph)` for a structural summary showing topo indices, shapes, predecessor/successor connections, and fork points.

- **`onnx_loader.py`** — Loads ONNX models into `ComputeGraph`. Parses ONNX nodes into the right `GraphNode` subclass via `OP_REGISTRY`, performs constant folding, topological sort, shape inference (`node.infer_shape()`), and BatchNorm folding into preceding Conv/Gemm.

- **`spec.py`** — Parses VNNLIB specifications (supports `.gz`). Extracts input bounds (`x_lo`, `x_hi`) and output constraints identifying `pred_label` and `competitors`.

- **`bounds.py`** — Thin dispatch loop: `ia_bounds_graph()` iterates topo order calling `node.ia_bounds()` on each node. Returns `dict[node_name -> (lo, hi)]`.

- **`zonotope.py`** — `DenseZonotope` represents sets as `{center + G @ e | ||e||_inf <= 1}`. Methods: `propagate_linear()` (FC/Conv), `apply_relu()` with three relaxation types (`min_area`, `y_bloat`, `box`), `copy()` for fork points, `add(other, shared_gens)` for skip connection merges. The add method splits generators into shared prefix (added element-wise) and branch-specific suffix (concatenated).

- **`verify.py`** — Thin dispatch loop: `zonotope_verify()` iterates topo order calling `node.zonotope_propagate()`. For ops without zonotope support, point zonotopes (0 generators) use IA; non-point zonotopes raise `NotImplementedError`.

- **`graph.py`** — Backward-compat shim re-exporting `ComputeGraph` and `GraphNode` from `network.py`.

- **`main.py`** — CLI entry point. Exit code 0 = verified, 1 = unknown.

## Testing

- **`test_zonotope.py`** — Unit tests for `DenseZonotope` (bounds, FC propagation, ReLU cases). No external data needed.
- **`test_graph.py`** — Graph loading tests (sequential, cersyve dual-branch, ResNet with BN folding), zonotope add/copy tests, and parametrized vnncomp benchmark tests. Benchmark tests run in subprocesses with 16GB memory caps and 120s timeouts. Tests use point zonotopes (x_lo == x_hi) at the spec center for fast propagation, with soundness verified against onnxruntime.
- **`test_acasxu.py`** — ACAS Xu integration tests using the graph-based pipeline.
- External data paths configured in `tests/paths.yaml` (gitignored) with key `vnncomp_benchmarks` pointing to the benchmarks directory. Loaded via fixtures in `conftest.py`.

## Key Design Decisions

- **Graph-based pipeline**: `ComputeGraph` is the core representation. Non-sequential architectures (ResNets, parallel branches) are supported via topological traversal with fork/merge tracking.
- **Fork/merge zonotope semantics**: At fork points, zonotopes are copied. At Add merges, shared generator columns (from before the fork) are added element-wise while branch-specific columns are concatenated. The `shared_gens` count is found by walking ancestors to the common fork point.
- **Point propagation fallback**: Ops without zonotope relaxations (Sigmoid, LeakyRelu, ConvTranspose, bilinear MatMul) can propagate point zonotopes (0 generators) via IA on a zero-width interval. Non-point zonotopes raise `NotImplementedError`.
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
