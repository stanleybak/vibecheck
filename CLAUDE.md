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

# Run a single benchmark test
.venv/bin/python -m pytest "tests/test_graph.py::test_vnncomp_benchmark[acasxu_2023/ACASXU_run2a_1_1_batch_2000]" -v -s

# Run the verifier
.venv/bin/vibecheck --net model.onnx --spec property.vnnlib
```

## Architecture

The verification pipeline flows: **ONNX loading → graph construction → zonotope propagation → spec check**.

The codebase uses **object-oriented dispatch**: each ONNX op type is a `GraphNode` subclass (in `network.py`) that implements `infer_shape()` and `zonotope_propagate()`. The verifier loop simply iterates in topological order and calls these methods.

- **`network.py`** — Core graph representation. `ComputeGraph` holds a DAG of `GraphNode` subclass instances keyed by tensor name. Each subclass (`ConvNode`, `ReluNode`, `AddNode`, `TransposeNode`, `SliceNode`, etc.) implements its own shape inference and zonotope propagation. Also contains: `OP_REGISTRY` (maps ONNX op strings to subclasses), `_find_shared_gens()` for fork/merge tracking. Use `print(graph)` for a structural summary showing topo indices, shapes, predecessor/successor connections, and fork points.

- **`onnx_loader.py`** — Loads ONNX models into `ComputeGraph`. Parses ONNX nodes into the right `GraphNode` subclass via `OP_REGISTRY`, performs constant folding, topological sort, shape inference (`node.infer_shape()`), and BatchNorm folding into preceding Conv/Gemm. Sets `SplitOutput` shapes from parent `Split` params.

- **`spec.py`** — OOP specification types. `VNNSpec` holds input bounds + disjunction of `Conjunct`s (DNF). Each `Conjunct` contains `Constraint` (threshold: `Y_i >= val`) or `PairwiseConstraint` (`Y_comp >= Y_pred`). `VNNSpec.check(output_lo, output_hi)` evaluates margins — positive means verified safe.

- **`vnnlib_loader.py`** — VNNLIB file parsing. `load_vnnlib(path)` reads `.vnnlib` / `.vnnlib.gz` files and returns a `VNNSpec`. Supports pairwise output constraints, threshold constraints, and `(or (and ...))` disjunctive normal form with mixed input/output constraints.

- **`zonotope.py`** — `DenseZonotope` represents sets as `{center + G @ e | ||e||_inf <= 1}`. Methods: `propagate_linear()` (FC/Conv), `apply_relu()` with three relaxation types (`min_area`, `y_bloat`, `box`), `copy()` for fork points, `add(other, shared_gens)` for skip connection merges. The add method splits generators into shared prefix (added element-wise) and branch-specific suffix (concatenated).

- **`verify.py`** — Thin dispatch loop: `zonotope_verify(graph, spec)` iterates topo order calling `node.zonotope_propagate()`, then calls `spec.check()` on the output bounds.

- **`main.py`** — CLI entry point. Exit code 0 = verified, 1 = unknown.

## Shapes

All tensor shapes **include the batch dimension** (always 1). For example:
- FC input: `(1, 5)` not `(5,)`
- Conv input: `(1, 3, 32, 32)` not `(3, 32, 32)`
- NHWC input: `(1, 30, 30, 3)`

This means ONNX axes and permutations work directly without batch-dim adjustment. Torch spatial ops (`conv2d`, `max_pool2d`, etc.) strip the batch via `shape[1:]` internally.

## Zonotope Propagation

- **Ops with full zonotope support** (work with any number of generators): Conv (1D/2D), Gemm/MatMul, Relu, Add, Sub, Mul (scale), Div (scale), Neg, BatchNorm, Concat, Split, Slice, Gather, Reduce, Transpose, Reshape, Flatten, passthrough ops.
- **Point-only ops** (require 0 generators, execute concretely on center): Sigmoid, Tanh, LeakyRelu, Clip, Sign, Softmax, Pow, Sin/Cos, ConvTranspose, MaxPool, AvgPool, Pad, Resize/Upsample.
- Conv/ConvTranspose/Pool use PyTorch (`torch.nn.functional`) for both zonotope propagation and point execution.

## Testing

Tests use pytest. Unit tests cover zonotope math and individual op propagation. Integration tests load real ONNX networks from vnncomp benchmarks (discovered via `instances.csv`), run point propagation, and validate against onnxruntime. On soundness failure, per-node comparison identifies the divergent op. External benchmark paths configured in `tests/paths.yaml` (gitignored, template at `tests/paths.yaml.template`).
