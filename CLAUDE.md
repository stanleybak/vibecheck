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

The verification pipeline flows: **ONNX loading → VNNLIB parsing → interval arithmetic bounds → zonotope propagation → margin check**.

- **`onnx_loader.py`** — Parses ONNX models into a flat list of layers. FC layers are `(W, b)` tuples; Conv layers are `(kernel, bias, params)` 3-tuples. Handles Gemm, MatMul+Add, Conv, Sub normalization folding, and FC→Conv composition. The `is_conv()` function distinguishes layer types by tuple length.

- **`spec.py`** — Parses VNNLIB specifications (supports `.gz`). Extracts input bounds (`x_lo`, `x_hi`) and output constraints identifying `pred_label` and `competitors`.

- **`bounds.py`** — Interval arithmetic (IA) bound propagation. Computes pre-ReLU and post-ReLU bounds per layer. Conv IA uses PyTorch's `F.conv2d` via numpy↔torch conversion.

- **`zonotope.py`** — Core verification engine. `DenseZonotope` represents sets as `{center + G @ e | ||e||_inf <= 1}`. `zonotope_verify()` runs multiple ReLU relaxation strategies (`min_area`, `y_bloat`, `box`) and intersects their results to tighten output bounds. Verification succeeds if `output_lo[pred] - output_hi[comp] > 0` for all competitors.

- **`main.py`** — CLI entry point. Exit code 0 = verified, 1 = unknown.

## Testing

- **`test_zonotope.py`** — Unit tests for `DenseZonotope` (bounds, FC propagation, ReLU cases). No external data needed.
- **`test_acasxu.py`** — Integration tests against ACAS Xu benchmarks. Requires `tests/paths.yaml` with a `vnncomp_benchmarks` path pointing to benchmark data. Tests skip automatically if not configured.
- External data paths are configured in `tests/paths.yaml` (gitignored) and loaded via fixtures in `conftest.py`.

## Key Design Decisions

- Layer type dispatch uses tuple length (`len(layer) == 3` → Conv), not class hierarchy.
- Conv operations in both bounds and zonotope propagation go through PyTorch (`torch.nn.functional.conv2d`) even though the rest is pure numpy.
- The multi-zonotope strategy intersects bounds from different ReLU relaxation types rather than picking one — this is the main precision mechanism.
