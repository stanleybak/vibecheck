"""VibeCheck — Vibe-Coded Neural Network Verification Tool.

Clean-slate verifier core (the 1.1.0 rewrite; see
docs/clean_slate_design.md): one IR, one forward propagator, one backward
propagator, one attack engine, one BaB search, with the battle-tested 1.0
front end (ONNX loading, VNNLIB parsing, CE validation) ported in under
frontend/. Soundness > design > size > speed, in that order.
"""

# Force single-threaded BLAS — multi-threaded OpenBLAS causes massive
# overhead on the small matrices typical in verification workloads.
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
# CUDA allocator: expandable segments drastically reduce fragmentation for
# the zonotope workflow, where G matrices grow by concatenation each ReLU
# layer. Without this, a 2.3 GB working set fragments so badly that
# allocating the next 1 GB G-cat fails even with 5+ GB formally free.
_os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
del _os

# Disable TF32 on Ampere+ GPUs. By default cuDNN runs convolutions in TF32
# (10-bit mantissa), which introduces ~4e-3 forward error on conv nets.
# That error swamps knife-edge counterexample margins (PGD chases spurious
# witnesses and misses real CEXes) AND makes verification BOUNDS unsound at
# tight margins (a `verified` with certified margin < ~4e-3 may not hold
# for the true model). LinMap computes convs via F.conv2d, so this is the
# single switch its exactness depends on; matmul TF32 already defaults off
# in recent torch, cudnn does not — set both explicitly.
import torch as _torch
_torch.backends.cuda.matmul.allow_tf32 = False
_torch.backends.cudnn.allow_tf32 = False
del _torch

from .api import Spec, VerifyResult, verify
