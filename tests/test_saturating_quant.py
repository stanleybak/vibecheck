"""Saturating quantized-GEMM twins (handlers/saturating_quant).

These kernels reproduce the non-VNNI MLAS behavior (VPMADDUBSW adjacent
pair-sums saturating at int16) that made smart_turn platform-dependent.
Tests pin: exact-regime agreement with the true int32 quantized matmul,
the saturation divergence itself, numpy/torch twin equivalence (what the
PGD surrogate relies on), and conv-vs-reference equivalence.
"""
import numpy as np
import torch

from vibecheck.handlers.saturating_quant import (
    sat_conv, sat_conv_torch, sat_matmul, sat_matmul_torch)

_rng = np.random.default_rng(7)


def _exact_qmatmul(a_u8, b_s8, a_zp, b_zp, a_s, b_s, y_s, y_zp):
    """The VNNI/int32 reference: exact accumulation, then requantize."""
    acc = (a_u8.astype(np.int64) - a_zp) @ (b_s8.astype(np.int64) - b_zp)
    y = np.rint(acc.astype(np.float64) * (a_s * b_s / y_s)) + y_zp
    return np.clip(y, 0, 255).astype(np.uint8)


def test_sat_matmul_matches_exact_when_no_saturation():
    """Small codes keep every adjacent pair-sum far below +-32767, so the
    saturating twin must agree with the exact int32 GEMM bit-for-bit."""
    a = _rng.integers(120, 136, size=(5, 8)).astype(np.uint8)
    b = _rng.integers(-4, 5, size=(8, 3)).astype(np.int8)
    got = sat_matmul(a, b, a_zp=128, b_zp=0, a_s=0.02, b_s=0.05,
                     y_s=0.003, y_zp=10)
    want = _exact_qmatmul(a, b, 128, 0, 0.02, 0.05, 0.003, 10)
    assert (got == want).all()


def test_sat_matmul_saturates_pair_sums():
    """255*127 + 255*127 = 64770 per pair clips to 32767: the twin must
    reproduce the clipped accumulator, not the exact one."""
    K = 4
    a = np.full((1, K), 255, np.uint8)
    b = np.full((K, 1), 127, np.int8)
    scale = 1.0 / 1024.0                       # a_s*b_s/y_s
    got = sat_matmul(a, b, a_zp=0, b_zp=0, a_s=1.0, b_s=1.0,
                     y_s=1024.0, y_zp=0)
    exact = _exact_qmatmul(a, b, 0, 0, 1.0, 1.0, 1024.0, 0)
    # two saturated pairs: acc = 2*32767 = 65534 -> rint(63.998) = 64
    assert got.ravel()[0] == round(2 * 32767 * scale) == 64
    assert exact.ravel()[0] == round(4 * 255 * 127 * scale) == 127
    assert got.ravel()[0] != exact.ravel()[0]


def test_sat_matmul_odd_k_tail_unpaired():
    """Odd K: the last product is a lone int16 term (no pair to saturate
    with); against small codes this equals the exact GEMM."""
    a = _rng.integers(0, 6, size=(3, 7)).astype(np.uint8)
    b = _rng.integers(-5, 6, size=(7, 4)).astype(np.int8)
    got = sat_matmul(a, b, a_zp=2, b_zp=1, a_s=0.1, b_s=0.1,
                     y_s=0.01, y_zp=100)
    want = _exact_qmatmul(a, b, 2, 1, 0.1, 0.1, 0.01, 100)
    assert (got == want).all()


def test_sat_matmul_torch_twin_matches_numpy():
    """hard=True torch forward must be bit-identical to the numpy twin in
    BOTH regimes (mixed saturating and small rows in one batch)."""
    a = np.vstack([np.full((2, 8), 255, np.uint8),
                   _rng.integers(0, 9, size=(3, 8)).astype(np.uint8)])
    b = np.vstack([np.full((4, 3), 127, np.int8),
                   _rng.integers(-6, 7, size=(4, 3)).astype(np.int8)])
    args = dict(a_zp=3, b_zp=-2, a_s=0.7, b_s=0.9, y_s=41.0, y_zp=17)
    want = sat_matmul(a, b, **args)
    got = sat_matmul_torch(torch.tensor(a, dtype=torch.float64),
                           torch.tensor(b, dtype=torch.float64), **args)
    assert (got.numpy().astype(np.uint8) == want).all()


def test_sat_matmul_torch_soft_mode_differentiable():
    """hard=False is the PGD surrogate: continuous output, gradients flow
    through the soft saturation."""
    a = torch.tensor(np.full((1, 8), 200, np.float64), requires_grad=True)
    b = torch.tensor(np.full((8, 1), 100, np.float64))
    y = sat_matmul_torch(a, b, a_zp=0, b_zp=0, a_s=1.0, b_s=1.0,
                         y_s=1000.0, y_zp=0, hard=False)
    y.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_sat_conv_matches_exact_conv_when_no_saturation():
    """Small codes: the saturating conv equals a plain integer conv computed
    independently via F.conv2d on the zero-point-shifted codes."""
    x = _rng.integers(6, 14, size=(1, 2, 5, 5)).astype(np.uint8)
    w = _rng.integers(-5, 6, size=(3, 2, 3, 3)).astype(np.int8)
    bias = _rng.integers(-40, 40, size=3).astype(np.int32)
    x_zp, w_zp, x_s, w_s, y_s, y_zp = 9, 0, 0.15, 0.1, 0.02, 30
    got = sat_conv(x, w, x_zp, w_zp, x_s, w_s, y_s, y_zp,
                   bias=bias, stride=2, pad=1)
    acc = torch.nn.functional.conv2d(
        torch.tensor(x.astype(np.float64) - x_zp).double(),
        torch.tensor(w.astype(np.float64) - w_zp).double(),
        stride=2, padding=1).numpy() + bias.reshape(1, -1, 1, 1)
    want = np.clip(np.rint(acc * (x_s * w_s / y_s)) + y_zp,
                   0, 255).astype(np.uint8)
    assert (got == want).all()


def test_sat_conv_torch_twin_matches_numpy():
    x = _rng.integers(0, 256, size=(1, 2, 6, 6)).astype(np.uint8)
    w = np.full((2, 2, 2, 2), 127, np.int8)   # saturating against big x rows
    args = dict(x_zp=5, w_zp=0, x_s=0.3, w_s=0.2, y_s=7.0, y_zp=12)
    want = sat_conv(x, w, stride=2, pad=0, **args)
    got = sat_conv_torch(torch.tensor(x, dtype=torch.float64),
                         torch.tensor(w, dtype=torch.float64),
                         stride=2, pad=0, **args)
    assert (got.numpy().astype(np.uint8) == want).all()
