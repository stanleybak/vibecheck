"""PatchAdjoint parity vs the dense conv adjoint (M5 step 1)."""
import numpy as np
import torch

from vibecheck2.core.linmap import Conv2d
from vibecheck2.core.patches import PatchAdjoint

RNG = np.random.default_rng(7)


def _conv(ci, co, k, stride, pad, ish):
    kern = RNG.standard_normal((co, ci, k, k)).astype(np.float32) * 0.5
    H2 = (ish[1] + 2 * pad - k) // stride + 1
    W2 = (ish[2] + 2 * pad - k) // stride + 1
    lm = Conv2d(kern, None, ish, (co, H2, W2), (stride, stride), (pad, pad))
    return lm, (co, H2, W2)


def _dense_adjoint(lms, edge_shape, channel):
    """Identity rows at (channel, :, :) pulled back via lin_t."""
    C, H, W = edge_shape
    n = C * H * W
    Q = H * W
    A = torch.zeros(Q, n)
    idx = channel * H * W + torch.arange(Q)
    A[torch.arange(Q), idx] = 1.0
    for lm in reversed(lms):
        A = lm.lin_t(A)
    return A.unsqueeze(0)                    # (1, Q, n_in)


def test_patch_conv_parity_stride1():
    lm1, s1 = _conv(3, 5, 3, 1, 1, (3, 8, 8))
    lm2, s2 = _conv(5, 4, 3, 1, 1, s1)
    for ch in (0, 3):
        pa = PatchAdjoint.identity(s2, ch)
        pa = pa.through_conv(lm2.kernel, lm2.stride, lm2.padding, s1)
        pa = pa.through_conv(lm1.kernel, lm1.stride, lm1.padding, (3, 8, 8))
        dense = _dense_adjoint([lm1, lm2], s2, ch)
        got = pa.to_dense()
        assert torch.allclose(got, dense, atol=1e-5), \
            float((got - dense).abs().max())


def test_patch_conv_parity_stride2():
    lm1, s1 = _conv(2, 6, 5, 2, 2, (2, 12, 12))
    lm2, s2 = _conv(6, 3, 3, 2, 1, s1)
    for ch in (0, 2):
        pa = PatchAdjoint.identity(s2, ch)
        pa = pa.through_conv(lm2.kernel, lm2.stride, lm2.padding, s1)
        pa = pa.through_conv(lm1.kernel, lm1.stride, lm1.padding, (2, 12, 12))
        dense = _dense_adjoint([lm1, lm2], s2, ch)
        got = pa.to_dense()
        assert torch.allclose(got, dense, atol=1e-5), \
            float((got - dense).abs().max())


def test_patch_memory_is_window_sized():
    lm1, s1 = _conv(3, 8, 3, 1, 1, (3, 32, 32))
    pa = PatchAdjoint.identity(s1, 1)
    pa = pa.through_conv(lm1.kernel, lm1.stride, lm1.padding, (3, 32, 32))
    # 1024 queries x 3 ch x 3x3 window vs dense 1024 x 3072
    assert pa.v.numel() == 1024 * 3 * 3 * 3
