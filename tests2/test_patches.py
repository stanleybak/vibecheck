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


def test_patch_planes_parity():
    """conv -> planes -> conv: patch lower-bound adjoint + intercepts
    match the dense computation elementwise."""
    lm1, s1 = _conv(2, 4, 3, 1, 1, (2, 7, 7))
    lm2, s2 = _conv(4, 3, 3, 1, 1, s1)
    n1 = int(np.prod(s1))
    ll = torch.rand(1, n1) * 0.9
    lh = torch.rand(1, n1) * 0.9 + 0.1
    bl = torch.randn(1, n1) * 0.1
    bh = torch.randn(1, n1) * 0.1 + 0.3
    ch = 1
    pa = PatchAdjoint.identity(s2, ch)
    d = torch.zeros(1, pa.v.shape[1])
    pa = pa.through_conv(lm2.kernel, lm2.stride, lm2.padding, s1)
    pa, d = pa.through_planes(ll, bl, lh, bh, d)
    pa = pa.through_conv(lm1.kernel, lm1.stride, lm1.padding, (2, 7, 7))
    got = pa.to_dense()
    # dense reference
    A = _dense_adjoint([lm2], s2, ch)[0]         # (Q, n1)
    pos = A > 0
    d_ref = (A * torch.where(pos, bl, bh)).sum(dim=1, keepdim=True).T
    A = A * torch.where(pos, ll, lh)
    A = lm1.lin_t(A).unsqueeze(0)
    assert torch.allclose(got, A, atol=1e-5), float((got - A).abs().max())
    assert torch.allclose(d, d_ref, atol=1e-5), \
        float((d - d_ref).abs().max())


def test_patch_refine_parity_conv_relu_conv():
    """patch_refine == the dense identity-crown refinement on a
    conv->relu->conv net (both bounds, all elements)."""
    import torch
    from vibecheck2.core.graph import Net, Op
    from vibecheck2.core import backward
    from vibecheck2.core.patches import patch_refine

    lm1, s1 = _conv(2, 4, 3, 1, 1, (2, 7, 7))
    lm2, s2 = _conv(4, 3, 3, 2, 1, s1)
    n_in = 2 * 7 * 7
    ops = {
        'x': Op('x', 'input', (), (n_in,), n_in),
        'c1': Op('c1', 'linmap', ('x',), s1, int(np.prod(s1)), lm=lm1),
        'r1': Op('r1', 'nonlin', ('c1',), s1, int(np.prod(s1)), fn='relu'),
        'c2': Op('c2', 'linmap', ('r1',), s2, int(np.prod(s2)), lm=lm2),
    }
    net = Net(ops, ['c1', 'r1', 'c2'], 'x', 'c2')
    lo = -torch.rand(1, n_in)
    hi = torch.rand(1, n_in)
    inter = backward.intermediates(net, lo, hi)
    lb_p, ub_p = patch_refine(net, 'c2', lo, hi, inter)
    # dense reference: +/- identity rows through backward.crown
    n2 = int(np.prod(s2))
    Wc = torch.cat([torch.eye(n2), -torch.eye(n2)])
    out = backward.crown(net, lo, hi, Wc.unsqueeze(0), inter, start='c2')
    lb_d, ub_d = out[:, :n2], -out[:, n2:]
    assert torch.allclose(lb_p, lb_d, atol=1e-4), \
        float((lb_p - lb_d).abs().max())
    assert torch.allclose(ub_p, ub_d, atol=1e-4), \
        float((ub_p - ub_d).abs().max())
