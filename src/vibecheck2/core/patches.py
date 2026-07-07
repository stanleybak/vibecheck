"""Patch-structured adjoints for conv chains (M5, design doc section).

A dense backward adjoint for identity queries at a conv edge is
(B, Q, n_prev) with Q up to millions -- it never fits on ImageNet-scale
nets (vgg16-7: 3.2M queries x 150k inputs). But a conv adjoint of an
identity query is a LOCAL WINDOW: it starts as a 1x1 one-hot patch and
each conv composition grows it by (k-1)*step, staying tiny for the
shallow chains that intermediate-bound computation walks.

PatchAdjoint stores the window values for a REGULAR GRID of queries
(every spatial position of one query channel; channels are the caller's
chunk axis) plus the affine anchor bookkeeping:

    anchor(qy, qx) = base + (qy, qx) * step        (edge coordinates)

through_conv is EXACT (conv_transpose2d on the window values; the same
adjoint the dense path computes, verified by the parity tests), and
scale/elementwise ops gather their per-position factors at the patch
footprint. to_dense materializes (B, Q, n) for parity tests and for the
fallback at the first op that cannot stay patched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PatchAdjoint:
    """values (B, Q, C, ph, pw) over a (gh, gw) query grid; Q = gh*gw.

    base/step map grid indices to spatial anchors in the CURRENT edge's
    (C, H, W) coordinates: window element (c, u, v) of query (qy, qx)
    addresses edge position (c, base_y + qy*step_y + u,
    base_x + qx*step_x + v); out-of-range positions (conv padding) are
    structural zeros.
    """

    def __init__(self, values, grid, base, step, edge_shape):
        self.v = values                      # (B, Q, C, ph, pw)
        self.grid = tuple(grid)              # (gh, gw)
        self.base = tuple(base)              # (by, bx), may be negative
        self.step = tuple(step)              # (sy, sx)
        self.edge_shape = tuple(edge_shape)  # (C, H, W) of the edge

    @staticmethod
    def identity(edge_shape, channel, B=1, device='cpu', dtype=torch.float32):
        """One query per spatial position of `channel`: 1x1 one-hot."""
        C, H, W = edge_shape
        v = torch.zeros(B, H * W, 1, 1, 1, device=device, dtype=dtype)
        v[:, :, 0, 0, 0] = 1.0
        pa = PatchAdjoint(v, (H, W), (0, 0), (1, 1), edge_shape)
        pa.channel = channel                 # window C-axis = [channel]
        pa.chan_lo = channel                 # first edge channel in window
        return pa

    def through_conv(self, kernel, stride, padding, in_shape):
        """Compose with y = conv2d(x, kernel): adjoint w.r.t. x.

        kernel (Co, Ci, kh, kw); self's window channels must cover the
        conv's OUTPUT channels (full Co window, or the 1-channel identity
        start whose channel selects a kernel slice)."""
        B, Q, Cw, ph, pw = self.v.shape
        k = torch.as_tensor(kernel, device=self.v.device, dtype=self.v.dtype)
        if Cw != k.shape[0]:
            assert Cw == 1, 'window channels must be full Co or the 1-ch start'
            k = k[self.chan_lo:self.chan_lo + 1]
        sy, sx = stride
        # clip window elements overhanging THIS edge (conv-padding
        # positions are structural zeros; without the clip, boundary
        # queries leak contributions "through the padding" -- measured:
        # interior queries exact, every boundary query wrong)
        _, H, W = self.edge_shape
        dev = self.v.device
        gh, gw = self.grid
        qy = torch.arange(gh, device=dev).repeat_interleave(gw)
        qx = torch.arange(gw, device=dev).repeat(gh)
        yy = (self.base[0] + qy * self.step[0]).unsqueeze(1) \
            + torch.arange(ph, device=dev)                     # (Q, ph)
        xx = (self.base[1] + qx * self.step[1]).unsqueeze(1) \
            + torch.arange(pw, device=dev)                     # (Q, pw)
        mask = (((yy >= 0) & (yy < H)).unsqueeze(2)
                & ((xx >= 0) & (xx < W)).unsqueeze(1))         # (Q, ph, pw)
        v = self.v * mask.unsqueeze(0).unsqueeze(2).to(self.v.dtype)
        vals = v.reshape(B * Q, Cw, ph, pw)
        out = F.conv_transpose2d(vals, k, stride=stride)
        ph2, pw2 = out.shape[2], out.shape[3]
        pa = PatchAdjoint(
            out.reshape(B, Q, -1, ph2, pw2), self.grid,
            (self.base[0] * sy - padding[0], self.base[1] * sx - padding[1]),
            (self.step[0] * sy, self.step[1] * sx), tuple(in_shape))
        pa.chan_lo = 0
        return pa

    def to_dense(self, B_out=None):
        """Materialize (B, Q, C*H*W); padding overhang clips to zero."""
        B, Q, Cw, ph, pw = self.v.shape
        C, H, W = self.edge_shape
        gh, gw = self.grid
        dev = self.v.device
        out = torch.zeros(B, Q, C, H, W, device=dev, dtype=self.v.dtype)
        qy = torch.arange(gh, device=dev).repeat_interleave(gw)
        qx = torch.arange(gw, device=dev).repeat(gh)
        ay = self.base[0] + qy * self.step[0]          # (Q,)
        ax = self.base[1] + qx * self.step[1]
        cw0 = getattr(self, 'chan_lo', 0)
        for u in range(ph):
            yy = ay + u
            ok_y = (yy >= 0) & (yy < H)
            for v in range(pw):
                xx = ax + v
                ok = ok_y & (xx >= 0) & (xx < W)
                if not bool(ok.any()):
                    continue
                qi = torch.nonzero(ok, as_tuple=False).flatten()
                # advanced indices (dims 1, 3, 4) separated by the
                # channel slice -> the indexed result is (Nq, B, Cw)
                out[:, qi, cw0:cw0 + Cw, yy[qi], xx[qi]] = \
                    self.v[:, qi, :, u, v].permute(1, 0, 2)
        return out.reshape(B, Q, C * H * W)

    def gather_edge(self, t):
        """Window-aligned values of an edge tensor t (B, C*H*W) ->
        (B, Q, Cw, ph, pw): element (q, c, u, v) reads t at this window
        element's edge position (out-of-range -> 0). The regular anchor
        grid makes this one padded unfold."""
        B, Q, Cw, ph, pw = self.v.shape
        C, H, W = self.edge_shape
        gh, gw = self.grid
        sy, sx = self.step
        by, bx = self.base
        t4 = t.reshape(B, C, H, W)
        # pad so every anchor lands in-range: left/top by -base (if
        # negative), right/bottom to cover the last window
        pl, pt = max(0, -bx), max(0, -by)
        pr = max(0, bx + (gw - 1) * sx + pw - W)
        pb = max(0, by + (gh - 1) * sy + ph - H)
        t4 = F.pad(t4, (pl, pr, pt, pb))
        u = F.unfold(t4, (ph, pw), stride=(sy, sx))    # (B, C*ph*pw, L)
        u = u.reshape(B, C, ph, pw, -1)
        gh_a = (t4.shape[2] - ph) // sy + 1
        gw_a = (t4.shape[3] - pw) // sx + 1
        u = u.reshape(B, C, ph, pw, gh_a, gw_a)
        # anchor (0,0) of the query grid sits at padded coord
        # (by+pt, bx+pl), stride-aligned by construction
        oy, ox = (by + pt) // sy, (bx + pl) // sx
        u = u[:, :, :, :, oy:oy + gh, ox:ox + gw]
        u = u.permute(0, 4, 5, 1, 2, 3).reshape(B, Q, C, ph, pw)
        cw0 = getattr(self, 'chan_lo', 0)
        return u[:, :, cw0:cw0 + Cw]

    def through_planes(self, lam_lo, b_lo, lam_hi, b_hi, d):
        """Compose with a nonlin's two-sided linear planes (lower-bound
        adjoint): positive window coefficients take the lower plane,
        negative the upper; intercepts accumulate into d (B, Q).
        Returns (new PatchAdjoint, d)."""
        ll = self.gather_edge(lam_lo)
        lh = self.gather_edge(lam_hi)
        bl = self.gather_edge(b_lo)
        bh = self.gather_edge(b_hi)
        pos = self.v > 0
        v2 = self.v * torch.where(pos, ll, lh)
        d = d + (self.v * torch.where(pos, bl, bh)).sum(dim=(2, 3, 4))
        pa = PatchAdjoint(v2, self.grid, self.base, self.step,
                          self.edge_shape)
        pa.chan_lo = getattr(self, 'chan_lo', 0)
        return pa, d
