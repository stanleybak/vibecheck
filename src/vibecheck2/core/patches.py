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
        ci = getattr(self, 'cidx', None)
        if ci is not None:
            k = k[ci]
        elif Cw != k.shape[0]:
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
        ci = getattr(self, 'cidx', None)
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
                if ci is not None:
                    for kci, ch in enumerate(ci.tolist()
                                             if hasattr(ci, 'tolist')
                                             else ci):
                        out[:, qi, ch, yy[qi], xx[qi]] = \
                            self.v[:, qi, kci, u, v]
                else:
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
        ci = getattr(self, 'cidx', None)
        if ci is not None:
            return u[:, :, ci]
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
        if getattr(self, 'cidx', None) is not None:
            pa.cidx = self.cidx
        return pa, d



    def through_block_sel(self, blocks, n_blocks):
        """Backward through a pair-block gather (decompose_maxpool's
        u/v Selects): the edge's leading blocks fold into the window's
        channel axis, so the gather is a pure channel REMAP -- values,
        anchors and grid are untouched. Current edge (P*Cb, OH, OW) with
        P = len(blocks) maps to input edge (n_blocks*Cb, OH, OW)."""
        P = len(blocks)
        Cf, H, W = self.edge_shape
        assert Cf % P == 0, (Cf, P)
        Cb = Cf // P
        ci = getattr(self, 'cidx', None)
        if ci is None:
            cw0 = getattr(self, 'chan_lo', 0)
            ci = torch.arange(cw0, cw0 + self.v.shape[2],
                              device=self.v.device)
        blk = torch.as_tensor(blocks, device=self.v.device,
                              dtype=torch.long)
        ci2 = blk[ci // Cb] * Cb + (ci % Cb)
        pa = PatchAdjoint(self.v, self.grid, self.base, self.step,
                          (n_blocks * Cb, H, W))
        pa.cidx = ci2
        return pa

    def through_pool_win(self, in_shape, stride, offsets):
        """Backward through the window-stacking Select (decompose_maxpool
        `/w`, REGULAR pools only): window channel (p, c) at element
        (u, v) maps to input channel c at (u*sh + i_p, v*sw + j_p);
        blocks with equal c ACCUMULATE (adjoint linearity). Anchors and
        steps scale by the pool stride; the window materializes densely
        (zero interleave), growing (ph-1)*(s-1) + max offset."""
        B, Q, Cw, ph, pw = self.v.shape
        C, H, W = in_shape
        sh, sw = stride
        cols = len(offsets)
        Cf = self.edge_shape[0]
        assert Cf == cols * C, (Cf, cols, C)
        ci = getattr(self, 'cidx', None)
        if ci is None:
            cw0 = getattr(self, 'chan_lo', 0)
            ci = torch.arange(cw0, cw0 + Cw, device=self.v.device)
        p_of = (ci // C).tolist()
        c_of = ci % C
        cmap = torch.unique(c_of, sorted=True)
        pos_of = {int(c): k for k, c in enumerate(cmap.tolist())}
        kh = max(o[0] for o in offsets) + 1
        kw = max(o[1] for o in offsets) + 1
        ph2 = (ph - 1) * sh + kh
        pw2 = (pw - 1) * sw + kw
        v2 = torch.zeros(B, Q, cmap.numel(), ph2, pw2,
                         device=self.v.device, dtype=self.v.dtype)
        for k in range(Cw):
            i_p, j_p = offsets[p_of[k]]
            tgt = pos_of[int(c_of[k])]
            v2[:, :, tgt, i_p::sh, j_p::sw][:, :, :ph, :pw] += \
                self.v[:, :, k]
        pa = PatchAdjoint(v2, self.grid,
                          (self.base[0] * sh, self.base[1] * sw),
                          (self.step[0] * sh, self.step[1] * sw),
                          (C, H, W))
        pa.cidx = cmap
        return pa


def patch_refine(net, edge, lo, hi, inter, chan_chunk=8, device=None):
    """Bounds for EVERY element of conv edge `edge` via patch-structured
    backward CROWN: identity queries per (channel, y, x), chunked by
    channel, walked back to the network input while every op stays
    patchable (Conv2d / Scale / ScaleShift linmaps, relu planes, add of
    two aligned patch paths). Raises NotImplementedError at the first op
    it cannot keep patched -- the caller falls back to the dense path.

    Returns (lb, ub) each (B, n_edge). Memory is O(chunk x window),
    never O(chunk x n_layer): the dense identity refinement on
    vgg16-7's 3.2M-neuron edges is unrunnable at any chunk size.
    """
    from .linmap import Conv2d, Scale, ScaleShift, Select
    from .relax import REL
    op_of = dict(net.ops)      # net.order excludes the input op
    B = lo.shape[0]
    dev = lo.device
    eop = op_of[edge]
    if eop.kind != 'linmap' or not isinstance(eop.lm, Conv2d):
        # the grid query needs a (C, H, W) edge; conv outputs carry it
        raise NotImplementedError(f'patch_refine: {edge} is not a conv edge')
    C, H, W = eop.lm.out_shape
    c_in = (hi + lo) / 2
    r_in = (hi - lo) / 2

    def _cidx(pa):
        ci = getattr(pa, 'cidx', None)
        if ci is None:
            cw0 = getattr(pa, 'chan_lo', 0)
            ci = torch.arange(cw0, cw0 + pa.v.shape[2], device=pa.v.device)
        return ci

    def _merge(pa_a, pa_b):
        """Sum two adjoints on the SAME edge (crown accumulates incoming
        adjoints BEFORE relaxing the edge's producer -- per-path plane
        choices are sound but measurably looser through the pool tree's
        reconvergent adds). Requires aligned anchors; unions channels."""
        assert pa_a.grid == pa_b.grid and pa_a.base == pa_b.base \
            and pa_a.step == pa_b.step \
            and pa_a.edge_shape == pa_b.edge_shape, 'adjoint merge misaligned'
        ph = max(pa_a.v.shape[3], pa_b.v.shape[3])
        pw = max(pa_a.v.shape[4], pa_b.v.shape[4])
        ca, cb = _cidx(pa_a), _cidx(pa_b)
        cu = torch.unique(torch.cat([ca, cb]), sorted=True)
        pos = {int(c): k for k, c in enumerate(cu.tolist())}
        v = torch.zeros(pa_a.v.shape[0], pa_a.v.shape[1], cu.numel(),
                        ph, pw, device=pa_a.v.device, dtype=pa_a.v.dtype)
        for pa_x, cx in ((pa_a, ca), (pa_b, cb)):
            tgt = torch.as_tensor([pos[int(c)] for c in cx.tolist()],
                                  device=v.device)
            v[:, :, tgt, :pa_x.v.shape[3], :pa_x.v.shape[4]] += pa_x.v
        pa = PatchAdjoint(v, pa_a.grid, pa_a.base, pa_a.step,
                          pa_a.edge_shape)
        pa.cidx = cu
        return pa

    def walk(pa0, d, nm0):
        """Reverse-topological accumulation from edge nm0's OUTPUT patch
        to the input: incoming adjoints per edge are MERGED before the
        edge's producer op is applied (matching the dense crown exactly),
        then concretized at the input."""
        order_pos = {nm: i for i, nm in enumerate(net.order)}
        acc = {nm0: pa0}
        total = None
        while acc:
            nm = max(acc, key=lambda k: order_pos.get(k, -1))
            pa = acc.pop(nm)
            op = op_of[nm]
            if op.kind == 'input':
                cw = pa.gather_edge(c_in)
                rw = pa.gather_edge(r_in)
                lb = (pa.v * cw).sum(dim=(2, 3, 4)) \
                    - (pa.v.abs() * rw).sum(dim=(2, 3, 4))
                total = lb if total is None else total + lb
                continue

            def _push(nm_in, pa_in):
                if nm_in in acc:
                    acc[nm_in] = _merge(acc[nm_in], pa_in)
                else:
                    acc[nm_in] = pa_in

            if op.kind == 'linmap':
                lm = op.lm
                if isinstance(lm, Conv2d):
                    if lm.b is not None:
                        bv = torch.as_tensor(
                            lm.bias_vec(pa.v), device=dev,
                            dtype=pa.v.dtype).unsqueeze(0).expand(B, -1)
                        d = d + (pa.v * pa.gather_edge(bv)).sum(
                            dim=(2, 3, 4))
                    _push(op.inputs[0],
                          pa.through_conv(lm.kernel, lm.stride,
                                          lm.padding, lm.in_shape))
                    continue
                if isinstance(lm, Scale):
                    pa2 = PatchAdjoint(pa.v * lm.a, pa.grid, pa.base,
                                       pa.step, pa.edge_shape)
                    pa2.cidx = _cidx(pa)
                    _push(op.inputs[0], pa2)
                    continue
                if isinstance(lm, ScaleShift):
                    n_e = op_of[op.inputs[0]].n
                    if lm.b is not None:
                        sht = torch.as_tensor(
                            lm.b, device=dev,
                            dtype=pa.v.dtype).reshape(1, -1).expand(B, n_e)
                        d = d + (pa.v * pa.gather_edge(sht)).sum(
                            dim=(2, 3, 4))
                    if lm.a is not None:
                        sct = torch.as_tensor(
                            lm.a, device=dev,
                            dtype=pa.v.dtype).reshape(1, -1).expand(B, n_e)
                        pa2 = PatchAdjoint(pa.v * pa.gather_edge(sct),
                                           pa.grid, pa.base, pa.step,
                                           pa.edge_shape)
                        pa2.cidx = _cidx(pa)
                        pa = pa2
                    _push(op.inputs[0], pa)
                    continue
                if isinstance(lm, Select):
                    if 'block_sel' in op.params:
                        bs = op.params['block_sel']
                        _push(op.inputs[0],
                              pa.through_block_sel(bs['blocks'],
                                                   bs['n_blocks']))
                        continue
                    if 'pool_win' in op.params:
                        pwn = op.params['pool_win']
                        _push(op.inputs[0],
                              pa.through_pool_win(pwn['in_shape'],
                                                  pwn['stride'],
                                                  pwn['offsets']))
                        continue
                    raise NotImplementedError(
                        'patch_refine: unannotated Select (irregular '
                        'pool window or non-pool gather)')
                raise NotImplementedError(
                    f'patch_refine: linmap {type(lm).__name__}')
            if op.kind == 'nonlin' and op.fn == 'relu':
                l0, h0 = inter[nm]
                al, bl, au, bu = REL['relu'].planes(l0, h0)
                pa, d = pa.through_planes(al, bl, au, bu, d)
                _push(op.inputs[0], pa)
                continue
            if op.kind == 'add':
                pa_b = PatchAdjoint(pa.v.clone(), pa.grid, pa.base,
                                    pa.step, pa.edge_shape)
                pa_b.cidx = _cidx(pa).clone()
                pa.cidx = _cidx(pa)
                _push(op.inputs[0], pa)
                _push(op.inputs[1], pa_b)
                continue
            raise NotImplementedError(
                f'patch_refine: op {op.kind}/{getattr(op, "fn", "")}')
        return total + d

    lbs = torch.empty(B, C, H * W, device=dev, dtype=lo.dtype)
    ubs = torch.empty(B, C, H * W, device=dev, dtype=lo.dtype)
    for c0 in range(0, C, chan_chunk):
        for ch in range(c0, min(c0 + chan_chunk, C)):
            pa = PatchAdjoint.identity((C, H, W), ch, B=B, device=dev,
                                       dtype=lo.dtype)
            d0 = torch.zeros(B, H * W, device=dev, dtype=lo.dtype)
            lbs[:, ch] = walk(pa, d0, edge)
            pa2 = PatchAdjoint.identity((C, H, W), ch, B=B, device=dev,
                                        dtype=lo.dtype)
            pa2.v = -pa2.v
            ubs[:, ch] = -walk(pa2, torch.zeros_like(d0), edge)
    return lbs.reshape(B, -1), ubs.reshape(B, -1)
