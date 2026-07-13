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

import time

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
    def identity(edge_shape, channel, B=1, device='cpu',
                 dtype=torch.float32, bbox=None):
        """One query per spatial position of `channel`: 1x1 one-hot.
        bbox=(y0, y1, x0, x1) restricts the query grid to a subrectangle
        (the anchors are affine, so a cropped grid is just a base shift
        -- the cascaded refiner queries only the unstable cluster)."""
        C, H, W = edge_shape
        y0, y1, x0, x1 = (0, H, 0, W) if bbox is None else bbox
        gh, gw = y1 - y0, x1 - x0
        v = torch.zeros(B, gh * gw, 1, 1, 1, device=device, dtype=dtype)
        v[:, :, 0, 0, 0] = 1.0
        pa = PatchAdjoint(v, (gh, gw), (y0, x0), (1, 1), edge_shape)
        pa.channel = channel                 # window C-axis = [channel]
        pa.chan_lo = channel                 # first edge channel in window
        return pa

    @staticmethod
    def identity_batch(edge_shape, chans, device='cpu',
                       dtype=torch.float32):
        """Identity queries for k channels batched along B as k x {+,-}:
        B = 2k walks share one geometry (anchors identical); the first
        conv composition resolves the per-B kernel slice via a GROUPED
        conv_transpose (bchan). Rows 0..k-1 bound lb, rows k..2k-1 carry
        negated windows for ub (caller negates the result)."""
        C, H, W = edge_shape
        k = len(chans)
        v = torch.ones(2 * k, H * W, 1, 1, 1, device=device, dtype=dtype)
        v[k:] = -1.0
        pa = PatchAdjoint(v, (H, W), (0, 0), (1, 1), edge_shape)
        pa.bchan = torch.as_tensor(list(chans) + list(chans),
                                   device=device, dtype=torch.long)
        return pa

    def through_conv(self, kernel, stride, padding, in_shape):
        """Compose with y = conv2d(x, kernel): adjoint w.r.t. x.

        kernel (Co, Ci, kh, kw); self's window channels must cover the
        conv's OUTPUT channels (full Co window, or the 1-channel identity
        start whose channel selects a kernel slice)."""
        B, Q, Cw, ph, pw = self.v.shape
        k = torch.as_tensor(kernel, device=self.v.device, dtype=self.v.dtype)
        bch = getattr(self, 'bchan', None)
        ci = getattr(self, 'cidx', None)
        if bch is not None:
            pass                        # grouped path slices k[bchan] below
        elif ci is not None:
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
        if bch is not None:
            # per-B kernel slice via grouped conv_transpose: input
            # (Q, B, ph, pw) with groups=B, weight k[bchan] (B, Ci, kh, kw)
            assert Cw == 1, 'bchan start must be 1-channel windows'
            kb = k[bch]                              # (B, Ci, kh, kw)
            vin = v.squeeze(2).permute(1, 0, 2, 3)   # (Q, B, ph, pw)
            og = F.conv_transpose2d(vin, kb, stride=stride, groups=B)
            ph2, pw2 = og.shape[2], og.shape[3]
            og = og.reshape(Q, B, -1, ph2, pw2).permute(1, 0, 2, 3, 4)
            pa = PatchAdjoint(
                og, self.grid,
                (self.base[0] * sy - padding[0],
                 self.base[1] * sx - padding[1]),
                (self.step[0] * sy, self.step[1] * sx), tuple(in_shape))
            pa.chan_lo = 0
            return pa
        vals = v.reshape(B * Q, Cw, ph, pw)
        # Q-chunked: deep-edge walks carry ~4GB per conv_transpose output
        # (784 queries x 64ch x 141^2); chunking bounds the transient peak
        n_bq = vals.shape[0]
        qc = max(1, int(3e8 // max(1, k.shape[1] * (ph + k.shape[2])
                                   * (pw + k.shape[3]))))
        outs_c = [F.conv_transpose2d(vals[s0:s0 + qc], k, stride=stride)
                  for s0 in range(0, n_bq, qc)]
        out = torch.cat(outs_c) if len(outs_c) > 1 else outs_c[0]
        del outs_c
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
        element's edge position (out-of-range -> 0). Indexed per-QUERY
        gather: the previous full-frame F.unfold materialized windows for
        every image position (L), not just the Q anchors -- 54GB at
        relu8-depth walks (Q=784 vs L=12544) and the reason deep vgg
        edges OOMed. Chunked over Q to bound the peak."""
        B, Q, Cw, ph, pw = self.v.shape
        C, H, W = self.edge_shape
        gh, gw = self.grid
        dev = self.v.device
        Bt = t.shape[0] if t.dim() == 2 else 1
        t4 = t.reshape(Bt, C, H * W)
        ci = getattr(self, 'cidx', None)
        if ci is None:
            cw0 = getattr(self, 'chan_lo', 0)
            if getattr(self, '_gather_full', False):
                tc = t4
            else:
                tc = t4[:, cw0:cw0 + Cw]
        elif getattr(self, '_gather_full', False):
            tc = t4
        else:
            tc = t4[:, ci]
        Cs = tc.shape[1]
        qy = torch.arange(gh, device=dev).repeat_interleave(gw)
        qx = torch.arange(gw, device=dev).repeat(gh)
        yy = (self.base[0] + qy * self.step[0]).unsqueeze(1) \
            + torch.arange(ph, device=dev)                    # (Q, ph)
        xx = (self.base[1] + qx * self.step[1]).unsqueeze(1) \
            + torch.arange(pw, device=dev)                    # (Q, pw)
        ok = (((yy >= 0) & (yy < H)).unsqueeze(2)
              & ((xx >= 0) & (xx < W)).unsqueeze(1))          # (Q, ph, pw)
        flat = (yy.clamp(0, H - 1).unsqueeze(2) * W
                + xx.clamp(0, W - 1).unsqueeze(1))            # (Q, ph, pw)
        out = torch.empty(Bt, Q, Cs, ph, pw, device=dev, dtype=t.dtype)
        # Q-chunks: peak extra memory ~ chunk*Cs*ph*pw
        qc = max(1, int(2e8 // max(1, Cs * ph * pw)))
        for s0 in range(0, Q, qc):
            e0 = min(Q, s0 + qc)
            g = tc[:, :, flat[s0:e0].reshape(-1)]
            g = g.reshape(Bt, Cs, e0 - s0, ph, pw).permute(0, 2, 1, 3, 4)
            out[:, s0:e0] = g * ok[s0:e0].unsqueeze(0).unsqueeze(2) \
                .to(t.dtype)
        return out

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


def patch_refine(net, edge, lo, hi, inter, chan_chunk=8, device=None,
                 channels=None, unstable=None):
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

    import os as _os
    _prof = {} if _os.environ.get('VC2_PATCH_PROF') else None

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
            if _prof is not None:
                torch.cuda.synchronize()
                _now = time.time()
                if walk._last is not None:
                    k0, t0_ = walk._last
                    _prof[k0] = _prof.get(k0, 0.0) + (_now - t0_)
                _kind = (type(op.lm).__name__ if op.kind == 'linmap'
                         else op.kind + ':' + getattr(op, 'fn', ''))
                walk._last = (_kind, _now)
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
                key = (nm, pa.grid, pa.base, pa.step,
                       pa.v.shape[3], pa.v.shape[4])
                got = _gcache.get(key)
                if got is None:
                    if nm not in _planes:
                        l0, h0 = inter[nm]
                        _planes[nm] = REL['relu'].planes(l0, h0)
                    al, bl, au, bu = _planes[nm]
                    pa._gather_full = True
                    got = (pa.gather_edge(al), pa.gather_edge(bl),
                           pa.gather_edge(au), pa.gather_edge(bu))
                    pa._gather_full = False
                    nbytes = sum(t.numel() * t.element_size() for t in got)
                    if pa.v.shape[3] * pa.v.shape[4] > 4096:
                        nbytes = float('inf')   # huge-window: never cache
                    # cap the cache (the early wide edges' gathers are
                    # GBs; they are also the ones every walk repays)
                    if _gcache_sz[0] + nbytes < 3e9:
                        _gcache[key] = got
                        _gcache_sz[0] += nbytes
                ll_f, bl_f, lh_f, bh_f = got
                ci = getattr(pa, 'cidx', None)
                if ci is None:
                    cw0 = getattr(pa, 'chan_lo', 0)
                    sl = slice(cw0, cw0 + pa.v.shape[2])
                    ll, blg, lh, bhg = (ll_f[:, :, sl], bl_f[:, :, sl],
                                        lh_f[:, :, sl], bh_f[:, :, sl])
                else:
                    ll, blg, lh, bhg = (ll_f[:, :, ci], bl_f[:, :, ci],
                                        lh_f[:, :, ci], bh_f[:, :, ci])
                pos = pa.v > 0
                v2 = pa.v * torch.where(pos, ll, lh)
                d = d + (pa.v * torch.where(pos, blg, bhg)).sum(
                    dim=(2, 3, 4))
                pa2 = PatchAdjoint(v2, pa.grid, pa.base, pa.step,
                                   pa.edge_shape)
                pa2.chan_lo = getattr(pa, 'chan_lo', 0)
                if ci is not None:
                    pa2.cidx = ci
                _push(op.inputs[0], pa2)
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

    # channels: restrict the (C x HW identity queries) to the channels
    # that still matter -- the cascaded caller passes the set with any
    # unstable element (vgg relu4: 122 unstable of 802k spread over a
    # few dozen channels; refining all 256 cost 99.5s for nothing).
    # Unrefined channels return (-inf, +inf): callers intersect.
    walk._last = None
    _gcache = {}
    _gcache_sz = [0]
    _planes = {}
    if channels is None:
        chan_iter = list(range(C))
    else:
        chan_iter = sorted(int(c) for c in channels)
    # unstable (B-agnostic bool over the edge, reshaped (C, H, W)):
    # crop each channel's query grid to its unstable BOUNDING BOX --
    # unstable positions cluster spatially, and full-frame grids paid
    # 3136 queries for ~4 useful ones on vgg relu5. Unqueried elements
    # return (-inf, +inf); callers intersect with existing bounds.
    um = None
    if unstable is not None:
        um = unstable.reshape(C, H, W)
    lbs = torch.full((B, C, H * W), -torch.inf, device=dev, dtype=lo.dtype)
    ubs = torch.full((B, C, H * W), torch.inf, device=dev, dtype=lo.dtype)
    assert B == 1, 'patch_refine batches CHANNELS along the walk batch'
    if um is not None:
        # bbox mode stays per-channel (quadrant callers)
        for ch in chan_iter:
            ys, xs = torch.nonzero(um[ch], as_tuple=True)
            if ys.numel() == 0:
                continue
            bbox = (int(ys.min()), int(ys.max()) + 1,
                    int(xs.min()), int(xs.max()) + 1)
            pa = PatchAdjoint.identity((C, H, W), ch, B=1, device=dev,
                                       dtype=lo.dtype, bbox=bbox)
            nq = pa.v.shape[1]
            d0 = torch.zeros(1, nq, device=dev, dtype=lo.dtype)
            lb_c = walk(pa, d0, edge)
            pa2 = PatchAdjoint.identity((C, H, W), ch, B=1, device=dev,
                                        dtype=lo.dtype, bbox=bbox)
            pa2.v = -pa2.v
            ub_c = -walk(pa2, torch.zeros_like(d0), edge)
            y0, y1, x0, x1 = bbox
            lbs[:, ch].reshape(1, H, W)[:, y0:y1, x0:x1] = \
                lb_c.reshape(1, y1 - y0, x1 - x0)
            ubs[:, ch].reshape(1, H, W)[:, y0:y1, x0:x1] = \
                ub_c.reshape(1, y1 - y0, x1 - x0)
    else:
        # CHANNEL-BATCHED walks: k channels x {+,-} share one walk along
        # the batch axis (per-channel Python/launch overhead dominated
        # the vgg cascade; the grouped first-hop resolves per-channel
        # kernels). OOM halves k and retries the remaining channels.
        k = max(1, int(chan_chunk))
        pend = list(chan_iter)
        while pend:
            grp = pend[:k]
            try:
                pa = PatchAdjoint.identity_batch((C, H, W), grp,
                                                 device=dev,
                                                 dtype=lo.dtype)
                d0 = torch.zeros(2 * len(grp), H * W, device=dev,
                                 dtype=lo.dtype)
                res = walk(pa, d0, edge)           # (2k, HW)
                lbs[0, grp] = res[:len(grp)]
                ubs[0, grp] = -res[len(grp):]
                pend = pend[k:]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if k == 1:
                    raise
                k = max(1, k // 2)
    if _prof is not None:
        torch.cuda.synchronize()
        if walk._last is not None:
            k0, t0_ = walk._last
            _prof[k0] = _prof.get(k0, 0.0) + (time.time() - t0_)
        print('[patch-prof] ' + ' '.join(
            f'{k}={v:.2f}s' for k, v in
            sorted(_prof.items(), key=lambda x: -x[1])), flush=True)
    return lbs.reshape(B, -1), ubs.reshape(B, -1)
