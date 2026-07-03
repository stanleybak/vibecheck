"""The forward propagator: point, interval, and zonotope in one DAG sweep.

One implementation, DAG-native (forks and residual merges are the normal
case), batched over a leading domain dimension B. The three modes share the
same traversal; only the per-op state transformer differs:

  point:    x                              exact evaluation
  interval: (lo, hi)                       IBP
  zono:     (c, G) affine over shared noise symbols; relu adds one fresh
            symbol per (batch-anywhere-unstable) element, so the generator
            layout stays rectangular across the batch (a stable sample just
            carries a zero column).

Generator lifecycle (reduce / drop-and-continue) hooks in here (M5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .relax import OUT_RANGE as REL_RANGE, REL



def _op_const(op, key, dev, build):
    """Per-(op, key, device) tensor cache for concat bases/positions: these
    numpy constants were re-uploaded on every forward call, which dominated
    tight BaB loops (lsnc: 3.2s of a 20s slice in torch.as_tensor)."""
    cache = getattr(op, '_tcache', None)
    if cache is None:
        cache = {}
        op._tcache = cache
    k = (key, str(dev))
    if k not in cache:
        cache[k] = build()
    return cache[k]

def _as2d(x):
    return x if x.dim() == 2 else x.unsqueeze(0)


def clamped_bounds(inter_lh, clamp):
    """Intersect pre-activation bounds with a BaB sign clamp (+1 forces
    z >= 0, -1 forces z <= 0, 0 free). Sound: the clamp is the domain."""
    l, h = inter_lh
    l = torch.where(clamp > 0, l.clamp_min(0.0), l)
    h = torch.where(clamp > 0, h.clamp_min(0.0), h)
    l = torch.where(clamp < 0, l.clamp_max(0.0), l)
    h = torch.where(clamp < 0, h.clamp_max(0.0), h)
    return l, h


def _maxpool_point(op, x):
    B = x.shape[0]
    p = op.params
    x4 = x.reshape(B, *p['in_shape'])
    y = F.max_pool2d(x4, kernel_size=p['kernel_shape'], stride=p['stride'],
                     padding=p['padding'])
    return y.reshape(B, -1)


def point(net, x: torch.Tensor) -> torch.Tensor:
    """Exact forward evaluation, (B, n_in) -> (B, n_out). Edges free as
    their last consumer runs (a 640x640 YOLO batch would otherwise hold
    all ~180 edges live)."""
    x = _as2d(x)
    state = {net.input_name: x}
    remaining = {e: len(c) for e, c in net.consumers().items()}
    for name in net.order:
        op = net.ops[name]
        if op.kind == 'linmap':
            state[name] = op.lm.point(state[op.inputs[0]])
        elif op.kind == 'nonlin':
            state[name] = REL[op.fn].point(state[op.inputs[0]], op.params)
        elif op.kind == 'add':
            state[name] = state[op.inputs[0]] + state[op.inputs[1]]
        elif op.kind == 'mul':
            state[name] = state[op.inputs[0]] * state[op.inputs[1]]
        elif op.kind == 'concat':
            B = x.shape[0]
            out = _op_const(op, 'base', x.device,
                            lambda: torch.as_tensor(
                                op.params['base'], device=x.device,
                                dtype=x.dtype)).expand(B, -1).clone()
            for si, (src, pos) in enumerate(zip(op.inputs,
                                                op.params['positions'])):
                p = _op_const(op, ('pos', si), x.device,
                              lambda: torch.as_tensor(pos, device=x.device))
                out[:, p] = state[src]
            state[name] = out
        elif op.kind == 'maxpool':
            state[name] = _maxpool_point(op, state[op.inputs[0]])
        elif op.kind == 'bmm':
            B = x.shape[0]
            a = state[op.inputs[0]].reshape(B, *op.params['a_shape'])
            bmat = state[op.inputs[1]].reshape(B, *op.params['b_shape'])
            state[name] = torch.matmul(a, bmat).reshape(B, -1)
        else:
            raise NotImplementedError(f'point: op kind {op.kind!r}')
        for e in op.inputs:
            remaining[e] -= 1
            if remaining[e] == 0 and e != net.output_name:
                del state[e]
    return state[net.output_name]


def interval(net, lo: torch.Tensor, hi: torch.Tensor, return_state=False,
             clamps=None, range_clamps=None):
    """IBP bounds, (B, n_in) boxes -> (B, n_out) bounds (per-edge if asked).
    clamps: BaB sign splits per relu op; intersecting the pre-activation
    range here REFRESHES all downstream bounds under the split (the
    reforward-IBP regime of relu-split BaB)."""
    lo, hi = _as2d(lo), _as2d(hi)
    c, r = (hi + lo) / 2, (hi - lo) / 2
    state = {net.input_name: (c, r)}
    for name in net.order:
        op = net.ops[name]
        if op.kind == 'linmap':
            ci, ri = state[op.inputs[0]]
            state[name] = (op.lm.point(ci), op.lm.lin_abs(ri))
        elif op.kind == 'nonlin':
            ci, ri = state[op.inputs[0]]
            if clamps and name in clamps:
                xl, xh = clamped_bounds((ci - ri, ci + ri), clamps[name])
                ci, ri = (xl + xh) / 2, (xh - xl) / 2
            if range_clamps and name in range_clamps:
                rlo, rhi = range_clamps[name]
                xl = torch.maximum(ci - ri, rlo)
                xh = torch.minimum(ci + ri, torch.maximum(rhi, xl))
                ci, ri = (xl + xh) / 2, (xh - xl) / 2
            f = REL[op.fn].point
            flo, fhi = f(ci - ri, op.params), f(ci + ri, op.params)
            if op.fn == 'reciprocal':
                # monotone ONLY on sign-definite ranges; across 0 the true
                # range is unbounded both ways (endpoint eval there was
                # silently unsound). The declared out_range param (the
                # softmax emission knows its sum >= exp(0) = 1) or a
                # vacuous +/-inf keeps it bracketing.
                crossing = (ci - ri <= 0) & (ci + ri >= 0)
                flo = torch.where(crossing, torch.full_like(flo, -torch.inf),
                                  flo)
                fhi = torch.where(crossing, torch.full_like(fhi, torch.inf),
                                  fhi)
            elif op.fn in ('relu', 'leaky_relu', 'sigmoid', 'tanh', 'exp',
                           'floor', 'sign'):
                pass                      # monotone: endpoint eval is exact
            elif op.fn in ('sin', 'cos', 'pow'):
                flo, fhi = _nonmono_interval(op, ci - ri, ci + ri, flo, fhi)
            else:
                raise NotImplementedError(f'interval: nonlin {op.fn!r}')
            lo_hi = torch.minimum(flo, fhi), torch.maximum(flo, fhi)
            rlo, rhi = REL_RANGE.get(op.fn, (None, None))
            if 'out_lo' in op.params:      # emission-declared output range
                rlo = op.params['out_lo'] if rlo is None \
                    else max(rlo, op.params['out_lo'])
            if 'out_hi' in op.params:
                rhi = op.params['out_hi'] if rhi is None \
                    else min(rhi, op.params['out_hi'])
            l2 = lo_hi[0] if rlo is None else lo_hi[0].clamp_min(rlo)
            h2 = lo_hi[1] if rhi is None else lo_hi[1].clamp_max(rhi)
            h2 = torch.maximum(h2, l2)
            # the (c, r) encoding NaNs on infinite widths (inf - inf);
            # anchor the center at the finite side and let r = inf carry
            # the unboundedness -- reads back as (-inf, inf) downstream,
            # loose but FINITE arithmetic (vit: exp past fp32 max)
            wide = torch.isinf(l2) | torch.isinf(h2)
            c_mid = torch.where(
                wide,
                torch.where(torch.isfinite(l2), l2,
                            torch.where(torch.isfinite(h2), h2,
                                        torch.zeros_like(l2))),
                (h2 + l2) / 2)
            r_mid = torch.where(wide, torch.full_like(h2, torch.inf),
                                (h2 - l2) / 2)
            state[name] = (c_mid, r_mid)
        elif op.kind == 'add':
            (c1, r1), (c2, r2) = state[op.inputs[0]], state[op.inputs[1]]
            state[name] = (c1 + c2, r1 + r2)
        elif op.kind == 'mul':
            (c1, r1), (c2, r2) = state[op.inputs[0]], state[op.inputs[1]]
            cands = torch.stack([(c1 - r1) * (c2 - r2), (c1 - r1) * (c2 + r2),
                                 (c1 + r1) * (c2 - r2), (c1 + r1) * (c2 + r2)])
            mlo, mhi = cands.min(dim=0).values, cands.max(dim=0).values
            state[name] = ((mhi + mlo) / 2, (mhi - mlo) / 2)
        elif op.kind == 'concat':
            B = c.shape[0]
            bc = _op_const(op, 'base', c.device,
                           lambda: torch.as_tensor(
                               op.params['base'], device=c.device,
                               dtype=c.dtype)).expand(B, -1).clone()
            br = torch.zeros_like(bc)
            for si, (src, pos) in enumerate(zip(op.inputs,
                                                op.params['positions'])):
                p = _op_const(op, ('pos', si), c.device,
                              lambda: torch.as_tensor(pos, device=c.device))
                bc[:, p], br[:, p] = state[src][0], state[src][1]
            state[name] = (bc, br)
        elif op.kind == 'maxpool':
            ci, ri = state[op.inputs[0]]
            flo = _maxpool_point(op, ci - ri)
            fhi = _maxpool_point(op, ci + ri)
            state[name] = ((fhi + flo) / 2, (fhi - flo) / 2)
        elif op.kind == 'bmm':
            mlo, mhi = _bmm_interval(op, state[op.inputs[0]],
                                     state[op.inputs[1]])
            state[name] = ((mhi + mlo) / 2, (mhi - mlo) / 2)
        else:
            raise NotImplementedError(f'interval: op kind {op.kind!r}')
    if return_state:
        return {k: (v[0] - v[1], v[0] + v[1]) for k, v in state.items()}
    co, ro = state[net.output_name]
    return co - ro, co + ro


def _bmm_interval(op, sa_state, sb_state):
    lo_c, hi_c = _bmm_interval_corners(op, sa_state, sb_state)
    if op.params.get('simplex_left'):
        # left rows are convex-combination weights (softmax): the output
        # is inside the coordinatewise hull of the right factor's rows
        (cb, rb) = sb_state
        B = cb.shape[0]
        sb = op.params['b_shape']
        k, p = sb[-2], sb[-1]
        bl = (cb - rb).reshape(B, -1, k, p)
        bh = (cb + rb).reshape(B, -1, k, p)
        sl = bl.min(dim=2, keepdim=True).values           # (B, G, 1, p)
        sh = bh.max(dim=2, keepdim=True).values
        m = op.params['a_shape'][-2]
        sl = sl.expand(B, sl.shape[1], m, p).reshape(B, -1)
        sh = sh.expand(B, sh.shape[1], m, p).reshape(B, -1)
        lo_c = torch.maximum(lo_c, sl)
        hi_c = torch.maximum(torch.minimum(hi_c, sh), lo_c)
    return lo_c, hi_c


def _bmm_interval_corners(op, sa_state, sb_state):
    """Sound interval matmul via per-product corner enumeration, summed
    over the contraction axis. Memory is (B, ..., m, k, n); attention-sized
    operands only (the McCormick adjoint version arrives with M6)."""
    (ca, ra), (cb, rb) = sa_state, sb_state
    B = ca.shape[0]
    sa, sb = op.params['a_shape'], op.params['b_shape']
    al = (ca - ra).reshape(B, *sa).unsqueeze(-1)
    ah = (ca + ra).reshape(B, *sa).unsqueeze(-1)
    bl = (cb - rb).reshape(B, *sb).unsqueeze(-3)
    bh = (cb + rb).reshape(B, *sb).unsqueeze(-3)
    cands = torch.stack([al * bl, al * bh, ah * bl, ah * bh])
    plo = cands.min(dim=0).values.sum(dim=-2)
    phi = cands.max(dim=0).values.sum(dim=-2)
    return plo.reshape(B, -1), phi.reshape(B, -1)


def _nonmono_interval(op, xlo, xhi, flo, fhi):
    """Exact interval images for the non-monotone elementwise ops."""
    if op.fn == 'pow':
        p = op.params['exponent']
        if p == int(p) and int(p) % 2 == 0:
            crosses = (xlo < 0) & (xhi > 0)
            m = torch.maximum(flo, fhi)
            return torch.where(crosses, torch.zeros_like(flo),
                               torch.minimum(flo, fhi)), m
        return flo, fhi           # odd integer / monotone on their domains
    # sin/cos: check whether an interior extremum (+/-1) lies in [xlo, xhi]
    two_pi = 2 * torch.pi
    shift = 0.0 if op.fn == 'sin' else torch.pi / 2
    lo_ = torch.minimum(flo, fhi)
    hi_ = torch.maximum(flo, fhi)
    # max at x = pi/2 + 2k pi (sin) / 0 + 2k pi (cos)
    kmax = torch.ceil((xlo - (torch.pi / 2 - shift)) / two_pi)
    has_max = (torch.pi / 2 - shift) + kmax * two_pi <= xhi
    kmin = torch.ceil((xlo - (-torch.pi / 2 - shift)) / two_pi)
    has_min = (-torch.pi / 2 - shift) + kmin * two_pi <= xhi
    hi_ = torch.where(has_max, torch.ones_like(hi_), hi_)
    lo_ = torch.where(has_min, -torch.ones_like(lo_), lo_)
    return lo_, hi_


class ZonoState:
    """Batched zonotope: c (B,n), G (B,n,g) over shared noise symbols.

    Column layout is identical across the batch (input symbols first, then
    one column per relu-introduced symbol); a sample where the neuron was
    stable simply has a zero column. `sym` names the op/element each column
    came from so BaB splitting can address them.
    """

    def __init__(self, c, G, sym):
        self.c, self.G, self.sym = c, G, sym

    def bounds(self):
        r = self.G.abs().sum(dim=2)
        return self.c - r, self.c + r


def zono(net, lo, hi, return_state=False, record=None, clamp_bounds=None,
         slope_override=None):
    """DeepZ forward. Boxes (B, n_in) -> output bounds (+ per-edge states).

    record: optional dict; when given, each relu op stores its
    pre-activation snapshot (center, generator rows, symbols, band coeffs)
    for the dual-ascent LP state builder (core.dual_lp).
    clamp_bounds: optional {nonlin op: (lo, hi)} EXTERNAL pre-activation
    bounds (e.g. CROWN-refined) intersected before each band; sound, and
    the resulting bands/state get much tighter.
    slope_override: optional {relu op: (B, n) lam in [0, 1]} per-neuron
    slopes (e.g. a query's optimized alphas); the band offset is recomputed
    for the given slope (relu deviation spans [0, max(-lam*l, (1-lam)*h)]),
    so ANY slope in [0, 1] stays sound."""
    lo, hi = _as2d(lo), _as2d(hi)
    B, n = lo.shape
    dev, dt = lo.device, lo.dtype
    # cheap parallel interval state: every nonlin pre-activation is
    # intersected with it, so zono is never looser than IBP and
    # structurally-positive chains survive band negativity (the softmax
    # difference form: sum of exps >= exp(0) = 1, which the exp band's
    # generators alone cannot see)
    _iv = interval(net, lo, hi, return_state=True)
    c = (hi + lo) / 2
    # generators only for dims that are WIDE somewhere in the batch: a
    # zero-radius dim needs no symbol (dist_shift: 8 wide of 792 dims,
    # so the input block shrinks 100x)
    r = (hi - lo) / 2
    wide = torch.nonzero((r > 0).any(dim=0), as_tuple=False).flatten()
    G = torch.zeros(B, n, wide.numel(), device=dev, dtype=dt)
    G[:, wide, torch.arange(wide.numel(), device=dev)] = r[:, wide]
    sym = [('input', int(i)) for i in wide.tolist()]
    state = {net.input_name: ZonoState(c, G, sym)}

    def lin_cols(lmap, G):
        Bv, nv, g = G.shape
        if g == 0:
            # a degenerate (point) zonotope has no generators; the mapped
            # width comes from a zero probe (metaroom's tiny-eps boxes
            # hit this through the wide-dims optimization)
            n_out = lmap.point(torch.zeros(Bv, nv, device=G.device,
                                           dtype=G.dtype)).shape[1]
            return torch.zeros(Bv, n_out, 0, device=G.device,
                               dtype=G.dtype)
        cols = G.permute(0, 2, 1).reshape(Bv * g, nv)
        out = lmap.lin(cols)
        return out.reshape(Bv, g, -1).permute(0, 2, 1)

    for name in net.order:
        op = net.ops[name]
        if op.kind == 'linmap':
            z = state[op.inputs[0]]
            state[name] = ZonoState(op.lm.point(z.c), lin_cols(op.lm, z.G), z.sym)
        elif op.kind == 'add':
            za, zb = state[op.inputs[0]], state[op.inputs[1]]
            ga, gb = za.G.shape[2], zb.G.shape[2]
            # shared prefix of symbols is summed; distinct tails concatenate
            k = 0
            while k < min(ga, gb) and za.sym[k] == zb.sym[k]:
                k += 1
            if k == ga == gb:
                G = za.G + zb.G
                sym = za.sym
            else:
                G = torch.cat([za.G[:, :, :k] + zb.G[:, :, :k],
                               za.G[:, :, k:], zb.G[:, :, k:]], dim=2)
                sym = za.sym[:k] + za.sym[k:] + zb.sym[k:]
            state[name] = ZonoState(za.c + zb.c, G, sym)
        elif op.kind == 'nonlin':
            rel = REL[op.fn]
            if not hasattr(rel, 'band'):
                raise NotImplementedError(
                    f'zono: no affine band for {op.fn!r} yet (design 3.4)')
            z = state[op.inputs[0]]
            zl, zh = z.bounds()
            _il, _ih = _iv[op.inputs[0]]
            zl = torch.maximum(zl, _il)
            zh = torch.minimum(zh, torch.maximum(_ih, zl))
            if clamp_bounds and name in clamp_bounds:
                cl, ch = clamp_bounds[name]
                zl = torch.maximum(zl, cl)
                zh = torch.minimum(zh, torch.maximum(ch, zl))
            # generic DeepZ affine band: y = lam*x + mu + delta*e_new
            # (relu: DeepZ triangle; sigmoid/tanh: chord band; each op's
            # RelaxLib entry owns its closed-form construction)
            try:
                lam, mu, delta = rel.band(zl, zh, op.params)
            except NotImplementedError:
                if 'out_lo' not in op.params:
                    raise
                lam = torch.zeros_like(zl)
                mu = torch.full_like(zl, (op.params['out_lo']
                                          + op.params['out_hi']) / 2)
                delta = torch.full_like(zl, (op.params['out_hi']
                                             - op.params['out_lo']) / 2)
            if (slope_override and op.fn == 'relu'
                    and name in slope_override):
                lam_o = slope_override[name].clamp(0.0, 1.0)
                unst = (zl < 0) & (zh > 0)
                dev_max = torch.maximum(-lam_o * zl, (1 - lam_o) * zh)
                lam = torch.where(unst, lam_o, lam)
                mu = torch.where(unst, dev_max / 2, mu)
                delta = torch.where(unst, dev_max / 2, delta)
            if record is not None and op.fn == 'relu':
                # pre-activation snapshot for dual_lp.build_state:
                # z_j = c_pre[j] + G_pre[j] . e  and  y = lam z + mu + mu e_new
                record[name] = {'c_pre': z.c.detach(), 'G_pre': z.G.detach(),
                                'sym': list(z.sym), 'lam': lam.detach(),
                                'mu': mu.detach(), 'lo': zl.detach(),
                                'hi': zh.detach()}
            c2 = lam * z.c + mu
            G2 = lam.unsqueeze(2) * z.G
            # fresh symbol per element with a nonzero band ANYWHERE in batch
            new_idx = torch.nonzero((delta > 0).any(dim=0),
                                    as_tuple=False).flatten()
            if new_idx.numel():
                cols = torch.zeros(B, z.c.shape[1], new_idx.numel(),
                                   device=dev, dtype=dt)
                cols[:, new_idx, torch.arange(new_idx.numel(), device=dev)] = \
                    delta[:, new_idx]
                G2 = torch.cat([G2, cols], dim=2)
            sym = z.sym + [(name, int(i)) for i in new_idx.tolist()]
            state[name] = ZonoState(c2, G2, sym)
        elif op.kind == 'concat':
            z_parts = [state[s] for s in op.inputs]
            base = _op_const(op, 'base', dev,
                             lambda: torch.as_tensor(op.params['base'],
                                                     device=dev, dtype=dt))
            n_out = op.params['n_out']
            # union the symbol lists (shared prefix + tails, as in add)
            syms, gmap = [], []
            for zp in z_parts:
                cols = []
                for s in zp.sym:
                    if syms and s in syms:      # rare; only shared prefixes
                        cols.append(syms.index(s))
                    else:
                        syms.append(s)
                        cols.append(len(syms) - 1)
                gmap.append(cols)
            c2 = base.expand(B, -1).clone()
            G2 = torch.zeros(B, n_out, len(syms), device=dev, dtype=dt)
            for zp, cols, pos in zip(z_parts, gmap,
                                     op.params['positions']):
                p = torch.as_tensor(pos, device=dev)
                c2[:, p] = zp.c
                G2[:, p.unsqueeze(1), torch.as_tensor(cols, device=dev)] = zp.G
            state[name] = ZonoState(c2, G2, syms)
        elif op.kind == 'mul':
            # bilinear product: sound box collapse (correlation through the
            # product is dropped; McCormick-in-zono can tighten later)
            za, zb = state[op.inputs[0]], state[op.inputs[1]]
            (la, ha), (lb_, hb_) = za.bounds(), zb.bounds()
            cands = torch.stack([la * lb_, la * hb_, ha * lb_, ha * hb_])
            mlo = cands.min(dim=0).values
            mhi = cands.max(dim=0).values
            c2 = (mhi + mlo) / 2
            delta = (mhi - mlo) / 2
            G2 = torch.diag_embed(delta)
            sym = [(name, i) for i in range(c2.shape[1])]
            state[name] = ZonoState(c2, G2, sym)
        elif op.kind == 'bmm':
            za, zb = state[op.inputs[0]], state[op.inputs[1]]
            mlo, mhi = _bmm_interval(op, ((za.c + 0), za.G.abs().sum(dim=2)),
                                     ((zb.c + 0), zb.G.abs().sum(dim=2)))
            c2 = (mhi + mlo) / 2
            delta = (mhi - mlo) / 2
            state[name] = ZonoState(c2, torch.diag_embed(delta),
                                    [(name, i) for i in range(c2.shape[1])])
        elif op.kind == 'maxpool':
            raise NotImplementedError(
                'zono: maxpool arrives with M5 (relu decomposition)')
        else:
            raise NotImplementedError(f'zono: op kind {op.kind!r}')
    zout = state[net.output_name]
    lo_o, hi_o = zout.bounds()
    if return_state:
        return lo_o, hi_o, state
    return lo_o, hi_o
