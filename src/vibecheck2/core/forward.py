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



def _op_const(op, key, dev, build, dtype=None):
    """Per-(op, key, device, dtype) tensor cache for concat bases/positions:
    these numpy constants were re-uploaded on every forward call, which
    dominated tight BaB loops (lsnc: 3.2s of a 20s slice in torch.as_tensor).
    dtype must be part of the key for value tensors: alpha_zono runs the
    same net in float64 after float32 phases cached float32 bases."""
    cache = getattr(op, '_tcache', None)
    if cache is None:
        cache = {}
        op._tcache = cache
    k = (key, str(dev), str(dtype))
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
                                dtype=x.dtype),
                            dtype=x.dtype).expand(B, -1).clone()
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
            if 'in_lo' in op.params or 'in_hi' in op.params:
                # emission-DECLARED input range (a theorem about the
                # subgraph, e.g. a shifted-softmax denominator lies in
                # [1, k] always); intersecting is sound by declaration
                xl = (ci - ri).clamp_min(op.params.get('in_lo', -torch.inf))
                xh = torch.maximum(
                    (ci + ri).clamp_max(op.params.get('in_hi', torch.inf)),
                    xl)
                ci, ri = (xl + xh) / 2, (xh - xl) / 2
            if op.fn == 'softmax':
                # fused row transform: the EXACT coordinatewise interval
                # image (relax.softmax_interval); endpoint eval would be
                # wrong (y_i is anti-monotone in the rival logits)
                from .relax import softmax_interval
                flo, fhi = softmax_interval(ci - ri, ci + ri, op.params)
                state[name] = ((fhi + flo) / 2, (fhi - flo) / 2)
                continue
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
            if 'out_lo' in op.params:      # emission-declared output range
                mlo = mlo.clamp_min(op.params['out_lo'])
            if 'out_hi' in op.params:
                mhi = torch.maximum(mhi.clamp_max(op.params['out_hi']), mlo)
            state[name] = ((mhi + mlo) / 2, (mhi - mlo) / 2)
        elif op.kind == 'concat':
            B = c.shape[0]
            bc = _op_const(op, 'base', c.device,
                           lambda: torch.as_tensor(
                               op.params['base'], device=c.device,
                               dtype=c.dtype),
                           dtype=c.dtype).expand(B, -1).clone()
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

    rad (B, n) >= 0, optional: BOX REMAINDER -- aggregate magnitude of
    unlabeled noise (bmm remainders, non-relu band deltas) kept as a
    per-element radius instead of dense generator columns. Equivalent to
    carrying each source as its own diagonal column EXCEPT that distinct
    sources merge (their cross-element correlation is dropped), which is
    sound: |sum_s a_s u_s| <= sum_s |a_s| for independent u_s in [-1, 1].
    Measured on vit 2157: the dense form carries ~7.7k such columns of a
    13.4k-generator state, and the per-domain BaB zono pass OOMs at B=8.
    """

    def __init__(self, c, G, sym, rad=None):
        self.c, self.G, self.sym, self.rad = c, G, sym, rad

    def bounds(self):
        r = self.G.abs().sum(dim=2)
        if self.rad is not None:
            r = r + self.rad
        return self.c - r, self.c + r


def zono(net, lo, hi, return_state=False, record=None, clamp_bounds=None,
         slope_override=None, box_remainder=False, sym_budget=None):
    """DeepZ forward. Boxes (B, n_in) -> output bounds (+ per-edge states).

    record: optional dict; when given, each relu op stores its
    pre-activation snapshot (center, generator rows, symbols, band coeffs)
    for the dual-ascent LP state builder (core.dual_lp).
    clamp_bounds: optional {nonlin op: (lo, hi)} EXTERNAL pre-activation
    bounds (e.g. CROWN-refined) intersected before each band; sound, and
    the resulting bands/state get much tighter.
    slope_override: optional {nonlin op: (B, n) alpha in [0, 1]} per-neuron
    band-slope parameters for any op with a RelaxLib band_alpha (relu,
    sigmoid/tanh, sin/cos/pow, exp, reciprocal): lam = (1-a) f'(lo) +
    a f'(hi) with the offsets recomputed in closed form for that lam, so
    ANY alpha in [0, 1] stays sound (v1's nl_alpha mechanism).
    box_remainder: when True, unlabeled fresh noise (bmm/mul remainders,
    band deltas of every op EXCEPT relu and the softmax exp stage, whose
    columns the BaB scores and splits) accumulates in ZonoState.rad
    instead of dense generator columns. Sound (see ZonoState); trades the
    dropped cross-element correlations for the memory/FLOPs of ~60% of
    the generator columns (the per-domain BaB regime).
    sym_budget: cap on INPUT symbol columns (requires box_remainder).
    Only the top-budget wide dims by radius get columns; the rest start
    in the remainder. This is ab-crown's vggnet16 recipe
    (bound_prop_method dynamic-forward, forward.max_dim 100): a 150k-dim
    ImageNet box cannot carry one column per pixel, and a ~100-symbol
    forward + input-split BaB is what officially closes that class."""
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
    rad0 = torch.zeros(B, n, device=dev, dtype=dt) if box_remainder else None
    if sym_budget is not None and wide.numel() > sym_budget:
        assert box_remainder, 'sym_budget needs the box remainder'
        keep = r.amax(dim=0)[wide].argsort(descending=True)[:sym_budget]
        keep_idx = wide[keep.sort().values]
        spill = torch.ones(n, dtype=torch.bool, device=dev)
        spill[keep_idx] = False
        rad0 = rad0 + r * spill                 # boxed input noise
        wide = keep_idx
    G = torch.zeros(B, n, wide.numel(), device=dev, dtype=dt)
    G[:, wide, torch.arange(wide.numel(), device=dev)] = r[:, wide]
    sym = [('input', int(i)) for i in wide.tolist()]
    state = {net.input_name: ZonoState(c, G, sym, rad0)}
    # free edges as their last consumer runs (same discipline as point():
    # holding every edge's (B, n, g) matrix alive put vit's zono at ~5 GiB
    # per leaf and made per-domain BaB bounding hopeless; v1's ~32 ms/domain
    # beta-bab rate implies a live-frontier footprint)
    remaining = {e: len(c2) for e, c2 in net.consumers().items()}

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
            state[name] = ZonoState(
                op.lm.point(z.c), lin_cols(op.lm, z.G), z.sym,
                op.lm.lin_abs(z.rad) if z.rad is not None else None)
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
            state[name] = ZonoState(za.c + zb.c, G, sym,
                                    za.rad + zb.rad
                                    if za.rad is not None else None)
        elif op.kind == 'nonlin':
            rel = REL[op.fn]
            if op.fn != 'softmax' and not hasattr(rel, 'band'):
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
            if 'in_lo' in op.params or 'in_hi' in op.params:
                # emission-declared input range (see interval()); sound
                zl = zl.clamp_min(op.params.get('in_lo', -torch.inf))
                zh = torch.maximum(
                    zh.clamp_max(op.params.get('in_hi', torch.inf)), zl)
            if op.fn == 'softmax':
                if record is not None:
                    record[name] = {'c_pre': z.c.detach(),
                                    'G_pre': z.G.detach(),
                                    'sym': list(z.sym)}
                    if z.rad is not None:
                        record[name]['rad'] = z.rad.detach()
                state[name] = _softmax_zono(op, z, zl, zh, B, dev, dt)
                continue
            # generic DeepZ affine band: y = lam*x + mu + delta*e_new
            # (relu: DeepZ triangle; sigmoid/tanh: chord band; each op's
            # RelaxLib entry owns its closed-form construction). An override
            # routes through band_alpha: the slope becomes the caller's
            # optimizable parameter, offsets recomputed for that slope.
            ov = (slope_override.get(name) if slope_override else None)
            try:
                if ov is not None and hasattr(rel, 'band_alpha'):
                    lam, mu, delta = rel.band_alpha(zl, zh, ov, op.params)
                else:
                    lam, mu, delta = rel.band(zl, zh, op.params)
            except NotImplementedError:
                if 'out_lo' not in op.params:
                    raise
                lam = torch.zeros_like(zl)
                mu = torch.full_like(zl, (op.params['out_lo']
                                          + op.params['out_hi']) / 2)
                delta = torch.full_like(zl, (op.params['out_hi']
                                             - op.params['out_lo']) / 2)
            if record is not None and op.fn == 'relu':
                # pre-activation snapshot for dual_lp.build_state:
                # z_j = c_pre[j] + G_pre[j] . e  and  y = lam z + mu + mu e_new
                record[name] = {'c_pre': z.c.detach(), 'G_pre': z.G.detach(),
                                'sym': list(z.sym), 'lam': lam.detach(),
                                'mu': mu.detach(), 'lo': zl.detach(),
                                'hi': zh.detach()}
                if z.rad is not None:
                    # the constraint row z = c + G.e (+ rad.u) is short its
                    # rad noise; consumers (beta) must charge |beta|*rad
                    record[name]['rad'] = z.rad.detach()
            c2 = lam * z.c + mu
            G2 = lam.unsqueeze(2) * z.G
            rad2 = lam.abs() * z.rad if z.rad is not None else None
            if rad2 is not None and (op.fn != 'relu'
                                     or box_remainder == 'all'):
                # unlabeled band delta -> remainder. relu fresh columns
                # stay dense at box_remainder=True (they are the relu
                # BaB's split/score/beta handles) but spill too under
                # 'all': the input-split regime (ab's vggnet16
                # dynamic-forward, max_dim ~100) carries INPUT symbols
                # only -- vgg's 25 relu layers would otherwise add
                # millions of columns and the pass OOMs at any budget
                rad2 = rad2 + delta
                state[name] = ZonoState(c2, G2, list(z.sym), rad2)
                continue
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
            state[name] = ZonoState(c2, G2, sym, rad2)
        elif op.kind == 'concat':
            z_parts = [state[s] for s in op.inputs]
            base = _op_const(op, 'base', dev,
                             lambda: torch.as_tensor(op.params['base'],
                                                     device=dev, dtype=dt),
                             dtype=dt)
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
            rad2 = (torch.zeros(B, n_out, device=dev, dtype=dt)
                    if box_remainder else None)
            for zp, cols, pos in zip(z_parts, gmap,
                                     op.params['positions']):
                p = torch.as_tensor(pos, device=dev)
                c2[:, p] = zp.c
                # dtype=long explicitly: an EMPTY cols list (a part whose
                # symbols all live in the remainder, e.g. a mul output
                # under box_remainder='all') otherwise infers float and
                # crashes the advanced indexing
                G2[:, p.unsqueeze(1),
                   torch.as_tensor(cols, device=dev,
                                   dtype=torch.long)] = zp.G
                if rad2 is not None:
                    rad2[:, p] = zp.rad
            state[name] = ZonoState(c2, G2, syms, rad2)
        elif op.kind == 'mul':
            # bilinear product: sound box collapse. The correlation-exact
            # first-order product (v1 _torch_zono_mul_bilinear) was built
            # and MEASURED on ml4acopf 0298: bit-identical alpha_zono
            # margin, 2x slower (extra generator columns) -- the mul band
            # is not binding at the optimum there. Reintroduce only with a
            # case it wins.
            za, zb = state[op.inputs[0]], state[op.inputs[1]]
            (la, ha), (lb_, hb_) = za.bounds(), zb.bounds()
            cands = torch.stack([la * lb_, la * hb_, ha * lb_, ha * hb_])
            mlo = cands.min(dim=0).values
            mhi = cands.max(dim=0).values
            if 'out_lo' in op.params:      # emission-declared output range
                mlo = mlo.clamp_min(op.params['out_lo'])
            if 'out_hi' in op.params:
                mhi = torch.maximum(mhi.clamp_max(op.params['out_hi']), mlo)
            c2 = (mhi + mlo) / 2
            delta = (mhi - mlo) / 2
            if box_remainder:
                state[name] = ZonoState(
                    c2, torch.zeros(B, c2.shape[1], 0, device=dev, dtype=dt),
                    [], delta)
            else:
                G2 = torch.diag_embed(delta)
                sym = [(name, i) for i in range(c2.shape[1])]
                state[name] = ZonoState(c2, G2, sym)
        elif op.kind == 'bmm':
            za, zb = state[op.inputs[0]], state[op.inputs[1]]
            state[name] = _bmm_zono(op, name, za, zb, B, dev, dt)
        elif op.kind == 'maxpool':
            raise AssertionError(                       # unreachable
                'zono reached a maxpool op: decompose_maxpool must run at '
                'load (graph.py) so no maxpool survives into relaxation')
        else:
            raise NotImplementedError(f'zono: op kind {op.kind!r}')
        for e in op.inputs:
            remaining[e] -= 1
            if (remaining[e] == 0 and e != net.output_name
                    and not return_state):
                del state[e]
    zout = state[net.output_name]
    lo_o, hi_o = zout.bounds()
    if return_state:
        return lo_o, hi_o, state
    return lo_o, hi_o


def _bmm_zono(op, name, za, zb, B, dev, dt):
    """Correlation-exact zonotope matmul (the bilinear first-order product
    generalized over the contraction axis; algebra mirrors v1's proven
    elementwise form): out = X @ Y with X = cx + Gx e, Y = cy + Gy e over
    a SHARED symbol prefix,

      c_out   = cx cy + 0.5 sum_col (Gx_col @ Gy_col)   [e^2 in [0,1]]
      G_out_c = Gx_col @ cy + cx @ Gy_col               [exact 1st order]
      box     = (radX @ radY - 0.5 sum_col |Gx_col @ Gy_col|)  >= 0

    then the per-element HYBRID against the interval corner product (with
    the simplex hull for softmax-left attention): where the affine form
    is wider, that element collapses to the interval box -- both enclose
    the true set. The old pure-interval bmm severed the attention-weight
    correlations entirely (measured: vit 1151 zono floor -0.44 regardless
    of softmax representation; v1 with correlated products sits at -0.04).
    """
    ash, bsh = op.params['a_shape'], op.params['b_shape']
    m, k, pp = ash[-2], ash[-1], bsh[-1]
    Gr = 1
    for dd_ in ash[:-2]:
        Gr *= int(dd_)
    # align both factors over the union symbol list (shared prefix + tails)
    ga, gb = za.G.shape[2], zb.G.shape[2]
    kpre = 0
    while kpre < min(ga, gb) and za.sym[kpre] == zb.sym[kpre]:
        kpre += 1
    sym = za.sym[:kpre] + za.sym[kpre:] + zb.sym[kpre:]
    g = kpre + (ga - kpre) + (gb - kpre)
    Ga = torch.cat([za.G, torch.zeros(B, za.c.shape[1], gb - kpre,
                                      device=dev, dtype=dt)], dim=2)
    Gb = torch.cat([zb.G[:, :, :kpre],
                    torch.zeros(B, zb.c.shape[1], ga - kpre, device=dev,
                                dtype=dt),
                    zb.G[:, :, kpre:]], dim=2)
    caM = za.c.reshape(B, Gr, m, k)
    cbM = zb.c.reshape(B, Gr, k, pp)
    GaM = Ga.permute(0, 2, 1).reshape(B, g, Gr, m, k)
    GbM = Gb.permute(0, 2, 1).reshape(B, g, Gr, k, pp)
    c_out = torch.matmul(caM, cbM)                     # (B, Gr, m, pp)
    D = torch.matmul(GaM, GbM)                         # (B, g, Gr, m, pp)
    diag_sum = D.sum(dim=1)
    diag_abs = D.abs().sum(dim=1)
    c_out = c_out + 0.5 * diag_sum
    G_out = (torch.matmul(GaM, cbM.unsqueeze(1))
             + torch.matmul(caM.unsqueeze(1), GbM))    # (B, g, Gr, m, pp)
    radA = Ga.abs().sum(dim=2).reshape(B, Gr, m, k)
    radB = Gb.abs().sum(dim=2).reshape(B, Gr, k, pp)
    box = (torch.matmul(radA, radB) - 0.5 * diag_abs).clamp_min(0.0)
    rad_out = None
    if za.rad is not None:
        # box-remainder cross terms: every product involving a remainder
        # factor is first-order in ITS noise but unlabeled, so the whole
        # family lands in the output remainder (|a.u| <= |a|):
        #   c*rb + ra*c  (center x remainder)
        #   G*rb + ra*G  (generator x remainder)   ra*rb (quadratic)
        ra = za.rad.reshape(B, Gr, m, k)
        rb = zb.rad.reshape(B, Gr, k, pp)
        rad_out = (torch.matmul(caM.abs() + radA + ra, rb)
                   + torch.matmul(ra, cbM.abs() + radB))
    n_out = Gr * m * pp
    y_c = c_out.reshape(B, n_out)
    y_G = G_out.reshape(B, g, n_out).permute(0, 2, 1)
    box = box.reshape(B, n_out)
    # hybrid vs the interval corner product (+ simplex hull when tagged):
    # the interval path is exact per element; collapse where affine+box
    # is wider (both sound)
    ra_tot = za.G.abs().sum(dim=2)
    rb_tot = zb.G.abs().sum(dim=2)
    if za.rad is not None:
        ra_tot = ra_tot + za.rad
        rb_tot = rb_tot + zb.rad
    ilo, ihi = _bmm_interval(op, (za.c, ra_tot), (zb.c, rb_tot))
    slack = box + (0 if rad_out is None
                   else rad_out.reshape(B, n_out))
    aff_lo = y_c - y_G.abs().sum(dim=2) - slack
    aff_hi = y_c + y_G.abs().sum(dim=2) + slack
    # threshold 4x (measured: 1x -0.380, 4x -0.149, 100x identical to
    # 4x on vit 1151 -- collapse only guards genuine blow-ups)
    worse = ((aff_hi - aff_lo) > 4.0 * (ihi - ilo) + 1e-12).any(dim=0) \
        .unsqueeze(0).expand(B, n_out)
    y_c = torch.where(worse, (ihi + ilo) / 2, y_c)
    y_G = torch.where(worse.unsqueeze(2), torch.zeros_like(y_G), y_G)
    box = torch.where(worse, (ihi - ilo) / 2, box)
    if rad_out is not None:
        # remainder mode: the box term joins the remainder; a collapsed
        # element is EXACTLY the interval half-width (replace, not add)
        rad_out = torch.where(worse, (ihi - ilo) / 2,
                              rad_out.reshape(B, n_out) + box)
        return ZonoState(y_c, y_G, sym, rad_out)
    bidx = torch.nonzero((box > 0).any(dim=0), as_tuple=False).flatten()
    if bidx.numel():
        cols = torch.zeros(B, n_out, bidx.numel(), device=dev, dtype=dt)
        cols[:, bidx, torch.arange(bidx.numel(), device=dev)] = box[:, bidx]
        y_G = torch.cat([y_G, cols], dim=2)
    sym = sym + [(name, int(i)) for i in bidx.tolist()]
    return ZonoState(y_c, y_G, sym)


def _softmax_zono(op, z, zl, zh, B, dev, dt):
    """Fused softmax zonotope transformer (v1's approach, rebuilt on vc2
    primitives after the graph-level rewrite measured WORSE):

      1. shift by the CONSTANT per-row c = max_j zh_j (exact: softmax is
         shift-invariant; c from live bounds means no graph max tree and
         its relu-band slack). The shifted upper bounds satisfy eh <= 0
         STRUCTURALLY, so exp lands in (0, 1].
      2. exp via the RelaxLib band on the shifted range (generators scale
         by lam -- input correlations survive).
      3. denominator s = row-sum of e (affine: sums of centers and
         generator rows; bounds intersected with the interval sums, which
         keep s positive and <= k).
      4. reciprocal via its band on [s_l, s_h] (sign-definite by 3).
      5. y = e * r with the CORRELATION-EXACT bilinear product (e and r
         share the full symbol prefix): first-order terms stay linear,
         the quadratic remainder is boxed with the shared-diagonal
         e^2-in-[0,1] tightening. This is where the box-collapse mul
         killed the graph-level variant.

    Output bounds land in [0, 1] by construction of the planes plus the
    declared out range on the op."""
    from .relax import REL
    p = op.params
    pre, k, post = p['pre'], p['k'], p['post']
    n = z.c.shape[1]
    n_rows = pre * post
    c_row = zh.reshape(B, pre, k, post).max(dim=2, keepdim=True).values
    c_bc = c_row.expand(B, pre, k, post).reshape(B, n)
    el, eh = zl - c_bc, zh - c_bc
    eh = eh.clamp_max(0.0)                       # exact: c >= every zh
    lam, mu, delta = REL['exp'].band(el, eh)
    e_c = lam * (z.c - c_bc) + mu
    e_G = lam.unsqueeze(2) * z.G
    e_rad = lam.abs() * z.rad if z.rad is not None else None
    new_idx = torch.nonzero((delta > 0).any(dim=0),
                            as_tuple=False).flatten()
    if new_idx.numel():
        cols = torch.zeros(B, n, new_idx.numel(), device=dev, dtype=dt)
        cols[:, new_idx, torch.arange(new_idx.numel(), device=dev)] = \
            delta[:, new_idx]
        e_G = torch.cat([e_G, cols], dim=2)
    e_sym = z.sym + [(op.name + '/e', int(i)) for i in new_idx.tolist()]
    g_e = e_G.shape[2]
    # exact elementwise exp range (monotone): the band bounds are looser
    e_lo, e_hi = torch.exp(el), torch.exp(eh)
    # denominator: affine row sum + interval intersection
    s_c = e_c.reshape(B, pre, k, post).sum(dim=2).reshape(B, n_rows)
    s_G = e_G.reshape(B, pre, k, post, g_e).sum(dim=2).reshape(B, n_rows,
                                                               g_e)
    s_rad = (e_rad.reshape(B, pre, k, post).sum(dim=2).reshape(B, n_rows)
             if e_rad is not None else None)
    rad_s = s_G.abs().sum(dim=2)
    if s_rad is not None:
        rad_s = rad_s + s_rad
    s_l = torch.maximum(s_c - rad_s,
                        e_lo.reshape(B, pre, k, post).sum(dim=2)
                        .reshape(B, n_rows))
    s_h = torch.minimum(s_c + rad_s,
                        e_hi.reshape(B, pre, k, post).sum(dim=2)
                        .reshape(B, n_rows))
    s_h = torch.maximum(s_h, s_l)
    lam_r, mu_r, d_r = REL['reciprocal'].band(s_l, s_h)
    r_c = lam_r * s_c + mu_r
    r_G = lam_r.unsqueeze(2) * s_G
    r_rad = None
    if s_rad is not None:
        # remainder mode: the reciprocal delta is unlabeled -> remainder
        r_rad = lam_r.abs() * s_rad + d_r
        sym = list(e_sym)
    else:
        ridx = torch.nonzero((d_r > 0).any(dim=0), as_tuple=False).flatten()
        if ridx.numel():
            cols = torch.zeros(B, n_rows, ridx.numel(), device=dev, dtype=dt)
            cols[:, ridx, torch.arange(ridx.numel(), device=dev)] = \
                d_r[:, ridx]
            r_G = torch.cat([r_G, cols], dim=2)
        sym = e_sym + [(op.name + '/r', int(i)) for i in ridx.tolist()]
    # broadcast r back over the rows and pad e with the recip columns so
    # both factors live over the SAME symbol list
    grid = torch.arange(n_rows, device=dev).reshape(pre, 1, post)
    bidx = grid.expand(pre, k, post).reshape(-1)
    rb_c = r_c[:, bidx]
    rb_G = r_G[:, bidx, :]
    rb_rad = r_rad[:, bidx] if r_rad is not None else None
    if r_rad is None and ridx.numel():
        e_G = torch.cat([e_G, torch.zeros(B, n, ridx.numel(), device=dev,
                                          dtype=dt)], dim=2)
    # correlation-exact product y = e * r (shared symbols throughout)
    prod = e_G * rb_G
    y_c = e_c * rb_c + 0.5 * prod.sum(dim=2)
    y_G = e_G * rb_c.unsqueeze(2) + e_c.unsqueeze(2) * rb_G
    rad_e = e_G.abs().sum(dim=2)
    rad_r = rb_G.abs().sum(dim=2)
    box = (rad_e * rad_r - 0.5 * prod.abs().sum(dim=2)).clamp_min(0.0)
    y_rad = None
    if rb_rad is not None:
        # remainder cross terms of the product (see _bmm_zono)
        y_rad = (box + (e_c.abs() + rad_e + e_rad) * rb_rad
                 + e_rad * (rb_c.abs() + rad_r))
    else:
        yidx = torch.nonzero((box > 0).any(dim=0), as_tuple=False).flatten()
        if yidx.numel():
            cols = torch.zeros(B, n, yidx.numel(), device=dev, dtype=dt)
            cols[:, yidx, torch.arange(yidx.numel(), device=dev)] = \
                box[:, yidx]
            y_G = torch.cat([y_G, cols], dim=2)
        sym = sym + [(op.name, int(i)) for i in yidx.tolist()]
    # per-element hybrid against the EXACT interval image: on wide rows
    # the composed bands blow past [0, 1] (measured 5.8e34 on a scale-8
    # toy) while softmax_interval is exact -- where the affine form is
    # WIDER, collapse that element to the exact box (both representations
    # enclose the true set, so taking the tighter one per element is
    # sound; correlations survive exactly where they are competitive)
    from .relax import softmax_interval
    yl_ex, yh_ex = softmax_interval(zl, zh, p)
    rad_y = y_G.abs().sum(dim=2)
    if y_rad is not None:
        rad_y = rad_y + y_rad
    # threshold 4x, MEASURED on vit 1151 with the correlated bmm
    # downstream: 1x collapse -> zono floor -0.38 (correlations severed),
    # 4x -> -0.0421 = v1 parity (v1: -0.0423). The affine form's local
    # over-width is repaid downstream where the products correlate.
    worse = ((rad_y > 2.0 * (yh_ex - yl_ex) + 1e-12).any(dim=0)
             .unsqueeze(0).expand(B, n))
    y_c = torch.where(worse, (yh_ex + yl_ex) / 2, y_c)
    y_G = torch.where(worse.unsqueeze(2), torch.zeros_like(y_G), y_G)
    if y_rad is not None:
        y_rad = torch.where(worse, (yh_ex - yl_ex) / 2, y_rad)
        return ZonoState(y_c, y_G, sym, y_rad)
    widx = torch.nonzero(worse.any(dim=0), as_tuple=False).flatten()
    if widx.numel():
        cols = torch.zeros(B, n, widx.numel(), device=dev, dtype=dt)
        cols[:, widx, torch.arange(widx.numel(), device=dev)] = \
            ((yh_ex - yl_ex) / 2)[:, widx]
        y_G = torch.cat([y_G, cols], dim=2)
        sym = sym + [(op.name + '/x', int(i)) for i in widx.tolist()]
    return ZonoState(y_c, y_G, sym)


def alpha_zono(net, lo, hi, W, iters=200, lr=0.5, thresholds=None,
               budget=None, patience=40, clamp_bounds=None, disj_idx=None,
               return_alphas=False, known=None, init_alphas=None):
    """Adam-optimized band slopes over the forward zonotope (v1's nl_alpha,
    verify_graph.py _nonlinear_alpha_opt).

    One alpha per element of EVERY nonlin op with a RelaxLib band_alpha
    (relu, sigmoid/tanh, sin/cos/pow, exp, reciprocal), shared across query
    rows -- the whole forward state is differentiable in the slopes, so the
    optimizer tightens every relaxation jointly against the spec margin
    (backward CROWN relaxes each op once per direction and its per-op alpha
    measured bit-identical on ml4acopf; the forward composition is where
    the slope leverage lives: v1 climbs -12.76 -> +2.7e-7 on 0298 with this
    exact mechanism).

    disj_idx: optional (q,) disjunct index per query row. When given, the
    objective is v1's: maximize min over disjuncts of (max over the
    disjunct's rows) -- all gradient goes to the worst OPEN disjunct, and
    the phase exits as soon as every disjunct has a positive row. Without
    it, a hinged sum over rows.

    Returns (B, q) lower bounds on W @ y. Sound: every alpha in [0, 1]
    yields a bracketing zonotope (offsets recomputed per slope), so the
    elementwise-best iterate is a valid bound; iterate 0 uses the default
    bands, so the result is never worse than plain zono. clamp_bounds
    (CROWN-refined intermediates) mirrors v1's tight_bounds."""
    lo, hi = _as2d(lo), _as2d(hi)
    B = lo.shape[0]
    q = W.shape[-2]

    def margins(override):
        _, _, st = zono(net, lo, hi, return_state=True,
                        slope_override=override, clamp_bounds=clamp_bounds)
        z = st[net.output_name]
        return z.c @ W.T - torch.matmul(W, z.G).abs().sum(-1)

    best = margins(None).detach()
    raw0 = best
    if known is not None:
        # the pipeline's already-proven bound (crown chain). Seeding best
        # with it makes the stall rule DOMINANCE-aware: progress counts
        # only when the zono frame pushes past what is already known, so
        # a genuinely-climbing-but-dominated run stops after `patience`
        # iters (vit 2157: 55s of climb toward -0.043 with -0.029 in
        # hand) while a leading run (ml4acopf: crown is far behind) is
        # untouched. Returning the max is sound: both are valid bounds.
        best = torch.maximum(best, known.to(best.dtype))
    from .relax import REL
    a_ops = [nm for nm in net.order if net.ops[nm].kind == 'nonlin'
             and hasattr(REL[net.ops[nm].fn], 'band_alpha')]
    if not a_ops or iters <= 0:
        return best
    alphas = {}
    for nm in a_ops:
        if init_alphas is not None and nm in init_alphas:
            # warm start (the f32 stage's optimum feeding the f64 stage)
            t = init_alphas[nm].detach().to(lo.device, lo.dtype) \
                .clamp(0.0, 1.0).clone().requires_grad_(True)
        else:
            t = torch.full((B, net.ops[nm].n), 0.5, device=lo.device,
                           dtype=lo.dtype, requires_grad=True)
        alphas[nm] = t
    opt = torch.optim.Adam(list(alphas.values()), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15, min_lr=1e-3)
    thr = (torch.zeros(q, device=lo.device, dtype=lo.dtype)
           if thresholds is None else thresholds)
    gidx = None
    if disj_idx is not None:
        groups = {}
        for i, d in enumerate(disj_idx.tolist()):
            groups.setdefault(d, []).append(i)
        gidx = [torch.tensor(v, device=lo.device) for v in groups.values()]
    def _worst_open(bnd):
        if gidx is not None:
            return float(torch.stack([(bnd[0][g] - thr[g]).max()
                                      for g in gidx]).min())
        return float((bnd - thr.unsqueeze(0)).min())

    if known is not None:
        # start-ratio gate: when the DEFAULT-band zono starts an order
        # of magnitude further from closing than the bound already in
        # hand, the frame does not overtake within any observed budget
        # (vit: raw -0.38 vs crown -0.029, 13x, optimum -0.043 still
        # dominated after 55s; ml4acopf p3: raw -12 vs crown -9, 1.3x,
        # overtakes and closes). 3x splits those regimes.
        kw_ = _worst_open(known.to(best.dtype).expand_as(best))
        rw_ = _worst_open(raw0)
        import os as _os
        if _os.environ.get('VC2_FZ_DEBUG'):
            print(f'[fz-gate] raw={rw_:+.4f} known={kw_:+.4f} '
                  f'ratio={(-rw_) / max(-kw_, 1e-30):.2f}', flush=True)
        if kw_ < 0 and rw_ < 0 and -rw_ > 3.0 * -kw_:
            return (best, None) if return_alphas else best

    stall = 0
    best_alphas = None
    best_obj = -torch.inf
    prev_gap = float('inf')
    for _ in range(max(1, iters)):
        if budget is not None and budget.over():
            break
        lb = margins(alphas)
        best = torch.maximum(best, lb.detach())
        if gidx is not None:
            done = all(bool((best[0][g] > thr[g]).any()) for g in gidx)
            ob = torch.stack([(best[0][g] - thr[g]).max()
                              for g in gidx]).min()
        else:
            done = bool((best > thr.unsqueeze(0)).all())
            ob = (best - thr.unsqueeze(0)).min()
        if done:
            break                       # every disjunct has a positive row
        # progress = RELATIVE shrink of the worst-open gap (on the
        # monotone best). An any-element `gained` test never stalls on a
        # crawling tail: vit 2157 measured 55s of +1e-6/iter crawl on a
        # bound the crown chain already dominated, starving the dual/BaB
        # endgame to 8s each. Relative-to-gap keeps the ml4acopf closers
        # alive: near zero the required absolute progress shrinks too.
        gap = float((-ob).clamp_min(0.0))
        stall = 0 if gap < prev_gap * (1.0 - 1e-3) else stall + 1
        prev_gap = min(prev_gap, gap)
        if stall > patience:
            break
        if gidx is not None:
            m = lb[0] - thr             # margins past threshold, worst first
            obj = torch.stack([m[g].max() for g in gidx]).min()
        else:
            obj = torch.minimum(lb, thr.unsqueeze(0) + 1.0).sum()
        objf = float(obj.detach())
        if objf != objf:                # NaN iterate: best (detached) stands
            break
        if return_alphas and objf > best_obj:
            best_obj = objf
            best_alphas = {nm: t.detach().clone()
                           for nm, t in alphas.items()}
        opt.zero_grad(set_to_none=True)
        (-obj).backward()
        with torch.no_grad():
            for t in alphas.values():
                if t.grad is not None:
                    # crit-point closed forms (arccos, sqrt) have INFINITE
                    # gradient at their clamp boundary; the true gradient
                    # through a stationary point is 0 (envelope theorem),
                    # so zeroing the non-finite entries is exact there and
                    # merely freezes the rare boundary-pinned element
                    torch.nan_to_num_(t.grad, nan=0.0,
                                      posinf=0.0, neginf=0.0)
        opt.step()
        sched.step(objf)
        with torch.no_grad():
            for t in alphas.values():
                t.clamp_(0.0, 1.0)
    if return_alphas:
        return best, best_alphas
    return best
