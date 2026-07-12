"""The backward propagator: CROWN adjoint walk over the DAG (design 2.2).

One implementation. Computes sound LOWER bounds on q linear output rows
W (q, n_out) over B input boxes, walking the flat DAG in reverse topological
order. Per edge the state is an adjoint tensor A (B, q, n_edge) plus an
accumulated offset d (B, q); a fork's consumers sum their adjoints (exact),
a LinMap applies lin_t, a nonlinearity applies its RelaxLib planes split by
adjoint sign, and the input concretizes A over the box.

alpha: optimizable lower-plane slopes for unstable relus, one tensor per
relu op of shape (B, q, n) in [0,1]; when absent the sound adaptive default
from RelaxLib is used. The whole walk is differentiable w.r.t. alpha, so
`alpha_crown` just Adam-ascends the bound. beta (split constraints) and
gamma (output constraints) extend this same walk in later milestones.

Intermediate (pre-activation) bounds come from the forward propagator and
are treated as constants (fixed-intermediate mode).
"""
from __future__ import annotations

import os

import torch

from . import forward as fwd
from .relax import REL


def _neg_part(A):
    return A.clamp(max=0)


def _pos_part(A):
    return A.clamp(min=0)


_WARNED = set()


def _warn_once(msg):
    """Loud, non-spamming: every silent degradation prints ONCE per
    process so the run log always says what quality level produced the
    verdict (fail-loud policy; never eat an exception quietly)."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f'[vc2/degrade] {msg}', flush=True)


def _zono_cost_bytes(net, B):
    """Projected peak cost of a dense forward zonotope: max over edges of
    B * n_edge * (n_in + #relu elements so far) * 4 bytes. Cheap shape-only
    estimate for the generator-lifecycle decision (design 3.3); the full
    reduce/drop-and-continue lifecycle replaces this in M5."""
    g = net.n_in
    worst = 0
    for name in net.order:
        op = net.ops[name]
        worst = max(worst, op.n * g)
        if op.kind == 'nonlin':
            g += op.n         # worst case: every element adds a fresh gen
                              # (relu unstable, sigmoid/sign/exp bands)
    return worst * B * 4


def _mccormick_planes(lx, hx, ly, hy):
    """Per-term McCormick planes for a product x*y over box bounds, with
    the midpoint-tightness selection between the two valid planes on each
    side. One engine; `mul` (elementwise) and `bmm` (contracted) are shape
    instances -- future bilinear ops (attention Q@K) reuse it by
    broadcasting their bounds to the term layout.

    Returns (alx, aly, clo, aux, auy, cup): lower plane
    x*y >= alx*x + aly*y + clo and upper x*y <= aux*x + auy*y + cup.
    """
    cx, cy = (lx + hx) / 2, (ly + hy) / 2
    lo1_v = ly * cx + lx * cy - lx * ly
    lo2_v = hy * cx + hx * cy - hx * hy
    pick_lo = (lo1_v >= lo2_v)
    alx = torch.where(pick_lo, ly, hy)
    aly = torch.where(pick_lo, lx, hx)
    clo = torch.where(pick_lo, -lx * ly, -hx * hy)
    up1_v = ly * cx + hx * cy - hx * ly
    up2_v = hy * cx + lx * cy - lx * hy
    pick_up = (up1_v <= up2_v)
    aux = torch.where(pick_up, ly, hy)
    auy = torch.where(pick_up, hx, lx)
    cup = torch.where(pick_up, -hx * ly, -lx * hy)
    return alx, aly, clo, aux, auy, cup


def _inter_from_state(net, bounds_of):
    """{op: pre-activation bounds} for nonlin ops; a bilinear mul stores the
    bound pair of BOTH factors (its McCormick planes need them)."""
    inter = {}
    for name in net.order:
        op = net.ops[name]
        if op.kind == 'nonlin':
            inter[name] = bounds_of(op.inputs[0])
        elif op.kind in ('mul', 'bmm'):
            (lx, hx) = bounds_of(op.inputs[0])
            (ly, hy) = bounds_of(op.inputs[1])
            inter[name] = (lx, hx, ly, hy)     # flat: slices uniformly
    return inter


def intermediates(net, lo, hi):
    """Pre-activation bounds for every nonlinear edge: dense forward
    zonotope when its projected cost fits the memory budget, else the
    BUDGET forward (top-K input symbols, every fresh delta in the box
    remainder -- ab-crown's vggnet16 dynamic-forward, max_dim ~100);
    interval only when zono lacks an op's relaxation. The budget pass is
    bounded-memory on any net and tighter than interval on the early
    conv layers, which is what the relu planes there read (the dense
    pass on vgg16-7 OOMs at any cap; the budget pass runs in ~10 GB)."""
    from . import memory
    B = lo.shape[0]
    ist = fwd.interval(net, lo, hi, return_state=True)
    iiv = _inter_from_state(net, lambda e: ist[e])
    dense_fits = (_zono_cost_bytes(net, B)
                  < memory.free_bytes(lo.device) * memory.SAFETY)
    try:
        if dense_fits:
            _lo, _hi, state = fwd.zono(net, lo, hi, return_state=True)
        else:
            _lo, _hi, state = fwd.zono(net, lo, hi, return_state=True,
                                       box_remainder='all',
                                       sym_budget=96)
        izo = _inter_from_state(net, lambda e: state[e].bounds())
        # intersect with interval per entry (both sound; interval
        # keeps structurally-positive chains positive where the zono
        # concretization of band generators goes negative -- the
        # softmax difference form's sum >= exp(0) = 1)
        out = {}
        for k2, zt in izo.items():
            it = iiv[k2]
            merged = []
            for j2 in range(0, len(zt), 2):
                lo2 = torch.maximum(zt[j2], it[j2])
                hi2 = torch.minimum(zt[j2 + 1], it[j2 + 1])
                merged.append(lo2)
                merged.append(torch.maximum(hi2, lo2))
            out[k2] = tuple(merged)
        return out
    except NotImplementedError as e:
        _warn_once(f'intermediates: zono unavailable ({e}); '
                   f'interval bounds only')
    except torch.cuda.OutOfMemoryError:
        # the cost model missed (band nonlins whose fresh gens only
        # materialize for crossing elements); interval is the
        # sanctioned degradation, not a crash
        torch.cuda.empty_cache()
    return iiv


from .forward import clamped_bounds  # single definition (forward.py)


def _softmax_adjoint(op, Ao, l, h, d):
    """Backward-CROWN adjoint THROUGH the fused softmax op, composed from
    the same relaxations the rest of the core uses (Exp planes,
    Reciprocal planes, McCormick product) over the chain
        e = exp(x - c),  s = rowsum(e),  r = 1/s,  y = e * r_row ,
    with c the constant per-row shift from the pre-activation bounds
    (softmax is shift-invariant, so the rewrite is exact). Mathematically
    identical to running CROWN over the decomposed graph with those
    per-op planes -- the fused op previously exposed only interval-
    constant planes, which KILLED the adjoint at every attention block
    (root backward -9.9 where v1's attention backward reaches -3.0/-0.0156
    with alphas; the whole joint-alpha chain was blind to the logits).

    Ao: (B, q, n_sm) adjoint arriving at y. l, h: logit bounds (B, n_sm).
    Returns (A_x (B, q, n_sm), d updated with the intercept terms).
    """
    p = op.params
    pre, k, post = p['pre'], p['k'], p['post']
    B = l.shape[0]
    n_rows = pre * post

    def rows(t):                    # (B, n_sm) -> (B, pre, k, post)
        return t.reshape(B, pre, k, post)

    def rows_q(t):                  # (B, q, n_sm) -> (B, q, pre, k, post)
        return t.reshape(t.shape[0], t.shape[1], pre, k, post)

    c = rows(h).max(dim=2, keepdim=True).values          # (B,pre,1,post)
    c_bc = c.expand(B, pre, k, post).reshape(B, -1)
    zl, zh = l - c_bc, h - c_bc
    zh = zh.clamp_max(0.0)                               # exact: c >= h
    el, eh = torch.exp(zl), torch.exp(zh)                # e in (0, 1]
    sl = rows(el).sum(dim=2).reshape(B, n_rows).clamp_min(1e-30)
    sh = rows(eh).sum(dim=2).reshape(B, n_rows).clamp_max(float(k))
    sh = torch.maximum(sh, sl)
    rl, rh = 1.0 / sh, 1.0 / sl
    bidx = torch.arange(n_rows, device=l.device).reshape(
        pre, 1, post).expand(pre, k, post).reshape(-1)
    rl_bc, rh_bc = rl[:, bidx], rh[:, bidx]
    # stage 1: y = e * r (McCormick over the factor boxes)
    alx, aly, clo, aux, auy, cup = (
        t.unsqueeze(1) for t in _mccormick_planes(el, eh, rl_bc, rh_bc))
    Ap, An = _pos_part(Ao), _neg_part(Ao)
    A_e = Ap * alx + An * aux
    A_rb = Ap * aly + An * auy                           # onto r broadcast
    d = d + (Ap * clo + An * cup).sum(dim=2)
    A_r = rows_q(A_rb).sum(dim=3).reshape(Ao.shape[0], Ao.shape[1], n_rows)
    # stage 2: r = 1/s (sign-definite planes; s > 0 by construction)
    arl, brl, aru, bru = REL['reciprocal'].planes(sl, sh)
    Ap2, An2 = _pos_part(A_r), _neg_part(A_r)
    A_s = Ap2 * arl.unsqueeze(1) + An2 * aru.unsqueeze(1)
    d = d + (Ap2 * brl.unsqueeze(1) + An2 * bru.unsqueeze(1)).sum(dim=2)
    # stage 3: s = rowsum(e) (linear)
    A_e = A_e + A_s[:, :, bidx]
    # stage 4: e = exp(z), z = x - c: fold the constant shift into the
    # intercepts (e >= a*z + b = a*x + (b - a*c))
    ael, bel, aeu, beu = REL['exp'].planes(zl, zh)
    bel = bel - ael * c_bc
    beu = beu - aeu * c_bc
    Ap3, An3 = _pos_part(A_e), _neg_part(A_e)
    A_x = Ap3 * ael.unsqueeze(1) + An3 * aeu.unsqueeze(1)
    d = d + (Ap3 * bel.unsqueeze(1) + An3 * beu.unsqueeze(1)).sum(dim=2)
    return A_x, d


def _simplex_hull_wins(op, bounds4, Ao):
    """Per-op branch choice for a softmax-left bmm: hull-stop vs McCormick
    propagation, by the branches' ACTUAL per-element uncertainty mass
    (no tuned threshold). Hull cost: the hull box width. McCormick cost:
    the intercept spread cup - clo plus the propagated-slope slack, which
    scales with width(attn) x width(V) x k and dominates near-[0,1]
    weights. Ties break to McCormick (it keeps the adjoint alive)."""
    lx, hx, ly, hy = bounds4
    ash, bsh = op.params['a_shape'], op.params['b_shape']
    m, k, p = ash[-2], ash[-1], bsh[-1]
    G = 1
    for dd in ash[:-2]:
        G *= int(dd)
    Bv = lx.shape[0]
    hull_w = (hy.reshape(Bv, G, k, p).max(dim=2).values
              - ly.reshape(Bv, G, k, p).min(dim=2).values)   # (Bv, G, p)
    # per output element, the McCormick sum over the contraction picks up
    # ~ sum_l w(attn_l) * w(V_l) of quadratic slack
    wx = (hx - lx).reshape(Bv, G, m, k)
    wy = (hy - ly).reshape(Bv, G, k, p)
    mc_w = torch.matmul(wx, wy)                              # (Bv, G, m, p)
    return bool(mc_w.mean() > hull_w.mean())


def crown(net, lo, hi, W, inter=None, alpha=None, start=None,
          return_input_adjoint=False, clamps=None, beta=None,
          collect_adjoints=None, range_clamps=None, gamma=None,
          gamma_rows=None):
    """Lower bounds on W @ y_edge for x in [lo, hi], where y_edge is the
    value of edge `start` (default: the network output). Bounding an
    INTERMEDIATE edge is the same walk seeded there; ops after it never
    accumulate an adjoint and are skipped naturally.

    lo, hi: (B, n_in); W: (q, n_edge) or (B, q, n_edge).
    inter: {nonlin_op_name: (lo, hi)} pre-activation bounds ((B, n) each).
    alpha: {relu_op_name: (B, q, n) in [0, 1]} optimizable lower slopes.
    clamps: {relu_op_name: (B, n) in {-1, 0, +1}} BaB sign splits; the
        relaxation becomes exact identity/zero on clamped neurons.
    beta: {relu_op_name: (B, q, n) >= 0} split-constraint multipliers:
        a pos split z>=0 adds -beta to the pre-activation adjoint, a neg
        split +beta (Lagrangian of the split constraint; sound for beta>=0,
        beta=0 recovers the plain bound).
    collect_adjoints: optional dict; on return holds the pre-activation
        adjoint per relu op named in it (for BaBSR-style action scoring).
    Returns lb (B, q).
    """
    B = lo.shape[0]
    dev, dt = lo.device, lo.dtype
    if inter is None:
        inter = intermediates(net, lo, hi)
    if W.dim() == 2:
        W = W.unsqueeze(0).expand(B, -1, -1)
    q = W.shape[1]

    A = {start or net.output_name: W.to(device=dev, dtype=dt)}
    d = torch.zeros(B, q, device=dev, dtype=dt)
    if gamma is not None:
        # INVPROP / gamma: any counterexample satisfies the spec's output
        # rows w_m.y + b_m <= 0, so adding gamma_m * (w_m.y + b_m) with
        # gamma >= 0 to the objective only lowers it ON THE CE REGION; its
        # lower bound therefore stays a sound refutation bound there.
        Wg, bg = gamma_rows
        Wg = torch.as_tensor(Wg, device=dev, dtype=dt)
        bg = torch.as_tensor(bg, device=dev, dtype=dt)
        g = gamma.clamp_min(0.0)
        contrib = torch.einsum('bqm,mn->bqn', g, Wg)
        nm_out = net.output_name
        A[nm_out] = (A[nm_out] + contrib) if nm_out in A else contrib
        d = d + torch.einsum('bqm,m->bq', g, bg)

    def take(name):
        """Pop the accumulated adjoint for edge `name` (zeros if unused)."""
        return A.pop(name)

    def put(name, val):
        A[name] = A[name] + val if name in A else val

    for name in reversed(net.order):
        if name not in A:
            continue                     # edge does not influence the queries
        op = net.ops[name]
        Ao = take(name)
        if op.kind == 'linmap':
            d = d + Ao @ op.lm.bias_vec(Ao)
            Ain = op.lm.lin_t(Ao.reshape(B * q, -1)).reshape(B, q, -1)
            put(op.inputs[0], Ain)
        elif op.kind == 'add':
            put(op.inputs[0], Ao)
            put(op.inputs[1], Ao)
        elif op.kind == 'concat':
            # the forward OVERWRITES covered slots with input values, so
            # the base constant contributes only at uncovered slots --
            # adding it everywhere double-counts covered positions
            # (caught by the op-coverage suite with a nonzero base)
            base = torch.as_tensor(op.params['base'], device=dev,
                                   dtype=dt).clone()
            for pos in op.params['positions']:
                base[torch.as_tensor(pos, device=dev)] = 0.0
            d = d + Ao @ base
            for src, pos in zip(op.inputs, op.params['positions']):
                put(src, Ao[:, :, torch.as_tensor(pos, device=dev)])
        elif op.kind == 'nonlin':
            l, h = inter[name]
            cl = clamps.get(name) if clamps else None
            if cl is not None:
                l, h = clamped_bounds((l, h), cl)
            if range_clamps and name in range_clamps:
                rlo, rhi = range_clamps[name]
                l = torch.maximum(l, rlo)
                h = torch.minimum(h, torch.maximum(rhi, l))
            rel = REL[op.fn]
            if op.fn == 'softmax':
                Ain_sm, d = _softmax_adjoint(op, Ao, l, h, d)
                if collect_adjoints is not None:
                    collect_adjoints[name] = Ao.detach()
                put(op.inputs[0], Ain_sm)
                continue
            if not hasattr(rel, 'planes'):
                raise NotImplementedError(
                    f'crown: no planes for nonlinearity {op.fn!r} yet')
            if (alpha and name in alpha and op.fn != 'relu'
                    and hasattr(rel, 'alpha_planes')):
                # S-shaped alpha: optimizer-controlled tangent positions
                al, bl, au, bu = rel.alpha_planes(
                    l.unsqueeze(1), h.unsqueeze(1),
                    alpha[name].clamp(0.0, 1.0), op.params)
            else:
                try:
                    al, bl, au, bu = rel.planes(l, h, op.params)
                except NotImplementedError:
                    if 'out_lo' not in op.params:
                        raise
                    # emission-declared output range as constant planes:
                    # the softmax reciprocal is in (0, 1] because its sum
                    # contains exp(0) = 1, no matter how loose the
                    # propagated input range got
                    al = torch.zeros_like(l)
                    au = torch.zeros_like(l)
                    bl = torch.full_like(l, op.params['out_lo'])
                    bu = torch.full_like(l, op.params['out_hi'])
                if alpha and name in alpha:
                    # relu: optimizable lower slope on unstable neurons only
                    unstable = ((l < 0) & (h > 0)).unsqueeze(1)
                    al = torch.where(unstable, alpha[name].clamp(0.0, 1.0),
                                     al.unsqueeze(1))
            if al.dim() == 2:
                al = al.unsqueeze(1)
            if au.dim() == 2:
                au = au.unsqueeze(1)
            if bl.dim() == 2:
                bl = bl.unsqueeze(1)
            if bu.dim() == 2:
                bu = bu.unsqueeze(1)
            Ap, An = _pos_part(Ao), _neg_part(Ao)
            # lower bound: positive adjoint takes the lower plane,
            # negative adjoint the upper plane
            Ain = Ap * al + An * au
            d = d + (Ap * bl + An * bu).sum(dim=2)
            if beta and name in beta and cl is not None:
                # split-constraint Lagrangian: pos split (z>=0) adds -beta*z,
                # neg split (z<=0) adds +beta*z to the objective (beta>=0)
                sgn = -cl.sign().unsqueeze(1).to(dt)
                Ain = Ain + sgn * beta[name].clamp_min(0.0)
            if collect_adjoints is not None:
                # the adjoint ARRIVING at the nonlin output (ew): the sign
                # tells which plane the backward binds (dir-adaptive states)
                collect_adjoints[name] = Ao.detach()
            put(op.inputs[0], Ain)
        elif op.kind == 'mul':
            # McCormick planes for z = x*y over the factor boxes:
            #   z >= ly*x + lx*y - lx*ly     z >= hy*x + hx*y - hx*hy
            #   z <= ly*x + hx*y - hx*ly     z <= hy*x + lx*y - lx*hy
            # per element pick the plane pair that is tighter at the box
            # center; adjoint sign selects lower (A+) vs upper (A-).
            lx, hx, ly, hy = inter[name]
            if range_clamps:
                # MUL-FACTOR range splits (BaB): entries keyed by the
                # FACTOR EDGE name constrain that factor's box -- the
                # McCormick error is (hx-lx)(hy-ly)/4, so halving a
                # factor quarters it where relu/smooth splits cannot
                # move it at all (ml4acopf's frozen -8.7)
                e_a, e_b = op.inputs[0], op.inputs[1]
                if e_a in range_clamps:
                    lx = torch.maximum(lx, range_clamps[e_a][0])
                    hx = torch.minimum(hx, torch.maximum(
                        range_clamps[e_a][1], lx))
                if e_b in range_clamps:
                    ly = torch.maximum(ly, range_clamps[e_b][0])
                    hy = torch.minimum(hy, torch.maximum(
                        range_clamps[e_b][1], ly))
            alx, aly, clo, aux, auy, cup = (
                t.unsqueeze(1)
                for t in _mccormick_planes(lx, hx, ly, hy))
            Ap, An = _pos_part(Ao), _neg_part(Ao)
            put(op.inputs[0], Ap * alx + An * aux)
            put(op.inputs[1], Ap * aly + An * auy)
            d = d + (Ap * clo + An * cup).sum(dim=2)
        elif op.kind == 'bmm' and op.params.get('simplex_left') \
                and _simplex_hull_wins(op, inter[name], Ao):
            # attention @ V, LOOSE-weights regime: the left rows are
            # softmax weights (simplex), so the output is inside the
            # coordinatewise hull of the right factor's rows -- constant
            # planes, adjoint terminates here (sound). Chosen per op by
            # comparing the two branches' actual intercept mass (no
            # thresholds): near-[0,1] weights make McCormick blow up by
            # the token count, while TIGHT fused-softmax weights make the
            # McCormick route carry the adjoint through the whole
            # attention stack -- measured on vit 2157: root crown -10.77
            # (hull-stop) -> -0.0397, alpha -9.89 -> -0.031; the entire
            # backward chain was blind upstream of attention before.
            _, _, ly, hy = inter[name]
            bsh = op.params['b_shape']
            k, p = bsh[-2], bsh[-1]
            Bv = Ao.shape[0]
            sl = ly.reshape(Bv, -1, k, p).min(dim=2, keepdim=True).values
            sh = hy.reshape(Bv, -1, k, p).max(dim=2, keepdim=True).values
            m = op.params['a_shape'][-2]
            sl = sl.expand(Bv, sl.shape[1], m, p).reshape(Bv, 1, -1)
            sh = sh.expand(Bv, sh.shape[1], m, p).reshape(Bv, 1, -1)
            Ap, An = _pos_part(Ao), _neg_part(Ao)
            d = d + (Ap * sl + An * sh).sum(dim=2)
        elif op.kind == 'bmm':
            # bmm = the contracted instance of the same McCormick engine:
            # out[g,i,j] = sum_l X[g,i,l] * Y[g,l,j]; the adjoint on each
            # (g,i,l,j) term distributes sign-wise onto the factor planes
            lx, hx, ly, hy = inter[name]
            ash, bsh = op.params['a_shape'], op.params['b_shape']
            m, k, p = ash[-2], ash[-1], bsh[-1]
            G = 1
            for dd in ash[:-2]:
                G *= int(dd)
            B = Ao.shape[0]
            q = Ao.shape[1]
            lxr = lx.reshape(B, 1, G, m, k, 1)
            hxr = hx.reshape(B, 1, G, m, k, 1)
            lyr = ly.reshape(B, 1, G, 1, k, p)
            hyr = hy.reshape(B, 1, G, 1, k, p)
            alx, aly, clo, aux, auy, cup = _mccormick_planes(
                lxr, hxr, lyr, hyr)
            Aot = Ao.reshape(B, q, G, m, 1, p)
            Ap, An = _pos_part(Aot), _neg_part(Aot)
            put(op.inputs[0],
                (Ap * alx + An * aux).sum(dim=5).reshape(B, q, -1))
            put(op.inputs[1],
                (Ap * aly + An * auy).sum(dim=3).reshape(B, q, -1))
            d = d + (Ap * clo + An * cup).sum(dim=(2, 3, 4, 5))
        elif op.kind == 'maxpool':
            raise AssertionError(                       # unreachable
                'crown reached a maxpool op: decompose_maxpool must run at '
                'load (graph.py) so no maxpool survives into relaxation')
        else:
            raise NotImplementedError(f'crown: op kind {op.kind!r}')

    Ain = A.pop(net.input_name, None)
    if Ain is None:
        # every adjoint path terminated early (a simplex-hull bmm whose
        # constant planes absorbed all mass): the bound is d alone
        Ain = torch.zeros(lo.shape[0], W.shape[-2], net.n_in,
                          device=lo.device, dtype=lo.dtype)
    assert not A, f'unconsumed adjoints: {list(A)}'
    c = (hi + lo) / 2
    r = (hi - lo) / 2
    lb = d + torch.einsum('bqn,bn->bq', Ain, c) \
           - torch.einsum('bqn,bn->bq', Ain.abs(), r)
    if return_input_adjoint:
        return lb, Ain
    return lb


def intermediates_crown(net, lo, hi, base_inter=None, budget=None,
                        clamps=None, range_clamps=None, gamma_rows=None,
                        gamma_iters=8, alpha_iters=0):
    """Pre-activation bounds per nonlin edge via per-edge backward CROWN
    (chunked identity queries, both signs in one pass). Strictly tighter
    than interval; the regime for conv nets whose dense zonotope does not
    fit (until patches land in M5). Earlier edges' CROWN bounds feed later
    edges' relaxations (topo order)."""
    from . import memory
    B = lo.shape[0]
    dev, dt = lo.device, lo.dtype
    # interval bounds seed every edge; CROWN refines ONLY the neurons whose
    # interval sign is ambiguous (planes are already exact on stable ones),
    # which is a small fraction on certified/robust nets
    if base_inter is None:
        state = fwd.interval(net, lo, hi, return_state=True)
        base_inter = _inter_from_state(net, lambda e: state[e])
    inter = dict(base_inter)
    widest = max(net.ops[o].n for o in net.order)
    refined = {}                      # factor edges already refined
    for name in net.order:
        if budget is not None:
            budget.check()
        op = net.ops[name]
        if op.kind in ('mul', 'bmm'):
            # refine BOTH factor edges (McCormick quality tracks them);
            # bounds land back in the (lx, hx, ly, hy) flat tuple
            lx, hx, ly, hy = inter[name]
            outs = []
            for e2, (l2, h2) in ((op.inputs[0], (lx, hx)),
                                 (op.inputs[1], (ly, hy))):
                if range_clamps and e2 in range_clamps:
                    # BaB mul-factor range split (keyed by factor EDGE)
                    l2 = torch.maximum(l2, range_clamps[e2][0])
                    h2 = torch.minimum(h2, torch.maximum(
                        range_clamps[e2][1], l2))
                if e2 in refined:
                    r_l, r_h = refined[e2]
                    if range_clamps and e2 in range_clamps:
                        r_l = torch.maximum(r_l, range_clamps[e2][0])
                        r_h = torch.minimum(r_h, torch.maximum(
                            range_clamps[e2][1], r_l))
                    outs.append((r_l, r_h))
                    continue
                idx2 = torch.arange(net.ops[e2].n, device=dev)
                lb2, ub2 = l2.clone(), h2.clone()
                per_row2 = B * widest * 4 * 12

                def refine2(sel, _e=e2, _lb=lb2, _ub=ub2):
                    m = sel.numel()
                    Wc = torch.zeros(2 * m, net.ops[_e].n, device=dev,
                                     dtype=dt)
                    ar = torch.arange(m, device=dev)
                    Wc[ar, sel] = 1.0
                    Wc[m + ar, sel] = -1.0
                    out = crown(net, lo, hi,
                                Wc.unsqueeze(0).expand(B, -1, -1), inter,
                                start=_e, clamps=clamps,
                                range_clamps=range_clamps)
                    # a refinement through non-finite planes (exp past
                    # fp32) must be a NO-OP, not a poison
                    out = torch.where(torch.isfinite(out), out,
                                      torch.full_like(out, -torch.inf))
                    _lb[:, sel] = torch.maximum(_lb[:, sel], out[:, :m])
                    _ub[:, sel] = torch.minimum(_ub[:, sel], -out[:, m:])

                memory.chunked_indices(refine2, idx2, per_row2)
                refined[e2] = (lb2, ub2)
                outs.append((lb2, ub2))
            inter[name] = (outs[0][0], outs[0][1], outs[1][0], outs[1][1])
            continue
        if op.kind != 'nonlin':
            continue
        e = op.inputs[0]
        n = net.ops[e].n
        l0, h0 = inter[name]
        if clamps and name in clamps:
            l0, h0 = clamped_bounds((l0, h0), clamps[name])
        if range_clamps and name in range_clamps:
            rlo, rhi = range_clamps[name]
            l0 = torch.maximum(l0, rlo)
            h0 = torch.minimum(h0, torch.maximum(rhi, l0))
        inter[name] = (l0, h0)
        idx = torch.nonzero(((l0 < 0) & (h0 > 0)).any(dim=0),
                            as_tuple=False).flatten()
        if op.fn != 'relu' and (clamps or range_clamps):
            # SMOOTH ops in a per-domain (BaB) context: the sign-crossing
            # filter is the RELU criterion; a smooth band's slack scales
            # with WIDTH, and skipping wide-but-positive inputs meant no
            # split kind could ever move the spec bound through them
            # (ml4acopf 118-linear: mul-factor splits fired, downstream
            # sigmoid/sin inputs never re-tightened, bound frozen -8.78).
            # Refine the widest elements too, capped.
            wdt = (h0 - l0).amax(dim=0)
            k_w = min(256, wdt.numel())
            widx = wdt.topk(k_w).indices
            widx = widx[wdt[widx] > 1e-6]
            idx = torch.unique(torch.cat([idx, widx]))
        if not idx.numel():
            continue
        if net.ops[e].n > 65536 and clamps is None and range_clamps is None:
            # huge conv edge: the dense identity block (idx x n_prev) is
            # unrunnable (vgg16-7: 3.2M queries x 150k); the patch walk
            # refines ALL elements at window cost (M5). Falls through to
            # dense on the first unpatchable op.
            try:
                from .patches import patch_refine
                lb_p, ub_p = patch_refine(net, e, lo, hi, inter)
                l0 = torch.maximum(l0, lb_p)
                h0 = torch.minimum(h0, torch.maximum(ub_p, l0))
                inter[name] = (l0, h0)
                continue
            except NotImplementedError:
                pass                          # dense path below
        # identity blocks per chunk (never a full n x n eye); both signs in
        # one walk so lo and hi share it. A deep walk holds several live
        # adjoints plus conv temporaries, hence the generous per-row factor;
        # chunked_indices halves on an OOM anyway (the one sanctioned catch).
        per_row = B * widest * 4 * 12
        lb = l0.clone()
        ub = h0.clone()

        def refine(sel, _e=e, _n=n, _lb=lb, _ub=ub):
            m = sel.numel()
            Wc = torch.zeros(2 * m, _n, device=dev, dtype=dt)
            ar = torch.arange(m, device=dev)
            Wc[ar, sel] = 1.0
            Wc[m + ar, sel] = -1.0
            Wb = Wc.unsqueeze(0).expand(B, -1, -1)
            out = crown(net, lo, hi, Wb, inter, start=_e, clamps=clamps,
                        range_clamps=range_clamps)
            out = torch.where(torch.isfinite(out), out,
                              torch.full_like(out, -torch.inf))
            if alpha_iters > 0:
                # joint-intermediate alpha (v1 phase-0.5 alpha-refresh): the
                # identity rows themselves get optimized slopes, which is
                # what lifts the ROOT bound (measured: v1 -4.5 vs -11.3 on
                # dist_shift with fixed-slope refinement only)
                oa = alpha_crown(net, lo, hi, Wc, inter, iters=alpha_iters,
                                 budget=budget, start=_e)
                out = torch.maximum(out, oa)
            if gamma_rows is not None:
                # gamma (INVPROP): Adam-ascend output-row multipliers; the
                # refined bounds are CONDITIONAL on the CE region of the
                # spec rows supplied -- callers must scope them to that
                # disjunct's refutation only
                mg = len(gamma_rows[1])
                gam = torch.zeros(B, 2 * m, mg, device=dev, dtype=dt,
                                  requires_grad=True)
                opt = torch.optim.Adam([gam], lr=0.5)
                for _ in range(max(1, gamma_iters)):
                    ob = crown(net, lo, hi, Wb, inter, start=_e,
                               clamps=clamps, range_clamps=range_clamps,
                               gamma=gam, gamma_rows=gamma_rows)
                    out = torch.maximum(out, ob.detach())
                    (-ob.sum()).backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        gam.clamp_(min=0.0)
            _lb[:, sel] = torch.maximum(_lb[:, sel], out[:, :m])
            _ub[:, sel] = torch.minimum(_ub[:, sel], -out[:, m:])

        memory.chunked_indices(refine, idx, per_row)
        inter[name] = (lb, ub)
    return inter


def alpha_beta_crown(net, lo, hi, W, inter, clamps, iters=15, lr=0.1,
                     thresholds=None, budget=None, share_q=None,
                     range_clamps=None, init_alpha=None, init_beta=None,
                     return_beta=False, return_alpha=False, lr_beta=None):
    """Jointly Adam-optimized alpha (relaxation slopes) + beta (split
    multipliers) lower bounds for a batch of BaB domains under sign clamps.
    Every iterate is a sound bound (beta projected to >= 0); returns the
    elementwise best.

    share_q: share the alpha/beta tensors across query rows ((B,1,n)
    broadcast instead of (B,q,n)). Slightly looser, q-times smaller; the
    default shares whenever the full tensors would be large."""
    B = lo.shape[0]
    q = W.shape[-2]
    dev, dt = lo.device, lo.dtype
    n_relu_total = sum(net.ops[nm].n for nm in net.order
                       if net.ops[nm].kind == 'nonlin'
                       and net.ops[nm].fn == 'relu')
    if share_q is None:
        share_q = B * q * n_relu_total * 4 * 8 > 1 << 30
    qd = 1 if share_q else q
    alpha, beta = {}, {}
    for name in net.order:
        op = net.ops[name]
        if op.kind != 'nonlin':
            continue
        if op.fn == 'relu':
            l, h = inter[name]
            cl = clamps.get(name)
            if cl is not None:
                l, h = clamped_bounds((l, h), cl)
            if init_alpha is not None and name in init_alpha:
                # warm start: either the ROOT's optimized alphas
                # ((1, q, n): reduce/expand) or PER-DOMAIN transferred
                # state ((B, qd, n): use as-is -- ab-crown's set_bounds
                # regime; measured on vit 2157, per-domain bound quality
                # tracks optimization effort and cold 12-iter runs stall
                # the tree at -0.014 where 30 iters reach -0.009)
                a0 = init_alpha[name].detach().to(l.device, l.dtype)
                if a0.shape[0] != B or a0.shape[1] != qd:
                    a0 = (a0.mean(dim=1, keepdim=True) if qd == 1
                          else a0[:, :qd]).expand(B, qd, l.shape[1])
                alpha[name] = a0.contiguous().requires_grad_(True)
            else:
                al0 = REL['relu'].planes(l, h)[0]
                alpha[name] = al0.detach().clone().unsqueeze(1) \
                    .expand(B, qd, l.shape[1]).contiguous() \
                    .requires_grad_(True)
            if cl is not None and bool((cl != 0).any()):
                b0 = torch.zeros(B, qd, l.shape[1], device=dev, dtype=dt)
                if init_beta is not None and name in init_beta:
                    # warm start from the PARENT domain's optimized betas
                    # (ab-crown transfers this state; re-optimizing from
                    # zero in a dozen iters loses depth-20 trees)
                    b0 = b0 + init_beta[name].to(dev, dt)
                beta[name] = b0.requires_grad_(True)
        elif hasattr(REL[op.fn], 'alpha_planes'):
            l, h = inter[name]
            if init_alpha is not None and name in init_alpha \
                    and init_alpha[name].dim() == 4:
                a0 = init_alpha[name].detach().to(l.device, l.dtype)
                a0 = (a0.mean(dim=1, keepdim=True) if qd == 1
                      else a0[:, :qd]).expand(B, qd, 2, l.shape[1])
                alpha[name] = a0.contiguous().requires_grad_(True)
            else:
                crossing = ((l < 0) & (h > 0)).to(l.dtype)
                t0 = (0.5 * (1 - crossing)).unsqueeze(1).unsqueeze(1) \
                    .expand(B, qd, 2, l.shape[1]).contiguous()
                alpha[name] = t0.requires_grad_(True)
    params = list(alpha.values()) + list(beta.values())
    if not params:
        lb0_ = crown(net, lo, hi, W, inter, clamps=clamps,
                     range_clamps=range_clamps)
        extras = ([{}] if return_beta else []) \
            + ([{}] if return_alpha else [])
        return (lb0_, *extras) if extras else lb0_
    # split learning rates (ab-crown's beta-crown defaults: lr_alpha
    # 0.01, lr_beta 0.1): betas grow FROM ZERO and need large steps;
    # warm alphas need small refining ones. A single shared lr measured
    # both failure modes (vit: lr 0.1 thrashed transferred alphas;
    # yolo: beta contributions ~0 at small lrs).
    if lr_beta is None or not beta:
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.Adam([
            {'params': list(alpha.values()), 'lr': lr},
            {'params': list(beta.values()), 'lr': lr_beta}])
    thr = (torch.zeros(q, device=dev, dtype=dt) if thresholds is None
           else thresholds.to(dev, dt))
    best = None
    for _ in range(max(1, iters)):
        if budget is not None and budget.over():
            break
        lb = crown(net, lo, hi, W, inter, alpha=alpha, clamps=clamps,
                   beta=beta, range_clamps=range_clamps)
        best = lb.detach() if best is None else torch.maximum(best, lb.detach())
        loss = -(torch.minimum(lb, thr.unsqueeze(0) + 1.0)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for t in alpha.values():
                t.clamp_(0.0, 1.0)
            for t in beta.values():
                t.clamp_(min=0.0)
    lb = crown(net, lo, hi, W, inter, alpha=alpha, clamps=clamps, beta=beta,
               range_clamps=range_clamps)
    best = torch.maximum(best, lb.detach())
    extras = []
    if return_beta:
        extras.append({k: v.detach().clamp_min(0.0)
                       for k, v in beta.items()})
    if return_alpha:
        extras.append({k: v.detach().clamp(0.0, 1.0)
                       for k, v in alpha.items()})
    return (best, *extras) if extras else best


def alpha_crown(net, lo, hi, W, inter=None, iters=20, lr=0.25,
                thresholds=None, budget=None, return_alpha=False,
                start=None):
    """Adam-optimized alpha-CROWN lower bounds (fixed intermediates).

    Maximizes each query's lb independently (sum of hinged bounds: a query
    already past its threshold contributes nothing, focusing the optimizer
    on the still-open ones). Returns the elementwise-best lb seen (sound:
    every iterate is a valid bound).
    """
    B = lo.shape[0]
    if inter is None:
        inter = intermediates(net, lo, hi)
    q = W.shape[-2]
    alpha = {}
    upstream = None
    if start is not None:
        # only ops that can influence the start edge get alphas
        upstream = set()
        pending = {start}
        for name in reversed(net.order):
            if name in pending:
                upstream.add(name)
                pending.update(net.ops[name].inputs)
    for name in net.order:
        op = net.ops[name]
        if op.kind != 'nonlin':
            continue
        if upstream is not None and name not in upstream:
            continue
        if op.fn == 'relu':
            l, h = inter[name]
            al0 = REL['relu'].planes(l, h)[0]           # adaptive default
            alpha[name] = al0.detach().clone().unsqueeze(1) \
                .expand(B, q, l.shape[1]).contiguous().requires_grad_(True)
        elif hasattr(REL[op.fn], 'alpha_planes'):
            l, h = inter[name]
            crossing = ((l < 0) & (h > 0)).to(l.dtype)
            t0 = (0.5 * (1 - crossing)).unsqueeze(1).unsqueeze(1) \
                .expand(B, q, 2, l.shape[1]).contiguous()
            alpha[name] = t0.requires_grad_(True)
    if not alpha:
        lb = crown(net, lo, hi, W, inter, start=start)
        return (lb, {}) if return_alpha else lb
    opt = torch.optim.Adam(list(alpha.values()), lr=lr)
    best = None
    thr = (torch.zeros(q, device=lo.device, dtype=lo.dtype)
           if thresholds is None else thresholds)
    for _ in range(max(1, iters)):
        if budget is not None and budget.over():
            break
        lb = crown(net, lo, hi, W, inter, alpha, start=start)
        best = lb.detach() if best is None \
            else torch.maximum(best, lb.detach())
        loss = -(torch.minimum(lb, thr.unsqueeze(0) + 1.0)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for t in alpha.values():
                t.clamp_(0.0, 1.0)
    lb = crown(net, lo, hi, W, inter, alpha, start=start)
    # best stays None if the budget was already exhausted on entry (the loop
    # broke on iter 0 before setting it); the fresh lb above is then the bound.
    best = lb.detach() if best is None else torch.maximum(best, lb.detach())
    if return_alpha:
        return best, {k: v.detach() for k, v in alpha.items()}
    return best
