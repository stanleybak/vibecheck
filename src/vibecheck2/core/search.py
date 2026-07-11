"""The BaB search: one batched frontier over one domain type (design 2.3).

Milestone M4a: input-dimension splitting. A domain is an input subbox plus
its per-disjunct openness; the whole frontier lives in flat tensors, a
memory-budgeted batch of the worst domains is bounded per iteration
(forward intermediates + backward CROWN, batched over domains), verified
domains are dropped, the rest split on the best-scoring action. ReLU/
nonlinear clamp actions join the same ranking in M4b, sharing this loop.

Scoring: the action's estimated lb improvement from the SAME backward pass
that produced the bound: input dim k scores |A_in[:, k]| * width_k / 2
(Smart-Branching), giving the unified currency later action types share.

Falsification interleaves: the attack engine runs on the worst domain's
subbox every few rounds with the frontier's worst point as a seed; any
validated hit ends the search with 'sat'.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import attack, backward, debug, memory
from .bab import (Domain as _Dom, disjunct_selector, materialize_clamps,
                  refuted_matrix)


def _beta_tighten(lbq, W, bias, zo, WG, rec, batch_doms, inter, dev,
                  iters=8, lr=0.05):
    """beta-CROWN's split Lagrangian ON THE ZONOTOPE margin form (v1
    attn_crown's own analysis: without beta, a value split only tightens
    local planes and the concretization still ranges over the full box --
    'deep splits stagnate at the linearized-network bound', which is
    EXACTLY the flat -0.04 curve measured on vit 2157; v1 credits betas
    with 30-100x node collapse vs clamp-only splits).

    Each split neuron's pre-activation is itself affine in the shared
    symbols (z = z_c + z_G . e, from the zono record hook). On the
    subdomain the constraint value is nonnegative, so for beta >= 0:
        min_sub m  >=  min_full [m - beta * constraint],
    and the right side concretizes in closed form over e in [-1, 1]:
        [W y_c + b - sum beta (z_c - off)] - |W y_G - sum beta z_G|_1.
    Optimized per domain with Adam; every iterate is sound (best-of).
    Returns lbq elementwise-maxed with the tightened bounds."""
    B, q, g = WG.shape
    out = lbq.clone()
    for bi, dom in enumerate(batch_doms):
        splits = dom.splits
        if not splits:
            continue
        zc_rows, zg_rows, zr_rows, off = [], [], [], []
        for nm, j, spec_ in splits:
            r = rec.get(nm)
            if r is None:
                continue
            gp = r['G_pre'].shape[2]
            zg = torch.zeros(g, device=dev)
            zg[:gp] = r['G_pre'][bi, j]
            zc = float(r['c_pre'][bi, j])
            # box-remainder noise of the pre-activation is not in zg;
            # its worst case charges |beta| * rad per constraint
            zr = float(r['rad'][bi, j]) if 'rad' in r else 0.0
            if isinstance(spec_, tuple):     # range split: two constraints
                lo_c = (spec_[0] if spec_[0] > -1e30
                        else float(inter[nm][0][bi, j]))
                hi_c = (spec_[1] if spec_[1] < 1e30
                        else float(inter[nm][1][bi, j]))
                zc_rows += [zc, -zc]
                zg_rows += [zg, -zg]
                zr_rows += [zr, zr]
                off += [lo_c, -hi_c]         # z - lo >= 0 ; (-z) - (-hi) >= 0
            else:                            # sign split: sign * z >= 0
                zc_rows += [spec_ * zc]
                zg_rows += [spec_ * zg]
                zr_rows += [zr]
                off += [0.0]
        if not zc_rows:
            continue
        zC = torch.tensor(zc_rows, device=dev)           # (S,)
        zG = torch.stack(zg_rows)                        # (S, g)
        zR = torch.tensor(zr_rows, device=dev)           # (S,)
        offs = torch.tensor(off, device=dev)             # (S,)
        beta = torch.zeros(q, zC.shape[0], device=dev, requires_grad=True)
        opt = torch.optim.Adam([beta], lr=lr)
        base_c = torch.matmul(zo.c[bi], W.T) + bias      # (q,)
        if zo.rad is not None:
            base_c = base_c - torch.matmul(W.abs(), zo.rad[bi])
        best = out[bi] + bias
        for _ in range(iters):
            b_ = beta.clamp_min(0.0)
            cterm = base_c - (b_ * (zC - offs).unsqueeze(0)).sum(1) \
                - (b_ * zR.unsqueeze(0)).sum(1)
            Gterm = WG[bi] - torch.einsum('qs,sg->qg', b_, zG)
            lb = cterm - Gterm.abs().sum(1)
            best = torch.maximum(best, lb.detach())
            loss = -(torch.minimum(lb, torch.zeros_like(lb) + 1.0)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        out[bi] = best - bias
    return out


def stabilize_intermediates(net, W, lo1, hi1, inter, budget, device='cpu',
                            passes=3, alpha_iters=8, max_branch=512,
                            log=lambda m: None):
    """Split-and-tighten stabilization (v1 bab_refine, measured on the box:
    it closes the relusplitter model_2_2 hard instance in 33s where frontier
    BaB explodes). No domain frontier: sweep the relu layers in topo order;
    for each layer, split EVERY still-unstable neuron (one +/- clamp pair
    per neuron, batched), bound all intermediates under the clamps with
    alpha-CROWN refinement, and take per-neuron branch ENVELOPES (min lo /
    max hi over the two branches -- the input domain is covered by
    {z_j<=0} u {z_j>=0}, so the envelope is a valid parent bound);
    envelopes from different neurons are intersected (each is
    independently a parent bound). Tightened layers stabilize downstream
    neurons, so later sweeps shrink further (v1's trace: layer-3 unstable
    79 -> 60 -> 41 over three passes); a 48-neuron global top-k variant
    measured unstable 222 -> 221 in 4 rounds -- splitting the WHOLE layer
    is the lever, not scoring. The caller reruns root alpha-CROWN after.

    Returns the tightened `inter` (same layout as backward.intermediates).
    """
    dev = torch.device(device)
    relu_edges = [nm for nm in net.order
                  if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']
    for ps in range(passes):
        prev = sum(int(((inter[nm][0] < 0) & (inter[nm][1] > 0)).sum())
                   for nm in relu_edges)
        for nm in relu_edges:
            if budget.over():
                return inter
            l, h = inter[nm][0], inter[nm][1]
            js = torch.nonzero((l[0] < 0) & (h[0] > 0),
                               as_tuple=False).flatten()
            if not js.numel():
                continue
            js = js[:max_branch // 2]
            B = 2 * js.numel()
            cl = torch.zeros(B, net.ops[nm].n, device=dev, dtype=torch.int8)
            ar = torch.arange(js.numel(), device=dev)
            cl[2 * ar, js] = 1
            cl[2 * ar + 1, js] = -1
            blo, bhi = lo1.expand(B, -1), hi1.expand(B, -1)
            base = {k2: tuple(t.expand(B, -1) for t in v)
                    for k2, v in inter.items()}
            ib = backward.intermediates_crown(net, blo, bhi,
                                              base_inter=base,
                                              clamps={nm: cl},
                                              alpha_iters=alpha_iters,
                                              budget=budget)
            new_inter = {}
            for k2, v in inter.items():
                merged = []
                for j2 in range(0, len(v), 2):
                    bl, bh = ib[k2][j2], ib[k2][j2 + 1]
                    env_l = torch.minimum(bl[0::2], bl[1::2])
                    env_h = torch.maximum(bh[0::2], bh[1::2])
                    nl = torch.maximum(
                        v[j2], env_l.max(dim=0, keepdim=True).values)
                    nh = torch.minimum(
                        v[j2 + 1], env_h.min(dim=0, keepdim=True).values)
                    merged.append(nl)
                    merged.append(torch.maximum(nh, nl))
                new_inter[k2] = tuple(merged)
            inter = new_inter
        n_unst = sum(int(((inter[nm][0] < 0) & (inter[nm][1] > 0)).sum())
                     for nm in relu_edges)
        log(f'[vc2/stab] pass={ps + 1} unstable={n_unst}')
        if n_unst >= prev:                      # converged: passes stop paying
            break
    return inter


def _requeue(f_lo, f_hi, f_worst, f_row, batch, blo, bhi, brow):
    """Push an OOM'd batch back onto the host frontier and halve the
    batch size (the sanctioned round-level OOM recovery)."""
    torch.cuda.empty_cache()
    f_lo = torch.cat([f_lo, blo.cpu()])
    f_hi = torch.cat([f_hi, bhi.cpu()])
    f_worst = torch.cat([f_worst,
                         torch.full((blo.shape[0],), -torch.inf)])
    if brow is not None:
        f_row = torch.cat([f_row, brow.cpu()])
    return f_lo, f_hi, f_worst, f_row, max(64,
                                           min(batch, blo.shape[0]) // 2)


def input_split_bab(net, spec, W, bias, disj_idx, lo, hi, deadline,
                    device='cpu', batch=4096, split_dims=2, alpha_iters=8,
                    onnx_path=None, attack_every=8, heuristic=None,
                    roots=None, row_groups=None, mini_group=None,
                    full_refine=None, log=lambda m: None):
    """Returns (verdict, info): 'unsat' | 'sat' (+witness) | 'timeout'.

    W (q, n_out), bias (q,), disj_idx (q,): the spec query rows.
    lo, hi: (n_in,) root box. Each open domain splits its top `split_dims`
    scoring dims simultaneously (2^k children); domains whose plain-CROWN
    bound lands near zero get a short per-batch alpha pass before splitting.

    roots=(roots_lo (B0, n), roots_hi (B0, n), root_row (B0,)): multi-sub
    mode (v1 multi-sub BaB; nn4sys mega-disjunct). Every root box is its
    own single-row sub-instance -- the domain is refuted when ITS row's lb
    goes positive -- and they all share one frontier, so the batched bound
    amortizes across the 960 subs instead of a 0.2s-per-group serial loop.

    row_groups (G, r): generalizes root_row from one W row to a row GROUP
    per root (root_row then holds group ids). A group is one disjunct's
    conjunctive rows: the domain is refuted when ANY of them goes positive,
    and the clip intersects ALL of their halfspaces (abcrown's or-clause
    decomposition -- lsnc's 13-disjunct spec explodes as one joint frontier
    but drains per-disjunct). Ragged groups are padded by repeating a row
    (idempotent for refutation and clipping). Single-row multi-sub is the
    r=1 case and keeps its dedicated gathers.
    """
    dev = torch.device(device)
    dt = torch.float32
    W = W.to(dev, dt)
    bias = bias.to(dev, dt)
    if heuristic is None:
        # |A|-sensitivity scoring is informative through relu adjoints but
        # actively misleading through smooth bands (dist_shift index112:
        # widest closes in 53 splits where sb dies at 450k domains; v1
        # ships sb disabled for exactly that class and enabled for the
        # relu families)
        n_band = sum(op.n for op in net.ops.values()
                     if op.kind == 'nonlin'
                     and op.fn not in ('relu', 'leaky_relu'))
        n_relu = sum(op.n for op in net.ops.values()
                     if op.kind == 'nonlin'
                     and op.fn in ('relu', 'leaky_relu'))
        heuristic = 'widest' if n_band > n_relu else 'sb'
    _row_map = weight_of = None
    if roots is not None and row_groups is None:
        # dedupe by WEIGHT only for the CROWN cost. Cardinality specs (lindex:
        # 120k rows) carry ~q DISTINCT thresholds sharing ONE weight, so a
        # (w, b)-dedup left q identical-weight columns and CROWN ran O(B x q).
        # Keep the per-ROW bias per-domain and add it after a gather: f_row
        # indexes the original rows, weight_of maps a row to its unique weight.
        uniqW, weight_of = torch.unique(W, dim=0, return_inverse=True)
        W = uniqW.contiguous()
        weight_of = weight_of.to(dev)
    if row_groups is not None:
        # group mode bounds each domain's OWN rows via a batched per-domain
        # W (B, r, n) gather instead of the global deduped matrix (lsnc:
        # r=3 of 15 unique rows -- 5x less CROWN adjoint per box)
        assert roots is not None, 'row_groups requires roots mode'
        row_groups = row_groups.to(dev)
    q = W.shape[0] if row_groups is None else int(row_groups.shape[1])
    if roots is not None:
        D, sel = 0, None       # per-domain rows replace the disjunct map
    else:
        D, sel = disjunct_selector(disj_idx, q, dev)

    # the frontier lives on HOST: a stuck instance grows it to millions of
    # (n_in,) rows (dist_shift index112 hit 250k x 792 and OOM-crashed the
    # GPU mid-bookkeeping); only the popped batch goes to the device
    pend_lo = pend_hi = pend_row = None
    if roots is not None:
        R_lo = roots[0].to('cpu', dt)
        R_hi = roots[1].to('cpu', dt)
        R_row = roots[2].cpu()          # original row idx (bias is per-row)
        # MINI-GROUP admission: keep only `mini_group` subboxes active on the
        # frontier at once, holding the rest in a pending pool and admitting
        # the next wave as the active set closes (see _admit below). One shared
        # frontier over all N roots splits every open sub each round and
        # explodes (mscn: 145k leaves); this caps the peak while paying the
        # weight-dedup/setup cost ONCE. mini_group=None -> all roots at once.
        if mini_group and R_lo.shape[0] > mini_group:
            f_lo, pend_lo = R_lo[:mini_group], R_lo[mini_group:]
            f_hi, pend_hi = R_hi[:mini_group], R_hi[mini_group:]
            f_row, pend_row = R_row[:mini_group], R_row[mini_group:]
        else:
            f_lo, f_hi, f_row = R_lo, R_hi, R_row
        f_worst = torch.full((f_lo.shape[0],), -torch.inf)
        log(f'[vc2/bab] multi-sub: {R_lo.shape[0]} roots over '
            f'{W.shape[0]} unique rows'
            + (f', mini-group {mini_group}' if pend_lo is not None else ''))
    else:
        f_lo = lo.reshape(1, -1).to('cpu', dt)
        f_hi = hi.reshape(1, -1).to('cpu', dt)
        f_row = torch.zeros(1, dtype=torch.long)
        f_worst = torch.full((1,), -torch.inf)
    n_bounded = n_split = rounds = 0
    tol_witness = None
    _round_wall, _round_B = 1.0, 1
    # box-remainder mode: flips permanently after the first dense-zono
    # OOM, and starts ON when the dense estimate cannot fit at even a
    # modest batch (mscn_2048d: the dense attempt + OOM burned 7.8s of a
    # 20s budget before the switch; the estimate only ever SKIPS a doomed
    # attempt, rad-mode bounds stay sound)
    z_rad_mode = (backward._zono_cost_bytes(net, 16)
                  > memory.free_bytes(dev) * memory.SAFETY)
    if z_rad_mode:
        log('[vc2/bab] dense zono projected over budget at B=16; '
            'starting in box-remainder mode')
    t0 = time.time()
    n_nonlin = sum(net.ops[nm].n for nm in net.order
                   if net.ops[nm].kind == 'nonlin')
    # tiny nets (acasxu class): full per-batch identity-CROWN refinement.
    # bigger ones: joint-alpha refine ONCE at the root, then per batch only
    # a cheap reforward intersected with the root bounds (subbox is a
    # subset, so the intersection is sound) -- the abcrown cost model.
    forced_refine = full_refine is not None
    if not forced_refine:
        full_refine = n_nonlin <= 1200
    same_box = True
    if roots is not None:
        same_box = (bool((R_lo == R_lo[0]).all())
                    and bool((R_hi == R_hi[0]).all()))
        if not same_box and not forced_refine:
            full_refine = False          # SCATTERED per-sub boxes (mscn):
                                         # root_ref on the UNION box, cheap
                                         # reforward per batch (still sound).
                                         # Same-box roots (disjunct
                                         # decomposition) keep the probe: the
                                         # forced cheap bound dropped lsnc's
                                         # per-box closure 59% -> 29%.
    if full_refine and not forced_refine:
        # identity-CROWN vs forward-zono planes flips by net class (acasxu
        # amplifying weights favor backward refinement; dist_shift sigmoid
        # bands close at depth 1 under zono but need 100k+ splits under
        # identity-CROWN), so measure both on the root's first two
        # children and keep the winner.
        # probe on the (shared) root box -- f_lo may hold several identical
        # rows in decomposed-roots mode, and argmax over all of them would
        # return a flattened index
        wax = int((f_hi[0] - f_lo[0]).argmax())
        mid = (f_lo[0, wax] + f_hi[0, wax]) / 2
        plo = f_lo[:1].repeat(2, 1).to(dev)
        phi = f_hi[:1].repeat(2, 1).to(dev)
        phi[0, wax] = mid
        plo[1, wax] = mid
        try:
            # deduped-W mode (roots): bias is per ORIGINAL row and no longer
            # aligns with W's columns; the probe only compares the two
            # bounds' tightness, so compare bias-free there
            pb = bias if bias.shape[0] == W.shape[0] else 0.0
            _l, _h, zst = backward.fwd.zono(net, plo, phi, return_state=True)
            zi = backward._inter_from_state(net, lambda e: zst[e].bounds())
            # the zono route's real bound is max(crown-over-zono-seeds,
            # direct spec concretization through the state) -- see the
            # in-loop zout max
            zo_p = zst[net.output_name]
            W3p = W.unsqueeze(0).expand(2, -1, -1)
            zsp = torch.bmm(W3p, zo_p.c.unsqueeze(2)).squeeze(2) \
                - torch.bmm(W3p, zo_p.G).abs().sum(dim=2)
            if zo_p.rad is not None:
                zsp = zsp - torch.bmm(W3p.abs(),
                                      zo_p.rad.unsqueeze(2)).squeeze(2)
            z_lb = float((torch.maximum(
                backward.crown(net, plo, phi, W, zi), zsp) + pb).min())
            ci = backward.intermediates_crown(net, plo, phi, base_inter=zi)
            c_lb = float((backward.crown(net, plo, phi, W, ci) + pb).min())
            full_refine = c_lb >= z_lb
            log(f'[vc2/bab] refine probe: crown={c_lb:.3f} zono={z_lb:.3f} '
                f'-> full_refine={full_refine}')
        except (NotImplementedError, torch.cuda.OutOfMemoryError) as _pe:
            backward._warn_once(f'refine probe zono unavailable '
                                f'({type(_pe).__name__}); keeping crown')
    root_ref = None
    if not full_refine and same_box:
        # multi-sub with SCATTERED roots skips this (the union of 960
        # subboxes is nearly the whole space, so refining it is GBs of
        # pure cost; the per-subbox reforward below is the tight part
        # there) -- but when every root shares ONE box (disjunct
        # decomposition) the root refinement is as valid as at roots=None.
        root_ref = backward.intermediates_crown(
            net, f_lo[:1].to(dev), f_hi[:1].to(dev),
            alpha_iters=(12 if n_nonlin <= 20000 else 0))

    def _grp_vals(lbq, brow):
        """(B, r): each domain's own rows' bounds. In group mode lbq's r
        columns ARE the domain's rows (per-domain W); add their biases."""
        return lbq + bias[row_groups[brow]]

    def dom_bound(lbq, brow):
        """(B,) each domain's OWN row bound = its weight column + its bias
        (multi-sub, weight-deduped W). Group mode: the group's best FINITE
        row (any positive row refutes, so max is the closure margin)."""
        if row_groups is not None:
            vals = _grp_vals(lbq, brow)
            return vals.masked_fill(~torch.isfinite(vals),
                                    -torch.inf).max(dim=1).values
        wrow = weight_of[brow]
        return lbq.gather(1, wrow.unsqueeze(1)).squeeze(1) + bias[brow]

    def domain_refuted(lbq, brow=None):
        """(B, q) query lbs -> (B, D) refutation matrix; in multi-sub
        mode a (B, 1) matrix from each domain's OWN row (group mode: ANY
        of its rows). Only FINITE positive bounds refute (+inf is an
        artifact, never a proof)."""
        if brow is not None:
            if row_groups is not None:
                vals = _grp_vals(lbq, brow)
                return ((vals > 0) & torch.isfinite(vals)).any(
                    dim=1, keepdim=True)
            db = dom_bound(lbq, brow)
            return ((db > 0) & torch.isfinite(db)).unsqueeze(1)
        return refuted_matrix(lbq, bias, sel)

    def clip_domains(olo, ohi, A, lbb, rows):
        """ABC-style input clipping: every row r in `rows` must satisfy its
        linear lower form L_r(x) = A_r.x + d_r <= 0 for a counterexample to
        exist in the domain, so tighten the box to those halfspaces per dim.
        A (B, q, n); lbb (B, q) = lbq + bias = min of L_r over the box."""
        for r in rows:
            Ar = A[:, r]                                    # (B, n)
            mn = Ar.clamp(max=0) * ohi + Ar.clamp(min=0) * olo
            d = lbb[:, r].unsqueeze(1) - mn.sum(dim=1, keepdim=True)
            slack = -d - (mn.sum(dim=1, keepdim=True) - mn)  # max A_k x_k
            new_hi = torch.minimum(ohi, slack / Ar.clamp_min(1e-30))
            new_lo = torch.maximum(olo, slack / Ar.clamp_max(-1e-30))
            ohi = torch.where(Ar > 0, new_hi, ohi)
            olo = torch.where(Ar < 0, new_lo, olo)
        return olo, ohi

    while f_lo.shape[0] or (pend_lo is not None and pend_lo.shape[0]):
        if time.time() > deadline:
            return 'timeout', {'frontier': int(f_lo.shape[0]),
                               'pending': int(pend_lo.shape[0])
                               if pend_lo is not None else 0,
                               'bounded': n_bounded, 'splits': n_split,
                               'tol_witness': tol_witness}
        # admit the next wave once the active set has drained below the group
        # size, so the peak frontier stays ~mini_group (the descendants of the
        # active subs) instead of the whole root set. Refill to the group size.
        if (pend_lo is not None and pend_lo.shape[0]
                and f_lo.shape[0] < mini_group):
            na = min(mini_group - f_lo.shape[0], pend_lo.shape[0])
            f_lo = torch.cat([f_lo, pend_lo[:na]])
            f_hi = torch.cat([f_hi, pend_hi[:na]])
            f_row = torch.cat([f_row, pend_row[:na]])
            f_worst = torch.cat([f_worst,
                                 torch.full((na,), -torch.inf)])
            pend_lo, pend_hi, pend_row = (pend_lo[na:], pend_hi[na:],
                                          pend_row[na:])
        rounds += 1
        # adaptive batch growth: when leaves are cheap (microseconds) and
        # the frontier dwarfs the batch, the loop is launch-overhead-bound
        # and doubling the batch is free throughput (lsnc: v1 ships
        # batch_size 65536 for exactly this; 350k leaves/s at 64k vs 7k/s
        # at 4096). Expensive-leaf nets (iso: ~40us/leaf, where bigger
        # batches dilute the boundary-alpha cap) never trigger.
        if (rounds > 4 and batch < 65536 and f_lo.shape[0] > 4 * batch
                and _round_wall / max(_round_B, 1) < 2e-5):
            batch *= 2
        _round_t0 = time.time()
        zout = None            # this batch's forward-zono OUTPUT state
        # pick the worst-first batch, sized by the memory budget: the
        # reforward holds (lo, hi) for EVERY edge (sum, not max -- mscn's
        # 116 edges held 3GB at the max-based estimate), the crown adjoint
        # scales with q x widest
        total_n = sum(net.ops[o].n for o in net.order)
        per_dom = (q * max(net.ops[o].n for o in net.order) * 8
                   + total_n * 4) * 4
        bs = min(batch, memory.chunk_size(f_lo.shape[0], per_dom, dev))
        # worst-first pop via topk (a full argsort over a multi-million
        # frontier cost ~12s of lsnc's 39s run)
        take = torch.topk(f_worst, bs, largest=False, sorted=False).indices
        keep = torch.ones(f_worst.shape[0], dtype=torch.bool)
        keep[take] = False
        blo, bhi = f_lo[take].to(dev), f_hi[take].to(dev)
        brow = f_row[take].to(dev) if roots is not None else None
        f_lo, f_hi, f_worst = f_lo[keep], f_hi[keep], f_worst[keep]
        f_row = f_row[keep]

        try:
            if full_refine:
                # zono-SEEDED identity-CROWN refinement: seeding from forward
                # zono instead of interval reproduces v1's per-leaf bounds AND
                # its input linearization A exactly (measured 72/72 split-dim
                # agreement on iso leaf boxes vs 62/72 interval-seeded; the
                # equally-tight-but-different A compounded into a 2.3x tree)
                try:
                    _l, _h, zst = backward.fwd.zono(net, blo, bhi,
                                                    return_state=True)
                    zi = backward._inter_from_state(
                        net, lambda e: zst[e].bounds())
                    zout = zst[net.output_name]
                    del zst
                    inter = backward.intermediates_crown(net, blo, bhi,
                                                         base_inter=zi)
                except (NotImplementedError, torch.cuda.OutOfMemoryError):
                    inter = backward.intermediates_crown(net, blo, bhi)
            else:
                B = blo.shape[0]
                ib = None
                zono_oom = False
                try:
                    # true per-subbox planes from the wide-dims zonotope.
                    # After a dense OOM the run switches PERMANENTLY to
                    # the box-remainder form: mul/reciprocal collapse
                    # into the rad vector instead of diagonal columns,
                    # so the generator count stays ~n_wide (mscn_2048d:
                    # the dense state needs 130+ GiB and OOMed EVERY
                    # batch into the slow chunked-CROWN degrade; official
                    # budget there is 20s)
                    _l, _h, zst = backward.fwd.zono(
                        net, blo, bhi, return_state=True,
                        box_remainder='all' if z_rad_mode else False)
                    ib = backward._inter_from_state(
                        net, lambda e: zst[e].bounds())
                    zout = zst[net.output_name]
                    del zst
                except torch.cuda.OutOfMemoryError:
                    if not z_rad_mode:
                        z_rad_mode = True
                        log('[vc2/bab] dense zono OOM; switching to '
                            'box-remainder zono for this run')
                        if dev.type == 'cuda':
                            torch.cuda.empty_cache()
                        try:
                            _l, _h, zst = backward.fwd.zono(
                                net, blo, bhi, return_state=True,
                                box_remainder='all')
                            ib = backward._inter_from_state(
                                net, lambda e: zst[e].bounds())
                            zout = zst[net.output_name]
                            del zst
                        except (torch.cuda.OutOfMemoryError,
                                NotImplementedError):
                            zono_oom = True
                    else:
                        zono_oom = True
                    # empty_cache OUTSIDE this handler's live traceback would
                    # still pin the zono partial state; do it here and let the
                    # retry below run cache-clean
                    if dev.type == 'cuda':
                        torch.cuda.empty_cache()
                except NotImplementedError as _ze:
                    backward._warn_once(
                        f'input-split per-batch zono unavailable '
                        f'({type(_ze).__name__}: {str(_ze)[:60]}); '
                        f'interval reforward only')
                if ib is None and zono_oom and roots is None:
                    # the BATCH does not fit but per-leaf zono may (vit:
                    # ~5 GiB/leaf at 5671 generators; interval degrade here
                    # made the frontier explode to 16k boxes where v1's
                    # SERIAL zono-input-split closes in 21). Re-run in
                    # halving chunks via the sanctioned memory helper,
                    # keeping only each chunk's per-edge bounds -- the
                    # generator matrices free chunk-wise. Seed the chunk at
                    # half the batch: the single shot just proved B over.
                    outs = {}

                    def _zchunk(idx_c):
                        _zl, _zh, zc = backward.fwd.zono(
                            net, blo[idx_c], bhi[idx_c], return_state=True)
                        ibc = backward._inter_from_state(
                            net, lambda e: zc[e].bounds())
                        for k2, v in ibc.items():
                            outs.setdefault(k2, []).append(
                                tuple(t.detach() for t in v))

                    try:
                        memory.chunked_indices(
                            _zchunk, torch.arange(B, device=dev),
                            bytes_per_item=(memory.free_bytes(dev)
                                            * memory.SAFETY
                                            / max(1, B // 2)))
                        ib = {k2: tuple(
                            torch.cat([o[j] for o in v], dim=0)
                            for j in range(len(v[0])))
                            for k2, v in outs.items()}
                    except torch.cuda.OutOfMemoryError:
                        # even one leaf's zono does not fit: the halving
                        # floor re-raised; fall through to the degrades
                        torch.cuda.empty_cache()
                        backward._warn_once(
                            'per-leaf zono does not fit even at chunk=1; '
                            'interval reforward only')
                if ib is not None:
                    pass                        # zono bounds stand
                elif zono_oom and roots is not None:
                    # multi-sub: the zono generator count EXPLODES on mul/
                    # reciprocal nets (mscn_2048d: 130+ GiB, OOMs at every
                    # batch), yet interval reforward is far too loose to close
                    # cardinality subboxes (frontier -> 100k+). Memory-bounded
                    # backward CROWN (chunked identity queries) is tighter than
                    # interval and fits at narrow width (mscn_128d). If CROWN
                    # itself OOMs, the outer handler halves the batch. (At 2048d
                    # its per-mul-factor refinement is O(width^2) and too slow;
                    # that family needs a forward-LiRPA propagator, see notes.)
                    ib = backward.intermediates_crown(net, blo, bhi)
                else:
                    if dev.type == 'cuda':
                        torch.cuda.empty_cache()
                    ib_state = backward.fwd.interval(net, blo, bhi,
                                                     return_state=True)
                    ib = backward._inter_from_state(net,
                                                    lambda e: ib_state[e])
                if root_ref is None:
                    inter = ib
                else:
                    inter = {}
                    for k2, v in root_ref.items():
                        rv = tuple(t.expand(B, -1) for t in v)
                        iv = ib[k2]
                        merged = []
                        for j2 in range(0, len(rv), 2):
                            merged.append(torch.maximum(rv[j2], iv[j2]))
                            merged.append(torch.minimum(rv[j2 + 1],
                                                        torch.maximum(iv[j2 + 1],
                                                                      merged[-1])))
                        inter[k2] = tuple(merged)
            Wq = W if row_groups is None else W[row_groups[brow]]
            lbq, Ain = backward.crown(net, blo, bhi, Wq, inter,
                                      return_input_adjoint=True)
        except torch.cuda.OutOfMemoryError:
            # v1's round-level pattern: push the popped batch back, halve
            # the batch, retry (one guard covers every stage of the bound;
            # per-site chunking got mscn from 1-2 to 366 domains/round,
            # this catches whatever stage trips next)
            if batch <= 64:
                return 'timeout', {'frontier': int(f_lo.shape[0]),
                                   'bounded': n_bounded,
                                   'splits': n_split,
                                   'reason': 'oom_floor'}
            f_lo, f_hi, f_worst, f_row, batch = _requeue(
                f_lo, f_hi, f_worst, f_row, batch, blo, bhi, brow)
            continue
        _round_wall, _round_B = time.time() - _round_t0, blo.shape[0]
        if debug.enabled() and rounds == 1:
            debug.add('bab_root_lb', lbq[0] + bias)
            debug.add('bab_root_A', Ain[0])
        if debug.enabled() and rounds <= 64:
            debug.add_seq('bab_queue', int(f_lo.shape[0]))
        # the clip bias reconstruction below must use THIS lbq -- it is the
        # exact concretization of Ain; the alpha-improved bound has no
        # matching linearization and an inflated bias would over-clip
        # (cutting real counterexamples: unsound)
        lbq_lin = lbq.clone()
        if zout is not None:
            # forward-zono spec concretization: w.c - |w.G|.1 - |w|.rad is
            # a sound lower bound that KEEPS the state's cross-factor
            # correlations end-to-end; extracting per-edge boxes for the
            # backward crown severs them at every mul (lsnc quadrotor2d:
            # the correlated product state was invisible to the crown).
            # lbq_lin above stays the PLAIN crown concretization -- the
            # clip's (A, c) reconstruction must match Ain exactly.
            Wq3 = Wq if Wq.dim() == 3 \
                else Wq.unsqueeze(0).expand(blo.shape[0], -1, -1)
            zm = torch.bmm(Wq3, zout.c.unsqueeze(2)).squeeze(2) \
                - torch.bmm(Wq3, zout.G).abs().sum(dim=2)
            if zout.rad is not None:
                zm = zm - torch.bmm(Wq3.abs(),
                                    zout.rad.unsqueeze(2)).squeeze(2)
            lbq = torch.maximum(lbq, zm)
            zout = None
        n_bounded += blo.shape[0]
        refuted = domain_refuted(lbq, brow)
        open_mask = ~refuted.all(dim=1)
        if os.environ.get('VC2_DEBUG_CLIP') and rounds <= 60:
            _alloc = (torch.cuda.memory_allocated(dev) / 1e9
                      if dev.type == 'cuda' else 0)
            log(f'[vc2/round] round={rounds} popped={blo.shape[0]} '
                f'closed_crown={int(refuted.all(dim=1).sum())} '
                f'queue={f_lo.shape[0]} gpu={_alloc:.2f}GB')
        try:
            if open_mask.any() and alpha_iters > 0:
                # alpha on BOUNDARY domains only (v1 boundary_eps=10 with a
                # 2048 cap): a domain whose worst row is far below zero will
                # split anyway, and alpha on the whole batch halves the loop
                # throughput (iso: 33k -> 75k leaves/s)
                if brow is not None:
                    margin = dom_bound(lbq, brow)
                else:
                    margin = (lbq + bias).max(dim=1).values
                bmask = open_mask & (margin > -10.0)
                oi = torch.nonzero(bmask, as_tuple=False).flatten()
                if oi.numel() > 2048:
                    oi = oi[margin[oi].argsort(descending=True)[:2048]]
            if open_mask.any() and alpha_iters > 0 and oi.numel():
                inter_o = {k2: tuple(t[oi] for t in v)
                           for k2, v in inter.items()}
                if row_groups is not None:
                    # per-domain rows: no shared columns to threshold; let
                    # alpha optimize every row fully (sound, no early-stop)
                    thr = None
                elif brow is not None:
                    # W is weight-deduped (q_W cols); the alpha loss threshold
                    # must be per-COLUMN, not per original row. Target each
                    # column's HARDEST domain (max of -bias over rows mapping
                    # to it) so alpha keeps optimizing until every sharing
                    # sub-row could cross zero.
                    thr = torch.full((W.shape[0],), -float('inf'), device=dev)
                    thr.scatter_reduce_(0, weight_of, -bias, reduce='amax',
                                        include_self=True)
                else:
                    thr = -bias
                Wq_o = Wq[oi] if row_groups is not None else W
                lb_a, al = backward.alpha_crown(net, blo[oi], bhi[oi], Wq_o,
                                                inter_o, iters=alpha_iters,
                                                thresholds=thr,
                                                return_alpha=True)
                # one extra pass with the final alphas yields a linearization
                # whose exact concretization is its own bound -- a SOUND
                # (A, b) upgrade for the clip below (the plain-CROWN pair is
                # much weaker; pairing alpha bounds with the plain-CROWN A
                # would over-clip and was the unsound variant)
                lb_al, Ain_a = backward.crown(net, blo[oi], bhi[oi], Wq_o,
                                              inter_o, al,
                                              return_input_adjoint=True)
                # adopt the alpha linearization ONLY where it beats the plain
                # one, rowwise (each row's (A_r, b_r) is an independent valid
                # pair). The final alpha iterate can be WORSE than the plain
                # slopes (dist_shift: zono-informed chords beat 8-iter alpha);
                # overwriting unconditionally weakened the clip there and blew
                # the frontier from 1 split to an OOM.
                better = lb_al > lbq_lin[oi]
                lbq_lin[oi] = torch.where(better, lb_al, lbq_lin[oi])
                Ain[oi] = torch.where(better.unsqueeze(-1), Ain_a, Ain[oi])
                lbq[oi] = torch.maximum(lbq[oi],
                                        torch.maximum(lb_a, lb_al))
                refuted = domain_refuted(lbq, brow)
                if os.environ.get('VC2_DEBUG_CLIP') and rounds % 16 == 1:
                    log(f'[vc2/alpha] round={rounds} open_pre={oi.numel()} '
                        f'flipped={int(oi.numel() - (~refuted.all(dim=1)).sum())} '
                        f'gain={float((lb_a - lbq[oi]).abs().max()):.4f}')
                open_mask = ~refuted.all(dim=1)
        except torch.cuda.OutOfMemoryError:
            # same round-level recovery for the alpha pass (autograd
            # graphs over the boundary subset)
            if batch <= 64:
                return 'timeout', {'frontier': int(f_lo.shape[0]),
                                   'bounded': n_bounded,
                                   'splits': n_split,
                                   'reason': 'oom_floor'}
            f_lo, f_hi, f_worst, f_row, batch = _requeue(
                f_lo, f_hi, f_worst, f_row, batch, blo, bhi, brow)
            continue
        if open_mask.any():
            olo, ohi = blo[open_mask], bhi[open_mask]
            oA = Ain[open_mask]
            # ABC clip, per disjunct (v1 / abcrown clip_input_domain): a CE
            # satisfying disjunct d must lie in ALL of d's halfspaces
            # L_r(x) <= 0, so clip the box against each open disjunct's
            # rows; the domain shrinks to the UNION of the per-disjunct
            # boxes and is VERIFIED outright when every polytope is empty
            # (iso instance_3: v1 with this clip 55s / ~2.5k queue, without
            # it timeout -- it is the load-bearing certifier, not the alpha)
            oref = refuted[open_mask]
            if brow is not None:
                # multi-sub: each domain clips against ITS OWN rows'
                # halfspaces A_r.x + c_r <= 0 (a pure conjunction; group
                # mode iterates the disjunct's rows, each tightening the
                # box the next clips further). The affine constant c_r is
                # recovered over the ORIGINAL box the crown ran on (the
                # per-row concretization lbq_lin is a min over THAT box);
                # re-deriving it over a shrunk box would be unsound.
                orow = brow[open_mask]
                og = (row_groups[orow] if row_groups is not None
                      else orow.unsqueeze(1))            # (B, r) W rows
                if row_groups is not None:
                    # per-domain W: lbq_lin's r columns ARE the own rows
                    A_own = oA                           # (B, r, n)
                    d_own = lbq_lin[open_mask] + bias[og]
                else:
                    wrow_o = weight_of[og]               # (B, 1)
                    A_own = oA.gather(1, wrow_o.unsqueeze(-1)
                                      .expand(-1, -1, oA.shape[2]))
                    d_own = (lbq_lin[open_mask].gather(1, wrow_o)
                             + bias[og])
                # affine constants recovered over the box the crown ran on
                # (d_own is a min over THAT box; re-deriving one over a
                # shrunk box would be unsound)
                mn0 = (A_own.clamp(min=0) * olo.unsqueeze(1)
                       + A_own.clamp(max=0) * ohi.unsqueeze(1)).sum(-1)
                c_own = d_own - mn0                      # (B, r)
                xl_u, xh_u = olo, ohi
                for j_g in range(og.shape[1]):
                    Ar = A_own[:, j_g]
                    c = c_own[:, j_g].unsqueeze(1)
                    mn = Ar.clamp(max=0) * xh_u + Ar.clamp(min=0) * xl_u
                    base = mn.sum(dim=1, keepdim=True)
                    slack = -c - (base - mn)
                    new_hi = torch.minimum(xh_u, slack / Ar.clamp_min(1e-30))
                    new_lo = torch.maximum(xl_u, slack / Ar.clamp_max(-1e-30))
                    xh_u = torch.where(Ar > 0, new_hi, xh_u)
                    xl_u = torch.where(Ar < 0, new_lo, xl_u)
                feas_any = (xh_u - xl_u).min(dim=1).values >= 0
                # post-clip re-check (abcrown clip_n_verify): the same rows
                # concretized over the SHRUNK box; a positive row closes
                # the domain without a split
                mn1 = (A_own.clamp(min=0) * xl_u.unsqueeze(1)
                       + A_own.clamp(max=0) * xh_u.unsqueeze(1)).sum(-1)
                lb_clip = c_own + mn1
                feas_any &= ~((lb_clip > 0)
                              & torch.isfinite(lb_clip)).any(dim=1)
            else:
                lbb = (lbq_lin + bias)[open_mask]
                xl_u = torch.full_like(olo, torch.inf)
                xh_u = torch.full_like(ohi, -torch.inf)
                feas_any = torch.zeros(olo.shape[0], dtype=torch.bool,
                                       device=dev)
                for dd in range(D):
                    rows = torch.nonzero(sel[dd], as_tuple=False).flatten()
                    clo, chi = clip_domains(olo, ohi, oA, lbb,
                                            rows.tolist())
                    feas_d = (((chi - clo).min(dim=1).values >= 0)
                              & ~oref[:, dd])
                    xl_u = torch.where(feas_d.unsqueeze(1),
                                       torch.minimum(xl_u, clo), xl_u)
                    xh_u = torch.where(feas_d.unsqueeze(1),
                                       torch.maximum(xh_u, chi), xh_u)
                    feas_any |= feas_d
            if os.environ.get('VC2_DEBUG_CLIP') and rounds % 16 == 1:
                w_pre = (ohi - olo).sum()
                w_post = ((torch.where(feas_any.unsqueeze(1), xh_u, ohi)
                           - torch.where(feas_any.unsqueeze(1), xl_u, olo))
                          .clamp(min=0).sum())
                log(f'[vc2/clip] round={rounds} open={olo.shape[0]} '
                    f'clip_closed={int((~feas_any).sum())} '
                    f'shrink={float(w_post / w_pre.clamp(min=1e-30)):.4f} '
                    f'open_disj_mean={float((~oref).float().sum(1).mean()):.2f}')
            olo = torch.where(feas_any.unsqueeze(1), xl_u, olo)
            ohi = torch.where(feas_any.unsqueeze(1), xh_u, ohi)
            olo, ohi, oA = olo[feas_any], ohi[feas_any], oA[feas_any]
            if brow is not None:
                orow = orow[feas_any]
                # per-domain query bound = the domain's OWN row (weight-deduped
                # gather + per-domain bias); lbq columns are unique weights.
                lbq_open = dom_bound(lbq[open_mask][feas_any], orow)
            else:
                lbq_open = (lbq + bias)[open_mask][feas_any]
            if not olo.shape[0]:
                continue
            # Smart-Branching: estimated improvement per input dim; split the
            # top `split_dims` dims simultaneously -> 2^k children.
            # 'widest' ignores sensitivity (v1's dist_shift setting).
            if heuristic == 'widest':
                score = ohi - olo
            else:
                # (group mode: oA's rows are the domain's OWN rows already,
                # so the sum scores exactly the right polytope)
                score = oA.abs().sum(dim=1) * (ohi - olo) / 2
            kdims = min(split_dims, olo.shape[1])
            if row_groups is not None and f_lo.shape[0] < bs // 4:
                # frontier-starved ramp (decomposed mode): 2-way splitting
                # from a handful of roots pays ~25 rounds of fixed per-round
                # cost before a batch fills (lsnc: ~4s of a 19s slice).
                # Splitting 3 extra levels while starved fills the frontier
                # in a few rounds; those shallow boxes would split anyway.
                kdims = min(split_dims + 3, olo.shape[1])
            topk = score.topk(kdims, dim=1).indices          # (B, kdims)
            if debug.enabled() and rounds <= 64:
                debug.add_seq('bab_split_dim', int(topk[0, 0]))
            ch_lo, ch_hi = olo, ohi
            for j in range(kdims):
                reps = ch_lo.shape[0] // olo.shape[0]
                kk = topk[:, j].repeat(reps).unsqueeze(1)
                mid = (ch_lo.gather(1, kk) + ch_hi.gather(1, kk)) / 2
                left_hi = ch_hi.clone()
                left_hi.scatter_(1, kk, mid)
                right_lo = ch_lo.clone()
                right_lo.scatter_(1, kk, mid)
                ch_lo = torch.cat([ch_lo, right_lo])
                ch_hi = torch.cat([left_hi, ch_hi])
            from .dual_lp import _host_ram_room
            need = 2 * 4 * (f_lo.shape[0] + ch_lo.shape[0]) * ch_lo.shape[1]
            if need > _host_ram_room() * 0.5:
                # graceful stop beats the cgroup OOM kill (which loses the
                # verdict); the caller treats timeout as unknown
                return 'timeout', {'frontier': int(f_lo.shape[0]),
                                   'bounded': n_bounded,
                                   'splits': n_split,
                                   'reason': 'host_ram_cap'}
            if brow is not None:
                w = lbq_open                    # already per-domain (No,)
                # child pre-refutation (abcrown's concretize step): the
                # parent's own linear forms (A, c) are valid on any subset,
                # so evaluate them over each child box and never push a
                # child they already prove empty -- a refuted child costs
                # ~nothing here vs a full zono+crown pass next round. The
                # surviving child's linear margin doubles as a TIGHTER
                # frontier priority than the parent's repeated bound.
                reps_c = ch_lo.shape[0] // olo.shape[0]
                A_f = A_own[feas_any].repeat(reps_c, 1, 1)
                c_f = c_own[feas_any].repeat(reps_c, 1)
                mnc = (A_f.clamp(min=0) * ch_lo.unsqueeze(1)
                       + A_f.clamp(max=0) * ch_hi.unsqueeze(1)).sum(-1)
                lb_ch = c_f + mnc                        # (C, r)
                ch_keep = ~((lb_ch > 0) & torch.isfinite(lb_ch)).any(dim=1)
                ch_pri = lb_ch.masked_fill(~torch.isfinite(lb_ch),
                                           -torch.inf).max(dim=1).values
                f_lo = torch.cat([f_lo, ch_lo[ch_keep].cpu()])
                f_hi = torch.cat([f_hi, ch_hi[ch_keep].cpu()])
                f_row = torch.cat([f_row,
                                   orow.repeat(reps_c)[ch_keep].cpu()])
                f_worst = torch.cat([f_worst, ch_pri[ch_keep].cpu()])
            else:
                w = lbq_open.min(dim=1).values
                f_lo = torch.cat([f_lo, ch_lo.cpu()])
                f_hi = torch.cat([f_hi, ch_hi.cpu()])
                f_row = torch.cat([f_row,
                                   torch.zeros(ch_lo.shape[0],
                                               dtype=torch.long)])
                f_worst = torch.cat([f_worst,
                                     w.repeat(ch_lo.shape[0] // w.shape[0])
                                     .cpu()])
            n_split += int(w.shape[0])

            if onnx_path is not None and (
                    rounds == 1
                    or time.time() - last_atk > 8.0
                    or (rounds % attack_every == 1
                        and time.time() - last_atk > 2.0)):
                # cadence is time-gated: the bare round-modulo fired a 0.5s
                # pgd every few hundred ms during the fast early rounds
                # (lsnc: ~3s of a 19s BaB slice burned before the frontier
                # ramped). Round 1 stays unconditional -- the first
                # post-clip subboxes are the attack's best shot on sat rows.
                last_atk = time.time()
                # attack the worst open subboxes as one batched-box PGD
                widx = torch.argsort(w)[:64]
                cand, _ = attack.pgd(net, spec, lo=olo[widx], hi=ohi[widx],
                                     restarts=256, iters=60, device=device,
                                     time_budget=0.5, seed=rounds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand,
                                                log=log)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
                    if vinfo.get('within_tol_witness') is not None:
                        tol_witness = vinfo['within_tol_witness']
        if rounds % 32 == 0:
            log(f'[vc2/bab] round={rounds} frontier={int(f_lo.shape[0])} '
                f'bounded={n_bounded} t={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'splits': n_split,
                     'rounds': rounds}



def _kfsb_pick(net, W, bias, lo1, hi1, inter, clamps, range_clamps,
               relu_maps, open_idx, lbq, dev, root_alphas=None,
               dom_alphas=None, k=8, chunk=256):
    """kFSB branching (ab-crown's default, its vit 2157 trace: 786 domains
    visited, frontier 51->29->9->2->0, ~half of each round spent in
    `decision`): the BaBSR proxy NOMINATES the top-k relu candidates per
    open domain, a real planes-only crown pass bounds both children of
    every candidate, and the pick maximizes min(child lb) (reduceop min).
    The proxy alone measured a flat 6k-domain tree at -0.015 on the same
    row the probe-based pick closes.

    Returns {domain_row: (edge, neuron)} for the open domains (may omit
    rows whose candidates all scored empty)."""
    from . import backward as bwd
    edges = sorted(relu_maps)
    widths = [relu_maps[nm].shape[1] for nm in edges]
    flat = torch.cat([relu_maps[nm] for nm in edges], dim=1)   # (B, ncat)
    k_eff = min(k, flat.shape[1])
    top_v, top_i = flat[open_idx].topk(k_eff, dim=1)           # (Bo, k)
    edge_of = torch.repeat_interleave(
        torch.arange(len(edges), device=dev),
        torch.tensor(widths, device=dev))
    j_of = torch.cat([torch.arange(w, device=dev) for w in widths])
    Bo = open_idx.numel()
    # child rows: (bo, cand, sign) flattened; sign alternates fastest
    rows = open_idx.repeat_interleave(2 * k_eff)               # (Bo*2k,)
    cand = top_i.repeat_interleave(2, dim=1).reshape(-1)       # (Bo*2k,)
    signs = torch.tensor([1, -1], device=dev, dtype=torch.int8) \
        .repeat(Bo * k_eff)
    valid = top_v.repeat_interleave(2, dim=1).reshape(-1) > 0
    m_ch = torch.full((Bo * 2 * k_eff,), -torch.inf, device=dev)
    for c0 in range(0, rows.numel(), chunk):
        sl = slice(c0, min(c0 + chunk, rows.numel()))
        r = rows[sl]
        if not bool(valid[sl].any()):
            continue
        cc = {nm: cl[r].clone() for nm, cl in clamps.items()}
        for ei, nm in enumerate(edges):
            if nm not in cc:
                cc[nm] = torch.zeros(r.numel(), net.ops[nm].n,
                                     device=dev, dtype=torch.int8)
        e_sel = edge_of[cand[sl]]
        j_sel = j_of[cand[sl]]
        ar = torch.arange(r.numel(), device=dev)
        for ei, nm in enumerate(edges):
            m = e_sel == ei
            if bool(m.any()):
                cc[nm][ar[m], j_sel[m]] = signs[sl][m]
        ic = {k2: tuple(t[r] for t in v) for k2, v in inter.items()}
        rc = {k2: tuple(t[r] for t in v)
              for k2, v in (range_clamps or {}).items()}
        lo_r = lo1.expand(r.numel(), -1)
        hi_r = hi1.expand(r.numel(), -1)
        # probe with the batch's DOMAIN alphas when available (each
        # parent's optimized state, gathered per child row -- strictly
        # better probe context than the root's), else the root alphas.
        # A plain-planes probe measured every child at the parent floor
        # (all candidates tied); candidates only separate at a bound
        # quality near the domain's.
        if dom_alphas:
            q_w = W.shape[0]
            al = {}
            for nm, a in dom_alphas.items():
                if a.dim() != 3:
                    continue             # relu alphas only (4-dim
                    # S-shaped entries need their own expand shape)
                a_r = a.detach()[r]
                if a_r.shape[1] != q_w:
                    a_r = a_r.mean(dim=1, keepdim=True) \
                        .expand(-1, q_w, -1)
                al[nm] = a_r
        elif root_alphas is not None:
            al = {nm: a.expand(r.numel(), *a.shape[1:])
                  for nm, a in root_alphas.items()}
        else:
            al = None
        lb_c = bwd.crown(net, lo_r, hi_r, W, ic, al, clamps=cc,
                         range_clamps=rc)
        # NO parent floor here: the floor is a sound BOUND but it masks
        # candidate separation (measured: with it, every candidate's
        # children scored exactly the parent bound and the pick
        # degenerated to the proxy). The probe wants the RAW effect of
        # each clamp on the crown pass.
        m_ch[sl] = (lb_c + bias).min(dim=1).values
    m_ch = torch.where(valid, m_ch, torch.full_like(m_ch, -torch.inf))
    sc = m_ch.reshape(Bo, k_eff, 2).min(dim=2).values          # (Bo, k)
    best = sc.argmax(dim=1)                                    # (Bo,)
    picks = {}
    for i, bo in enumerate(open_idx.tolist()):
        if not bool(torch.isfinite(sc[i, best[i]])):
            continue
        ci = int(top_i[i, best[i]])
        picks[bo] = (edges[int(edge_of[ci])], int(j_of[ci]))
    return picks


def relu_split_bab(net, spec, W, bias, disj_idx, lo, hi, deadline,
                   device='cpu', batch=256, beta_iters=None, beta_lr=0.1,
                   onnx_path=None,
                   attack_every=16, root_inter=None, bound='crown',
                   warm_alphas=None, root_alphas=None,
                   log=lambda m: None):
    """ReLU-phase splitting BaB (no-reforward): intermediates stay ROOT
    bounds; each domain carries sign clamps, and the bound comes from
    alpha+beta CROWN under those clamps (v1 _crown_bab_noreforward / abcrown
    beta-CROWN style). Action score is BaBSR: |pre-act adjoint| x triangle
    intercept, from the same backward pass that produced the bound.

    Domains are (worst_lb, splits) with splits a tuple of
    (relu_name, neuron, sign); clamps materialize densely per batch.
    """
    import heapq
    dev = torch.device(device)
    dt = torch.float32
    W = W.to(dev, dt)
    bias = bias.to(dev, dt)
    q = W.shape[0]
    D, sel = disjunct_selector(disj_idx, q, dev)
    lo1 = lo.reshape(1, -1).to(dev, dt)
    hi1 = hi.reshape(1, -1).to(dev, dt)
    if beta_iters is None:
        # each beta iteration is a full backward pass; on a 460k-neuron
        # conv net 12 iterations make ONE round cost ~40s (bounded=0 at
        # the halt). Scale effort to the net so rounds complete -- but
        # 4 iterations optimizes NOTHING (TinyYOLO: alpha/beta both
        # flat); the 50k-1M tier takes 12. (A 20-iter tier for small
        # nets REGRESSED sb048: rounds 1.4x slower starved the
        # round-interleaved CE attack -- the sat row needs round 64
        # by t=82.)
        n_nl = sum(net.ops[nm].n for nm in net.order
                   if net.ops[nm].kind == 'nonlin')
        beta_iters = 12 if n_nl <= 1_000_000 else 4

    if root_inter is None:
        root_inter = backward.intermediates(net, lo1, hi1)
    relu_edges = [nm for nm in net.order
                  if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']
    relu_set = set(relu_edges)
    softmax_set = {nm for nm in net.order
                   if net.ops[nm].kind == 'nonlin'
                   and net.ops[nm].fn == 'softmax'}
    # smooth single-input nonlins are RANGE-splittable: same action ranking,
    # children constrain the pre-activation interval instead of a sign
    smooth_edges = [nm for nm in net.order
                    if net.ops[nm].kind == 'nonlin'
                    and net.ops[nm].fn in ('sigmoid', 'tanh', 'sin', 'cos',
                                           'exp', 'reciprocal', 'pow')]

    # (worst_lb, tiebreak, splits, lb_floor): lb_floor is the parent's
    # per-query bound vector. A child domain is a SUBSET of its parent,
    # so the parent's bound holds for it; flooring the child's computed
    # bound there keeps bounds monotone down the tree (measured on vit
    # 4493: without it the frontier worst DRIFTED -0.034 -> -0.074 as
    # looser deep-domain beta optima re-opened queries the parent had
    # already refuted)
    # 5th slot: the parent's optimized split betas {(edge, j): float};
    # 6th: the parent's optimized alpha state {edge: (qd, n) fp16 numpy}
    # -- ab-crown transfers both (set_bounds); measured on vit 2157,
    # per-domain bound quality tracks optimization effort (cold 12-iter
    # -0.0138 vs 30-iter -0.0094 at equal domains) and the transfer
    # buys converged-quality starts at low iteration counts
    heap = [_Dom(-float('inf'), 0, (), None, {}, None)]
    last_atk = time.time()   # attack cadence is TIME-based too: slow
    # bound rounds must not starve the CE hunt (sb048's hidden CE)
    tick = 1
    n_bounded = rounds = 0
    tol_witness = None
    t0 = time.time()


    while heap:
        if time.time() > deadline:
            return 'timeout', {'frontier': len(heap), 'bounded': n_bounded,
                               'tol_witness': tol_witness}
        rounds += 1
        n_relu_total = sum(net.ops[nm].n for nm in relu_edges)
        widest = max(net.ops[o].n for o in net.order)
        per_dom = (n_relu_total * 10 + q * widest * 12) * 4   # alpha/beta/adam + adjoints
        bs = min(batch, memory.chunk_size(len(heap), per_dom, dev))
        batch_doms = [heapq.heappop(heap) for _ in range(min(bs, len(heap)))]
        B = len(batch_doms)
        if os.environ.get('VC2_DEBUG_CLIP') and rounds <= 6:
            log(f'[vc2/rbab] round={rounds} start B={B} bs={bs} '
                f'heap={len(heap)} t={time.time() - t0:.1f}s')
        try:
            blo = lo1.expand(B, -1)
            bhi = hi1.expand(B, -1)
            clamps, range_clamps = materialize_clamps(
                [dom.splits for dom in batch_doms],
                lambda nm: net.ops[nm].n, B, dev)
            # reforward-IBP under the clamps, intersected with the (tighter at
            # the root, clamp-blind) root intermediates: best of both regimes
            if bound == 'zono':
                # lean round: the sign clamps act INSIDE the zono via
                # clamp_bounds; the reforward-IBP merge and the crown
                # scoring pass cost more than the bound itself (measured
                # 0.7 s/domain vs the zono's 95 ms at B=16)
                inter = {k2: tuple(t.expand(B, -1) for t in v)
                         for k2, v in root_inter.items()}
            else:
                ib_state = backward.fwd.interval(net, blo, bhi,
                                                 return_state=True,
                                                 clamps=clamps,
                                                 range_clamps=range_clamps)
                ib = backward._inter_from_state(net, lambda e: ib_state[e])
                inter = {}
                for k2, v in root_inter.items():
                    rv = tuple(t.expand(B, -1) for t in v)
                    iv = ib[k2]
                    merged = []
                    for j2 in range(0, len(rv), 2):
                        merged.append(torch.maximum(rv[j2], iv[j2]))
                        merged.append(torch.minimum(rv[j2 + 1], iv[j2 + 1]))
                    inter[k2] = tuple(merged)
            if bound != 'zono' and n_relu_total <= 2000:
                # small net (acasxu class): per-batch CROWN refinement of the
                # merged bounds under the clamps. NOT for wide layers: the
                # identity blocks scale with unstable count x batch (malbeware,
                # 7842 unstable: 6 domains/s vs the fixed-root regime; abcrown
                # runs fixed root intermediates + beta only on this class;
                # vit measured 145 -> 5 domains/s when this ran through its
                # bmm factor edges)
                inter = backward.intermediates_crown(net, blo, bhi,
                                                     base_inter=inter,
                                                     clamps=clamps,
                                                     range_clamps=range_clamps)
            adj = {}
            beta_out = {}
            alpha_out = {}
            zono_scores = {}
            zono_range_scores = {}
            if bound == 'zono':
                # v1's vit beta-bab regime (its trace: 1583 domains close
                # 2157's last query at ~32 ms/domain): the FORWARD zono --
                # the only bound that is tight on attention nets (backward
                # crown: -9.9 vs zono -0.04 at the root) -- evaluated
                # under the domain's sign clamps via clamp_bounds. The
                # BaBSR adjoint signal comes from a crown pass over the
                # same clamps (scoring only; its bound is not used).
                cb = {}
                for nm, cl in clamps.items():
                    l0, h0 = inter[nm][0], inter[nm][1]
                    cb[nm] = (torch.where(cl > 0, l0.clamp_min(0.0), l0),
                              torch.where(cl < 0, h0.clamp_max(0.0), h0))
                for nm, (rlo, rhi) in (range_clamps or {}).items():
                    l0, h0 = inter[nm][0], inter[nm][1]
                    cb[nm] = (torch.maximum(l0, rlo),
                              torch.maximum(torch.minimum(h0, rhi),
                                            torch.maximum(l0, rlo)))
                _dbg = os.environ.get('VC2_DEBUG_CLIP')
                _tz = time.time()
                so = None
                if warm_alphas is not None:
                    # the root fzono's optimized slopes, expanded per
                    # domain: default bands measured ~10x looser
                    so = {nm: t.to(dev).expand(B, -1)
                          for nm, t in warm_alphas.items()}
                rec = {}
                _zl, _zh, zst = backward.fwd.zono(net, blo, bhi,
                                                  return_state=True,
                                                  clamp_bounds=cb,
                                                  slope_override=so,
                                                  record=rec)
                if _dbg:
                    torch.cuda.synchronize() if dev.type == 'cuda' else None
                    log(f'[vc2/rbab] round={rounds} zono {time.time()-_tz:.2f}s B={B}')
                    _tz = time.time()
                zo = zst[net.output_name]
                WG = torch.matmul(W, zo.G)               # (B, q, g)
                lbq = torch.matmul(zo.c, W.T) - WG.abs().sum(-1)
                if zo.rad is not None:
                    lbq = lbq - torch.matmul(zo.rad, W.abs().T)
                lbq = _beta_tighten(lbq, W, bias, zo, WG, rec, batch_doms,
                                    inter, dev)
                # SLACK ATTRIBUTION scoring (v1's BBPS intent, exact in
                # the zono frame): each relu fresh column's |W @ G[:,col]|
                # IS that neuron's relaxation contribution to the margin
                zono_scores = {}
                zono_range_scores = {}
                WGa = WG.abs().amax(dim=1)               # (B, g), one kernel
                for col, sm in enumerate(zo.sym):
                    nm_s, j_s = sm
                    if nm_s in relu_set:
                        zono_scores.setdefault(nm_s, []).append(
                            (j_s, WGa[:, col]))
                    elif nm_s.endswith('/e') and nm_s[:-2] in softmax_set:
                        # fused-softmax exp band columns map 1:1 to the
                        # softmax INPUT elements -> range-splittable
                        # (v1 splits the attention internals; the fused
                        # op hid them from the edge lists)
                        zono_range_scores.setdefault(nm_s[:-2], []).append(
                            (j_s, WGa[:, col]))
                a_in = None
                del zst, zo, WG
                if _dbg:
                    torch.cuda.synchronize() if dev.type == 'cuda' else None
                    log(f'[vc2/rbab] round={rounds} scoring {time.time()-_tz:.2f}s '
                        f'g_cols={len(zono_scores) and sum(len(v) for v in zono_scores.values())}')
            else:
                lbq, a_in = backward.crown(net, blo, bhi, W, inter,
                                           clamps=clamps,
                                           range_clamps=range_clamps,
                                           collect_adjoints=adj,
                                           return_input_adjoint=True)
                ib_beta = {}
                for bi, dom in enumerate(batch_doms):
                    for (nm, j), bv in (dom.betas or {}).items():
                        if nm not in ib_beta:
                            ib_beta[nm] = torch.zeros(
                                B, 1, net.ops[nm].n, device=dev)
                        ib_beta[nm][bi, 0, j] = bv
                # per-domain alpha transfer: batch rows with stored state
                # start there, the rest from the root alphas
                ib_alpha = None
                if any(dom.alphas is not None for dom in batch_doms):
                    ib_alpha = {}
                    for bi, dom in enumerate(batch_doms):
                        for nm, a16 in (dom.alphas or {}).items():
                            if nm not in ib_alpha:
                                qd_s = a16.shape[0]   # stored qd governs
                                if root_alphas and nm in root_alphas:
                                    r_a = root_alphas[nm]
                                    base = (r_a.mean(dim=1, keepdim=True)
                                            if qd_s == 1
                                            else r_a[:, :qd_s])
                                else:
                                    base = torch.full(
                                        (1, qd_s, net.ops[nm].n), 0.5)
                                ib_alpha[nm] = base.to(dev).float() \
                                    .expand(B, -1, -1).contiguous()
                            a_t = torch.as_tensor(a16, device=dev,
                                                  dtype=torch.float32)
                            qd_b = ib_alpha[nm].shape[1]
                            if a_t.shape[0] != qd_b:
                                # share_q flips with batch size, so
                                # stored qd differs ACROSS generations
                                # (sb048 crashed on a 12-vs-1 mix);
                                # mean-reduce/broadcast -- any [0,1]
                                # init is sound
                                a_t = a_t.mean(dim=0, keepdim=True) \
                                    .expand(qd_b, -1)
                            ib_alpha[nm][bi] = a_t
                lb_ab, beta_out, alpha_out = backward.alpha_beta_crown(
                    net, blo, bhi, W, inter, clamps, iters=beta_iters,
                    lr=beta_lr,
                    # WARM state refines at the small lr; big steps are
                    # only for cold-from-zero betas (yolo). Inherited
                    # betas at lr 0.1 thrash exactly like inherited
                    # alphas did (2157 replays: -0.0075 regime lost to
                    # -0.0134 when the split-lr landed)
                    lr_beta=(beta_lr if ib_beta else 0.1),
                    thresholds=-bias, range_clamps=range_clamps,
                    init_alpha=(ib_alpha if ib_alpha is not None
                                else root_alphas),
                    init_beta=ib_beta or None,
                    return_beta=True, return_alpha=True)
                lbq = torch.maximum(lbq, lb_ab)
        except (torch.cuda.OutOfMemoryError,
                torch.AcceleratorError):
            # round-level OOM recovery (mirrors input_split_bab): push
            # the popped domains back, halve the batch, continue --
            # covers raw CUDA allocation failures (compile workspace)
            # that bypass the caching allocator's OutOfMemoryError
            torch.cuda.empty_cache()
            for dom in batch_doms:
                heapq.heappush(heap, dom)
            if batch <= 1:
                return 'timeout', {'frontier': len(heap),
                                   'bounded': n_bounded,
                                   'reason': 'oom_floor'}
            batch = max(1, B // 2)
            log(f'[vc2/rbab] round={rounds} OOM at B={B}; batch -> {batch}')
            continue
        n_bounded += B
        floors = [d.floor for d in batch_doms]
        if any(f is not None for f in floors):
            fl = torch.stack([
                torch.as_tensor(f, device=dev, dtype=lbq.dtype)
                if f is not None
                else torch.full((q,), -torch.inf, device=dev,
                                dtype=lbq.dtype)
                for f in floors])
            lbq = torch.maximum(lbq, fl)
        refuted = refuted_matrix(lbq, bias, sel)
        open_mask = ~refuted.all(dim=1)
        if open_mask.any():
            # unified action ranking: relu sign splits score by BaBSR
            # (|adjoint| x triangle intercept), smooth range splits by
            # |adjoint| x band delta -- both estimate removable slack
            best_score = torch.full((B,), -torch.inf, device=dev)
            best_edge = [None] * B
            best_j = torch.zeros(B, dtype=torch.long, device=dev)
            best_mid = torch.zeros(B, device=dev)
            best_kind = [''] * B
            if bound == 'zono' and (zono_scores or zono_range_scores):
                for nm, cols in zono_scores.items():
                    sc = torch.zeros(B, net.ops[nm].n, device=dev)
                    for j_s, v_s in cols:
                        sc[:, j_s] = v_s
                    cl = clamps.get(nm)
                    if cl is not None:
                        sc = sc * (cl == 0)      # already-split: not again
                    v, j = sc.max(dim=1)
                    better = v > best_score
                    best_score = torch.where(better, v, best_score)
                    best_j[better] = j[better]
                    for bi in torch.nonzero(better,
                                            as_tuple=False).flatten().tolist():
                        best_edge[bi] = nm
                        best_kind[bi] = 'sign'
                for nm, cols in zono_range_scores.items():
                    sc = torch.zeros(B, net.ops[nm].n, device=dev)
                    for j_s, v_s in cols:
                        sc[:, j_s] = torch.maximum(sc[:, j_s], v_s)
                    l_r, h_r = inter[nm][0], inter[nm][1]
                    if nm in range_clamps:
                        l_r = torch.maximum(l_r, range_clamps[nm][0])
                        h_r = torch.minimum(h_r, range_clamps[nm][1])
                    mid_r = (l_r + h_r) / 2
                    v, j = sc.max(dim=1)
                    better = v > best_score
                    best_score = torch.where(better, v, best_score)
                    best_j[better] = j[better]
                    best_mid[better] = mid_r.gather(
                        1, j.unsqueeze(1))[:, 0][better]
                    for bi in torch.nonzero(better,
                                            as_tuple=False).flatten().tolist():
                        best_edge[bi] = nm
                        best_kind[bi] = 'range'

            skip_babsr = bound == 'zono' and bool(zono_scores)

            def consider(nm, s_scores, kind, mid=None):
                nonlocal best_score
                v, j = s_scores.max(dim=1)
                better = v > best_score
                best_score = torch.where(better, v, best_score)
                best_j[better] = j[better]
                if mid is not None:
                    best_mid[better] = mid.gather(1, j.unsqueeze(1))[:, 0][better]
                for bi in torch.nonzero(better,
                                        as_tuple=False).flatten().tolist():
                    best_edge[bi] = nm
                    best_kind[bi] = kind

            relu_maps = {}
            for nm in (() if skip_babsr else relu_edges):
                l, h = inter[nm]
                cl = clamps.get(nm)
                if cl is not None:
                    l, h = backward.clamped_bounds((l, h), cl)
                unstable = (l < 0) & (h > 0)
                if not bool(unstable.any()):
                    continue
                intercept = (-h * l / (h - l).clamp_min(1e-30)).clamp_min(0.0)
                a = adj.get(nm)
                sc_nm = (a.abs().amax(dim=1) if a is not None
                         else torch.ones_like(l)) * intercept * unstable
                if bound != 'zono':
                    relu_maps[nm] = sc_nm
                consider(nm, sc_nm, 'sign')
            from .relax import REL
            for nm in (() if skip_babsr else smooth_edges):
                l, h = inter[nm]
                _lam, _mu, delta = REL[net.ops[nm].fn].band(
                    l, h, net.ops[nm].params)
                a = adj.get(nm)
                consider(nm, (a.abs().amax(dim=1) if a is not None
                              else torch.ones_like(l)) * delta,
                         'range', mid=(l + h) / 2)
            if bound != 'zono' and relu_maps:
                try:
                    op_idx = torch.nonzero(open_mask,
                                           as_tuple=False).flatten()
                    if op_idx.numel() > 64:
                        # probe only the WORST 64 open domains (they
                        # drive the tree; the rest keep the proxy pick):
                        # the probe was ~half the round cost at B=256
                        wq = (lbq + bias).min(dim=1).values[op_idx]
                        op_idx = op_idx[wq.argsort()[:64]]
                    picks = _kfsb_pick(net, W, bias, lo1, hi1, inter,
                                       clamps, range_clamps, relu_maps,
                                       op_idx, lbq, dev,
                                       root_alphas=root_alphas,
                                       dom_alphas=(alpha_out or None))
                    for bo, (nm, j) in picks.items():
                        best_edge[bo] = nm
                        best_j[bo] = j
                        best_kind[bo] = 'sign'
                except torch.cuda.OutOfMemoryError:
                    # probe pass is advisory; the proxy picks stand
                    torch.cuda.empty_cache()
            w_dom = (lbq + bias).min(dim=1).values
            if os.environ.get('VC2_DEBUG_CLIP') and rounds <= 8:
                _sd = [f'({best_edge[bi]},{int(best_j[bi])},'
                       f'{best_kind[bi]})'
                       for bi in torch.nonzero(open_mask, as_tuple=False)
                       .flatten().tolist()[:6] if best_edge[bi]]
                log(f'[vc2/rbab] round={rounds} splits: '
                    + ' '.join(_sd))
            for bi in torch.nonzero(open_mask, as_tuple=False).flatten().tolist():
                if best_edge[bi] is None:
                    # no unstable relu left: relaxation exact -> the domain
                    # can only be sat; try to falsify, else give up loudly
                    return 'unknown', {'reason': 'exhausted splits',
                                       'bounded': n_bounded}
                base = batch_doms[bi].splits
                if best_kind[bi] == 'range':
                    m = float(best_mid[bi])
                    children = ((-np.inf, m), (m, np.inf))
                else:
                    children = (1, -1)
                fl_ch = lbq[bi].detach().cpu().numpy()
                bd_ch = dict(batch_doms[bi].betas or {})
                for nm2, j2, sp2 in base:
                    if not isinstance(sp2, tuple) and nm2 in beta_out \
                            and beta_out[nm2].numel():
                        bd_ch[(nm2, j2)] = float(
                            beta_out[nm2][bi, :, j2].max())
                ad_ch = ({nm2: a[bi].detach().to(torch.float16)
                          .cpu().numpy()
                          for nm2, a in alpha_out.items()
                          if a.dim() == 3}    # relu alphas only: the
                         # (B, qd, 2, n) S-shaped entries would need
                         # their own base shape in the batch rebuild
                         if alpha_out else None)
                for ch in children:
                    heapq.heappush(heap, _Dom(
                        float(w_dom[bi]), tick,
                        base + ((best_edge[bi], int(best_j[bi]), ch),),
                        fl_ch, bd_ch, ad_ch))
                    tick += 1
            if onnx_path is not None and (
                    rounds % attack_every == 1
                    or time.time() - last_atk > 8.0):
                last_atk = time.time()
                # BaB-GUIDED seeds (v1 _pgd_refine): each open (domain,
                # query)'s relaxed-bound argmin x* = mid - sign(A_in) * rad
                # attains the CROWN lb under that domain's clamps. Hidden
                # CEs (soundnessbench) defeat box-uniform restarts BY
                # CONSTRUCTION; the bound's own primal point is where the
                # relaxation says the margin is lowest, and splitting
                # isolates it -- seed the attack there.
                m_bq = lbq + bias
                seeds = None
                nz = torch.nonzero(((m_bq <= 0)
                                    & open_mask.unsqueeze(1)).flatten(),
                                   as_tuple=False).flatten()
                if a_in is not None and nz.numel():
                    pick = nz[m_bq.flatten()[nz].argsort()[:64]]
                    Af = a_in.reshape(-1, a_in.shape[-1])[pick]
                    mid = (lo1[0] + hi1[0]) / 2
                    rad = (hi1[0] - lo1[0]) / 2
                    seeds = (mid.unsqueeze(0) - torch.sign(Af)
                             * rad.unsqueeze(0)).cpu().numpy()
                cand, _ = attack.pgd(net, spec, lo=lo1[0], hi=hi1[0],
                                     restarts=128, iters=60, device=device,
                                     time_budget=1.5, seed=rounds,
                                     seeds=seeds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand,
                                                log=log)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
                    if vinfo.get('within_tol_witness') is not None:
                        tol_witness = vinfo['within_tol_witness']
        if rounds % 16 == 0 or (bound == 'zono' and rounds % 4 == 0):
            _wl = float((lbq + bias).min()) if lbq.numel() else float('nan')
            log(f'[vc2/rbab] round={rounds} frontier={len(heap)} '
                f'bounded={n_bounded} worst={_wl:+.4f} '
                f't={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'rounds': rounds}
