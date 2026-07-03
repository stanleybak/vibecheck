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
                    roots=None, log=lambda m: None):
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
    _row_map = None
    if roots is not None:
        # dedupe the query rows BEFORE q feeds the memory sizing: nn4sys
        # carries 960 disjunct rows drawn from 2 unique (w, b) pairs, and
        # a stale q=960 shrank the pop batch to 1-2 domains per round
        key = torch.cat([W, bias.unsqueeze(1)], dim=1)
        uniq, _row_map = torch.unique(key, dim=0, return_inverse=True)
        W = uniq[:, :-1].contiguous()
        bias = uniq[:, -1].contiguous()
    q = W.shape[0]
    if roots is not None:
        D, sel = 0, None       # per-domain rows replace the disjunct map
    else:
        D = int(disj_idx.max()) + 1 if disj_idx.numel() else 0
        # per-disjunct row selector (D, q) for the batched refutation check
        sel = torch.zeros(D, q, device=dev, dtype=torch.bool)
        sel[disj_idx, torch.arange(q)] = True

    # the frontier lives on HOST: a stuck instance grows it to millions of
    # (n_in,) rows (dist_shift index112 hit 250k x 792 and OOM-crashed the
    # GPU mid-bookkeeping); only the popped batch goes to the device
    if roots is not None:
        f_lo = roots[0].to('cpu', dt)
        f_hi = roots[1].to('cpu', dt)
        f_row = _row_map.cpu()[roots[2].cpu()]
        f_worst = torch.full((f_lo.shape[0],), -torch.inf)
        log(f'[vc2/bab] multi-sub: {f_lo.shape[0]} roots over '
            f'{W.shape[0]} unique rows')
    else:
        f_lo = lo.reshape(1, -1).to('cpu', dt)
        f_hi = hi.reshape(1, -1).to('cpu', dt)
        f_row = torch.zeros(1, dtype=torch.long)
        f_worst = torch.full((1,), -torch.inf)
    n_bounded = n_split = rounds = 0
    _round_wall, _round_B = 1.0, 1
    t0 = time.time()
    n_nonlin = sum(net.ops[nm].n for nm in net.order
                   if net.ops[nm].kind == 'nonlin')
    # tiny nets (acasxu class): full per-batch identity-CROWN refinement.
    # bigger ones: joint-alpha refine ONCE at the root, then per batch only
    # a cheap reforward intersected with the root bounds (subbox is a
    # subset, so the intersection is sound) -- the abcrown cost model.
    full_refine = n_nonlin <= 1200
    if roots is not None:
        full_refine = False              # per-sub boxes vary; root_ref on
                                         # the UNION box, cheap reforward
                                         # per batch (still sound)
    if full_refine:
        # identity-CROWN vs forward-zono planes flips by net class (acasxu
        # amplifying weights favor backward refinement; dist_shift sigmoid
        # bands close at depth 1 under zono but need 100k+ splits under
        # identity-CROWN), so measure both on the root's first two
        # children and keep the winner.
        wax = int((f_hi - f_lo).argmax())
        mid = (f_lo[0, wax] + f_hi[0, wax]) / 2
        plo = f_lo.repeat(2, 1).to(dev)
        phi = f_hi.repeat(2, 1).to(dev)
        phi[0, wax] = mid
        plo[1, wax] = mid
        try:
            _l, _h, zst = backward.fwd.zono(net, plo, phi, return_state=True)
            zi = backward._inter_from_state(net, lambda e: zst[e].bounds())
            z_lb = float((backward.crown(net, plo, phi, W, zi) + bias).min())
            ci = backward.intermediates_crown(net, plo, phi, base_inter=zi)
            c_lb = float((backward.crown(net, plo, phi, W, ci) + bias).min())
            full_refine = c_lb >= z_lb
            log(f'[vc2/bab] refine probe: crown={c_lb:.3f} zono={z_lb:.3f} '
                f'-> full_refine={full_refine}')
        except (NotImplementedError, torch.cuda.OutOfMemoryError) as _pe:
            backward._warn_once(f'refine probe zono unavailable '
                                f'({type(_pe).__name__}); keeping crown')
    root_ref = None
    if not full_refine and roots is None:
        # (multi-sub skips this: the union of 960 scattered subboxes is
        # nearly the whole space, so refining it is GBs of pure cost;
        # the per-subbox reforward below is the tight part there)
        root_ref = backward.intermediates_crown(
            net, f_lo.to(dev), f_hi.to(dev),
            alpha_iters=(12 if n_nonlin <= 20000 else 0))

    def domain_refuted(lbq, brow=None):
        """(B, q) query lbs -> (B, D) refutation matrix; in multi-sub
        mode a (B, 1) matrix from each domain's OWN row. Only FINITE
        positive bounds refute (+inf is an artifact, never a proof)."""
        lbb = lbq + bias
        pos = (lbb > 0) & torch.isfinite(lbb)        # refuting rows
        if brow is not None:
            return pos.gather(1, brow.unsqueeze(1))
        refuted = torch.zeros(lbq.shape[0], D, device=dev, dtype=torch.bool)
        for dd in range(D):
            refuted[:, dd] = (pos & sel[dd]).any(dim=1)
        return refuted

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

    while f_lo.shape[0]:
        if time.time() > deadline:
            return 'timeout', {'frontier': int(f_lo.shape[0]),
                               'bounded': n_bounded, 'splits': n_split}
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
                    inter = backward.intermediates_crown(net, blo, bhi,
                                                         base_inter=zi)
                except (NotImplementedError, torch.cuda.OutOfMemoryError):
                    inter = backward.intermediates_crown(net, blo, bhi)
            else:
                B = blo.shape[0]
                zst = None
                try:
                    # true per-subbox planes from the wide-dims zonotope
                    _l, _h, zst = backward.fwd.zono(net, blo, bhi,
                                                    return_state=True)
                except (NotImplementedError,
                        torch.cuda.OutOfMemoryError) as _ze:
                    backward._warn_once(
                        f'input-split per-batch zono unavailable '
                        f'({type(_ze).__name__}: {str(_ze)[:60]}); '
                        f'interval reforward only')
                    # fall back BELOW: running the fallback inside
                            # this handler would pin the zono's partial state
                            # via the live traceback (mscn: 3GB held while
                            # the interval pass then OOMs)
                if zst is not None:
                    ib = backward._inter_from_state(
                        net, lambda e: zst[e].bounds())
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
            lbq, Ain = backward.crown(net, blo, bhi, W, inter,
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
                    margin = (lbq + bias).gather(1,
                                                 brow.unsqueeze(1)).squeeze(1)
                else:
                    margin = (lbq + bias).max(dim=1).values
                bmask = open_mask & (margin > -10.0)
                oi = torch.nonzero(bmask, as_tuple=False).flatten()
                if oi.numel() > 2048:
                    oi = oi[margin[oi].argsort(descending=True)[:2048]]
            if open_mask.any() and alpha_iters > 0 and oi.numel():
                inter_o = {k2: tuple(t[oi] for t in v)
                           for k2, v in inter.items()}
                lb_a, al = backward.alpha_crown(net, blo[oi], bhi[oi], W,
                                                inter_o, iters=alpha_iters,
                                                thresholds=-bias,
                                                return_alpha=True)
                # one extra pass with the final alphas yields a linearization
                # whose exact concretization is its own bound -- a SOUND
                # (A, b) upgrade for the clip below (the plain-CROWN pair is
                # much weaker; pairing alpha bounds with the plain-CROWN A
                # would over-clip and was the unsound variant)
                lb_al, Ain_a = backward.crown(net, blo[oi], bhi[oi], W, inter_o,
                                              al, return_input_adjoint=True)
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
            lbb = (lbq_lin + bias)[open_mask]
            if brow is not None:
                # multi-sub: each domain's CE must satisfy exactly ITS
                # row's halfspace -- a pure conjunction, no union step
                orow = brow[open_mask]
                xl_u, xh_u = olo.clone(), ohi.clone()
                for rr in orow.unique().tolist():
                    m = orow == rr
                    clo, chi = clip_domains(olo[m], ohi[m], oA[m],
                                            lbb[m], [rr])
                    xl_u[m], xh_u[m] = clo, chi
                feas_any = (xh_u - xl_u).min(dim=1).values >= 0
            else:
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
            lbq_open = (lbq + bias)[open_mask][feas_any]
            if not olo.shape[0]:
                continue
            # Smart-Branching: estimated improvement per input dim; split the
            # top `split_dims` dims simultaneously -> 2^k children.
            # 'widest' ignores sensitivity (v1's dist_shift setting).
            if heuristic == 'widest':
                score = ohi - olo
            else:
                score = oA.abs().sum(dim=1) * (ohi - olo) / 2
            kdims = min(split_dims, olo.shape[1])
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
            f_lo = torch.cat([f_lo, ch_lo.cpu()])
            f_hi = torch.cat([f_hi, ch_hi.cpu()])
            if brow is not None:
                w = lbq_open.gather(1, orow.unsqueeze(1)).squeeze(1)
                f_row = torch.cat([f_row,
                                   orow.repeat(ch_lo.shape[0]
                                               // orow.shape[0]).cpu()])
            else:
                w = lbq_open.min(dim=1).values
                f_row = torch.cat([f_row,
                                   torch.zeros(ch_lo.shape[0],
                                               dtype=torch.long)])
            f_worst = torch.cat([f_worst,
                                 w.repeat(ch_lo.shape[0] // w.shape[0])
                                 .cpu()])
            n_split += int(w.shape[0])

            if onnx_path is not None and rounds % attack_every == 1:
                # attack the worst open subboxes as one batched-box PGD
                widx = torch.argsort(w)[:64]
                cand, _ = attack.pgd(net, spec, lo=olo[widx], hi=ohi[widx],
                                     restarts=256, iters=60, device=device,
                                     time_budget=0.5, seed=rounds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
        if rounds % 32 == 0:
            log(f'[vc2/bab] round={rounds} frontier={int(f_lo.shape[0])} '
                f'bounded={n_bounded} t={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'splits': n_split,
                     'rounds': rounds}


def relu_split_bab(net, spec, W, bias, disj_idx, lo, hi, deadline,
                   device='cpu', batch=256, beta_iters=12, onnx_path=None,
                   attack_every=16, root_inter=None, log=lambda m: None):
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
    D = int(disj_idx.max()) + 1 if disj_idx.numel() else 0
    sel = torch.zeros(D, q, device=dev, dtype=torch.bool)
    sel[disj_idx, torch.arange(q)] = True
    lo1 = lo.reshape(1, -1).to(dev, dt)
    hi1 = hi.reshape(1, -1).to(dev, dt)

    if root_inter is None:
        root_inter = backward.intermediates(net, lo1, hi1)
    relu_edges = [nm for nm in net.order
                  if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']
    # smooth single-input nonlins are RANGE-splittable: same action ranking,
    # children constrain the pre-activation interval instead of a sign
    smooth_edges = [nm for nm in net.order
                    if net.ops[nm].kind == 'nonlin'
                    and net.ops[nm].fn in ('sigmoid', 'tanh', 'sin', 'cos',
                                           'exp', 'reciprocal', 'pow')]

    heap = [(-float('inf'), 0, ())]           # (worst_lb, tiebreak, splits)
    tick = 1
    n_bounded = rounds = 0
    t0 = time.time()


    def refuted_of(lbq):
        lbb = lbq + bias
        pos = (lbb > 0) & torch.isfinite(lbb)
        r = torch.zeros(lbq.shape[0], D, device=dev, dtype=torch.bool)
        for dd in range(D):
            r[:, dd] = (pos & sel[dd]).any(dim=1)
        return r

    while heap:
        if time.time() > deadline:
            return 'timeout', {'frontier': len(heap), 'bounded': n_bounded}
        rounds += 1
        n_relu_total = sum(net.ops[nm].n for nm in relu_edges)
        widest = max(net.ops[o].n for o in net.order)
        per_dom = (n_relu_total * 10 + q * widest * 12) * 4   # alpha/beta/adam + adjoints
        bs = min(batch, memory.chunk_size(len(heap), per_dom, dev))
        batch_doms = [heapq.heappop(heap) for _ in range(min(bs, len(heap)))]
        B = len(batch_doms)
        try:
            blo = lo1.expand(B, -1)
            bhi = hi1.expand(B, -1)
            clamps = {}
            range_clamps = {}
            for bi, (_, _, splits) in enumerate(batch_doms):
                for nm, j, spec_ in splits:
                    if isinstance(spec_, tuple):          # smooth range split
                        if nm not in range_clamps:
                            n_e = net.ops[nm].n
                            range_clamps[nm] = (
                                torch.full((B, n_e), -torch.inf, device=dev),
                                torch.full((B, n_e), torch.inf, device=dev))
                        rlo, rhi = range_clamps[nm]
                        rlo[bi, j] = max(float(rlo[bi, j]), spec_[0])
                        rhi[bi, j] = min(float(rhi[bi, j]), spec_[1])
                    else:                                  # relu sign split
                        if nm not in clamps:
                            clamps[nm] = torch.zeros(B, net.ops[nm].n, device=dev,
                                                     dtype=torch.int8)
                        clamps[nm][bi, j] = spec_
            # reforward-IBP under the clamps, intersected with the (tighter at
            # the root, clamp-blind) root intermediates: best of both regimes
            ib_state = backward.fwd.interval(net, blo, bhi, return_state=True,
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
            if n_relu_total <= 2000:
                # small net (acasxu class): per-batch CROWN refinement of the
                # merged bounds under the clamps. NOT for wide layers: the
                # identity blocks scale with unstable count x batch (malbeware,
                # 7842 unstable: 6 domains/s vs the fixed-root regime; abcrown
                # runs fixed root intermediates + beta only on this class)
                inter = backward.intermediates_crown(net, blo, bhi,
                                                     base_inter=inter,
                                                     clamps=clamps,
                                                     range_clamps=range_clamps)
            adj = {}
            lbq = backward.crown(net, blo, bhi, W, inter, clamps=clamps,
                                 range_clamps=range_clamps, collect_adjoints=adj)
            lb_ab = backward.alpha_beta_crown(net, blo, bhi, W, inter, clamps,
                                              iters=beta_iters, thresholds=-bias,
                                              range_clamps=range_clamps)
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
            if batch <= 8:
                return 'timeout', {'frontier': len(heap),
                                   'bounded': n_bounded,
                                   'reason': 'oom_floor'}
            batch = max(8, B // 2)
            continue
        n_bounded += B
        refuted = refuted_of(lbq)
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

            for nm in relu_edges:
                l, h = inter[nm]
                cl = clamps.get(nm)
                if cl is not None:
                    l, h = backward.clamped_bounds((l, h), cl)
                unstable = (l < 0) & (h > 0)
                if not bool(unstable.any()):
                    continue
                intercept = (-h * l / (h - l).clamp_min(1e-30)).clamp_min(0.0)
                a = adj.get(nm)
                consider(nm, (a.abs().amax(dim=1) if a is not None
                              else torch.ones_like(l)) * intercept * unstable,
                         'sign')
            from .relax import REL
            for nm in smooth_edges:
                l, h = inter[nm]
                _lam, _mu, delta = REL[net.ops[nm].fn].band(
                    l, h, net.ops[nm].params)
                a = adj.get(nm)
                consider(nm, (a.abs().amax(dim=1) if a is not None
                              else torch.ones_like(l)) * delta,
                         'range', mid=(l + h) / 2)
            w_dom = (lbq + bias).min(dim=1).values
            for bi in torch.nonzero(open_mask, as_tuple=False).flatten().tolist():
                if best_edge[bi] is None:
                    # no unstable relu left: relaxation exact -> the domain
                    # can only be sat; try to falsify, else give up loudly
                    return 'unknown', {'reason': 'exhausted splits',
                                       'bounded': n_bounded}
                base = batch_doms[bi][2]
                if best_kind[bi] == 'range':
                    m = float(best_mid[bi])
                    children = ((-np.inf, m), (m, np.inf))
                else:
                    children = (1, -1)
                for ch in children:
                    heapq.heappush(heap, (float(w_dom[bi]), tick,
                                          base + ((best_edge[bi],
                                                   int(best_j[bi]), ch),)))
                    tick += 1
            if onnx_path is not None and rounds % attack_every == 1:
                cand, _ = attack.pgd(net, spec, lo=lo1[0], hi=hi1[0],
                                     restarts=128, iters=60, device=device,
                                     time_budget=1.5, seed=rounds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
        if rounds % 16 == 0:
            log(f'[vc2/rbab] round={rounds} frontier={len(heap)} '
                f'bounded={n_bounded} t={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'rounds': rounds}
