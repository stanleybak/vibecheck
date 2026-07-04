"""vibecheck2 verification entry point.

Thin orchestration for the current milestone: load net + spec (v1 front end),
forward bounds for intermediates, alpha-CROWN on the spec query rows, verdict
by per-disjunct refutation. Grows into the scheduler of design 2.5; the
results-file discipline matches v1 (the file is the verdict authority).

Disjunct semantics (v1 spec.py): a counterexample must satisfy EVERY
constraint of SOME disjunct. A disjunct is refuted when ANY of its query
rows w.y + b has a positive proven lower bound; `unsat` when every disjunct
is refuted.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .core import backward, forward
from .core.graph import load as load_net


def _spec_queries(spec, n_out, dtype=torch.float32):
    """(W (q, n_out), bias (q,), disj_idx (q,)) from the v1 VNNSpec."""
    rows = spec.as_linear_queries(n_out)
    W = torch.tensor(np.stack([w for _, w, _ in rows]), dtype=dtype)
    b = torch.tensor([bias for _, _, bias in rows], dtype=dtype)
    di = torch.tensor([d for d, _, _ in rows])
    return W, b, di


def _verdict_from_lbs(lb_plus_bias, disj_idx, n_disjuncts):
    """'unsat' iff every disjunct has some strictly-positive FINITE query
    row (a +inf lower bound is always an arithmetic artifact, never a
    proof; NaN already fails the comparison)."""
    import torch as _t
    refuted = set()
    for d in range(n_disjuncts):
        rows = lb_plus_bias[disj_idx == d]
        if rows.numel() and bool(((rows > 0)
                                  & _t.isfinite(rows)).any()):
            refuted.add(d)
    open_d = [d for d in range(n_disjuncts) if d not in refuted]
    return ('unsat' if not open_d else 'unknown'), open_d


def _subbox_groups(spec):
    """Group disjuncts by their per-conjunct input subbox (acasxu prop_6,
    nn4sys lindex). Returns [(x_lo, x_hi, [disjunct indices])]; a single
    group with the global box when no disjunct declares one."""
    groups = {}
    for i, c in enumerate(spec.disjuncts):
        if c.input_lo is not None:
            key = (tuple(np.asarray(c.input_lo).ravel()),
                   tuple(np.asarray(c.input_hi).ravel()))
        else:
            key = None
        groups.setdefault(key, []).append(i)
    out = []
    for key, idxs in groups.items():
        if key is None:
            out.append((spec.x_lo, spec.x_hi, idxs))
        else:
            out.append((np.asarray(key[0]), np.asarray(key[1]), idxs))
    return out


def _log_flush(m):
    print(m, flush=True)


def verify(onnx_path, vnnlib_path, timeout=60.0, device='cpu',
           alpha_iters=20, pgd_budget=5.0, log=_log_flush):
    """Returns (verdict, details); details carries 'witness' for 'sat'.

    Disjuncts carrying their own input subboxes (acasxu prop_6) decompose
    into independent sub-instances: 'sat' if any, 'unsat' iff all."""
    from vibecheck.spec import VNNSpec
    from vibecheck.vnnlib_loader import load_vnnlib
    t0 = time.time()
    try:
        net = load_net(onnx_path)
    except Exception as e:                    # noqa: BLE001 - see re-raise
        # a net the graph loader cannot model: try the discrete-grid
        # handler (cctsdb); if the instance is not discrete either,
        # re-raise the ORIGINAL load error (never silently swallowed)
        from .handlers.discrete_enum import try_discrete_enum
        log(f'[vc2] graph load failed ({type(e).__name__}: {str(e)[:80]}); '
            f'trying discrete-enum handler')
        try:
            return try_discrete_enum(onnx_path, vnnlib_path, timeout, log)
        except NotImplementedError:
            # discrete-enum can't model it either. An unsupported OP in the
            # original load (e.g. smart_turn's quantized/dequantized conv
            # kernels) is a FEATURE BOUNDARY -> honest 'unknown', not a crash;
            # any other load error is a real bug and still becomes 'error'.
            if isinstance(e, NotImplementedError):
                log(f'[vc2] net not supported: {e}; verdict unknown')
                return 'unknown', {
                    'reason': f'not_implemented: {str(e)[:200]}',
                    'time': time.time() - t0}
            raise e
    try:
        spec = load_vnnlib(vnnlib_path)
    except NotImplementedError as e:
        # nonlinear v2 specs (ml4acopf 2.0) need the augment transpiler
        # (handlers); a spec the front end cannot represent is a feature
        # boundary, not a crash
        log(f'[vc2] spec not supported: {e}; verdict unknown')
        return 'unknown', {'reason': f'not_implemented: {e}',
                           'time': time.time() - t0}
    log(f'[vc2] {net}')

    groups = _subbox_groups(spec)
    try:
        return _verify_groups(net, spec, groups, onnx_path, timeout,
                              device, alpha_iters, pgd_budget, log, t0)
    except NotImplementedError as e:
        # an op-coverage gap (vit bmm adjoints until M6, etc.) is a
        # feature boundary, not a crash: log it loudly and return the
        # honest verdict
        log(f'[vc2] not implemented: {e}; verdict unknown')
        return 'unknown', {'reason': f'not_implemented: {e}',
                           'time': time.time() - t0}
    except torch.AcceleratorError as e:
        # raw CUDA allocation failure outside the caching allocator
        # (compile workspace exhaustion poisons subsequent allocs in this
        # process); the phases that ran stand, the verdict is honest
        log(f'[vc2] accelerator failure: {str(e)[:80]}; verdict unknown')
        return 'unknown', {'reason': 'accelerator_failure',
                           'time': time.time() - t0}


def _verify_groups(net, spec, groups, onnx_path, timeout, device,
                   alpha_iters, pgd_budget, log, t0):
    from vibecheck.spec import VNNSpec
    if len(groups) == 1:
        return _verify_one(net, spec, onnx_path, timeout, device,
                           alpha_iters, pgd_budget, log, t0)
    log(f'[vc2] {len(groups)} input-subbox groups (per-disjunct boxes)')
    if len(groups) > 16 and all(len(idxs) == 1 for _, _, idxs in groups):
        # single-row-per-group survivors (nn4sys cardinality: 960 subs
        # over 2 shared W rows): one MULTI-SUB input-split BaB over the
        # stacked subboxes -- the batched bound amortizes across subs,
        # where the serial per-group pipeline got 0.2s each and timed out
        W, b, di = _spec_queries(spec, net.n_out)
        per_d = {}
        for i in range(di.shape[0]):
            per_d.setdefault(int(di[i]), []).append(i)
        if all(len(per_d.get(idxs[0], ())) == 1 for _, _, idxs in groups):
            from .core.search import input_split_bab
            r_lo = torch.tensor(np.stack([np.asarray(g[0]).ravel()
                                          for g in groups]),
                                dtype=torch.float32)
            r_hi = torch.tensor(np.stack([np.asarray(g[1]).ravel()
                                          for g in groups]),
                                dtype=torch.float32)
            r_row = torch.tensor([per_d[idxs[0]][0]
                                  for _, _, idxs in groups])
            verdict, binfo = input_split_bab(
                net, spec, W, b, di, r_lo.min(dim=0).values,
                r_hi.max(dim=0).values,
                deadline=t0 + timeout - 2.0, device=device,
                onnx_path=onnx_path, roots=(r_lo, r_hi, r_row), log=log)
            log(f'[vc2] multi-sub bab: {verdict} '
                f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
            if verdict == 'sat':
                return 'sat', {'witness': binfo['witness'],
                               'time': time.time() - t0}
            if verdict == 'unsat':
                return 'unsat', {'time': time.time() - t0}
            return 'unknown', {'time': time.time() - t0}
    if len(groups) > 16:
        # mega-disjunct screening for the SERIAL per-group path (multi-sub
        # instances never reach here: their BaB's first rounds are the
        # screen, and this pass burned 90s refuting 0/960 on nn4sys)
        groups = _screen_subbox_groups(net, spec, groups, device, log)
        log(f'[vc2] {len(groups)} groups open after batched screening')
    share = ((timeout - (time.time() - t0)) / max(1, len(groups))
             if groups else 0.0)
    for glo, ghi, idxs in groups:
        sub = VNNSpec(x_lo=np.asarray(glo, dtype=np.float64),
                      x_hi=np.asarray(ghi, dtype=np.float64),
                      disjuncts=[spec.disjuncts[i] for i in idxs])
        verdict, details = _verify_one(net, sub, onnx_path, share, device,
                                       alpha_iters, pgd_budget, log,
                                       time.time())
        if verdict != 'unsat':
            details['time'] = time.time() - t0
            return verdict, details
    return 'unsat', {'time': time.time() - t0}


def _screen_subbox_groups(net, spec, groups, device, log):
    """Batched-CROWN refutation screen over subbox groups; returns the
    still-open subset. Sound: only provably-refuted groups are dropped."""
    from .core import backward, memory
    dev = torch.device(device)
    W_all, b_all, di_all = _spec_queries(spec, net.n_out)
    W_all, b_all = W_all.to(dev), b_all.to(dev)
    open_groups = []
    widest = max(net.ops[o].n for o in net.order)
    per_dom = W_all.shape[0] * widest * 4 * 8
    cs = memory.chunk_size(len(groups), per_dom, dev)
    i = 0
    while i < len(groups):
        chunk = groups[i:i + cs]
        lo = torch.tensor(np.stack([g[0] for g in chunk]),
                          dtype=torch.float32, device=dev)
        hi = torch.tensor(np.stack([g[1] for g in chunk]),
                          dtype=torch.float32, device=dev)
        try:
            lbq = backward.crown(net, lo, hi, W_all) + b_all
        except torch.cuda.OutOfMemoryError:
            # the shape-only estimate misses crown internals on mega-row
            # specs (nn4sys cardinality dual); halve and retry -- the one
            # sanctioned OOM catch pattern
            torch.cuda.empty_cache()
            if cs == 1:
                raise
            cs = max(1, cs // 2)
            continue
        i += len(chunk)
        for k, (glo, ghi, idxs) in enumerate(chunk):
            refuted = all(
                bool((lbq[k][di_all == d] > 0).any()) for d in idxs)
            if not refuted:
                open_groups.append((glo, ghi, idxs))
    return open_groups


def _verify_one(net, spec, onnx_path, timeout, device, alpha_iters,
                pgd_budget, log, t0):
    from .core import attack
    from .core.budget import Budget, OutOfTime
    budget = Budget(timeout, margin=0.0)
    budget.t0 = t0
    budget.deadline = t0 + timeout - 2.0

    # Phase A: falsification first (cheap, decides most sat instances).
    # A candidate is only a 'sat' after the ORT chokepoint accepts it.
    #
    # Restart diversity, not iteration count, is the lever on tight-eps sat
    # rows: cifar100 idx_8502 (eps 0.0039) plateaus at margin +0.011 with 100
    # OSI restarts even at 500+ iters, but crosses zero at r=256 -- each OSI
    # restart samples a different output-space basin and one lands in the CE
    # basin.  So escalate restarts on a near-zero MISS; clear cases exit on
    # the first (cheap) pass and never pay for the extra restarts.
    if pgd_budget > 0:
        restarts = 100
        for _ in range(3):                        # 100 -> 250 -> 625
            w, ainfo = attack.pgd(net, spec, device=device, restarts=restarts,
                                  iters=250, init='osi',
                                  time_budget=pgd_budget, log=log)
            if w is not None:
                ok, vinfo = attack.validate(onnx_path, spec, w)
                if ok:
                    w_emit = vinfo.get('witness_inbox', w)
                    return 'sat', {'witness': np.asarray(w_emit),
                                   'time': time.time() - t0}
                log('[vc2] pgd candidate rejected by ORT chokepoint; '
                    'continuing')
            # escalate only on a genuine near-miss (margin just above zero,
            # or a marginal below-zero candidate the chokepoint rejected) with
            # cheap-phase budget to spare; otherwise fall through to bounds.
            if ainfo['best_margin'] >= 0.05 or budget.remaining() < 15:
                break
            restarts = int(restarts * 2.5)

    dev = torch.device(device)
    lo = torch.tensor(spec.x_lo, dtype=torch.float32, device=dev).unsqueeze(0)
    hi = torch.tensor(spec.x_hi, dtype=torch.float32, device=dev).unsqueeze(0)
    W, b, di = _spec_queries(spec, net.n_out)
    W, b = W.to(dev), b.to(dev)

    try:
        inter = backward.intermediates(net, lo, hi)
    except OutOfTime:
        return 'timeout', {'time': time.time() - t0}
    lb0 = backward.crown(net, lo, hi, W, inter)[0]
    verdict, open_d = _verdict_from_lbs(lb0 + b, di, len(spec.disjuncts))
    log(f'[vc2] crown: worst={float((lb0 + b).min()):.4f} '
        f'open={len(open_d)}/{len(spec.disjuncts)}')
    # route by the number of WIDE input dims, not the raw input size:
    # dist_shift is 792-dim with only 8 non-degenerate dims (v1 config
    # split up to 800 dims for exactly this reason). Wide-route instances
    # close by per-leaf zono planes under input splits; the heavy root
    # phases (joint-inter alpha, lift, dual) are wasted work there.
    n_wide = int((hi[0] - lo[0] > 1e-6).sum())
    wide_route = n_wide <= 32
    if verdict != 'unsat':
        # per-edge backward-CROWN refinement whenever disjuncts stay
        # open -- QUALITY-triggered, not memory-triggered. (It was gated
        # on 'zono does not fit', which meant a 23GB card took a WEAKER
        # path than the memory-starved 8GB one: on the A10G the zono fit,
        # the refinement was skipped, the dual state lost 0.03 of root
        # bound and 6 unstable neurons, and idx_8945's 35M-node tree
        # became a 140M-node non-closing one.)
        try:
            inter = backward.intermediates_crown(net, lo, hi,
                                                 base_inter=inter,
                                                 budget=budget)
        except OutOfTime:
            return 'timeout', {'time': time.time() - t0}
        lb0 = torch.maximum(lb0, backward.crown(net, lo, hi, W, inter)[0])
        verdict, open_d = _verdict_from_lbs(lb0 + b, di,
                                            len(spec.disjuncts))
        log(f'[vc2] crown-inter: worst={float((lb0 + b).min()):.4f} '
            f'open={len(open_d)}/{len(spec.disjuncts)}')
    if verdict != 'unsat' and alpha_iters > 0:
        lb = backward.alpha_crown(net, lo, hi, W, inter,
                                  iters=alpha_iters, thresholds=-b,
                                  budget=budget)[0]
        lb = torch.maximum(lb, lb0)
        verdict, open_d = _verdict_from_lbs(lb + b, di, len(spec.disjuncts))
        log(f'[vc2] alpha-crown: worst={float((lb + b).min()):.4f} '
            f'open={len(open_d)}/{len(spec.disjuncts)}')
        worst = float((lb + b).min())
        n_nonlin = sum(net.ops[nm].n for nm in net.order
                       if net.ops[nm].kind == 'nonlin')
    from .core import debug as _dbg
    if _dbg.enabled():
        _dbg.add('W', W)
        _dbg.add('bias', b)
        _dbg.add('spec_lb', (lb + b) if alpha_iters > 0 else (lb0 + b))
        _dbg.add('inter', {k: (v[0], v[1]) for k, v in inter.items()
                           if isinstance(v, tuple) and len(v) == 2})
    # MILP eligibility (shallow relu-only nets: big-M is tight only for <=2
    # relu layers -- see the depth gate rationale). Computed here, before the
    # wide route and the dual, because the exact MILP is the DECISIVE tool for
    # this class (malbeware, safenlp) and must not be starved: the wide route's
    # input-split floor (max(20s,...)) or the dual can eat a short (T=20)
    # budget whole before it is ever reached.
    n_nonlin_m = sum(net.ops[nm].n for nm in net.order
                     if net.ops[nm].kind == 'nonlin')
    n_relu_layers = sum(1 for op in net.ops.values()
                        if op.kind == 'nonlin')
    relu_only = all(op.fn == 'relu' for op in net.ops.values()
                    if op.kind == 'nonlin')
    milp_eligible = (relu_only and n_nonlin_m <= 25_000
                     and n_relu_layers <= 2)

    def _try_milp(open_d, deadline):
        """Triangle-exact MILP escalation. Returns ('sat', witness_np) |
        ('unsat', None) | ('open', remaining_open_disjuncts). Refutation uses
        the solver DUAL bound (sound at any point); an incumbent is only a
        candidate and is validated through the ORT chokepoint."""
        from .core.milp import refute_rows_milp
        rows_open = [r for d in open_d
                     for r in torch.nonzero(di == d, as_tuple=False)
                     .flatten().tolist()]
        try:
            mrefuted, cand = refute_rows_milp(net, lo, hi, inter, W, b,
                                              rows_open, deadline=deadline,
                                              log=log)
        except NotImplementedError as e:
            log(f'[vc2/milp] skipped: {e}')
            return 'open', open_d
        if cand is not None and onnx_path is not None:
            from .core import attack
            okc, vinfo = attack.validate(onnx_path, spec, cand)
            if okc:
                return 'sat', np.asarray(vinfo.get('witness_inbox', cand))
            log('[vc2/milp] incumbent rejected by ORT chokepoint')
        rem = [d for d in open_d
               if not any(int(r) in mrefuted for r in
                          torch.nonzero(di == d, as_tuple=False)
                          .flatten().tolist())]
        log(f'[vc2] milp: {len(mrefuted)} rows refuted, {len(rem)} open')
        return ('unsat', None) if not rem else ('open', rem)

    if verdict != 'unsat' and open_d and milp_eligible \
            and budget.remaining() > 8:
        mv, mres = _try_milp(open_d, time.time()
                             + min(0.65 * budget.remaining(), 60.0))
        if mv == 'sat':
            return 'sat', {'witness': mres, 'time': time.time() - t0}
        if mv == 'unsat':
            return 'unsat', {'time': time.time() - t0}
        open_d = mres                             # partial: continue with rest

    if verdict != 'unsat' and open_d and wide_route:
        # wide route: most such instances close under input splits with
        # per-leaf zono planes in seconds (dist_shift, acasxu), so try
        # that FIRST on a budget slice; the borderline rows that instead
        # need lift + dual (index112: v1 spends 96.8s there) fall through
        # to the normal heavy pipeline below on a miss
        from .core.search import input_split_bab
        # two-stage alpha escalation: the boundary-alpha depth is a
        # knife-edge lever (iso instance_3 closes at 8 iters and dies at
        # 20; instance_31 the exact opposite -- deeper alpha both closes
        # marginal leaves AND steers splits off-tree via the adopted
        # linearizations). Stage 1 costs nothing when it wins.
        for ai, frac in ((8, 0.7), (20, 0.9)):
            slice_end = time.time() + max(20.0, frac * budget.remaining())
            verdict, binfo = input_split_bab(
                net, spec, W, b, di, lo[0], hi[0],
                deadline=min(t0 + timeout - 2.0, slice_end), device=device,
                alpha_iters=ai, onnx_path=onnx_path, log=log)
            log(f'[vc2] input_split_bab (wide slice, alpha={ai}): '
                f'{verdict} '
                f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
            if verdict == 'sat':
                return 'sat', {'witness': binfo['witness'],
                               'time': time.time() - t0}
            if verdict == 'unsat':
                return 'unsat', {'time': time.time() - t0}
        verdict = 'unknown'
    if verdict != 'unsat' and alpha_iters > 0:
        if verdict != 'unsat' and n_nonlin <= 20000 \
                and budget.remaining() > 15:
            # joint-intermediate alpha refresh (v1 phase-0.5): re-derive the
            # intermediate bounds with alpha-optimized identity rows, then
            # rerun the spec alpha (measured on dist_shift: root -11.3 ->
            # v1-level with this; fixed intermediates were the ceiling)
            inter = backward.intermediates_crown(net, lo, hi,
                                                 base_inter=inter,
                                                 alpha_iters=12,
                                                 budget=budget)
            lb_j = backward.alpha_crown(net, lo, hi, W, inter,
                                        iters=alpha_iters, thresholds=-b,
                                        budget=budget)[0]
            lb = torch.maximum(lb, lb_j)
            lb0 = torch.maximum(lb0, lb)
            verdict, open_d = _verdict_from_lbs(lb + b, di,
                                                len(spec.disjuncts))
            log(f'[vc2] joint-inter alpha: worst={float((lb + b).min()):.4f} '
                f'open={len(open_d)}/{len(spec.disjuncts)}')
            worst = float((lb + b).min())
        if verdict != 'unsat' and -1.0 < worst <= 0 and budget.remaining() > 20:
            # near-zero gap: a longer, lower-lr polish often closes it
            # outright (abcrown runs ~100 root iters; the quick pass is
            # 20). Iterations scale with the net: 150 crown passes on a
            # 460k-neuron conv net is ~2.5 minutes -- it silently ate the
            # whole budget on challenging_certified and the dual + BaB
            # got literally zero seconds.
            p_iters = 150 if n_nonlin <= 50_000 else 30
            lb2 = backward.alpha_crown(net, lo, hi, W, inter, iters=p_iters,
                                       lr=0.1, thresholds=-b,
                                       budget=budget)[0]
            lb = torch.maximum(lb, lb2)
            verdict, open_d = _verdict_from_lbs(lb + b, di,
                                                len(spec.disjuncts))
            log(f'[vc2] alpha-polish: worst={float((lb + b).min()):.4f} '
                f'open={len(open_d)}/{len(spec.disjuncts)}')
    if verdict != 'unsat' and len(open_d) == 1 and n_nonlin <= 20000:
        # zono-lift (v1 phase 2.5): exact box+halfspace LP tightening of
        # every pre-activation under the open disjunct's own output rows.
        # Region-conditional, hence scoped to the single-open-disjunct case
        # where refuting that region IS the instance.
        from .core.dual_lp import lift_intermediates
        rows_d = torch.nonzero(di == open_d[0],
                               as_tuple=False).flatten().tolist()
        try:
            inter = lift_intermediates(
                net, lo, hi, inter,
                cut_rows=[(W[r].cpu().numpy(), float(b[r]))
                          for r in rows_d],
                device=device, budget=budget, log=log)
            lb0 = torch.maximum(lb0, backward.crown(net, lo, hi, W, inter)[0])
            lb_l = backward.alpha_crown(net, lo, hi, W, inter,
                                        iters=alpha_iters, thresholds=-b,
                                        budget=budget)[0]
            lb0 = torch.maximum(lb0, lb_l)
            verdict, open_d = _verdict_from_lbs(lb0 + b, di,
                                                len(spec.disjuncts))
            log(f'[vc2] zono-lift: worst={float((lb0 + b).min()):.4f} '
                f'open={len(open_d)}/{len(spec.disjuncts)}')
            worst_l = float((lb0 + b).min())
            if verdict != 'unsat' and -1.0 < worst_l <= 0 \
                    and budget.remaining() > 20:
                # the pre-lift polish never saw the lifted intermediates;
                # a near-zero post-lift gap often closes under the same
                # long low-lr pass (cifar100 idx_8945: -0.23 after lift)
                lb2 = backward.alpha_crown(net, lo, hi, W, inter,
                                           iters=(150 if n_nonlin <= 50_000
                                                  else 30),
                                           lr=0.1, thresholds=-b,
                                           budget=budget)[0]
                lb0 = torch.maximum(lb0, lb2)
                verdict, open_d = _verdict_from_lbs(lb0 + b, di,
                                                    len(spec.disjuncts))
                log(f'[vc2] lift-polish: worst={float((lb0 + b).min()):.4f} '
                    f'open={len(open_d)}/{len(spec.disjuncts)}')
        except NotImplementedError as e:
            log(f'[vc2] zono-lift skipped ({e})')
    if verdict != 'unsat':
        # dual-ascent LP certifier (compiled GPU BaB over the alpha-zono
        # state, ported v1 fast_dual_ascent): the strongest per-query
        # refuter. The state builds BACKWARD (unstable rows only, chunked),
        # so no forward-zonotope gate; survivors fall through to BaB.
        # (MILP-eligible nets already had their exact shot ABOVE, before the
        # wide route, so the dual needs no reserve here.)
        from .core.dual_lp import certify_queries
        refuted = certify_queries(
            net, spec, W, b, di, lo, hi, inter, open_d,
            deadline=t0 + timeout - 2.0, device=device, log=log)
        open_d = [d for d in open_d if d not in refuted]
        log(f'[vc2] dual-lp: {len(refuted)} disjuncts refuted, '
            f'{len(open_d)} open')
        if not open_d:
            return 'unsat', {'time': time.time() - t0}
    if verdict != 'unsat' and open_d and milp_eligible \
            and budget.remaining() > 8:
        # second exact-MILP shot: the intermediates are now tighter (joint-
        # inter alpha + lift), so rows the early attempt left open may close.
        mv, mres = _try_milp(open_d, time.time()
                             + min(0.8 * budget.remaining(), 60.0))
        if mv == 'sat':
            return 'sat', {'witness': mres, 'time': time.time() - t0}
        if mv == 'unsat':
            return 'unsat', {'time': time.time() - t0}
        open_d = mres
    if verdict != 'unsat':
        # branch and bound: input splits for low-dimensional inputs, relu
        # phase splits otherwise (unified scoring across both is the design
        # target; the two loops share bound/attack machinery meanwhile)
        from .core.search import input_split_bab, relu_split_bab
        kw = {}
        if n_wide <= 32:
            bab = input_split_bab
        else:
            bab = relu_split_bab
            kw['root_inter'] = inter        # the crown-refined root bounds
        verdict, binfo = bab(
            net, spec, W, b, di, lo[0], hi[0],
            deadline=t0 + timeout - 2.0, device=device,
            onnx_path=onnx_path, log=log, **kw)
        log(f'[vc2] {bab.__name__}: {verdict} '
            f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
        if verdict == 'sat':
            return 'sat', {'witness': binfo['witness'],
                           'time': time.time() - t0}
        if verdict == 'timeout':
            verdict = 'unknown'
    return verdict, {'open_disjuncts': open_d, 'time': time.time() - t0}


def main(argv=None):
    """Minimal CLI mirroring v1's verdict conventions for parity harnesses."""
    import argparse
    p = argparse.ArgumentParser(prog='vibecheck2')
    p.add_argument('--net', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--results-file', default=None)
    a = p.parse_args(argv)
    if a.net.lstrip().startswith('['):
        # network-pair instance (isomorphic/monotonic acasxu): reuse the v1
        # front end to merge the pair into one onnx + v1 spec (exact,
        # ORT-oracle-gated), then verify normally (design: frontends port)
        from vibecheck import network_pair as npair
        a.net, a.spec = npair.build_merged_instance(a.net, a.spec)
    else:
        # nonlinear v2 spec (adaptive_cruise): v1's ORT-oracle-gated
        # transpile to an augmented onnx + linear v1 spec. NOTE: an unsat on
        # the augmented instance is sound for the original; a sat witness is
        # re-validated by the chokepoint on the AUGMENTED net here, and the
        # strict original-spec disposition is handler work (v1
        # _sat_disposition), so borderline CEs may differ from v1 for now.
        from vibecheck import nonlinear_augment as nla
        try:
            text = nla._read_vnnlib_text(a.spec)
        except (OSError, ValueError):
            text = ''
        if text and nla.is_nonlinear_v2_spec(text):
            a.net, a.spec = nla.build_augmented_instance(a.net, a.spec)
    if a.results_file:                        # pre-seed like v1
        with open(a.results_file, 'w') as f:
            f.write('timeout\n')
    try:
        verdict, details = verify(a.net, a.spec, a.timeout, a.device)
    except BaseException as e:                # crash -> 'error' (v1 discipline)
        import traceback
        traceback.print_exc()
        if a.results_file:
            with open(a.results_file, 'w') as f:
                f.write(f'error\n{type(e).__name__}: {str(e)[:300]}\n')
        return 2
    if a.results_file:
        ce = None
        if verdict == 'sat' and details.get('witness') is not None:
            # v1's CE formatting: version/io names resolved from the spec,
            # Y recomputed by the same ORT forward the scorer replays
            from vibecheck.main import (_counterexample_sexpr,
                                        _resolve_cex_io_meta,
                                        _vnnlib_version)
            from vibecheck.vnnlib_loader import load_vnnlib
            ce = _counterexample_sexpr(
                a.net, load_vnnlib(a.spec), details['witness'],
                version=_vnnlib_version(a.spec),
                io_meta=_resolve_cex_io_meta(a.spec))
        tmp = a.results_file + '.tmp'
        with open(tmp, 'w') as f:
            f.write(verdict + '\n')
            if ce is not None:
                f.write(ce + '\n')
        os.replace(tmp, a.results_file)
    print(f'[vc2] verdict: {verdict}  ({details["time"]:.2f}s)')
    return 0 if verdict == 'unsat' else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
