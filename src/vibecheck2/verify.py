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


def _peel_output_softmax(net, spec, log):
    """Drop a terminal softmax when every spec row is a pure difference
    (one +1, one -1, zero bias): softmax is strictly monotone per row, so
    softmax(z)_i - softmax(z)_j and z_i - z_j have the same sign and the
    property is EQUIVALENT on the logits (ab-crown's
    peel_off_last_softmax_layer). Measured need (traffic idx_10645, an
    OFFICIAL-ab sat row): the saturated softmax pins pgd at margin +1.0
    with zero gradient, and interval/crown explode to 1e30 through it.
    Returns the (possibly) rewritten net."""
    out_op = net.ops.get(net.output_name)
    if out_op is None or out_op.kind != 'nonlin' or out_op.fn != 'softmax':
        return net
    n_out = out_op.n
    try:
        rows = spec.as_linear_queries(n_out)
    except Exception:                # noqa: BLE001 -- conservatively skip
        return net
    for _, w, bias in rows:
        wv = np.asarray(w)
        nz = wv[wv != 0]
        if abs(float(bias)) > 0 or len(nz) != 2 \
                or sorted(nz.tolist()) != [-1.0, 1.0]:
            return net
    pre = out_op.inputs[0]
    order = [nm for nm in net.order if nm != net.output_name]
    ops = {nm: op for nm, op in net.ops.items() if nm != net.output_name}
    from .core.graph import Net as _Net
    net2 = _Net(ops, order, net.input_name, pre,
                onnx_path=getattr(net, 'onnx_path', None))
    log('[vc2] terminal softmax peeled (pure-difference spec rows: the '
        'property is equivalent on the logits)')
    return net2


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


_T0 = time.time()


def _log_flush(m):
    # elapsed-time stamp: phase budget forensics need to be readable off
    # any run log (the vit misses were diagnosed blind without this)
    print(f'[{time.time() - _T0:6.1f}s] {m}', flush=True)


def verify(onnx_path, vnnlib_path, timeout=60.0, device='cpu',
           alpha_iters=20, pgd_budget=5.0, net_cache=None, attack_off=False,
           log=_log_flush):
    """Returns (verdict, details); details carries 'witness' for 'sat'.

    Disjuncts carrying their own input subboxes (acasxu prop_6) decompose
    into independent sub-instances: 'sat' if any, 'unsat' iff all."""
    from .frontend.spec import VNNSpec
    from .frontend.vnnlib_loader import load_vnnlib
    t0 = time.time()
    try:
        if net_cache and os.path.exists(net_cache):
            # VNNCOMP prepare-stage cache: the conversion ran once in
            # prepare; the timed run deserializes (mscn_2048d measured
            # 5.7s of onnx load against a 20s budget)
            net = torch.load(net_cache, weights_only=False)
        else:
            net = load_net(onnx_path)
            if net_cache:
                tmp_c = net_cache + '.tmp'
                torch.save(net, tmp_c)
                os.replace(tmp_c, net_cache)
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
    net = _peel_output_softmax(net, spec, log)
    log(f'[vc2] {net}')

    groups = _subbox_groups(spec)
    if attack_off:
        # SOUNDNESS sweep: disable ALL falsification -- no root PGD
        # (pgd_budget 0) and no CE validation anywhere (onnx_path None
        # gates the in-BaB attack + every attack.validate). With no
        # attack a counterexample can never be found, so a truly-sat
        # instance can only return unknown/timeout; an 'unsat' here is a
        # false-unsat SOUNDNESS violation.
        onnx_path = None
        pgd_budget = 0.0
    from .core.budget import OutOfTime
    try:
        return _verify_groups(net, spec, groups, onnx_path, timeout,
                              device, alpha_iters, pgd_budget, log, t0)
    except OutOfTime:
        # the cooperative deadline fired inside a phase that does not
        # wrap its own budget.check() (stabilize_intermediates ->
        # intermediates_crown on a large ResNet). Budget exhausted =
        # honest 'unknown'/timeout, never a crash-to-error.
        log('[vc2] budget exhausted (OutOfTime); verdict unknown')
        return 'unknown', {'reason': 'out_of_time',
                           'time': time.time() - t0}
    except NotImplementedError as e:
        # an op-coverage gap (vit bmm adjoints until M6, etc.) is a
        # feature boundary, not a crash: log it loudly and return the
        # honest verdict
        log(f'[vc2] not implemented: {e}; verdict unknown')
        return 'unknown', {'reason': f'not_implemented: {e}',
                           'time': time.time() - t0}
    except (torch.AcceleratorError, torch.cuda.OutOfMemoryError) as e:
        # raw CUDA allocation failure outside the caching allocator
        # (compile workspace exhaustion poisons subsequent allocs) OR a
        # CUDA OOM the chunked memory service could not absorb (a large
        # ResNet's dense adjoint on a 22GB card -- tinyimagenet). The
        # phases that ran stand; the HONEST verdict is unknown (vc2 ran
        # out of memory), never a crash-to-error.
        torch.cuda.empty_cache()
        log(f'[vc2] cuda oom/accelerator failure: {str(e)[:80]}; '
            f'verdict unknown')
        return 'unknown', {'reason': 'cuda_oom_or_accelerator',
                           'time': time.time() - t0}


def _verify_groups(net, spec, groups, onnx_path, timeout, device,
                   alpha_iters, pgd_budget, log, t0):
    from .frontend.spec import VNNSpec
    if len(groups) == 1:
        return _verify_one(net, spec, onnx_path, timeout, device,
                           alpha_iters, pgd_budget, log, t0)
    log(f'[vc2] {len(groups)} input-subbox groups (per-disjunct boxes)')
    if len(groups) > 16:
        # MULTI-SUB input-split BaB: one root per (subbox, single-row
        # disjunct). The batched bound amortizes across all subs where a
        # serial per-group pipeline times out (nn4sys cardinality 960; lindex
        # 120k). A subbox may carry SEVERAL disjuncts (lindex: 2/box) -- each
        # becomes its OWN root, so the per-group disjunct count no longer
        # matters (the old code required exactly 1 and otherwise fell to a
        # screen+serial loop that bounded every group vs ALL query rows and
        # stalled). Requires every disjunct single-row: its row is the root's
        # refutation target.
        W, b, di = _spec_queries(spec, net.n_out)
        per_d = {}
        for i in range(di.shape[0]):
            per_d.setdefault(int(di[i]), []).append(i)
        if all(len(per_d.get(d, ())) == 1
               for _, _, idxs in groups for d in idxs):
            from .core.search import input_split_bab
            r_lo, r_hi, r_row = [], [], []
            for glo, ghi, idxs in groups:
                glo = np.asarray(glo).ravel()
                ghi = np.asarray(ghi).ravel()
                for d in idxs:
                    r_lo.append(glo)
                    r_hi.append(ghi)
                    r_row.append(per_d[d][0])
            r_lo = torch.tensor(np.stack(r_lo), dtype=torch.float32)
            r_hi = torch.tensor(np.stack(r_hi), dtype=torch.float32)
            r_row = torch.tensor(r_row)
            n_roots = r_lo.shape[0]
            # MINI-GROUP admission (v1 mini_group_size): input_split_bab keeps
            # only ~mg subboxes active on the frontier at once, admitting the
            # next wave as they close. One shared frontier over ALL roots splits
            # every open sub each round and explodes (mscn: 145k leaves, never
            # converges) even though the per-sub bound is fine -- v1's fast
            # CROWN also closes ~0% up front and wins purely by bounding the
            # peak frontier. Done INSIDE the BaB so the weight-dedup + setup is
            # paid ONCE, not per wave (the caller-loop repeated it and was 3x
            # slower). mg=200 matches v1; small root sets never explode anyway.
            # mg=200 (v1 parity). mg=500 MEASURED WORSE with the cheap
            # rad-mode bound (mscn_2048d: frontier exploded 4.6k -> 18.5k
            # while bounded stayed flat; the cap IS the tree discipline)
            mg = 200 if n_roots > 500 else None
            verdict, binfo = input_split_bab(
                net, spec, W, b, di, r_lo.min(dim=0).values,
                r_hi.max(dim=0).values, deadline=t0 + timeout - 2.0,
                device=device, onnx_path=onnx_path,
                roots=(r_lo, r_hi, r_row), mini_group=mg,
                # multi-sub closes by SPLITTING (v1's mini-group BaB is pure
                # forward-LiRPA + split, no per-leaf alpha); the 8-iter alpha
                # per round cost 15x the bound and closed nothing extra here.
                alpha_iters=0, log=log)
            log(f'[vc2] multi-sub bab: {verdict} '
                f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
            if verdict == 'sat':
                return 'sat', {'witness': binfo['witness'],
                               'time': time.time() - t0}
            if verdict == 'unsat':
                return 'unsat', {'time': time.time() - t0}
            det = {'time': time.time() - t0}
            if binfo.get('tol_witness') is not None:
                log('[vc2] verdict unknown BUT a within-tolerance CE was '
                    'found (strict-violation policy keeps unknown)')
                det['within_tol_ce'] = np.asarray(binfo['tol_witness'])
            return 'unknown', det
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
    has_mixed = any((op.kind == 'nonlin' and op.fn != 'relu')
                    or op.kind in ('mul', 'bmm')
                    for op in net.ops.values())
    fz_alphas = None      # fzono's optimized band slopes (BaB warm start)
    root_alphas = None    # the polish phase's optimized CROWN alphas (the
    # BaB's per-domain alpha/beta pass and the kFSB probe warm-start here)
    fz_gain = True        # did the zono frame ever beat the crown chain?
    # (True when fzono did not run: absence of evidence keeps the
    # measured-on-vit zono-BaB default for mixed nets)
    tol_w = None          # within-tolerance witness: emitted ONLY if the
                          # pipeline ends without a strict verdict (v1's
                          # variant sats land at timeout, not at phase A)

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
                ok, vinfo = attack.validate(onnx_path, spec, w, log=log)
                if ok:
                    w_emit = vinfo.get('witness_inbox', w)
                    return 'sat', {'witness': np.asarray(w_emit),
                                   'time': time.time() - t0}
                if vinfo.get('within_tol_witness') is not None:
                    tol_w = vinfo['within_tol_witness']
                log('[vc2] pgd candidate rejected by ORT chokepoint; '
                    'continuing')
            # escalate only on a genuine near-miss (margin just above zero,
            # or a marginal below-zero candidate the chokepoint rejected) with
            # cheap-phase budget to spare; otherwise fall through to bounds.
            if ainfo['best_margin'] >= 0.05 or budget.remaining() < 15:
                break
            if any((op.kind == 'nonlin' and op.fn != 'relu')
                   or op.kind in ('mul', 'bmm')
                   for op in net.ops.values()):
                # the near-miss escalation is the hidden-CE lesson and
                # those live in small pure-relu nets (soundnessbench:
                # crosses zero at r=256). Mixed nets sit near-zero
                # because they are HARD UNSAT rows (vit: +0.012 margins
                # on all 12 rows); two extra OSI rounds cost 8s of the
                # exact budget the BaB endgame needs.
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
    verdict, open_d = None, []

    def _phase(tag, lbv):
        """Uniform phase epilogue: verdict from the running bound + the
        one-line log every phase used to hand-roll."""
        nonlocal verdict, open_d
        verdict, open_d = _verdict_from_lbs(lbv + b, di,
                                            len(spec.disjuncts))
        log(f'[vc2] {tag}: worst={float((lbv + b).min()):.4f} '
            f'open={len(open_d)}/{len(spec.disjuncts)}')

    _phase('crown', lb0)
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
        _phase('crown-inter', lb0)
    if verdict != 'unsat' and alpha_iters > 0:
        lb = backward.alpha_crown(net, lo, hi, W, inter,
                                  iters=alpha_iters, thresholds=-b,
                                  budget=budget)[0]
        lb = torch.maximum(lb, lb0)
        _phase('alpha-crown', lb)
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
    relu_only = all(op.fn == 'relu' for op in net.ops.values()
                    if op.kind == 'nonlin')
    n_relu_layers = sum(1 for op in net.ops.values()
                        if op.kind == 'nonlin')
    # gate on the UNSTABLE count -- that is the MILP's binary count (milp.py),
    # not the full relu width. malbeware 16-25 has a wide layer but few
    # unstable neurons; keying on full width wrongly gated it out.
    n_unstable = sum(int(((inter[nm][0] < 0) & (inter[nm][1] > 0)).sum())
                     for nm in net.order
                     if net.ops[nm].kind == 'nonlin'
                     and net.ops[nm].fn == 'relu') if relu_only else 0
    milp_eligible = (relu_only and n_unstable <= 25_000
                     and n_relu_layers <= 2)

    def _try_milp(open_d, deadline):
        """Triangle-exact MILP escalation. Returns ('sat', witness_np) |
        ('unsat', None) | ('open', remaining_open_disjuncts). Refutation uses
        the solver DUAL bound (sound at any point); an incumbent is only a
        candidate and is validated through the ORT chokepoint."""
        from .core.milp import refute_rows_milp
        try:
            mrefuted, cand = refute_rows_milp(net, lo, hi, inter, W, b,
                                              di, open_d, deadline=deadline,
                                              log=log)
        except NotImplementedError as e:
            log(f'[vc2/milp] skipped: {e}')
            return 'open', open_d
        if cand is not None and onnx_path is not None:
            from .core import attack
            nonlocal tol_w
            okc, vinfo = attack.validate(onnx_path, spec, cand, log=log)
            if okc:
                return 'sat', np.asarray(vinfo.get('witness_inbox', cand))
            if vinfo.get('within_tol_witness') is not None:
                tol_w = vinfo['within_tol_witness']
            log('[vc2/milp] incumbent rejected by ORT chokepoint')
        rem = [d for d in open_d if int(d) not in mrefuted]
        log(f'[vc2] milp: {len(mrefuted)} disjuncts refuted, {len(rem)} open')
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
            if budget.remaining() < 5:
                # a 20s-budget instance (nn4sys mscn) measured 6.4s of
                # OVERRUN from stage 2 + the dual + the final BaB all
                # launching after the deadline had already passed
                break
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
            if binfo.get('tol_witness') is not None:
                tol_w = binfo['tol_witness']
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
            _phase('joint-inter alpha', lb)
            worst = float((lb + b).min())
        if verdict != 'unsat' and -5.0 < worst <= 0 and budget.remaining() > 20:
            # near-zero gap: a longer, lower-lr polish often closes it
            # outright (abcrown runs ~100 root iters; the quick pass is
            # 20). Gate widened -1 -> -5: TinyYOLO prop_000024 sat at
            # -4.43 with the 20-iter alpha barely moving (-4.448 ->
            # -4.433) while ab's ~100-iter root closes the row in 38s.
            # Iterations still scale with the net (150 crown passes on a
            # 460k-neuron conv net is ~2.5 minutes -- it silently ate the
            # whole budget on challenging_certified once).
            p_iters = 150 if n_nonlin <= 50_000 else 30
            lb2, root_alphas = backward.alpha_crown(net, lo, hi, W, inter,
                                                    iters=p_iters,
                                                    lr=0.1, thresholds=-b,
                                                    budget=budget,
                                                    return_alpha=True)
            lb = torch.maximum(lb, lb2[0])
            _phase('alpha-polish', lb)
    if verdict != 'unsat' and budget.remaining() > 15 \
            and any((op.kind == 'nonlin' and op.fn != 'relu')
                    or op.kind == 'mul' for op in net.ops.values()):
        # forward-zono alpha (v1's nl_alpha): jointly optimize the band
        # slope of EVERY nonlinearity (relu, sigmoid/tanh, sin/cos/pow,
        # exp, reciprocal) over the differentiable forward zonotope,
        # against the worst-open-disjunct margin. On mixed-nonlinearity
        # nets this beats the whole backward stack: 0298 measured -11.84
        # backward -> +2.2e-7 here (216/216 disjuncts, ~30s), matching
        # v1. float64 like v1: the closing margins are ~1e-7, below
        # float32 resolution at output scale. UNclamped: CROWN clamps
        # measurably trap the optimizer (-0.24 vs closed, same net).
        # Admission by ATTEMPT under the centralized OOM policy: no
        # faithful predictive estimate exists for the zonotope footprint
        # (its generator count depends on band widths only the propagation
        # knows: vit measured 10 GiB real vs ~130 GiB from BOTH a shape
        # estimate and an interval-replay estimate, each wrongly gating
        # the phase out while the dual sat one disjunct from closing).
        from .core import memory
        try:
            # slice + iters sized for the 300-bus nets (measured):
            # 300base_p3 closes 276/276 at ~95s, p4 needs 600/600 at
            # ~278s and >400 iterations -- a 150s/200-iter cap left 71
            # open. The loop self-terminates on close/plateau, so the
            # wide caps only cost time on nets it was losing anyway
            # (and the dual/BaB measurably never rescue this class).
            # TWO-STAGE by precision (measured on the A10G: fp64 runs
            # at ~1/30 of fp32 there, so 40 stall iterations of the f64
            # vit forward alone cost 56s and starved the dual/BaB to 8s
            # each). Stage 1 optimizes in f32; only when its optimum
            # lands within f32 forward noise of closing (the ml4acopf
            # regime: closing margins ~1e-7, far below f32 resolution
            # at output scale) does the f64 stage run, warm-started
            # from the f32 alphas. A row whose f32 optimum sits clearly
            # negative (vit: -0.04) can never be resolved by precision.
            sub = Budget(min(0.35 * budget.remaining(), 150.0),
                         margin=0.0)
            fz = memory.attempt(
                lambda: forward.alpha_zono(net, lo, hi, W, iters=1000,
                                           thresholds=-b,
                                           budget=sub, disj_idx=di,
                                           return_alphas=True,
                                           known=lb0.reshape(1, -1),
                                           abort_on_gain=True),
                tag='fzono-alpha')
            lb_f, fz_alphas = (None, None) if fz is None \
                else (fz[0][0].double(), fz[1])
            if lb_f is not None:
                gain32 = bool((lb_f > lb0.double() + 1e-12).any())
                # escalate to f64 iff the zono frame LEADS (gain32; on
                # crown-led nets precision cannot help a bound fzono
                # didn't produce -- vit 1151 measured 38s of waste) AND
                # either f32 closed MOST of the gap but stalled short
                # (relative: p3 went -36.36 -> -0.0145, then the f32
                # noise floor blocks the last 26 disjuncts -- an
                # absolute 1e-2 band missed this by 1.45x and REGRESSED
                # the sentinel) or it claims a razor-thin closure that
                # needs the f64 confirmation the always-f64 phase gave.
                if gain32:
                    # the zono frame LEADS: the f32 stage was only the
                    # probe (it aborts on first gain); the real
                    # optimization is the COLD f64 run with the full
                    # slice -- the regime that closes ml4acopf p3
                    # 276/276 at ~95s (warm f32->f64 measured 1/276
                    # short, and a near_done gate on the f32 optimum
                    # regressed the sentinel twice)
                    sub = Budget(min(0.7 * budget.remaining(), 300.0),
                                 margin=0.0)
                    fz = memory.attempt(
                        lambda: forward.alpha_zono(
                            net, lo.double(), hi.double(), W.double(),
                            iters=1000, thresholds=(-b).double(),
                            budget=sub, disj_idx=di, return_alphas=True,
                            known=lb0.double().reshape(1, -1)),
                        tag='fzono-alpha64')
                    if fz is not None:
                        lb_f, fz_alphas = fz[0][0], fz[1]
            if lb_f is None:
                log('[vc2] fzono-alpha skipped (oom)')
            else:
                # did the zono frame beat the crown chain ANYWHERE? This
                # decides the BaB bound engine below: fzono returns
                # max(own, known), so equality with lb0 means the crown
                # chain leads on this instance (the post-adjoint vit
                # regime; ab-crown closes those rows with relu-split
                # beta-CROWN, which is exactly the crown-mode BaB)
                fz_gain = bool((lb_f > lb0.double() + 1e-12).any())
                verdict, open_d = _verdict_from_lbs(
                    lb_f + b.double(), di, len(spec.disjuncts))
                # feed downstream f32 phases a DIRECTED-rounded cast (a
                # nearest-cast could round the bound up = unsound)
                lb0 = torch.maximum(lb0, torch.nextafter(
                    lb_f.to(lb0.dtype),
                    torch.full_like(lb0, -torch.inf)))
                log(f'[vc2] fzono-alpha: '
                    f'worst={float((lb_f + b.double()).min()):.4e} '
                    f'open={len(open_d)}/{len(spec.disjuncts)} '
                    f'gain={fz_gain}')
        except NotImplementedError as e:
            # an op without a forward band yet is a feature boundary
            # for this escalation only; the phases below still run
            log(f'[vc2] fzono-alpha skipped ({e})')
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
            _phase('zono-lift', lb0)
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
                _phase('lift-polish', lb0)
        except NotImplementedError as e:
            log(f'[vc2] zono-lift skipped ({e})')
    if verdict != 'unsat' and budget.remaining() > 5:
        # dual-ascent LP certifier (compiled GPU BaB over the alpha-zono
        # state, ported v1 fast_dual_ascent): the strongest per-query
        # refuter. (A temporary skip on the vit route was REVERTED: with
        # the attention backward adjoint live, the dual's state is tight
        # and it closed 1151's last disjunct in 324 nodes/0.02s; its
        # stall risk is capped by the 55% slice reserve below.) The state builds BACKWARD (unstable rows only, chunked),
        # so no forward-zonotope gate; survivors fall through to BaB.
        # (MILP-eligible nets already had their exact shot ABOVE, before the
        # wide route, so the dual needs no reserve here.)
        from .core.dual_lp import certify_queries
        # reserve a BaB slice: the dual's per-query BnB happily consumes
        # the whole remaining budget failing on a stubborn disjunct (vit
        # 1151: ~30s of time_limit nodes, then input_split_bab got 0
        # bounds where v1 closes it in 21 boxes/25s). The dual keeps 55%
        # of what is left; survivors fall through with real time.
        # On the crown-BaB route (mixed net, crown chain leads, few open
        # disjuncts) the slice is SHORT: measured on all 12 vit rows the
        # dual there either kills its disjunct instantly (7 rows, <=324
        # nodes, <1s) or diverges (5 rows: frontier 0.6M-4M and growing
        # at 25s), while the crown BaB bounds ~140 domains/s and is the
        # engine that officially closes this class -- every dual second
        # past the instant-kill window is stolen from the closer.
        # (12 -> 8: the kills cost ~6.5s state build + <1s search; the
        # extra 4s only ever fed diverging frontiers.)
        crown_bab_route = (len(open_d) <= 2 and has_mixed
                           and not fz_gain)
        dual_slice = (min(8.0, 0.3 * budget.remaining())
                      if crown_bab_route else 0.55 * budget.remaining())
        refuted = certify_queries(
            net, spec, W, b, di, lo, hi, inter, open_d,
            deadline=min(t0 + timeout - 2.0, time.time() + dual_slice),
            device=device, log=log)
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
    if verdict != 'unsat' and open_d and relu_only \
            and n_unstable <= 2000 and budget.remaining() > 20:
        # split-and-tighten stabilization (v1 bab_refine): tighten the
        # INTERMEDIATES under targeted sign splits until root alpha closes;
        # the frontier BaB below measurably explodes exactly where this
        # converges (relusplitter model_2_2: v1 33s, vc2 BaB frontier 39k)
        from .core.search import stabilize_intermediates
        inter = stabilize_intermediates(net, W, lo, hi, inter, budget,
                                        device=device, log=log)
        lb_s = backward.alpha_crown(net, lo, hi, W, inter,
                                    iters=max(alpha_iters, 45),
                                    thresholds=-b, budget=budget)[0]
        lb0 = torch.maximum(lb0, lb_s)
        _phase('stabilize', lb0)
    if verdict != 'unsat' and budget.remaining() > 3:
        # branch and bound: input splits for low-dimensional inputs, relu
        # phase splits otherwise (unified scoring across both is the design
        # target; the two loops share bound/attack machinery meanwhile).
        # EXCEPTION (v1 zono-input-split, vit_2023.yaml + measured on vit
        # 1151): when only a FEW disjuncts survive the root phases AND the
        # net has non-relu nonlinearities, input splitting wins even on
        # wide inputs -- v1 closes its last vit disjunct in 21 boxes/25s.
        # Relu splits cannot address bmm/exp looseness at all, while the
        # input-split scorer picks the handful of dims the open rows
        # depend on. Pure-relu wide nets keep the relu route (cifar
        # idx_8945's 35M-node relu tree is the counter-case).
        from .core.search import input_split_bab, relu_split_bab
        kw = {}
        if n_wide <= 32:
            bab = input_split_bab
        elif len(open_d) <= 2 and has_mixed and fz_gain:
            # zono-bounded BaB: only when the zono frame measurably LEADS
            # the crown chain on this instance (fz_gain). Where the crown
            # chain leads (vit post-softmax-adjoint: crown -0.029 vs zono
            # -0.043 root), the default crown-mode BaB below is the
            # ab-crown regime that officially closes those rows.
            bab = relu_split_bab
            kw['root_inter'] = inter
            kw['bound'] = 'zono'
            kw['batch'] = 16    # measured on vit: B=16 fits (8.4 GiB), B=64
            # OOMs. (box_remainder=True measured 2.9x faster/domain here
            # but HALVED bound quality: -0.083 vs -0.039 on 2157 -- the
            # merged columns' cross-element cancellation through the
            # residual adds is load-bearing. Dense stays.)
            # the root fzono's optimized band slopes: bounding domains
            # with DEFAULT bands measured ~10x looser (-0.46-quality vs
            # the root's -0.043) and the tree barely pruned
            if fz_alphas is not None:
                kw['warm_alphas'] = {nm: t.float()
                                     for nm, t in fz_alphas.items()}
        else:
            bab = relu_split_bab
            kw['root_inter'] = inter        # the crown-refined root bounds
        if bab is relu_split_bab and kw.get('bound') != 'zono' \
                and root_alphas:
            kw['root_alphas'] = root_alphas
            # refine regime for the deep-tree class: with per-domain
            # alpha transfer, lr 0.1's bias-corrected first steps THRASH
            # the inherited state back to the cold 12-iter plateau
            # (2157 ckpt replays: cold/transfer@0.1 both flat -0.0136;
            # transfer@0.03 reaches -0.0075 by round 32 and falling,
            # beating 30 cold iters at 12-iter cost)
            kw['beta_lr'] = 0.03
        ckpt = os.environ.get('VC2_BAB_CKPT')
        if ckpt:
            # dev harness: freeze everything the BaB consumes so search
            # changes iterate WITHOUT re-running ~50s of root phases
            # (scratch/clean_slate/bab_from_ckpt.py replays it)
            torch.save({'W': W, 'b': b, 'di': di, 'lo': lo[0],
                        'hi': hi[0], 'kw': kw, 'bab': bab.__name__,
                        'onnx_path': onnx_path}, ckpt)
            log(f'[vc2] BaB checkpoint saved to {ckpt}')
        verdict, binfo = bab(
            net, spec, W, b, di, lo[0], hi[0],
            deadline=t0 + timeout - 2.0, device=device,
            onnx_path=onnx_path, log=log, **kw)
        log(f'[vc2] {bab.__name__}: {verdict} '
            f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
        if verdict == 'sat':
            return 'sat', {'witness': binfo['witness'],
                           'time': time.time() - t0}
        if binfo.get('tol_witness') is not None:
            tol_w = binfo['tol_witness']
        if verdict == 'timeout':
            verdict = 'unknown'
    det = {'open_disjuncts': open_d, 'time': time.time() - t0}
    if verdict != 'unsat' and tol_w is not None:
        # a within-tolerance CE exists but the verdict stays HONEST
        # (strict violations only -- policy). Flag it loudly: the
        # official scorer would accept this witness at its
        # COUNTEREXAMPLE_ATOL, so the row is one acceptance-policy
        # decision away from sat (ml4acopf linear variants, vc1's
        # accepted CEs are exactly these box-vertex points).
        log('[vc2] verdict unknown BUT a within-tolerance CE was found '
            '(margin <= ce_tol; strict-violation policy keeps unknown)')
        det['within_tol_ce'] = np.asarray(tol_w)
    return verdict, det


def main(argv=None):
    """Minimal CLI mirroring v1's verdict conventions for parity harnesses."""
    import argparse
    p = argparse.ArgumentParser(prog='vibecheck2')
    p.add_argument('--net', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--results-file', default=None)
    p.add_argument('--ce-tol', type=float, default=1e-4,
                   help='within-tolerance CE DETECTION band (flag-only; '
                        'verdicts always require strict violation)')
    p.add_argument('--net-cache', default=None,
                   help='converted-net cache path: load when present, '
                        'else convert and save (VNNCOMP prepare stage)')
    p.add_argument('--no-attack', action='store_true',
                   help='disable ALL falsification (soundness sweep): no '
                        'counterexample can be found, so sat instances '
                        'return unknown/timeout; an unsat is a false-unsat')
    a = p.parse_args(argv)
    from .core import attack as _atk
    _atk.CE_TOL = a.ce_tol
    if a.net.lstrip().startswith('['):
        # network-pair instance (isomorphic/monotonic acasxu): reuse the v1
        # front end to merge the pair into one onnx + v1 spec (exact,
        # ORT-oracle-gated), then verify normally (design: frontends port)
        from .frontend import network_pair as npair
        a.net, a.spec = npair.build_merged_instance(a.net, a.spec)
    else:
        # nonlinear v2 spec (adaptive_cruise): v1's ORT-oracle-gated
        # transpile to an augmented onnx + linear v1 spec. NOTE: an unsat on
        # the augmented instance is sound for the original; a sat witness is
        # re-validated by the chokepoint on the AUGMENTED net here, and the
        # strict original-spec disposition is handler work (v1
        # _sat_disposition), so borderline CEs may differ from v1 for now.
        from .frontend import nonlinear_augment as nla
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
        verdict, details = verify(a.net, a.spec, a.timeout, a.device,
                                  net_cache=a.net_cache,
                                  attack_off=a.no_attack)
    except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError) as e:
        # a CUDA OOM anywhere the chunked memory service could not absorb
        # (a large ResNet's dense adjoint deep in the BaB, past verify()'s
        # own catch) is an HONEST 'unknown' -- vc2 ran out of memory --
        # never a crash-to-error. Belt to verify()'s suspenders.
        import traceback
        traceback.print_exc()
        try:
            torch.cuda.empty_cache()
        except Exception:                     # noqa: BLE001 - best effort
            pass
        if a.results_file:
            with open(a.results_file, 'w') as f:
                f.write('unknown\n')
        print(f'[vc2] cuda oom (top-level): verdict unknown ({str(e)[:80]})')
        return 1
    except BaseException as e:                # crash -> 'error' (v1 discipline)
        import traceback
        traceback.print_exc()
        # a RuntimeError carrying an OOM message (some CUDA paths raise the
        # base RuntimeError, not the OOM subclass) is also unknown-not-error
        if 'out of memory' in str(e).lower():
            if a.results_file:
                with open(a.results_file, 'w') as f:
                    f.write('unknown\n')
            print('[vc2] oom-message runtime error (top-level): unknown')
            return 1
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
            from .frontend.vnnlib_loader import load_vnnlib
            from .frontend.witness import (_counterexample_sexpr,
                                           _resolve_cex_io_meta,
                                           _vnnlib_version)
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
