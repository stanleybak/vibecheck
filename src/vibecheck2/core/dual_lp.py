"""Dual-ascent LP leaf certifier (design 3.1 dual_lp): the vc2 adapter for
v1's `fast_dual_ascent` GPU-compiled BaB verifier (log-bucket line search,
far-probe infeasibility cert, torch.compile-fused kernels; measured ~50x
over the Gurobi-backed racing in v1, hundreds of thousands to millions of
node bounds per second compiled).

The verifier consumes the alpha-zonotope LP state of one query: the query
value as c0 + d . e over generator coordinates e in [-1,1]^n, plus per
unstable-relu substitution data (pre-activation row, band coefficients,
fresh column). vc2's forward zonotope already carries exactly this
(ZonoState with symbol provenance); `build_state` snapshots it into the v1
schema.

COLUMN ORDER SOUNDNESS: v1's parser boxes only the first `n_input` columns
and the relu fresh columns to [-1,1] and PINS everything else to 0. Any
non-relu fresh generator (sigmoid/tanh bands, bilinear boxes) is therefore
moved INTO the leading free block, and `n_input` covers the whole block.
"""
from __future__ import annotations

import numpy as np
import torch

from . import forward as fwd


class _GenGeometry:
    """The alpha-zono generator geometry of one box: per-nonlin band
    coefficients (lam, mu, delta) from the given bounds, a global column
    layout (inputs first, then non-relu band columns -- the FREE block --
    then relu fresh columns), a slope-linear center pass, and a chunked
    backward row builder (LinMap-generic; no forward zonotope).

    Column-order soundness: v1's LP parser boxes [-1,1] only the leading
    `n_input` columns and the relu e_new columns; everything splittable
    must be a relu column and everything else must live in the free block.
    """

    def __init__(self, net, lo, hi, inter, slopes=None, device='cpu'):
        from .relax import REL
        self.net = net
        dev = torch.device(device)
        dt = torch.float32
        self.dev, self.dt = dev, dt
        lo2 = lo.reshape(1, -1).to(dev, dt)
        hi2 = hi.reshape(1, -1).to(dev, dt)
        self.radii = ((hi2 - lo2) / 2)[0]
        self.center_in = ((lo2 + hi2) / 2)[0]
        n_in = net.n_in

        self.nonlin = [nm for nm in net.order
                       if net.ops[nm].kind == 'nonlin']
        self.lam, self.mu, self.delta, self.fresh = {}, {}, {}, {}
        for nm in self.nonlin:
            op = net.ops[nm]
            rel = REL[op.fn]
            if not hasattr(rel, 'band'):
                raise NotImplementedError(f'gen geometry: no band for {op.fn}')
            entry = inter[nm]
            if len(entry) != 2:
                raise NotImplementedError(
                    f'gen geometry: {op.fn} inter entry has {len(entry)} '
                    f'fields (bilinear ops use the forward builder)')
            l, h = entry[0][0].to(dev, dt), entry[1][0].to(dev, dt)
            if slopes and nm in slopes and op.fn == 'relu':
                a = slopes[nm].reshape(-1).to(dev, dt).clamp(0.0, 1.0)
                lam = torch.where(l >= 0, torch.ones_like(l),
                                  torch.where(h <= 0, torch.zeros_like(l), a))
                mu = torch.where((l < 0) & (h > 0),
                                 torch.maximum((1 - lam) * h, -lam * l) / 2,
                                 torch.zeros_like(l))
                delta = mu
            else:
                lam, mu, delta = rel.band(l.unsqueeze(0), h.unsqueeze(0),
                                          op.params)
                lam, mu, delta = lam[0], mu[0], delta[0]
            self.lam[nm], self.mu[nm], self.delta[nm] = lam, mu, delta
            self.fresh[nm] = torch.nonzero(delta > 0,
                                           as_tuple=False).flatten()
        # column layout: inputs, then non-relu band cols (free), then relu
        self.e_col = {}
        col = n_in
        for nm in self.nonlin:
            if net.ops[nm].fn != 'relu':
                for j in self.fresh[nm].tolist():
                    self.e_col[(nm, j)] = col
                    col += 1
        self.n_free = col
        for nm in self.nonlin:
            if net.ops[nm].fn == 'relu':
                for j in self.fresh[nm].tolist():
                    self.e_col[(nm, j)] = col
                    col += 1
        self.n_gens = col
        self.n_in = n_in

        # slope-linear center pass (nonlin -> lam*z + mu)
        center = {net.input_name: self.center_in}
        self.pre_center = {}
        for name in net.order:
            op = net.ops[name]
            if op.kind == 'linmap':
                center[name] = op.lm.point(
                    center[op.inputs[0]].unsqueeze(0))[0]
            elif op.kind == 'nonlin':
                z = center[op.inputs[0]]
                self.pre_center[name] = z
                center[name] = self.lam[name] * z + self.mu[name]
            elif op.kind == 'add':
                center[name] = (center[op.inputs[0]]
                                + center[op.inputs[1]])
            elif op.kind == 'concat':
                out = torch.as_tensor(op.params['base'], device=dev,
                                      dtype=dt).clone()
                for src, pos in zip(op.inputs, op.params['positions']):
                    out[torch.as_tensor(pos, device=dev)] = center[src]
                center[name] = out
            else:
                raise NotImplementedError(
                    f'gen geometry center: {op.kind}/{op.fn}')
        self.center = center

    def rows(self, seed_edge, seed_idx, self_nonlin=None):
        """(len(seed_idx), n_gens) generator rows of the seeded neurons via
        one slope-linear backward pass (delta deposited on fresh cols)."""
        net, dev, dt = self.net, self.dev, self.dt
        ns = len(seed_idx)
        rowG = torch.zeros(ns, self.n_gens, device=dev, dtype=dt)
        sens = {seed_edge: torch.zeros(ns, net.ops[seed_edge].n,
                                       device=dev, dtype=dt)}
        sens[seed_edge][torch.arange(ns, device=dev),
                        torch.as_tensor(seed_idx, device=dev)] = 1.0
        for name in reversed(net.order):
            if name not in sens:
                continue
            sx = sens.pop(name)
            op = net.ops[name]
            if op.kind == 'linmap':
                add = op.lm.lin_t(sx)
            elif op.kind == 'nonlin':
                if name != self_nonlin:
                    u = self.fresh[name]
                    if u.numel():
                        cols = torch.as_tensor(
                            [self.e_col[(name, int(j))] for j in u.tolist()],
                            device=dev)
                        rowG[:, cols] += sx[:, u] \
                            * self.delta[name][u].unsqueeze(0)
                    sx = sx * self.lam[name].unsqueeze(0)
                add = sx
            elif op.kind == 'add':
                sens[op.inputs[1]] = sens.get(op.inputs[1], 0) + sx
                add = sx
            elif op.kind == 'concat':
                for src, pos in zip(op.inputs, op.params['positions']):
                    p = torch.as_tensor(pos, device=dev)
                    sens[src] = sens.get(src, 0) + sx[:, p]
                continue
            else:
                raise NotImplementedError(f'gen rows: {op.kind}/{op.fn}')
            sens[op.inputs[0]] = sens.get(op.inputs[0], 0) + add
        s_in = sens.get(net.input_name)
        if s_in is not None:
            rowG[:, :self.n_in] += s_in * self.radii.unsqueeze(0)
        return rowG

    def rows_chunked(self, seed_edge, seed_idx, self_nonlin=None):
        from . import memory
        widest = max(self.net.ops[o].n for o in self.net.order)
        acc = []

        def take(sel):
            acc.append(self.rows(seed_edge, sel.tolist(), self_nonlin))

        memory.chunked_indices(take, torch.as_tensor(seed_idx,
                                                     device=self.dev),
                               widest * 4 * 6)
        return torch.cat(acc)


def build_state_backward(net, lo, hi, inter, slopes=None, device='cpu'):
    """The alpha-zono LP state built backward (v1 reverse_g port), now for
    ANY banded net: non-relu band columns land in the free block (boxed
    [-1,1] by the parser via n_input), relu columns stay splittable."""
    import scipy.sparse as sp
    geo = _GenGeometry(net, lo, hi, inter, slopes=slopes, device=device)
    unstable_list = []
    for nm in geo.nonlin:
        if net.ops[nm].fn != 'relu' or not geo.fresh[nm].numel():
            continue
        u = geo.fresh[nm]
        rowG = geo.rows_chunked(net.ops[nm].inputs[0], u, nm).cpu().numpy()
        for i, j in enumerate(u.tolist()):
            nz = np.nonzero(rowG[i])[0]
            unstable_list.append({
                'layer_idx': nm, 'neuron_idx': int(j),
                'lam': float(geo.lam[nm][j]), 'mu': float(geo.mu[nm][j]),
                'c_in': float(geo.pre_center[nm][j]),
                'e_new_col': geo.e_col[(nm, int(j))],
                'row_indices': nz.tolist(),
                'row_values': rowG[i, nz].astype(np.float64).tolist(),
            })
    obj_G = geo.rows_chunked(net.output_name,
                             list(range(net.n_out))).cpu().numpy()
    state = {
        'n_gens': int(geo.n_gens), 'n_input': int(geo.n_free),
        'unstable_list': unstable_list,
        'obj_G_out_csr': sp.csr_matrix(obj_G.astype(np.float64)),
        'obj_c_out': geo.center[net.output_name].cpu().numpy()
        .astype(np.float64),
    }
    keys = [(u['layer_idx'], u['neuron_idx']) for u in unstable_list]
    return state, keys


def lift_intermediates(net, lo, hi, inter, cut_rows, rounds=3,
                       device='cpu', log=lambda m: None):
    """v1 phase-2.5 zono-lift port: tighten every nonlinearity's
    pre-activation bounds by the EXACT box+one-halfspace LP
    (v1 box_halfspace.lagrangian_min, closed form) under the spec cut
    "a counterexample satisfies w.y + b <= 0", then rebuild the geometry
    with the shrunken bands and repeat. Bounds only ever tighten; each
    round's LP is sound on the CE region, so the result is scoped to
    refuting the supplied rows (same discipline as gamma).

    cut_rows: [(w, b)] output rows that must be <= 0 at a counterexample.
    """
    from vibecheck.box_halfspace import lagrangian_min
    inter = dict(inter)
    for rnd in range(rounds):
        geo = _GenGeometry(net, lo, hi, inter, device=device)
        # the cut in generator coordinates (use the FIRST row; iterating
        # rows each round would also be valid)
        w, bcut = cut_rows[0]
        obj = geo.rows_chunked(net.output_name, list(range(net.n_out)))
        w_t = torch.as_tensor(np.asarray(w), device=geo.dev, dtype=geo.dt)
        a_cut = (w_t @ obj).cpu().numpy().astype(np.float64)
        c_out = geo.center[net.output_name]
        beta = float(-(float(w_t @ c_out) + float(bcut)))
        improved = 0.0
        for nm in geo.nonlin:
            e = net.ops[nm].inputs[0]
            n = net.ops[e].n
            rows = geo.rows_chunked(e, list(range(n)), nm).cpu().numpy()
            c_pre = geo.pre_center[nm].cpu().numpy()
            l0, h0 = inter[nm]
            l0 = l0.clone()
            h0 = h0.clone()
            for j in range(n):
                lo_j = lagrangian_min(rows[j], c_pre[j], a_cut, beta)
                hi_j = -lagrangian_min(-rows[j], -c_pre[j], a_cut, beta)
                if lo_j > float(l0[0, j]):
                    improved += lo_j - float(l0[0, j])
                    l0[0, j] = lo_j
                if hi_j < float(h0[0, j]):
                    improved += float(h0[0, j]) - hi_j
                    h0[0, j] = hi_j
            h0 = torch.maximum(h0, l0)
            inter[nm] = (l0, h0)
        log(f'[vc2/lift] round {rnd}: total tightening {improved:.3f}')
        if improved < 1e-3:
            break
    return inter


def build_state(net, lo, hi, inter=None, slopes=None):
    """One recorded zonotope pass -> the v1 gen-state dict (single box).
    `inter` (CROWN-refined pre-activation bounds) clamps every band, which
    is what makes the LP state competitive with v1's tightened states.

    Returns (state, scored_key_universe) where the universe lists every
    splittable (relu_name, neuron) key present in the state.
    """
    import scipy.sparse as sp
    record = {}
    clamp = None
    if inter is not None:
        clamp = {k: (v[0], v[1]) for k, v in inter.items()
                 if len(v) == 2}
    _lo, _hi, zstate = fwd.zono(net, lo, hi, return_state=True,
                                record=record, clamp_bounds=clamp,
                                slope_override=slopes)
    out = zstate[net.output_name]
    final_sym = out.sym
    n_gens = len(final_sym)

    # permute columns: all always-free generators (input + non-relu bands)
    # first, relu fresh columns last (see module docstring)
    relu_names = set(record)
    free_cols = [i for i, s in enumerate(final_sym) if s[0] not in relu_names]
    relu_cols = [i for i, s in enumerate(final_sym) if s[0] in relu_names]
    perm = free_cols + relu_cols
    colmap = {final_sym[i]: k for k, i in enumerate(perm)}
    n_input = len(free_cols)

    G_out = out.G[0].cpu().numpy()[:, perm]
    obj_c_out = out.c[0].cpu().numpy().astype(np.float64)

    unstable_list = []
    for name, rec in record.items():
        c_pre = rec['c_pre'][0].cpu().numpy()
        G_pre = rec['G_pre'][0].cpu().numpy()
        lam = rec['lam'][0].cpu().numpy()
        mu = rec['mu'][0].cpu().numpy()
        # columns of this snapshot in final coordinates
        cols = np.array([colmap[s] for s in rec['sym']], dtype=np.int64)
        fresh = [(j, colmap.get((name, j))) for j in range(c_pre.shape[0])]
        for j, col in fresh:
            if col is None:
                continue                      # stable neuron: no fresh col
            row = G_pre[j]
            nz = np.nonzero(row)[0]
            unstable_list.append({
                'layer_idx': name, 'neuron_idx': int(j),
                'lam': float(lam[j]), 'mu': float(mu[j]),
                'c_in': float(c_pre[j]), 'e_new_col': int(col),
                'row_indices': cols[nz].tolist(),
                'row_values': row[nz].astype(np.float64).tolist(),
            })
    state = {
        'n_gens': int(n_gens), 'n_input': int(n_input),
        'unstable_list': unstable_list,
        'obj_G_out_csr': sp.csr_matrix(G_out.astype(np.float64)),
        'obj_c_out': obj_c_out,
    }
    keys = [(u['layer_idx'], u['neuron_idx']) for u in unstable_list]
    return state, keys


_VERIFIER = {}


def _host_ram_room():
    """Bytes the process can still safely allocate on the host: global
    MemAvailable intersected with the cgroup-v2 ceiling when one is set
    (systemd-run -p MemoryMax=... scopes -- /proc/meminfo alone would
    happily report 50G inside a 6G scope and get us OOM-killed)."""
    avail = None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    try:
        with open('/proc/self/cgroup') as f:
            cg = f.read().strip().rsplit('::', 1)[-1]
        with open(f'/sys/fs/cgroup{cg}/memory.max') as f:
            mx = f.read().strip()
        if mx != 'max':
            with open(f'/sys/fs/cgroup{cg}/memory.current') as f:
                room = int(mx) - int(f.read())
            avail = room if avail is None else min(avail, room)
    except (OSError, ValueError, IndexError):
        pass
    return avail if avail is not None else (4 << 30)


def _make_host_frontier_verifier(base_cls):
    """Subclass v1's compiled dual Verifier to hold the BaB frontier in HOST
    memory. The stock loop keeps sides (B, d) int8 PLUS the per-node
    warm-start duals lam0/lam1 (B, d) float and nu (B, M) float resident on
    GPU, with masked + doubled copies at each depth -- ~3GB at the 7.8M-open
    frontier where cifar100 idx_8945 died (reason=oom) on the 8GB part.
    Here the frontier lives in RAM; the GPU sees per-chunk uploads inside
    `_bounds` (compute-bound kernel, upload is noise). Kernel math and
    chunk-halving are inherited unchanged."""
    import time as _time
    from vibecheck.fast_dual_ascent.fast_verify_topk import (
        _TOL, _PARENT_FLOOR, _DeadlineExceeded)

    class HostFrontierVerifier(base_cls):
        def _bounds(self, F, sides, lam0, lam1, nu, deadline=None):
            if sides.device.type != 'cpu':      # warmup / small direct calls
                return base_cls._bounds(self, F, sides, lam0, lam1, nu,
                                        deadline=deadline)
            B = sides.shape[0]
            best = torch.empty(B)
            o0 = torch.empty_like(lam0)
            o1 = torch.empty_like(lam1)
            onu = torch.empty_like(nu)
            dev = self.device
            i = 0
            while i < B:
                if deadline is not None and _time.perf_counter() > deadline:
                    raise _DeadlineExceeded()
                step = min(self.chunk, B - i)
                while True:
                    try:
                        bb, l0, l1, nv = self._kernel(
                            F,
                            sides[i:i + step].to(dev).long().contiguous(),
                            lam0[i:i + step].to(dev).contiguous(),
                            lam1[i:i + step].to(dev).contiguous(),
                            nu[i:i + step].to(dev).contiguous())
                        best[i:i + step] = bb.cpu()
                        o0[i:i + step] = l0.cpu()
                        o1[i:i + step] = l1.cpu()
                        onu[i:i + step] = nv.cpu()
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if step <= 256:
                            raise
                        step //= 2
                        self.chunk = max(256, min(self.chunk, step))
                i += step
            return best, o0, o1, onu

        def verify(self, prob, *, time_limit=120.0, verbose=False,
                   stop_event=None):
            # v1's loop with a HYBRID frontier: per-node tensors (sides,
            # lam0, lam1, nu, floor) stay on GPU at full stock throughput
            # until the projected footprint nears free VRAM, then spill to
            # host once and continue via the chunk-uploading _bounds above
            # (slower per node, but the alternative was reason=oom at ~8M
            # open; tinyimagenet-class runs that fit VRAM never spill and
            # keep the stock speed).
            dev = self.device
            G = self._upload(prob)
            M = int(G['hs_a'].shape[0])
            for D in self._warm_depths:
                if D <= prob.n_splits and D not in self._warmed:
                    z = torch.zeros(8, D, device=dev)
                    self._kernel(self._F(G, D),
                                 torch.zeros(8, D, device=dev,
                                             dtype=torch.long),
                                 z, z.clone(),
                                 torch.zeros(8, M, device=dev))
                    self._warmed.add(D)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
            elapsed = lambda: _time.perf_counter() - t0     # noqa: E731
            deadline = t0 + time_limit
            if prob.root_bound > 0:
                return 'unsat', dict(nodes=0, depth=0, peak_frontier=0,
                                     wall=0.0)
            sides = torch.tensor([[0], [1]], device=dev, dtype=torch.int8)
            lam0 = torch.zeros(2, 1, device=dev)
            lam1 = torch.zeros(2, 1, device=dev)
            nu = torch.zeros(2, M, device=dev)
            floor = torch.full((2,), float('-inf'), device=dev)
            on_host = dev.type != 'cuda'
            nodes_total = 0
            depth = 1
            peak = 2
            # segment stash: survivors awaiting their split, parked on CPU
            # when building all children at once would blow RAM. BFS breadth
            # doubles per depth, but it need not be RESIDENT: resolve one
            # segment to exhaustion (its subtrees certify and free), then
            # pop the next. Peak memory = segment size, not tree breadth.
            stash = []

            def _unknown(open_n, reason):
                return 'unknown', dict(nodes=nodes_total, depth=depth,
                                       peak_frontier=peak, open=int(open_n),
                                       reason=reason, wall=elapsed())

            def _open_total():
                return sides.shape[0] + sum(s[1].shape[0] * 2
                                            for s in stash)

            while sides.shape[0] > 0 or stash:
                if sides.shape[0] == 0:
                    d0, ss, l0, l1, nk, pb = stash.pop()
                    z8 = torch.zeros(ss.shape[0], 1, dtype=torch.int8)
                    zf = torch.zeros(ss.shape[0], 1)
                    sides = torch.cat([torch.cat([ss, z8], 1),
                                       torch.cat([ss, z8 + 1], 1)], 0)
                    lam0 = torch.cat([torch.cat([l0, zf], 1),
                                      torch.cat([l0, zf], 1)], 0)
                    lam1 = torch.cat([torch.cat([l1, zf], 1),
                                      torch.cat([l1, zf], 1)], 0)
                    nu = torch.cat([nk, nk], 0)
                    floor = torch.cat([pb, pb])
                    depth = d0 + 1
                    on_host = True
                if elapsed() > time_limit:
                    return _unknown(_open_total(), 'time_limit')
                if stop_event is not None and stop_event.is_set():
                    return _unknown(sides.shape[0], 'stopped')
                nodes_total += sides.shape[0]
                peak = max(peak, sides.shape[0])
                try:
                    best, o0, o1, onu = self._bounds(
                        self._F(G, depth), sides, lam0, lam1, nu,
                        deadline=deadline)
                    if _PARENT_FLOOR:
                        best = torch.maximum(best, floor)
                    keep = best <= _TOL
                    ss = sides[keep]
                    l0, l1, nk = o0[keep], o1[keep], onu[keep]
                    pb = best[keep]
                    if ss.shape[0] == 0:
                        return 'unsat', dict(nodes=nodes_total, depth=depth,
                                             peak_frontier=peak,
                                             wall=elapsed())
                    if depth >= prob.n_splits:
                        return _unknown(ss.shape[0], 'splits_exhausted')
                    # per-node bytes: sides d + lam0/lam1 8d + nu 4M +
                    # floor 4, x2 children x2 cat temporaries
                    need = 2 * 2 * ss.shape[0] * (9 * (depth + 1) + 4 * M + 4)
                    if not on_host:
                        free_b, _ = torch.cuda.mem_get_info(dev)
                        if need > 0.4 * free_b:
                            ss, l0, l1 = ss.cpu(), l0.cpu(), l1.cpu()
                            nk, pb = nk.cpu(), pb.cpu()
                            on_host = True
                    if on_host and need > _host_ram_room() * 0.4:
                        # split the survivors instead of bailing: build
                        # children for a RAM-sized head, stash the tail
                        # (bounded parents; their split is a cheap cat on
                        # resume). Bail only when even a minimal segment
                        # cannot fit.
                        per_node = 2 * 2 * (9 * (depth + 1) + 4 * M + 4)
                        h = int(_host_ram_room() * 0.4 // per_node)
                        if h < 1024:
                            return _unknown(_open_total(), 'host_ram_cap')
                        if h < ss.shape[0]:
                            stash.append((depth, ss[h:].cpu(), l0[h:].cpu(),
                                          l1[h:].cpu(), nk[h:].cpu(),
                                          pb[h:].cpu()))
                            ss, l0, l1 = ss[:h], l0[:h], l1[:h]
                            nk, pb = nk[:h], pb[:h]
                    z8 = torch.zeros(ss.shape[0], 1, device=ss.device,
                                     dtype=torch.int8)
                    zf = torch.zeros(ss.shape[0], 1, device=ss.device)
                    sides = torch.cat([torch.cat([ss, z8], 1),
                                       torch.cat([ss, z8 + 1], 1)], 0)
                    lam0 = torch.cat([torch.cat([l0, zf], 1),
                                      torch.cat([l0, zf], 1)], 0)
                    lam1 = torch.cat([torch.cat([l1, zf], 1),
                                      torch.cat([l1, zf], 1)], 0)
                    nu = torch.cat([nk, nk], 0)
                    floor = torch.cat([pb, pb])
                except _DeadlineExceeded:
                    return _unknown(sides.shape[0], 'time_limit')
                except torch.cuda.OutOfMemoryError:
                    # kernel could not fit even a 256-node chunk: resource
                    # failure of the sanctioned halving path, surfaced as
                    # 'oom' exactly like v1's loop
                    open_n = int(sides.shape[0])
                    torch.cuda.empty_cache()
                    return 'unknown', dict(nodes=nodes_total, depth=depth,
                                           peak_frontier=peak, open=open_n,
                                           reason='oom', wall=elapsed())
                depth += 1
            return 'unsat', dict(nodes=nodes_total, depth=depth,
                                 peak_frontier=peak, wall=elapsed())

    return HostFrontierVerifier


def _verifier(device):
    """One compiled Verifier per device (kernel warm-up is reused)."""
    if device not in _VERIFIER:
        from vibecheck.fast_dual_ascent import Verifier
        cls = _make_host_frontier_verifier(Verifier)
        _VERIFIER[device] = cls(device=device,
                                compile=(torch.device(device).type
                                         == 'cuda'))
    return _VERIFIER[device]


def score_keys(net, lo, hi, W_open, inter, keys):
    """BaBSR split order for the state's keys: |pre-activation adjoint| x
    triangle intercept from one collected-adjoint crown pass, descending."""
    from . import backward
    adj = {}
    backward.crown(net, lo, hi, W_open, inter, collect_adjoints=adj)
    scores = {}
    for name, j in keys:
        l, h = inter[name]
        icpt = float((-h[0, j] * l[0, j]
                      / max(float(h[0, j] - l[0, j]), 1e-30)))
        a = adj.get(name)
        w = float(a[0, :, j].abs().max()) if a is not None else 1.0
        scores[(name, j)] = w * max(icpt, 0.0)
    return sorted(keys, key=lambda k: -scores[k])


def _state_for(net, lo, hi, inter, slopes, device):
    """Backward state build with the forward-recorded fallback for nets
    carrying non-slope-linear ops (mul/sigmoid free-block generators)."""
    try:
        return build_state_backward(net, lo, hi, inter, device=device,
                                    slopes=slopes)
    except NotImplementedError:
        return build_state(net, lo, hi, inter=inter,
                           slopes={k: v.unsqueeze(0)
                                   for k, v in slopes.items()})


def range_split_dual(net, lo, hi, inter, qw, qb, extra, ver, deadline,
                     slopes, device, log, max_depth=10, leaf_time=3.0):
    """Refute one query on nets whose slack lives in SMOOTH free-block
    generators (sigmoid/tanh bands): a small best-first BaB over smooth-op
    RANGE splits, each leaf certified by the dual with its state rebuilt
    under the tightened pre-activation ranges (tighter range -> smaller
    band -> smaller free generator -> the dual can close).

    Sound: every leaf state is built from intersected TRUE bounds, and the
    disjunction of children covers the parent exactly.
    """
    import heapq
    import time

    from .relax import REL
    smooth = [nm for nm in net.order
              if net.ops[nm].kind == 'nonlin'
              and net.ops[nm].fn in ('sigmoid', 'tanh', 'exp', 'reciprocal')]
    if not smooth:
        return 'unknown', {'reason': 'no smooth edges'}

    def leaf(inter_d):
        # refresh DOWNSTREAM bounds under the tightened ranges (an interval
        # reforward intersected with the parent's refined bounds); without
        # this the relu substitutions stay stale and no leaf ever tightens
        from . import backward
        rc = {nm: inter_d[nm] for nm in smooth}
        ib_state = fwd.interval(net, lo, hi, return_state=True,
                                range_clamps=rc)
        ib = backward._inter_from_state(net, lambda e: ib_state[e])
        inter_leaf = {}
        for k, v in inter_d.items():
            iv = ib[k]
            merged = []
            for j2 in range(0, len(v), 2):
                merged.append(torch.maximum(v[j2], iv[j2]))
                merged.append(torch.minimum(v[j2 + 1],
                                            torch.maximum(iv[j2 + 1],
                                                          merged[-1])))
            inter_leaf[k] = tuple(merged)
        state, keys = _state_for(net, lo, hi, inter_leaf, slopes, device)
        if not keys:
            return 'unknown', {}
        sk = score_keys(net, lo, hi, torch.as_tensor(qw).unsqueeze(0),
                        inter_leaf, keys)
        return ver.verify_query(state, qw, qb, sk,
                                time_limit=min(leaf_time,
                                               deadline - time.time()),
                                extra_hs=extra)

    def pick_split(inter_d):
        best = None
        for nm in smooth:
            l, h = inter_d[nm]
            _lam, _mu, delta = REL[net.ops[nm].fn].band(l, h,
                                                        net.ops[nm].params)
            v, j = delta[0].max(dim=0)
            if best is None or float(v) > best[0]:
                best = (float(v), nm, int(j))
        return best

    heap = [(0, 0, inter)]                     # (depth, tiebreak, inter)
    tick = 1
    leaves = 0
    while heap:
        if time.time() > deadline:
            return 'unknown', {'reason': 'time', 'leaves': leaves}
        depth, _, inter_d = heapq.heappop(heap)
        verdict, info = leaf(inter_d)
        leaves += 1
        if verdict == 'unsat':
            continue                           # this region refuted
        if depth >= max_depth:
            return 'unknown', {'reason': 'depth', 'leaves': leaves}
        _v, nm, j = pick_split(inter_d)
        l, h = inter_d[nm]
        mid = float((l[0, j] + h[0, j]) / 2)
        for lo_j, hi_j in ((float(l[0, j]), mid), (mid, float(h[0, j]))):
            child = dict(inter_d)
            l2, h2 = l.clone(), h.clone()
            l2[0, j], h2[0, j] = lo_j, hi_j
            child[nm] = (l2, h2)
            heapq.heappush(heap, (depth + 1, tick, child))
            tick += 1
    return 'unsat', {'leaves': leaves}


def certify_queries(net, spec, W, bias, disj_idx, lo, hi, inter, open_d,
                    deadline, device='cpu', log=lambda m: None):
    """Refute the still-open disjuncts with the dual-ascent BaB, one query
    row at a time (sibling rows of the disjunct join as extra halfspaces).
    Returns the set of disjuncts refuted."""
    import time
    from . import backward
    # per-edge CROWN refinement of the pre-activation bounds first: the LP
    # state's bands inherit them, which is what makes the dual competitive
    # with v1's tightened states
    inter = backward.intermediates_crown(net, lo, hi, base_inter=inter)
    # per-query direction-adaptive slopes (v1 build_dir_adaptive_alpha):
    # per neuron, the OPTIMIZED alpha where the query's adjoint ew > 0
    # (lower plane binds) and the chord slope h/(h-l) where ew <= 0, so the
    # single-slope state reproduces the backward alpha-CROWN bound
    open_rows = [r for d in open_d
                 for r in torch.nonzero(disj_idx == d,
                                        as_tuple=False).flatten().tolist()]
    W_open = W[open_rows]
    _lb, alpha = backward.alpha_crown(net, lo, hi, W_open, inter,
                                      iters=60, thresholds=-bias[open_rows],
                                      return_alpha=True)
    adj = {}
    backward.crown(net, lo, hi, W_open, inter, alpha=alpha,
                   collect_adjoints=adj)
    row_pos = {r: i for i, r in enumerate(open_rows)}

    def dir_adaptive_slopes(row_i):
        slopes = {}
        for nm, a in alpha.items():
            if a.dim() != 3 or nm not in adj:
                continue                       # relu alphas only
            l, h = inter[nm]
            chord = (h[0] / (h[0] - l[0]).clamp_min(1e-30)).clamp(0.0, 1.0)
            ew = adj[nm][0, row_i]
            slopes[nm] = torch.where(ew > 0, a[0, row_i], chord)
        return slopes
    refuted = set()
    dev = str(torch.device(device))
    ver = _verifier(dev)
    state_cache = {}
    gamma_inter = {}
    for d in open_d:
        rows = torch.nonzero(disj_idx == d, as_tuple=False).flatten().tolist()
        left = deadline - time.time()
        if left <= 1.0:
            break
        per_q = max(2.0, left / max(1, len(open_d)) / max(1, len(rows)))
        for r in rows:
            qw = W[r].cpu().numpy()
            qb = float(bias[r])
            # NOTE: naively reusing CROWN alphas as zonotope slopes makes
            # the state LOOSER (measured: img96 dual went unsat 3.5s ->
            # frontier OOM); DeepZ slopes + refined bounds win. State comes
            # from the BACKWARD builder (v1 reverse_g port): no forward
            # zonotope, unstable rows only, LinMap-generic.
            if r not in state_cache:
                state_cache[r] = _state_for(
                    net, lo, hi, inter, dir_adaptive_slopes(row_pos[r]),
                    device)
            state, keys = state_cache[r]
            if not keys:
                continue
            sk = score_keys(net, lo, hi, W[r:r + 1], inter, keys)
            extra = [(W[r2].cpu().numpy(), float(bias[r2]))
                     for r2 in rows if r2 != r]
            verdict, info = ver.verify_query(
                state, qw, qb, sk, time_limit=min(per_q,
                                                  deadline - time.time()),
                extra_hs=extra)
            if (verdict != 'unsat'
                    and info.get('reason') != 'splits_exhausted'
                    and deadline - time.time() > 5.0):
                # gamma retry: refine THIS disjunct's intermediates under
                # its own output rows (INVPROP; conditional on the CE
                # region, so scoped strictly to this disjunct) and rerun
                if d not in gamma_inter:
                    Wg = W[rows].cpu().numpy()
                    bg = bias[rows].cpu().numpy()
                    gamma_inter[d] = backward.intermediates_crown(
                        net, lo, hi, base_inter=inter,
                        gamma_rows=(Wg, bg))
                g_state, g_keys = _state_for(
                    net, lo, hi, gamma_inter[d],
                    dir_adaptive_slopes(row_pos[r]), device)
                sk2 = score_keys(net, lo, hi, W[r:r + 1], gamma_inter[d],
                                 g_keys)
                verdict, info = ver.verify_query(
                    g_state, qw, qb, sk2,
                    time_limit=min(per_q, deadline - time.time()),
                    extra_hs=extra)
                log(f'[vc2/dual]   gamma retry: {verdict} '
                    f'nodes={info.get("nodes")} '
                    f'wall={info.get("wall", 0):.2f}s')
            log(f'[vc2/dual] disj {d} row {r}: {verdict} '
                f'nodes={info.get("nodes")} wall={info.get("wall", 0):.2f}s '
                f'reason={info.get("reason", "-")} open={info.get("open", 0)}')
            if verdict == 'unsat':
                refuted.add(d)
                break
            if (verdict != 'unsat'
                    and info.get('reason') == 'splits_exhausted'
                    and deadline - time.time() > 10.0):
                # relu splits alone cannot close: the slack lives in smooth
                # free-block generators. Range-split those OUTSIDE the dual,
                # rebuilding the leaf states under the tightened ranges.
                verdict, info = range_split_dual(
                    net, lo, hi, inter, qw, qb, extra, ver, deadline,
                    dir_adaptive_slopes(row_pos[r]), device, log)
                log(f'[vc2/dual]   range-split: {verdict} {info}')
                if verdict == 'unsat':
                    refuted.add(d)
                    break
                log('[vc2/dual] state too loose even range-split; bailing')
                return refuted
    return refuted
