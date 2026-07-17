"""Exact MILP refutation (design: the escalation for spread-slack nets).

Some nets place the certification slack thin across thousands of relu
triangles (malbeware: best split worth 0.007 of a 1.43 gap;
challenging_certified: 9.6k splits, zero closures): no relaxation split
tree converges, and v1 closes these only through its Gurobi routes. This
module is vc2's contained equivalent: the triangle-EXACT big-M encoding
of a query over the box, solved by Gurobi with a time limit.

Soundness: refutation uses the solver's DUAL bound (ObjBound), valid at
any point of the solve including timeouts, with a strictness margin. An
incumbent (primal) solution is only a candidate counterexample and is
handed to the caller for ORT validation, never trusted directly.

Scope (kept deliberately small): relu-only nonlinearities; linmaps
materialized via lin(eye) in chunks (dense/conv/select all work);
add/concat supported. Anything else raises NotImplementedError loudly.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def _linmap_matrix(lm, n_in, dev):
    """(n_out, n_in) dense matrix of a LinMap via identity probes."""
    cols = []
    step = max(1, 4_000_000 // max(1, lm.n_out))
    eye = torch.eye(n_in, device=dev)
    for i in range(0, n_in, step):
        cols.append(lm.lin(eye[i:i + step]).cpu().numpy())
    W = np.concatenate(cols, axis=0).T          # (n_out, n_in)
    b = lm.bias_vec(torch.zeros(1, n_in, device=dev)).cpu().numpy()
    return W, b


def refute_rows_milp(net, lo, hi, inter, W, bias, disj_idx, disjuncts,
                     deadline, log=lambda m: None, obj_margin=1e-5):
    """Refute each open DISJUNCT (conjunction of rows) by exact MILP.

    A counterexample to disjunct d satisfies EVERY row: w_r.y + b_r <= 0.
    Refuting d proves that region empty:
      - single row r:        min_x (w_r.y + b_r) > 0.
      - conjunction (k>1):   min_x max_r (w_r.y + b_r) > 0  (a per-row pass
        runs first, since one always-positive row already refutes the AND).
    This closes conjunctive specs (sat_relu) that per-row refutation cannot.

    disj_idx (q,): disjunct id of each query row. disjuncts: ids to try.
    Returns (refuted_disjunct_set, candidate_ce_or_None); a candidate is an
    incumbent where the whole conjunction holds, validated by the caller
    through the ORT chokepoint.
    """
    import gurobipy as gp

    di = disj_idx.cpu().numpy() if hasattr(disj_idx, 'cpu') else \
        np.asarray(disj_idx)
    rows_of = {int(d): np.nonzero(di == d)[0].tolist() for d in disjuncts}

    dev = lo.device
    n_bin = 0
    for nm in net.order:
        op = net.ops[nm]
        if op.kind == 'nonlin':
            if op.fn != 'relu':
                raise NotImplementedError(
                    f'milp: nonlinearity {op.fn!r} (relu-only encoding)')
            l, h = inter[nm]
            n_bin += int(((l < 0) & (h > 0)).sum())
        elif op.kind not in ('input', 'linmap', 'add', 'concat'):
            raise NotImplementedError(f'milp: op kind {op.kind!r}')
    log(f'[vibecheck/milp] encoding: {n_bin} binaries, '
        f'{len(disjuncts)} disjuncts, budget {deadline - time.time():.0f}s')

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 0                        # all cores
    lo1 = lo.reshape(-1).cpu().numpy()
    hi1 = hi.reshape(-1).cpu().numpy()
    x_in = m.addMVar(net.n_in, lb=lo1, ub=hi1)
    edge = {net.input_name: x_in}

    for nm in net.order:
        op = net.ops[nm]
        if op.kind == 'linmap':
            Wm, bm = _linmap_matrix(op.lm, net.ops[op.inputs[0]].n, dev)
            y = m.addMVar(op.n, lb=-gp.GRB.INFINITY)
            m.addConstr(y == Wm @ edge[op.inputs[0]] + bm)
            edge[nm] = y
        elif op.kind == 'add':
            y = m.addMVar(op.n, lb=-gp.GRB.INFINITY)
            m.addConstr(y == edge[op.inputs[0]] + edge[op.inputs[1]])
            edge[nm] = y
        elif op.kind == 'concat':
            y = m.addMVar(op.n, lb=-gp.GRB.INFINITY)
            base = np.asarray(op.params['base'], dtype=float).copy()
            covered = np.zeros(op.n, dtype=bool)
            for src, pos in zip(op.inputs, op.params['positions']):
                p = np.asarray(pos)
                m.addConstr(y[p] == edge[src])
                covered[p] = True
            if (~covered).any():
                m.addConstr(y[~covered] == base[~covered])
            edge[nm] = y
        elif op.kind == 'nonlin':
            z = edge[op.inputs[0]]
            l = inter[nm][0].reshape(-1).cpu().numpy().astype(float)
            h = inter[nm][1].reshape(-1).cpu().numpy().astype(float)
            y = m.addMVar(op.n, lb=0.0)
            pos = l >= 0
            neg = h <= 0
            uns = ~(pos | neg)
            if pos.any():
                m.addConstr(y[pos] == z[pos])
            if neg.any():
                m.addConstr(y[neg] == 0)
            if uns.any():
                a = m.addMVar(int(uns.sum()), vtype=gp.GRB.BINARY)
                zu, yu = z[uns], y[uns]
                lu, hu = l[uns], h[uns]
                m.addConstr(yu >= zu)
                m.addConstr(yu <= zu - lu * (1 - a))
                m.addConstr(yu <= hu * a)
            edge[nm] = y

    y_out = edge[net.output_name]
    Wn = W.cpu().numpy().astype(float)
    bn = bias.cpu().numpy().astype(float)
    refuted = set()
    candidate = None
    todo = list(disjuncts)
    for d in todo:
        left = deadline - time.time()
        if left < 3:
            break
        d = int(d)
        rows = rows_of[d]
        cs = []
        if len(rows) == 1:
            # OPTIMIZATION form for a single row: min w.y + b with
            # bound-driven stops -- BestBdStop fires the moment the DUAL
            # bound clears zero (sound refutation at any point of the
            # solve), BestObjStop the moment an incumbent violates. The
            # pure feasibility form has no objective to prune with and
            # grinds big-M trees to TIME_LIMIT on barely-infeasible
            # regions (safenlp medical 1988: 118 binaries, 13s, no
            # verdict).
            r = rows[0]
            m.setObjective(Wn[r] @ y_out + bn[r], gp.GRB.MINIMIZE)
            m.Params.BestBdStop = obj_margin
            m.Params.BestObjStop = -obj_margin
        else:
            # FEASIBILITY of the CE region {all rows w_r.y + b_r <= 0}
            # (v1's Gurobi route). INFEASIBLE proves no counterexample
            # exists -> refuted, and it is DECISIVE at any positive true
            # margin -- no razor-thin obj_margin trap (min-of-max got
            # stuck at a loose LP bound of 0.0 on sat_relu v56_c239,
            # where the exact margin is a tiny positive epsilon). A
            # feasible point satisfies every row -> a CE candidate the
            # caller validates through the ORT chokepoint.
            cs = [m.addConstr(Wn[r] @ y_out + bn[r] <= 0.0) for r in rows]
            m.setObjective(0.0)
            m.Params.BestBdStop = gp.GRB.INFINITY
            m.Params.BestObjStop = gp.GRB.INFINITY
        m.Params.TimeLimit = max(3.0, left / max(1, len(todo) - len(refuted)))
        t0 = time.time()
        m.optimize()
        if len(rows) == 1:
            try:
                ob = float(m.ObjBound)
            except (AttributeError, gp.GurobiError):
                ob = -float('inf')
            log(f'[vibecheck/milp] disj {d} (k=1): status={m.Status} '
                f'sols={m.SolCount} bd={ob:+.3e} t={time.time() - t0:.1f}s')
            if ob >= obj_margin:
                refuted.add(d)               # sound: dual bound on the min
            elif m.SolCount > 0 and candidate is None \
                    and float(m.ObjVal) <= 0.0:
                candidate = np.array(x_in.X, dtype=np.float64)
        else:
            log(f'[vibecheck/milp] disj {d} (k={len(rows)}): status={m.Status} '
                f'sols={m.SolCount} t={time.time() - t0:.1f}s')
            if m.Status == gp.GRB.INFEASIBLE:
                refuted.add(d)                      # sound: CE region empty
            elif m.SolCount > 0 and candidate is None:
                candidate = np.array(x_in.X, dtype=np.float64)  # rows <= 0
            m.remove(cs)
    return refuted, candidate
