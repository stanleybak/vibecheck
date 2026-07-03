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


def refute_rows_milp(net, lo, hi, inter, W, bias, rows, deadline,
                     log=lambda m: None, obj_margin=1e-5):
    """Try to refute each query row (w.y + b > 0 proven) by exact MILP.

    Returns (refuted_row_set, candidate_ce_or_None). A candidate CE is
    the incumbent of a row whose optimum went clearly negative; the
    caller MUST validate it through the ORT chokepoint.
    """
    import gurobipy as gp

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
    log(f'[vc2/milp] encoding: {n_bin} binaries, '
        f'{len(rows)} rows, budget {deadline - time.time():.0f}s')

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
    refuted = set()
    candidate = None
    for r in rows:
        left = deadline - time.time()
        if left < 3:
            break
        w = W[r].cpu().numpy().astype(float)
        b = float(bias[r])
        m.setObjective(w @ y_out + b, gp.GRB.MINIMIZE)
        m.Params.TimeLimit = max(3.0, left / max(1, len(rows) -
                                                 len(refuted)))
        # stop as soon as the sign is decided either way
        m.Params.BestBdStop = obj_margin
        m.Params.BestObjStop = -obj_margin
        t0 = time.time()
        m.optimize()
        bound = m.ObjBound if m.SolCount >= 0 else -np.inf
        log(f'[vc2/milp] row {r}: status={m.Status} '
            f'bound={bound:+.6f} incumbent='
            f'{m.ObjVal if m.SolCount > 0 else float("nan"):+.6f} '
            f't={time.time() - t0:.1f}s')
        if m.Status == gp.GRB.INFEASIBLE:
            # the box itself admits no feasible point: vacuous refutation
            refuted.add(r)
        elif bound > obj_margin:
            refuted.add(r)                      # sound: dual bound > 0
        elif m.SolCount > 0 and m.ObjVal < -obj_margin and candidate is None:
            candidate = np.array(x_in.X, dtype=np.float64)
    return refuted, candidate
