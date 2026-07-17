"""Row-group (disjunct-decomposed) input-split BaB: the lsnc shape.

A root's row GROUP is one disjunct's conjunctive rows: the domain is
refuted when ANY row's lb goes positive, and the clip intersects ALL of
the group's halfspaces sequentially. These tests pin the any-row
refutation, the conjunctive clip (an infeasible pair must close), ragged
padding, and the soundness direction (a satisfiable group stays open).
"""
import time

import numpy as np
import torch

from vibecheck.core.graph import Net, Op
from vibecheck.core.linmap import Dense
from vibecheck.core.search import input_split_bab


def _identity_net(n=2):
    return Net({'x': Op('x', 'input', (), (n,), n),
                'y': Op('y', 'linmap', ('x',), (n,), n,
                        lm=Dense(np.eye(n, dtype=np.float32),
                                 np.zeros(n, dtype=np.float32)))},
               ['y'], 'x', 'y')


def _contradictory_spec():
    """3 disjuncts over box [0,1]^2, all infeasible, none refuted by a
    single root row:
      d0: x0 <= 0.4 AND x0 >= 0.6   (rows 0, 1 -- contradictory pair)
      d1: x1 <= 0.4 AND x1 >= 0.6   (rows 2, 3)
      d2: x0 <= -0.5                (row 4 -- single row, pads to r=2)
    Row form w.y + b <= 0 for a CE; refuted when lb(w.y + b) > 0.
    """
    W = torch.tensor([[1.0, 0.0],     # x0 - 0.4 <= 0
                      [-1.0, 0.0],    # 0.6 - x0 <= 0
                      [0.0, 1.0],     # x1 - 0.4 <= 0
                      [0.0, -1.0],    # 0.6 - x1 <= 0
                      [1.0, 0.0]])    # x0 + 0.5 <= 0
    bias = torch.tensor([-0.4, 0.6, -0.4, 0.6, 0.5])
    disj_idx = torch.tensor([0, 0, 1, 1, 2])
    row_groups = torch.tensor([[0, 1], [2, 3], [4, 4]])   # d2 padded
    return W, bias, disj_idx, row_groups


def test_rowgroups_conjunctive_infeasible_unsat():
    """Each contradictory pair is only refutable jointly: at the root both
    rows' lbs are -0.4 (open), but either one split at 0.5 or the
    sequential conjunctive clip (row0 -> x0 <= 0.4, then row1's halfspace
    x0 >= 0.6 -> empty box) must close every domain -> unsat."""
    net = _identity_net()
    W, bias, disj_idx, rg = _contradictory_spec()
    lo, hi = torch.zeros(2), torch.ones(2)
    n_dj = rg.shape[0]
    roots = (lo.repeat(n_dj, 1), hi.repeat(n_dj, 1), torch.arange(n_dj))
    verdict, info = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 20, device='cpu', roots=roots,
        row_groups=rg, alpha_iters=0)
    assert verdict == 'unsat', (verdict, info)


def test_rowgroups_any_row_refutes_at_root():
    """A group with one strictly-positive row must close at the root even
    though its OTHER row stays negative (any-row semantics; an all-rows
    requirement would leave it open -> timeout)."""
    net = _identity_net()
    W = torch.tensor([[1.0, 0.0],     # x0 + 0.5 <= 0: lb=+0.5, refutes
                      [0.0, 1.0]])    # x1 - 5.0 <= 0: lb=-5.0, open
    bias = torch.tensor([0.5, -5.0])
    disj_idx = torch.tensor([0, 0])
    rg = torch.tensor([[0, 1]])
    lo, hi = torch.zeros(2), torch.ones(2)
    roots = (lo.repeat(1, 1), hi.repeat(1, 1), torch.arange(1))
    verdict, info = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 5, device='cpu', roots=roots,
        row_groups=rg, alpha_iters=0)
    assert verdict == 'unsat', (verdict, info)
    assert info.get('splits', 0) == 0, info    # no splitting needed


def test_rowgroups_feasible_group_stays_open():
    """A satisfiable pair (x0 <= 0.6 AND x0 >= 0.4) must NOT be refuted:
    'unsat' here would be a false-unsat soundness violation (the clip or
    the any-row gather reading across groups)."""
    net = _identity_net()
    W = torch.tensor([[1.0, 0.0],     # x0 - 0.6 <= 0
                      [-1.0, 0.0]])   # 0.4 - x0 <= 0
    bias = torch.tensor([-0.6, 0.4])
    disj_idx = torch.tensor([0, 0])
    rg = torch.tensor([[0, 1]])
    lo, hi = torch.zeros(2), torch.ones(2)
    roots = (lo.repeat(1, 1), hi.repeat(1, 1), torch.arange(1))
    verdict, _ = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 2, device='cpu', roots=roots,
        row_groups=rg, alpha_iters=0)
    assert verdict != 'unsat', "satisfiable group wrongly refuted (unsound)"
