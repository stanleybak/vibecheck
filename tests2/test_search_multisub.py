"""Multi-sub input-split BaB: the nn4sys mega-disjunct shape.

Cardinality specs (lindex: 120k rows) carry many DISTINCT thresholds that
all share ONE weight vector. CROWN must dedupe on the WEIGHT only (else
O(B x q) tiny batches), keep the per-row bias per-domain, and add it back
via a gather. These tests pin that path: refutation, margin, and input
clip must all read each domain's OWN (deduped-weight, per-row-bias) bound.
"""
import time

import numpy as np
import torch

from vibecheck2.core.graph import Net, Op
from vibecheck2.core.linmap import Dense
from vibecheck2.core.search import input_split_bab


def _identity_net(n=2):
    return Net({'x': Op('x', 'input', (), (n,), n),
                'y': Op('y', 'linmap', ('x',), (n,), n,
                        lm=Dense(np.eye(n, dtype=np.float32),
                                 np.zeros(n, dtype=np.float32)))},
               ['y'], 'x', 'y')


def test_multisub_dedup_weight_all_refuted_unsat():
    """One weight [1,0], many distinct positive biases, box [0,1]^2. Each
    sub-row r requires x0 + b_r <= 0 with min = b_r > 0 -> every sub refuted
    -> unsat. Exercises the weight-dedup (q_W==1) gather across every domain.
    """
    net = _identity_net()
    q = 32
    W = torch.zeros(q, 2)
    W[:, 0] = 1.0                              # identical weight for all rows
    bias = torch.linspace(0.05, 1.0, q)        # all strictly positive
    disj_idx = torch.arange(q)
    lo = torch.zeros(2)
    hi = torch.ones(2)
    roots = (lo.repeat(q, 1), hi.repeat(q, 1), torch.arange(q))
    verdict, info = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 20, device='cpu', roots=roots, alpha_iters=0)
    assert verdict == 'unsat', (verdict, info)


def test_multisub_mixed_weights_per_row_bias():
    """Two distinct weights interleaved (dedup to q_W==2) with per-row
    biases; all refutable. Confirms weight_of gathers the RIGHT column per
    domain (a wrong column would leave a domain open -> timeout/misverdict).
    """
    net = _identity_net()
    # rows alternate weight [1,0] and [0,1]; box [0,1]^2 so min(w.x)+b = b
    q = 16
    W = torch.zeros(q, 2)
    W[0::2, 0] = 1.0
    W[1::2, 1] = 1.0
    bias = torch.full((q,), 0.25)              # each sub refuted (0 + .25 > 0)
    disj_idx = torch.arange(q)
    lo = torch.zeros(2)
    hi = torch.ones(2)
    roots = (lo.repeat(q, 1), hi.repeat(q, 1), torch.arange(q))
    # alpha_iters>0 exercises the deduped per-column threshold branch (q_W==2)
    verdict, info = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 20, device='cpu', roots=roots, alpha_iters=8)
    assert verdict == 'unsat', (verdict, info)


def test_multisub_minigroups_all_refuted_unsat():
    """The mscn shape at the DRIVER level: many distinct subboxes, each a
    single refutable row. verify._verify_groups must chunk them into
    mini-groups and return unsat when every chunk closes (v1's mini_group
    strategy -- one shared frontier over all N would explode on real nets).
    N=1100 forces >1 mini-group (mg=1000), exercising the chunk loop +
    aggregation; the identity net closes each chunk in one round.
    """
    from vibecheck2.frontend.spec import VNNSpec, Conjunct, Constraint
    from vibecheck2.verify import _verify_groups, _subbox_groups
    net = _identity_net()                       # y = x over R^2
    N = 1100
    disj = []
    for d in range(N):
        lo0 = 0.1 + 0.001 * d                   # distinct box per disjunct
        ilo = np.array([lo0, 0.0], dtype=np.float64)
        ihi = np.array([lo0 + 0.3, 1.0], dtype=np.float64)
        # CE region y_0 <= -0.5; box has x_0 >= lo0 > 0 > -0.5 -> refuted
        disj.append(Conjunct(constraints=[Constraint(0, '<=', -0.5)],
                             input_lo=ilo, input_hi=ihi))
    spec = VNNSpec(x_lo=np.zeros(2), x_hi=np.ones(2) * 1.5, disjuncts=disj)
    groups = _subbox_groups(spec)
    assert len(groups) > 16, len(groups)
    verdict, _ = _verify_groups(net, spec, groups, None, 30.0, 'cpu',
                                8, 0.0, lambda _m: None, time.time())
    assert verdict == 'unsat', verdict


def test_multisub_open_domain_not_refuted():
    """A genuinely feasible sub (b_r < 0 -> min x0 + b_r < 0) must NOT be
    refuted: with no onnx_path the CE can't be validated, so the search
    cannot close it as unsat. It stays open until the (short) deadline.
    A wrong gather that read the feasible row as refuted would return unsat
    -- unsound -- so 'unsat' here is the failure signal.
    """
    net = _identity_net()
    q = 8
    W = torch.zeros(q, 2)
    W[:, 0] = 1.0
    bias = torch.full((q,), 0.5)
    bias[3] = -0.5                             # sub 3 is feasible (x0 <= .5)
    disj_idx = torch.arange(q)
    lo = torch.zeros(2)
    hi = torch.ones(2)
    roots = (lo.repeat(q, 1), hi.repeat(q, 1), torch.arange(q))
    verdict, _ = input_split_bab(
        net, None, W, bias, disj_idx, lo, hi,
        deadline=time.time() + 2, device='cpu', roots=roots, alpha_iters=0)
    assert verdict != 'unsat', "feasible sub wrongly refuted (unsound)"
