"""Unit tests for the shared BaB core (core/bab.py)."""
import numpy as np
import torch

from vibecheck.core.bab import (Domain, disjunct_selector,
                                 materialize_clamps, refuted_matrix)


def test_disjunct_selector():
    di = torch.tensor([0, 0, 1, 2, 2])
    D, sel = disjunct_selector(di, 5, torch.device('cpu'))
    assert D == 3
    assert sel.shape == (3, 5)
    assert sel[0].tolist() == [True, True, False, False, False]
    assert sel[2].tolist() == [False, False, False, True, True]
    # empty spec
    D0, sel0 = disjunct_selector(torch.zeros(0, dtype=torch.long), 0,
                                 torch.device('cpu'))
    assert D0 == 0


def test_refuted_matrix_any_row_refutes():
    # disjunct 0 = rows {0,1}, disjunct 1 = row {2}
    di = torch.tensor([0, 0, 1])
    D, sel = disjunct_selector(di, 3, torch.device('cpu'))
    bias = torch.zeros(3)
    # domain A: row1 positive -> disj 0 refuted, row2 negative -> disj 1 open
    # domain B: all negative -> nothing refuted
    lbq = torch.tensor([[-1.0, 0.5, -0.3],
                        [-1.0, -0.2, -0.3]])
    R = refuted_matrix(lbq, bias, sel)
    assert R[0].tolist() == [True, False]
    assert R[1].tolist() == [False, False]


def test_refuted_matrix_inf_never_refutes():
    di = torch.tensor([0])
    D, sel = disjunct_selector(di, 1, torch.device('cpu'))
    lbq = torch.tensor([[float('inf')]])
    R = refuted_matrix(lbq, torch.zeros(1), sel)
    assert R[0].tolist() == [False]        # +inf is an artifact, not a proof


def test_materialize_clamps_sign_and_range():
    dev = torch.device('cpu')
    n_of = lambda nm: {'r': 4, 's': 3}[nm]
    batch = [
        (('r', 1, 1), ('r', 2, -1)),               # relu sign fixes
        (('s', 0, (-0.5, 0.5)),),                   # range split
    ]
    clamps, rc = materialize_clamps(batch, n_of, 2, dev)
    assert clamps['r'][0].tolist() == [0, 1, -1, 0]
    assert clamps['r'][1].tolist() == [0, 0, 0, 0]
    rlo, rhi = rc['s']
    assert abs(float(rlo[1, 0]) - -0.5) < 1e-6 and abs(float(rhi[1, 0]) - 0.5) < 1e-6


def test_materialize_clamps_intersect():
    dev = torch.device('cpu')
    n_of = lambda nm: 2
    # same neuron split twice: intersect (max lo, min hi)
    batch = [(('s', 0, (-1.0, 1.0)), ('s', 0, (-0.3, 2.0)))]
    _, rc = materialize_clamps(batch, n_of, 1, dev)
    rlo, rhi = rc['s']
    assert abs(float(rlo[0, 0]) - -0.3) < 1e-6 and float(rhi[0, 0]) == 1.0


def test_domain_heap_order():
    import heapq
    h = []
    heapq.heappush(h, Domain(-0.5, 1, (), None, {}, None))
    heapq.heappush(h, Domain(-2.0, 2, (), None, {}, None))
    heapq.heappush(h, Domain(-0.1, 3, (), None, {}, None))
    # worst (most negative) pops first; payload never compared
    assert heapq.heappop(h).lb == -2.0
    assert heapq.heappop(h).lb == -0.5
