"""Nonlinear-v2 augment handler: strict-bound grid tightening.

The official 2026 checker (vnncomp2026_results 28455bb) casts the witness
assignment to the ONNX input dtype before evaluating input atoms, so the
legal witness set is the input-dtype grid and a STRICT direct bound
`X > a` is exactly `X >= succ_grid(a)`. These tests pin that fold."""
import numpy as np

from vibecheck2.handlers.nonlinear_augment import (_grid_pred, _grid_succ,
                                                   analyze)
from vibecheck2.frontend.vnnlib_loader import parse_vnnlib_v2


def test_grid_succ_pred_f32():
    s0 = _grid_succ(0.0, np.float32)
    assert s0 > 0.0 and np.float32(s0) > 0.0
    assert float(np.nextafter(np.float32(s0), np.float32(-np.inf))) <= 0.0
    p40 = _grid_pred(40.0, np.float32)
    assert p40 < 40.0 and float(np.nextafter(np.float32(p40),
                                             np.float32(np.inf))) >= 40.0
    # non-representable threshold: smallest f32 strictly above 0.1
    s = _grid_succ(0.1, np.float32)
    assert s > 0.1 and float(np.nextafter(np.float32(s),
                                          np.float32(-np.inf))) <= 0.1


def test_grid_succ_pred_f64_identity_grid():
    a = 1.25
    assert _grid_succ(a, np.float64) == float(np.nextafter(a, np.inf))
    assert _grid_pred(a, np.float64) == float(np.nextafter(a, -np.inf))


_SPEC = (
    '(vnnlib-version <2.0>)\n'
    '(declare-network f (declare-input X real [1,2])'
    ' (declare-output Y real [1,1]))\n'
    '(assert (and (>= X[0,0] 0.0) (<= X[0,0] 20.0)))\n'
    '(assert (and (>= X[0,1] 0.0) (<= X[0,1] 40.0)))\n'
    '(assert (> X[0,0] 0.0))\n'
    '(assert (< X[0,1] 40.0))\n'
    '(assert (>= (* X[0,0] 200.0) (* X[0,1] X[0,1])))\n'
    '(assert (> Y[0,0] 0.0))\n')


def test_analyze_strict_fold_snaps_direct_bounds():
    prop = parse_vnnlib_v2(_SPEC)
    _, _, _, xbox = analyze(prop, in_grid=np.float32)
    # strict > 0 beats the coexisting nonstrict >= 0 within the clause
    assert xbox[0][0] == _grid_succ(0.0, np.float32)
    # strict < 40 with |coef|==1 -> largest f32 below 40
    assert xbox[1][1] == _grid_pred(40.0, np.float32)


def test_analyze_per_clause_bounds_not_globalized():
    """A bound present in only ONE clause of the DNF must not cut the other
    clause's region (the box covers the union)."""
    prop = parse_vnnlib_v2(
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X real [1,1])'
        ' (declare-output Y real [1,1]))\n'
        '(assert (and (>= X[0,0] 0.0) (<= X[0,0] 20.0)))\n'
        '(assert (>= (* X[0,0] X[0,0]) 0.0))\n'
        '(assert (or (and (>= X[0,0] 5.0) (> Y[0,0] 0.0))\n'
        '            (and (< Y[0,0] -1.0))))\n')
    _, _, _, xbox = analyze(prop)
    assert xbox[0] == [0.0, 20.0]        # clause-2 has no X lower bound > 0


def test_analyze_strict_fold_closure_without_grid():
    prop = parse_vnnlib_v2(_SPEC)
    _, _, _, xbox = analyze(prop, in_grid=None)
    assert xbox[1][1] == 40.0            # closure: strictness dropped


def test_analyze_scaled_strict_not_snapped():
    prop = parse_vnnlib_v2(
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X real [1,1])'
        ' (declare-output Y real [1,1]))\n'
        '(assert (and (>= X[0,0] 0.0) (<= X[0,0] 20.0)))\n'
        '(assert (> (* X[0,0] 2.0) 1.0))\n'
        '(assert (>= (* X[0,0] X[0,0]) Y[0,0]))\n')
    _, _, _, xbox = analyze(prop, in_grid=np.float32)
    assert xbox[0][0] == 0.5             # closure value, no ulp snap
