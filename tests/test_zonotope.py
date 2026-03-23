"""Basic tests for DenseZonotope."""

import numpy as np
from vibecheck.zonotope import DenseZonotope


def test_from_input_bounds():
    x_lo = np.array([0.0, -1.0])
    x_hi = np.array([1.0, 1.0])
    z = DenseZonotope.from_input_bounds(x_lo, x_hi)
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, x_lo)
    np.testing.assert_allclose(hi, x_hi)


def test_propagate_fc():
    x_lo = np.array([0.0, 0.0])
    x_hi = np.array([1.0, 1.0])
    z = DenseZonotope.from_input_bounds(x_lo, x_hi)

    W = np.array([[1.0, 2.0], [-1.0, 1.0]])
    b = np.array([0.0, 0.0])
    z.propagate_linear((W, b))

    lo, hi = z.bounds()
    # center = W @ [0.5, 0.5] = [1.5, 0.0], abs_sum = [1.5, 1.0]
    np.testing.assert_allclose(lo, [0.0, -1.0])
    np.testing.assert_allclose(hi, [3.0, 1.0])


def test_relu_stable_positive():
    z = DenseZonotope(center=np.array([2.0]), generators=np.array([[0.5]]))
    z.apply_relu(np.array([1.5]), np.array([2.5]))
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, [1.5])
    np.testing.assert_allclose(hi, [2.5])


def test_relu_stable_negative():
    z = DenseZonotope(center=np.array([-2.0]), generators=np.array([[0.5]]))
    z.apply_relu(np.array([-2.5]), np.array([-1.5]))
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, [0.0])
    np.testing.assert_allclose(hi, [0.0])
