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


def test_add_shared_only():
    """Add two zonotopes that share all generators."""
    z1 = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    z2 = DenseZonotope(np.array([3.0, 4.0]), np.array([[0.1, 0.0], [0.0, 0.1]]))
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [4.0, 6.0])
    np.testing.assert_allclose(z3.generators, [[0.6, 0.0], [0.0, 0.6]])


def test_add_with_extra_gens():
    """Add two zonotopes where each branch added extra generators."""
    z1 = DenseZonotope(
        np.array([1.0]),
        np.array([[0.5, 0.3, 0.1]]),  # 2 shared + 1 extra
    )
    z2 = DenseZonotope(
        np.array([2.0]),
        np.array([[0.4, 0.2, 0.05, 0.02]]),  # 2 shared + 2 extra
    )
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [3.0])
    np.testing.assert_allclose(z3.generators, [[0.9, 0.5, 0.1, 0.05, 0.02]])


def test_copy_independent():
    z = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5], [0.3]]))
    z2 = z.copy()
    z2.center[0] = 99.0
    assert z.center[0] == 1.0
