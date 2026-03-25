"""Basic tests for DenseZonotope."""

import numpy as np
import pytest
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


# ---- Conv propagation ----

def test_propagate_conv_with_generators():
    """Conv propagation with actual generators."""
    from vibecheck.zonotope import is_conv, conv_output_shape
    kernel = np.random.randn(2, 1, 3, 3)
    bias = np.zeros(2)
    params = {'input_shape': (1, 4, 4), 'stride': (1, 1), 'padding': (0, 0)}
    layer = (kernel, bias, params)
    assert is_conv(layer)

    out_shape = conv_output_shape((1, 4, 4), kernel, params)
    assert out_shape == (2, 2, 2)

    z = DenseZonotope.from_input_bounds(np.zeros(16), np.ones(16))
    z.propagate_linear(layer)
    assert len(z.center) == 8  # 2*2*2
    assert z.generators.shape[0] == 8


def test_propagate_conv_point():
    """Conv propagation with 0 generators (point zonotope)."""
    kernel = np.ones((1, 1, 2, 2))
    bias = np.array([0.0])
    params = {'input_shape': (1, 3, 3), 'stride': (1, 1), 'padding': (0, 0)}
    layer = (kernel, bias, params)

    center = np.arange(9, dtype=float)
    z = DenseZonotope(center, np.zeros((9, 0)))
    z.propagate_linear(layer)
    assert len(z.center) == 4  # 1*2*2
    assert z.generators.shape == (4, 0)


# ---- ReLU relaxation types ----

def test_relu_unstable_min_area():
    """Unstable neuron with min_area relaxation (hi > -lo)."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'min_area')
    lo, hi = z.bounds()
    assert lo[0] >= -0.01  # should be near 0
    assert hi[0] <= 1.01

def test_relu_unstable_min_area_lo_dominant():
    """Unstable neuron with min_area where hi < -lo."""
    z = DenseZonotope(np.array([-0.3]), np.array([[0.5]]))
    z.apply_relu(np.array([-0.8]), np.array([0.2]), 'min_area')
    lo, hi = z.bounds()
    assert lo[0] >= -0.01
    assert hi[0] <= 0.3

def test_relu_y_bloat():
    """Unstable neuron with y_bloat relaxation."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'y_bloat')
    lo, hi = z.bounds()
    assert lo[0] >= -1.01
    assert hi[0] <= 2.01  # y_bloat gives wider bounds

def test_relu_invalid_type():
    """Invalid relu_type raises."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    with pytest.raises(AssertionError, match="Unknown relu_type"):
        z.apply_relu(np.array([-1.0]), np.array([1.0]), 'invalid')


def test_relu_box():
    """Unstable neuron with box relaxation."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'box')
    lo, hi = z.bounds()
    assert lo[0] >= -0.01
    assert hi[0] <= 1.01
