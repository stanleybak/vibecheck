"""Zonotope forward propagation — dense numpy implementation."""

import numpy as np


def is_conv(layer):
    """Check if layer is Conv (3-tuple) vs FC (2-tuple)."""
    return len(layer) == 3


def conv_output_shape(input_shape, kernel, params):
    """Compute output spatial shape for a Conv layer."""
    C_in, H_in, W_in = input_shape
    C_out = kernel.shape[0]
    kH, kW = kernel.shape[2], kernel.shape[3]
    sH, sW = params['stride']
    pH, pW = params['padding']
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    return (C_out, H_out, W_out)


class DenseZonotope:
    """Zonotope with dense numpy center and generator matrix.

    Representation: { center + G @ e | ||e||_inf <= 1 }

    Attributes:
        center: (n,) array
        generators: (n, k) array — one column per noise symbol
    """

    def __init__(self, center: np.ndarray, generators: np.ndarray):
        self.center = center
        self.generators = generators

    @classmethod
    def from_input_bounds(cls, x_lo: np.ndarray, x_hi: np.ndarray) -> 'DenseZonotope':
        center = (x_lo + x_hi) / 2
        radii = (x_hi - x_lo) / 2
        # Only create generator columns for dimensions with nonzero radius
        nonzero = np.nonzero(radii)[0]
        n = len(center)
        generators = np.zeros((n, len(nonzero)))
        for i, j in enumerate(nonzero):
            generators[j, i] = radii[j]
        return cls(center, generators)

    def bounds(self):
        """Compute element-wise lower and upper bounds."""
        abs_sum = np.abs(self.generators).sum(axis=1)
        return self.center - abs_sum, self.center + abs_sum

    def propagate_linear(self, layer):
        """Propagate through a linear layer (FC or Conv)."""
        if is_conv(layer):
            self._propagate_conv(layer)
        else:
            W, b = layer
            self.center = W @ self.center + b
            self.generators = W @ self.generators

    def _propagate_conv(self, layer):
        """Propagate through a Conv layer via torch conv2d."""
        import torch
        import torch.nn.functional as F
        kernel, bias, params = layer
        input_shape = params['input_shape']
        stride, padding = params['stride'], params['padding']
        k = torch.tensor(kernel, dtype=torch.float64)
        b = torch.tensor(bias, dtype=torch.float64)

        c_4d = torch.tensor(self.center, dtype=torch.float64).reshape(1, *input_shape)
        self.center = F.conv2d(c_4d, k, bias=b, stride=stride, padding=padding).flatten().numpy()

        n_gen = self.generators.shape[1]
        if n_gen == 0:
            out_shape = conv_output_shape(input_shape, kernel, params)
            self.generators = np.zeros((out_shape[0] * out_shape[1] * out_shape[2], 0))
        else:
            g_batch = torch.tensor(self.generators.T, dtype=torch.float64).reshape(n_gen, *input_shape)
            g_out = F.conv2d(g_batch, k, stride=stride, padding=padding)
            self.generators = g_out.reshape(n_gen, -1).numpy().T

    def apply_relu(self, pre_lo: np.ndarray, pre_hi: np.ndarray, relu_type: str = 'min_area'):
        """Apply ReLU relaxation, appending new error generators for unstable neurons.

        Args:
            pre_lo, pre_hi: pre-ReLU bounds (used to classify neurons)
            relu_type: 'min_area' | 'y_bloat' | 'box'
        """
        n = len(self.center)
        scale = np.ones(n)
        offsets = np.zeros(n)

        for j in range(n):
            lo, hi = pre_lo[j], pre_hi[j]
            if hi <= 0:
                scale[j] = 0.0
            elif lo < 0:
                if relu_type == 'min_area':
                    lam = hi / (hi - lo) if hi > -lo else 0.0
                    mu = -hi * lo / (2 * (hi - lo)) if lam > 0 else hi / 2
                elif relu_type == 'y_bloat':
                    lam = 1.0
                    mu = -lo / 2
                elif relu_type == 'box':
                    lam = 0.0
                    mu = hi / 2
                else:
                    raise ValueError(f"Unknown relu_type: {relu_type}")
                scale[j] = lam
                offsets[j] = mu

        self.center = scale * self.center + offsets

        # Scale existing generators, append one new column per unstable neuron with mu > 0
        new_cols = np.where((pre_lo < 0) & (pre_hi > 0) & (offsets > 0))[0]
        new_g = np.zeros((n, self.generators.shape[1] + len(new_cols)))
        new_g[:, :self.generators.shape[1]] = scale[:, None] * self.generators
        for i, j in enumerate(new_cols):
            new_g[j, self.generators.shape[1] + i] = offsets[j]
        self.generators = new_g

    def copy(self):
        """Return an independent copy of this zonotope."""
        return DenseZonotope(self.center.copy(), self.generators.copy())

    def add(self, other, shared_gens):
        """Element-wise addition with another zonotope (for skip connections).

        The first `shared_gens` generator columns are shared noise symbols
        (from before the fork point) — these are added element-wise.
        Remaining columns are branch-specific and get concatenated.
        """
        g_shared = self.generators[:, :shared_gens] + other.generators[:, :shared_gens]
        g_self_extra = self.generators[:, shared_gens:]
        g_other_extra = other.generators[:, shared_gens:]
        return DenseZonotope(
            self.center + other.center,
            np.hstack([g_shared, g_self_extra, g_other_extra]),
        )
