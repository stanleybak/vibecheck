"""Verification orchestration — multi-zonotope intersection strategy."""

import numpy as np
from .zonotope import DenseZonotope
from .network import _prod


def zonotope_verify(graph, spec, relu_types=None):
    """Run zonotope analysis on a ComputeGraph with a VNNSpec.

    Args:
        graph: ComputeGraph
        spec: VNNSpec with x_lo, x_hi, and disjuncts
        relu_types: list of ReLU relaxation types to intersect

    Returns:
        result: 'verified' or 'unknown'
        details: dict with output_lo, output_hi, margins, worst_margin
    """
    if relu_types is None:
        relu_types = ['min_area', 'y_bloat', 'box']

    forks = graph.fork_points()
    n_out = graph.flat_size(graph.output_name)
    best_lo = np.full(n_out, -np.inf)
    best_hi = np.full(n_out, np.inf)

    for relu_type in relu_types:
        zono_state = {}
        gen_count = {}

        zono_state[graph.input_name] = DenseZonotope.from_input_bounds(
            spec.x_lo, spec.x_hi)
        gen_count[graph.input_name] = zono_state[graph.input_name].generators.shape[1]

        def _get_input(inp_name):
            if inp_name in forks:
                return zono_state[inp_name].copy()
            return zono_state[inp_name]

        for name in graph.topo_order:
            if name in zono_state:
                continue
            node = graph.nodes[name]
            node.zonotope_propagate(
                zono_state, gen_count, _get_input, relu_type, graph)
            gen_count[name] = zono_state[name].generators.shape[1]

        z_out = zono_state[graph.output_name]
        z_lo, z_hi = z_out.bounds()
        best_lo = np.maximum(best_lo, z_lo)
        best_hi = np.minimum(best_hi, z_hi)

    result, check_details = spec.check(best_lo, best_hi)

    return result, {
        'output_lo': best_lo,
        'output_hi': best_hi,
        'margins': check_details['margins'],
        'worst_margin': check_details['worst_margin'],
    }
