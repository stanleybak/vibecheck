"""Verification orchestration — multi-zonotope intersection strategy."""

import numpy as np
from .zonotope import DenseZonotope
from .network import _prod


def zonotope_verify(graph, x_lo, x_hi, pred_label, competitors, relu_types=None):
    """Run zonotope analysis on a ComputeGraph.

    Dispatches to each node's zonotope_propagate() method.

    Args:
        graph: ComputeGraph
        x_lo, x_hi: input bounds
        pred_label: predicted class index
        competitors: list of competitor class indices
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

        zono_state[graph.input_name] = DenseZonotope.from_input_bounds(x_lo, x_hi)
        gen_count[graph.input_name] = zono_state[graph.input_name].generators.shape[1]

        def _get_input(inp_name):
            if inp_name in forks:
                return zono_state[inp_name].copy()
            return zono_state[inp_name]

        for name in graph.topo_order:
            if name in zono_state:  # already set by parent (e.g., Split)
                continue
            node = graph.nodes[name]
            node.zonotope_propagate(
                zono_state, gen_count, _get_input, relu_type, graph)
            gen_count[name] = zono_state[name].generators.shape[1]

        z_out = zono_state[graph.output_name]
        z_lo, z_hi = z_out.bounds()
        best_lo = np.maximum(best_lo, z_lo)
        best_hi = np.minimum(best_hi, z_hi)

    margins = {comp: float(best_lo[pred_label] - best_hi[comp])
               for comp in competitors}
    worst_margin = min(margins.values())

    return ('verified' if worst_margin > 0 else 'unknown'), {
        'output_lo': best_lo,
        'output_hi': best_hi,
        'margins': margins,
        'worst_margin': float(worst_margin),
    }
