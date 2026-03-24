"""CLI entry point for zonotope-based neural network verification."""

import argparse
import sys
import time

from .network import ComputeGraph
from .spec import parse_vnnlib
from .verify import zonotope_verify


def main():
    parser = argparse.ArgumentParser(
        description='VibeCheck — Neural Network Verification via Zonotope Analysis')
    parser.add_argument('--net', required=True, help='Path to ONNX network')
    parser.add_argument('--spec', required=True, help='Path to VNNLIB specification')
    parser.add_argument('--relu-types', nargs='+',
                        default=['min_area', 'y_bloat', 'box'],
                        choices=['min_area', 'y_bloat', 'box'],
                        help='ReLU relaxation types to intersect (default: min_area y_bloat box)')
    args = parser.parse_args()

    t_start = time.time()

    # Load network as graph
    print(f'Loading network: {args.net}')
    graph = ComputeGraph.from_onnx(args.net)
    n_relu = len(graph.relu_nodes())
    forks = graph.fork_points()
    print(f'  {len(graph.nodes)} ops, {n_relu} ReLU layers, '
          f'{len(forks)} fork points, input shape: {graph.input_shape}')

    # Load spec
    print(f'Loading spec: {args.spec}')
    x_lo, x_hi, pred_label, competitors = parse_vnnlib(args.spec)
    print(f'  Predicted class: {pred_label}, competitors: {competitors}')

    # Run zonotope verification
    print(f'Running zonotope analysis (relu types: {args.relu_types})...')
    result, details = zonotope_verify(
        graph, x_lo, x_hi, pred_label, competitors,
        relu_types=args.relu_types)

    t_total = time.time() - t_start

    # Report results
    print(f'\nResult: {result}')
    print(f'  Worst margin: {details["worst_margin"]:.6f}')
    for comp, margin in details['margins'].items():
        status = 'SAFE' if margin > 0 else 'UNKNOWN'
        print(f'  Class {pred_label} vs {comp}: margin={margin:.6f} [{status}]')
    print(f'  Time: {t_total:.2f}s')

    sys.exit(0 if result == 'verified' else 1)


if __name__ == '__main__':
    main()
