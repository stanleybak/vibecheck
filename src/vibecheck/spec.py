"""VNNLIB spec parsing."""

import numpy as np
import re
import gzip


def parse_vnnlib(vnnlib_path):
    """Parse VNNLIB spec. Returns (x_lo, x_hi, pred_label, competitors)."""
    if vnnlib_path.endswith('.gz'):
        with gzip.open(vnnlib_path, 'rt') as f:
            text = f.read()
    else:
        with open(vnnlib_path, 'r') as f:
            text = f.read()

    x_bounds = {}
    for m in re.finditer(r'\(assert\s+\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', text):
        x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(assert\s+\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', text):
        x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))

    if not x_bounds:
        for m in re.finditer(r'X_(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)', text):
            x_bounds[int(m.group(1))] = [float(m.group(2)), float(m.group(3))]

    n_input = max(x_bounds.keys()) + 1
    x_lo = np.array([x_bounds[i][0] for i in range(n_input)])
    x_hi = np.array([x_bounds[i][1] for i in range(n_input)])

    # Format 1: >= Y_comp Y_pred (competitor >= pred)
    output_constrs = re.findall(r'>=\s+Y_(\d+)\s+Y_(\d+)', text)
    if output_constrs:
        from collections import Counter
        pred_label = Counter(int(c[1]) for c in output_constrs).most_common(1)[0][0]
        competitors = sorted(set(int(c[0]) for c in output_constrs))
        return x_lo, x_hi, pred_label, competitors

    # Format 2: <= Y_pred Y_comp (pred <= competitor)
    output_constrs = re.findall(r'<=\s+Y_(\d+)\s+Y_(\d+)', text)
    if output_constrs:
        from collections import Counter
        pred_label = Counter(int(c[0]) for c in output_constrs).most_common(1)[0][0]
        competitors = sorted(set(int(c[1]) for c in output_constrs))
        return x_lo, x_hi, pred_label, competitors

    raise ValueError("Cannot parse output constraints from VNNLIB")
