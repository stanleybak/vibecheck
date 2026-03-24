"""Verification specification types.

A VNNSpec defines input bounds and output constraints (the unsafe region).
The unsafe region is a disjunction of conjuncts (DNF). Verification succeeds
if ALL disjuncts are provably unreachable.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Constraint:
    """Threshold constraint: Y[index] op value."""
    index: int
    op: str       # '>=' or '<='
    value: float

    def margin(self, output_lo, output_hi):
        """Positive margin = verified safe."""
        if self.op == '>=':
            # Unsafe if Y[idx] >= val. Safe if hi < val.
            return self.value - output_hi[self.index]
        else:
            # Unsafe if Y[idx] <= val. Safe if lo > val.
            return output_lo[self.index] - self.value

    def __str__(self):
        return f'Y_{self.index} {self.op} {self.value}'


@dataclass
class PairwiseConstraint:
    """Pairwise constraint: unsafe if Y[comp] >= Y[pred]."""
    pred: int
    comp: int

    def margin(self, output_lo, output_hi):
        """Positive margin = pred provably beats comp."""
        return output_lo[self.pred] - output_hi[self.comp]

    def __str__(self):
        return f'Y_{self.comp} >= Y_{self.pred}'


@dataclass
class Conjunct:
    """Conjunction of constraints. All must hold for the unsafe region."""
    constraints: list

    def margin(self, output_lo, output_hi):
        """Worst margin across constraints. Positive = conjunction verified safe."""
        return min(c.margin(output_lo, output_hi) for c in self.constraints)

    def __str__(self):
        return ' AND '.join(str(c) for c in self.constraints)


@dataclass
class VNNSpec:
    """VNNLIB specification: input bounds + disjunction of conjuncts.

    The unsafe region is OR of conjuncts. Verified if ALL disjuncts have
    positive margin (every unsafe region is provably unreachable).
    """
    x_lo: np.ndarray
    x_hi: np.ndarray
    disjuncts: list  # list of Conjunct

    def check(self, output_lo, output_hi):
        """Check spec against output bounds.

        Returns:
            result: 'verified' or 'unknown'
            details: dict with margins per disjunct and worst_margin
        """
        margins = {}
        for i, conj in enumerate(self.disjuncts):
            margins[i] = conj.margin(output_lo, output_hi)

        worst = min(margins.values()) if margins else 0.0
        return ('verified' if worst > 0 else 'unknown'), {
            'margins': margins,
            'worst_margin': float(worst),
        }

    @property
    def n_constraints(self):
        return sum(len(d.constraints) for d in self.disjuncts)

    def __str__(self):
        parts = [f'input: {len(self.x_lo)}D  '
                 f'[{self.x_lo.min():.4f}, {self.x_hi.max():.4f}]']
        if len(self.disjuncts) == 1:
            parts.append(f'unsafe if: {self.disjuncts[0]}')
        else:
            parts.append(f'unsafe if any of {len(self.disjuncts)} disjuncts:')
            for i, d in enumerate(self.disjuncts):
                parts.append(f'  [{i}] {d}')
        return '\n'.join(parts)
