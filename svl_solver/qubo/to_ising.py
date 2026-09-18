"""QUBO -> dimod BQM / Ising (h, J) for annealers and QAOA."""

from __future__ import annotations

import dimod
import numpy as np

from .formulation import QuboProblem


def to_bqm(problem: QuboProblem) -> dimod.BinaryQuadraticModel:
    Q = problem.Q
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu, ju):
        v = Q[i, j]
        if v != 0.0:
            quadratic[(int(i), int(j))] = float(v)
    return dimod.BinaryQuadraticModel(linear, quadratic, problem.offset, dimod.BINARY)


def to_ising(problem: QuboProblem) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
    """Return (h, J, offset) with spins s ∈ {−1,+1}, x = (1 + s) / 2 (dimod convention)."""
    h, J, off = to_bqm(problem).to_ising()
    return dict(h), dict(J), float(off)
