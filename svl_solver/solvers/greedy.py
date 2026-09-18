"""Classical baseline 1: greedy sequential assignment.

Agents are processed in index order; each takes the free node with the lowest
marginal energy given the agents already placed. O(A²·N) and always feasible,
which is why the game uses it as the in-process fallback when the sidecar
misses the latency budget. Unity's ``GreedySolver.cs`` must mirror this exact
rule so that the fallback path is reproducible.
"""

from __future__ import annotations

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, marginal_gains


class GreedySolver(Solver):
    variant_id = "greedy"

    def _solve(self, problem: QuboProblem, budget_ms, rng) -> np.ndarray:
        A, N = problem.n_agents, problem.n_nodes
        x = np.zeros(A * N, dtype=float)
        taken: set[int] = set()
        for a in range(A):
            gains = marginal_gains(problem, x)
            best_n, best_g = -1, np.inf
            for n in range(N):
                if n in taken:
                    continue
                g = gains[a * N + n]
                if g < best_g:
                    best_g, best_n = g, n
            x[a * N + best_n] = 1.0
            taken.add(best_n)
        return x.astype(np.int8)
