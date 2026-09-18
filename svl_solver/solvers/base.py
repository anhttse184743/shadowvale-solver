"""Common solver interface (research objective O2).

Every solver, classical or quantum-inspired, sees ONLY a ``QuboProblem``:
the upper-triangular Q matrix plus the (n_agents, n_nodes) layout. None of
them may import the Instance or the objective; that is what makes the
comparison fair and lets Unity ship a Q matrix over HTTP.

    solver.solve(problem, budget_ms) -> Solution

``Solution.energy`` always includes the problem offset, so for a feasible
bit-vector it equals −F (minus the reference objective).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..qubo.formulation import QuboProblem


@dataclass
class Solution:
    x: np.ndarray
    energy: float                     # E(x) + offset
    latency_ms: float
    variant: str
    feasible: bool
    assignment: list[int]             # decoded; repaired if x was infeasible
    hit_budget: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def objective(self) -> float:
        """−energy. Equals the reference objective F when ``feasible``."""
        return -self.energy


class Solver(ABC):
    variant_id: str = "abstract"
    available: bool = True            # False for planned-but-not-implemented variants

    @abstractmethod
    def _solve(self, problem: QuboProblem, budget_ms: int | None, rng: np.random.Generator) -> np.ndarray:
        """Return a bit-vector of length problem.n_vars."""

    def solve(self, problem: QuboProblem, budget_ms: int | None = None,
              seed: int | None = None) -> Solution:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        x = np.asarray(self._solve(problem, budget_ms, rng), dtype=np.int8).ravel()
        latency = (time.perf_counter() - t0) * 1000.0
        if x.shape[0] != problem.n_vars:
            raise RuntimeError(f"{self.variant_id} returned {x.shape[0]} bits, expected {problem.n_vars}")
        assignment, feasible = decode(problem, x)
        meta: dict = {}
        if not feasible:
            assignment = greedy_repair(problem, x)
            meta["repaired"] = True
        energy = problem.energy_with_offset(x)
        hit = budget_ms is not None and latency > budget_ms
        return Solution(x=x, energy=energy, latency_ms=latency, variant=self.variant_id,
                        feasible=feasible, assignment=assignment, hit_budget=hit, meta=meta)


# ------------------------------------------------------------------ helpers
def decode(problem: QuboProblem, x: np.ndarray) -> tuple[list[int], bool]:
    A, N = problem.n_agents, problem.n_nodes
    m = np.asarray(x).reshape(A, N)
    one_each = bool(np.all(m.sum(axis=1) == 1))
    no_clash = bool(np.all(m.sum(axis=0) <= 1))
    assignment = []
    for a in range(A):
        idx = np.flatnonzero(m[a])
        assignment.append(int(idx[0]) if idx.size else -1)
    return assignment, (one_each and no_clash)


def marginal_gains(problem: QuboProblem, x: np.ndarray) -> np.ndarray:
    """ΔE_i of flipping x_i from 0 to 1 given the current x (ignoring x_i's own state)."""
    Q = problem.Q
    S = Q + Q.T
    np.fill_diagonal(S, 0.0)
    return np.diag(Q) + S @ np.asarray(x, dtype=float)


def greedy_repair(problem: QuboProblem, x: np.ndarray) -> list[int]:
    """Q-only repair: per agent keep the bit with the lowest marginal energy,
    then resolve node clashes and empty agents by the cheapest free node.
    Guarantees a feasible assignment for any input."""
    A, N = problem.n_agents, problem.n_nodes
    m = np.asarray(x).reshape(A, N).copy()
    taken: set[int] = set()
    out = [-1] * A
    for a in range(A):
        cand = [int(n) for n in np.flatnonzero(m[a]) if int(n) not in taken]
        if cand:
            cur = np.zeros(A * N)
            for b in range(a):
                if out[b] >= 0:
                    cur[b * N + out[b]] = 1.0
            gains = marginal_gains(problem, cur)
            best = min(cand, key=lambda n: (gains[a * N + n], n))
            out[a] = best
            taken.add(best)
    for a in range(A):
        if out[a] == -1:
            cur = np.zeros(A * N)
            for b in range(A):
                if out[b] >= 0:
                    cur[b * N + out[b]] = 1.0
            gains = marginal_gains(problem, cur)
            free = [n for n in range(N) if n not in taken]
            best = min(free, key=lambda n: (gains[a * N + n], n))
            out[a] = best
            taken.add(best)
    return out


def assignment_to_x(assignment: list[int], n_agents: int, n_nodes: int) -> np.ndarray:
    x = np.zeros(n_agents * n_nodes, dtype=np.int8)
    for a, n in enumerate(assignment):
        x[a * n_nodes + int(n)] = 1
    return x
