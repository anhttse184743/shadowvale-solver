"""Exact optimum for small instances → denominator of the approximation ratio.

Two enumeration spaces:
  "assignments" (default): all injective maps agent→node, N!/(N−A)! candidates.
                           This IS the optimum of the QUBO because every
                           infeasible x is dominated (see qubo/penalties.py).
  "bitstrings":            all 2^n bit-vectors, n ≤ 22. Slower; used by tests
                           to prove the claim above rather than assume it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, assignment_to_x


class BruteForceSolver(Solver):
    variant_id = "brute"

    def __init__(self, space: str = "assignments", max_candidates: int = 5_000_000,
                 max_bits: int = 22):
        if space not in ("assignments", "bitstrings"):
            raise ValueError("space must be 'assignments' or 'bitstrings'")
        self.space = space
        self.max_candidates = max_candidates
        self.max_bits = max_bits

    def _solve(self, problem: QuboProblem, budget_ms, rng) -> np.ndarray:
        if self.space == "assignments":
            return self._enumerate_assignments(problem)
        return self._enumerate_bitstrings(problem)

    # ---------------------------------------------------------------- spaces
    def _enumerate_assignments(self, problem: QuboProblem) -> np.ndarray:
        A, N = problem.n_agents, problem.n_nodes
        count = math.perm(N, A)
        if count > self.max_candidates:
            raise ValueError(f"{count} candidate assignments exceed max_candidates={self.max_candidates}")
        Q = problem.Q
        diag = np.diag(Q)
        best_e = math.inf
        best: tuple[int, ...] = ()
        for perm in itertools.permutations(range(N), A):
            idx = [a * N + perm[a] for a in range(A)]
            e = 0.0
            for a in range(A):
                e += diag[idx[a]]
                for b in range(a + 1, A):
                    e += Q[idx[a], idx[b]]          # idx[a] < idx[b] since a < b
            if e < best_e:
                best_e, best = e, perm
        return assignment_to_x(list(best), A, N)

    def _enumerate_bitstrings(self, problem: QuboProblem) -> np.ndarray:
        n = problem.n_vars
        if n > self.max_bits:
            raise ValueError(f"{n} bits exceed max_bits={self.max_bits}")
        Q = problem.Q
        best_e = math.inf
        best_x = None
        chunk = 1 << min(n, 16)
        total = 1 << n
        ar = np.arange(n)
        for start in range(0, total, chunk):
            codes = np.arange(start, min(start + chunk, total), dtype=np.int64)
            X = ((codes[:, None] >> ar[None, :]) & 1).astype(float)   # (chunk, n)
            E = np.einsum("ki,ij,kj->k", X, Q, X)
            k = int(np.argmin(E))
            if E[k] < best_e:
                best_e, best_x = float(E[k]), X[k].astype(np.int8)
        return best_x


def all_bitstring_energies(problem: QuboProblem) -> tuple[np.ndarray, np.ndarray]:
    """(X, E) for every bit-vector; test helper, n ≤ 16."""
    n = problem.n_vars
    if n > 16:
        raise ValueError("too many bits for exhaustive listing")
    codes = np.arange(1 << n, dtype=np.int64)
    X = ((codes[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    E = np.einsum("ki,ij,kj->k", X.astype(float), problem.Q, X.astype(float)) + problem.offset
    return X, E
