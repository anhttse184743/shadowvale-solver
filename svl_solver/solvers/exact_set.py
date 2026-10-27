"""Exact optimum through the reduced formulation: choose a set of nodes, then assign (review M4).

In ShadowVale's QUBO only the travel cost d[a, n] depends on the agent: coverage, cover quality,
redundancy and flanking depend on the nodes alone. Then the energy of an assignment σ with node
set S = σ(agents) is

    E(σ) = G(S) + Σ_a U[a, σa] + offset,     G(S) = Σ_{n<m in S} W[n, m],

so the optimum is  min over |S| = A  of  G(S) + min-cost assignment of the agents onto S
(Hungarian method). There are C(N, A) sets instead of N!/(N−A)! assignments — 134,596 against
96,909,120 for 6 agents over 24 nodes.

The solver reads the structure from Q (it sees only the ``QuboProblem``, like every solver):
it checks that every pair of agents has the same symmetric off-diagonal block W and raises
``ValueError`` otherwise (QAPLIB, for instance), so a caller can fall back to branch and bound.
Sets are ranked by G(S) plus a lower bound on the assignment cost, and the Hungarian method is
run in that order until the bound passes the incumbent: the result is a proven optimum.
Reference only (not a real-time candidate, although it often finishes within a budget).
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, assignment_to_x


def hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost perfect matching of a square matrix (Kuhn–Munkres with potentials, O(k³));
    returns the column of each row. Written out to keep the bench free of scipy."""
    k = cost.shape[0]
    INF = math.inf
    u = [0.0] * (k + 1)
    v = [0.0] * (k + 1)
    p = [0] * (k + 1)
    way = [0] * (k + 1)
    c = cost.tolist()
    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (k + 1)
        used = [False] * (k + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            row = c[i0 - 1]
            for j in range(1, k + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col = np.empty(k, dtype=np.int64)
    for j in range(1, k + 1):
        col[p[j] - 1] = j - 1
    return col


def node_structure(problem: QuboProblem, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray] | None:
    """(U, W) when every agent pair shares one symmetric node-pair block W (diagonal excluded),
    else None. U is the (A, N) diagonal of Q."""
    A, N = problem.n_agents, problem.n_nodes
    Q = problem.Q
    U = np.diag(Q).reshape(A, N).copy()
    if A < 2:
        return U, np.zeros((N, N))
    W = Q[0:N, N:2 * N].copy()
    off = ~np.eye(N, dtype=bool)
    if not np.allclose(W[off], W.T[off], atol=tol, rtol=0):
        return None
    for a in range(A):
        for b in range(a + 1, A):
            blk = Q[a * N:(a + 1) * N, b * N:(b + 1) * N]
            if not np.allclose(blk[off], W[off], atol=tol, rtol=0):
                return None
    W = np.where(off, W, 0.0)
    return U, W


class SetAssignmentExactSolver(Solver):
    variant_id = "exact_set"

    def __init__(self, max_sets: int = 30_000_000, chunk: int = 400_000):
        self.max_sets = int(max_sets)
        self.chunk = int(chunk)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        A, N = problem.n_agents, problem.n_nodes
        st = node_structure(problem)
        if st is None:
            raise ValueError("agent pairs do not share one node-pair block: no set reduction")
        total = math.comb(N, A)
        if total > self.max_sets:
            raise ValueError(f"C({N},{A}) = {total} sets exceeds {self.max_sets}")
        U, W = st
        t0 = time.perf_counter()
        pairs = list(itertools.combinations(range(A), 2))
        pi = np.array([p[0] for p in pairs], dtype=np.int64)
        pj = np.array([p[1] for p in pairs], dtype=np.int64)
        col_min = U.min(axis=0)                                # cheapest agent per node

        # Pass 1: score every set by G(S) + a lower bound on its assignment cost.
        keys, sets = [], []
        it = itertools.combinations(range(N), A)
        while True:
            block = np.fromiter(itertools.chain.from_iterable(itertools.islice(it, self.chunk)),
                                dtype=np.int16)
            if block.size == 0:
                break
            S = block.reshape(-1, A).astype(np.int64)
            g = W[S[:, pi], S[:, pj]].sum(axis=1) if pairs else np.zeros(len(S))
            lb_rows = U[:, S].min(axis=2).sum(axis=0)          # each agent at its best node of S
            lb_cols = col_min[S].sum(axis=1)                    # each node by its best agent
            keys.append(g + np.maximum(lb_rows, lb_cols))
            sets.append(S.astype(np.int16))
        key = np.concatenate(keys)
        S_all = np.concatenate(sets)
        order = np.argsort(key, kind="stable")

        # Pass 2: exact assignment cost in bound order until the bound passes the incumbent.
        best_e, best_sigma, solved = math.inf, None, 0
        for k in order:
            if key[k] >= best_e - 1e-12:
                break
            S = S_all[k].astype(np.int64)
            cols = hungarian(U[:, S])
            rows = np.arange(A)
            e = float(W[S[pi], S[pj]].sum() + U[rows, S[cols]].sum()) if pairs else float(U[rows, S[cols]].sum())
            solved += 1
            if e < best_e - 1e-12:
                best_e = e
                best_sigma = np.empty(A, dtype=np.int64)
                best_sigma[rows] = S[cols]
        return assignment_to_x(list(best_sigma), A, N), {
            "proven_optimal": True, "sets": int(total), "assignments_solved": solved,
            "seconds": time.perf_counter() - t0}
