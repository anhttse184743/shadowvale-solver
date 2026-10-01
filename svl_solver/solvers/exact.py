"""Exact optimum for game-size instances by branch and bound → approximation ratio (O4).

Brute force stops at N!/(N−A)! ≈ 5·10⁶ candidates; a game re-plan with 6 guards over a
24-node subgraph has ~9.6·10⁷. This solver searches the same space — injective maps
agent → node, which contain the QUBO optimum (every infeasible bit-vector is dominated, see
qubo/penalties.py) — but prunes with a lower bound, and proves optimality when it finishes.

Energy of an assignment n: Σ_a U[a, n_a] + Σ_{a<b} W[a, n_a, b, n_b], with U the diagonal of Q
and W its off-diagonal blocks. Agents are placed in order; with agents 0..k−1 placed, the
bound for the rest is

    Σ_{r ≥ k}  min_{n free} ( C[r, n] + ½ Σ_{r' ≥ k, r' ≠ r} min_m W[r, n, r', m] )

where C[r, n] = U[r, n] + Σ_{b < k} W[r, n, b, n_b] is exact for the placed agents. Each pair
among unplaced agents is split in two halves and each half bounded from below by its own
minimum, so the bound never exceeds the true energy of any completion. The incumbent starts
from greedy, children are tried best bound first.

No dependency beyond numpy (OR-Tools CP-SAT would do the same job; this keeps the bench
installable offline). ``meta`` reports ``proven_optimal`` and ``nodes_explored``; with a time
limit the best assignment found so far is returned unproven.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, assignment_to_x
from .greedy import GreedySolver


class BranchAndBoundSolver(Solver):
    variant_id = "exact"

    def __init__(self, time_limit_s: float | None = 120.0):
        self.time_limit_s = time_limit_s

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        A, N = problem.n_agents, problem.n_nodes
        if N < A:
            raise ValueError("fewer nodes than agents: no injective assignment")
        Q = problem.Q
        U = np.diag(Q).reshape(A, N).copy()
        # W[a, n, b, m] for a ≠ b, symmetric; Q is upper triangular and index a·N+n < b·N+m when a < b.
        W = np.zeros((A, N, A, N))
        for a in range(A):
            for b in range(a + 1, A):
                block = Q[a * N:(a + 1) * N, b * N:(b + 1) * N]
                W[a, :, b, :] = block
                W[b, :, a, :] = block.T
        Mmin = W.min(axis=3)                                   # (A, N, A): min over the other's node
        for a in range(A):
            Mmin[a, :, a] = 0.0
        # H[k][r, n] = ½ Σ_{r' ≥ k, r' ≠ r} Mmin[r, n, r']  — the pair bound among unplaced agents.
        H = [0.5 * Mmin[:, :, k:].sum(axis=2) for k in range(A + 1)]

        greedy = GreedySolver().solve(problem)
        best_assign = list(greedy.assignment)
        best_e = self._energy(U, W, best_assign)

        deadline = None
        limit = self.time_limit_s if budget_ms is None else budget_ms / 1000.0
        if limit is not None:
            deadline = time.perf_counter() + limit
        nodes = 0
        complete = True

        C = U.copy()
        used = np.zeros(N, dtype=bool)
        assign = [-1] * A

        def search(k: int, cost: float) -> None:
            nonlocal best_e, best_assign, nodes, complete
            nodes += 1
            if k == A:
                if cost < best_e - 1e-12:
                    best_e, best_assign = cost, assign.copy()
                return
            if deadline is not None and (nodes & 255) == 0 and time.perf_counter() > deadline:
                complete = False
                return
            free = ~used
            h = H[k]
            # Bound for the agents still to place (k..A−1), each at its best free node.
            rest = (C[k:, :] + h[k:, :])[:, free]
            if cost + rest.min(axis=1).sum() >= best_e - 1e-12:
                return
            own = C[k] + h[k]
            for n in np.argsort(np.where(free, own, np.inf), kind="stable"):
                if not free[n]:
                    break
                if not complete:
                    return
                step = C[k, n]
                used[n] = True
                assign[k] = int(n)
                if k + 1 < A:
                    C[k + 1:, :] += W[k + 1:, :, k, n]
                search(k + 1, cost + step)
                if k + 1 < A:
                    C[k + 1:, :] -= W[k + 1:, :, k, n]
                used[n] = False
                assign[k] = -1

        search(0, 0.0)
        x = assignment_to_x(best_assign, A, N)
        return x, {"proven_optimal": complete, "nodes_explored": nodes,
                   "search_space": math.perm(N, A)}

    @staticmethod
    def _energy(U: np.ndarray, W: np.ndarray, assign: list[int]) -> float:
        A = len(assign)
        e = sum(U[a, assign[a]] for a in range(A))
        for a in range(A):
            for b in range(a + 1, A):
                e += W[a, assign[a], b, assign[b]]
        return float(e)
