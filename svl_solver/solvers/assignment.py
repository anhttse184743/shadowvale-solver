"""Assignment-space search: Greedy + local search, tabu search, and SA with one-hot-preserving moves.

Added for exp-v2 after the review of exp-v1 (M3): GA and QIEA search over assignments (one
node per agent, repaired), while SA and SQA flip raw bits of the QUBO. These three solvers move
between assignments only, so the constraints hold at every step and the comparison separates
the search strategy from the encoding. Like every solver they see only the ``QuboProblem``.

With U the diagonal of Q (shape A×N) and P[a, b, n, m] the coefficient of x[a,n]·x[b,m]
(symmetric in (a,n) ↔ (b,m)), the energy of an assignment σ is

    E(σ) = Σ_a U[a, σa] + Σ_{a<b} P[a, b, σa, σb] + offset,

and C[a, n] = U[a, n] + Σ_{b≠a} P[a, b, n, σb] is agent a's cost at node n given the others.
Two moves, both evaluated in O(1) from C and updated in O(A·N):

* reassign a → free node n:   Δ = C[a, n] − C[a, σa]
* swap a, b (σa = i, σb = j): Δ = C[a, j] − C[a, i] + C[b, i] − C[b, j]
                                  − P[a,b,j,j] − P[a,b,i,i] + P[a,b,i,j] + P[a,b,j,i]

All three start from the greedy assignment and, with a budget, stop at ``time_fraction`` of it
(the convention of every stochastic solver here); without one they run a fixed number of
iterations and are reproducible for a given seed.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, assignment_to_x
from .greedy import GreedySolver


class AssignmentModel:
    """Incremental energy of an injective assignment agent → node."""

    def __init__(self, problem: QuboProblem):
        A, N = problem.n_agents, problem.n_nodes
        if A > N:
            raise ValueError(f"{A} agents on {N} nodes")
        self.A, self.N = A, N
        Q = problem.Q
        self.U = np.diag(Q).reshape(A, N).copy()
        S = np.triu(Q, 1)
        S = S + S.T
        self.P = S.reshape(A, N, A, N).transpose(0, 2, 1, 3).copy()   # P[a, b, n, m]
        for a in range(A):
            self.P[a, a] = 0.0
        self.offset = float(problem.offset)

    def reset(self, sigma: np.ndarray) -> None:
        self.sigma = np.asarray(sigma, dtype=np.int64).copy()
        A = self.A
        self.C = self.U + self.P[np.arange(A)[:, None], np.arange(A)[None, :], :, self.sigma[None, :]].sum(axis=1)
        self.occupied = np.zeros(self.N, dtype=bool)
        self.occupied[self.sigma] = True
        self.energy = self._energy()

    def _energy(self) -> float:
        a = np.arange(self.A)
        unary = self.U[a, self.sigma].sum()
        pair = self.P[a[:, None], a[None, :], self.sigma[:, None], self.sigma[None, :]].sum() / 2.0
        return float(unary + pair + self.offset)

    # ---- move evaluation
    def reassign_deltas(self) -> np.ndarray:
        """(A, N) Δ of moving each agent to each node; +inf on occupied nodes."""
        cur = self.C[np.arange(self.A), self.sigma]
        d = self.C - cur[:, None]
        d[:, self.occupied] = np.inf
        return d

    def swap_deltas(self) -> np.ndarray:
        """(A, A) Δ of swapping two agents' nodes; +inf on the diagonal."""
        A, s = self.A, self.sigma
        a = np.arange(A)
        cur = self.C[a, s]
        c_at = self.C[:, s]                                   # c_at[a, b] = C[a, σb]
        P = self.P
        aa, bb = a[:, None], a[None, :]
        si, sj = s[:, None], s[None, :]
        d = (c_at - cur[:, None]) + (c_at.T - cur[None, :]) \
            - P[aa, bb, sj, sj] - P[aa, bb, si, si] + P[aa, bb, si, sj] + P[aa, bb, sj, si]
        np.fill_diagonal(d, np.inf)
        return d

    # ---- moves
    def move(self, a: int, n: int) -> None:
        old = int(self.sigma[a])
        delta = self.C[a, n] - self.C[a, old]
        self.C += self.P[:, a, :, n] - self.P[:, a, :, old]   # agent a's own row is untouched (P[a,a]=0)
        self.sigma[a] = n
        self.occupied[old] = False
        self.occupied[n] = True
        self.energy += float(delta)

    def swap(self, a: int, b: int) -> None:
        i, j = int(self.sigma[a]), int(self.sigma[b])
        delta = float(self.swap_deltas()[a, b])
        self.C += self.P[:, a, :, j] - self.P[:, a, :, i] + self.P[:, b, :, i] - self.P[:, b, :, j]
        self.sigma[a], self.sigma[b] = j, i
        self.energy += delta


def _greedy_sigma(problem: QuboProblem, rng) -> np.ndarray:
    x = GreedySolver()._solve(problem, None, rng)
    return np.argmax(np.asarray(x).reshape(problem.n_agents, problem.n_nodes), axis=1)


def _best_move(m: AssignmentModel):
    """Steepest move over both neighbourhoods: (Δ, kind, i, j)."""
    r = m.reassign_deltas()
    ra, rn = np.unravel_index(int(np.argmin(r)), r.shape)
    best = (float(r[ra, rn]), "move", int(ra), int(rn))
    if m.A > 1:
        s = m.swap_deltas()
        sa, sb = np.unravel_index(int(np.argmin(s)), s.shape)
        if s[sa, sb] < best[0]:
            best = (float(s[sa, sb]), "swap", int(sa), int(sb))
    return best


def _apply(m: AssignmentModel, kind: str, i: int, j: int) -> None:
    if kind == "move":
        m.move(i, j)
    else:
        m.swap(i, j)


class GreedyLocalSearchSolver(Solver):
    """Greedy, then steepest descent over reassign + swap moves until no move improves.
    Deterministic; usually well under a millisecond on game-size instances."""

    variant_id = "ls"

    def __init__(self, max_iterations: int = 10_000):
        self.max_iterations = int(max_iterations)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * 0.8 / 1000.0
        m = AssignmentModel(problem)
        m.reset(_greedy_sigma(problem, rng))
        steps = evals = 0
        for _ in range(self.max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            d, kind, i, j = _best_move(m)
            evals += m.A * m.N + m.A * m.A
            if d >= -1e-12:
                break
            _apply(m, kind, i, j)
            steps += 1
        return assignment_to_x(list(m.sigma), m.A, m.N), {"samples_evaluated": evals, "moves": steps,
                                                           "raw_feasible_samples": evals}


class TabuSolver(Solver):
    """Tabu search (Glover 1989; tenure drawn at random as in Taillard's robust tabu search, 1991)
    over reassign + swap moves. Each iteration takes the best non-tabu move, improving or not;
    a tabu move is allowed when it beats the best energy found (aspiration). After moving an
    agent away from a node, putting that agent back on that node is tabu for a random tenure in
    [tenure_min, tenure_max] · A iterations. After ``restart_after`` iterations without a new best
    it restarts from the best with a random perturbation of ``kick`` reassignments."""

    variant_id = "tabu"

    # Defaults from `svl-bench tune` (exp-v2, 28/10/2026, experiments/tuning-v2), 30 trials like every
    # solver: mean gap 0.00 % on the tuning set with the first defaults and with these.
    def __init__(self, tenure_min: float = 0.766, tenure_max: float = 1.250, restart_after: int = 52,
                 kick: int = 4, iterations: int = 2000, time_fraction: float = 0.8):
        self.tenure_min, self.tenure_max = float(tenure_min), float(max(tenure_min, tenure_max))
        self.restart_after = max(10, int(restart_after))
        self.kick = max(1, int(kick))
        self.iterations = max(1, int(iterations))
        self.time_fraction = float(time_fraction)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        m = AssignmentModel(problem)
        m.reset(_greedy_sigma(problem, rng))
        A, N = m.A, m.N
        best_sigma, best_e = m.sigma.copy(), m.energy
        tabu_until = np.zeros((A, N), dtype=np.int64)        # iteration until which agent a may not take node n
        it = since_best = evals = restarts = 0

        def tenure() -> int:
            return max(1, int(round(rng.uniform(self.tenure_min, self.tenure_max) * max(A, 2))))

        while True:
            if deadline is not None:
                if time.perf_counter() >= deadline:
                    break
            elif it >= self.iterations:
                break
            it += 1
            aspire = best_e - m.energy - 1e-12                # a move with Δ below this beats the best
            r = m.reassign_deltas()
            evals += A * N
            r_ok = np.where((tabu_until > it) & ~(r < aspire), np.inf, r)
            cand = []
            if np.isfinite(r_ok).any():
                ra, rn = np.unravel_index(int(np.argmin(r_ok + rng.random(r_ok.shape) * 1e-9)), r_ok.shape)
                cand.append((float(r_ok[ra, rn]), "move", int(ra), int(rn)))
            if A > 1:
                s = m.swap_deltas()
                evals += A * A
                s_tabu = (tabu_until[np.arange(A)[:, None], m.sigma[None, :]] > it) | \
                         (tabu_until[np.arange(A)[None, :], m.sigma[:, None]] > it)
                s_ok = np.where(s_tabu & ~(s < aspire), np.inf, s)
                if np.isfinite(s_ok).any():
                    sa, sb = np.unravel_index(int(np.argmin(s_ok + rng.random(s_ok.shape) * 1e-9)), s_ok.shape)
                    cand.append((float(s_ok[sa, sb]), "swap", int(sa), int(sb)))
            if not cand:                                        # everything tabu: let the oldest go
                tabu_until[:] = 0
                continue
            _, kind, i, j = min(cand)
            if kind == "move":
                tabu_until[i, m.sigma[i]] = it + tenure()
            else:
                tabu_until[i, m.sigma[i]] = it + tenure()
                tabu_until[j, m.sigma[j]] = it + tenure()
            _apply(m, kind, i, j)
            if m.energy < best_e - 1e-12:
                best_e, best_sigma, since_best = m.energy, m.sigma.copy(), 0
            else:
                since_best += 1
            if since_best >= self.restart_after and N > A:
                m.reset(best_sigma)
                for _ in range(self.kick):
                    a = int(rng.integers(A))
                    free = np.flatnonzero(~m.occupied)
                    m.move(a, int(rng.choice(free)))
                tabu_until[:] = 0
                since_best = 0
                restarts += 1
        return assignment_to_x(list(best_sigma), A, N), {"samples_evaluated": evals, "iterations": it,
                                                          "restarts": restarts, "raw_feasible_samples": evals}


class AssignmentAnnealingSolver(Solver):
    """Simulated annealing (Kirkpatrick et al. 1983) over assignments: a random reassign or swap
    move, Metropolis acceptance, geometric cooling from T0 to T1. T0 and T1 are set from the
    mean |Δ| of a few random moves at the start (T0 accepts an average worsening move with
    probability ``p_start``, T1 with ``p_end``), so no penalty weight enters the schedule.
    The counterpart of the bit-flip SA (``sa``, dwave-neal) with constraint-preserving moves."""

    variant_id = "sa1h"

    # Defaults from `svl-bench tune` (exp-v2, experiments/tuning-v2): mean gap 0.90 % → 0.33 %.
    def __init__(self, p_start: float = 0.424, p_end: float = 1.97e-5, iterations: int = 20_000,
                 swap_share: float = 0.398, time_fraction: float = 0.8):
        self.p_start, self.p_end = float(p_start), float(p_end)
        self.iterations = max(10, int(iterations))
        self.swap_share = float(swap_share)
        self.time_fraction = float(time_fraction)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        m = AssignmentModel(problem)
        m.reset(_greedy_sigma(problem, rng))
        A, N = m.A, m.N
        best_sigma, best_e = m.sigma.copy(), m.energy
        if N == A and A < 2:
            return assignment_to_x(list(best_sigma), A, N), {"samples_evaluated": 0}

        def propose():
            if A > 1 and (N == A or rng.random() < self.swap_share):
                a, b = rng.choice(A, size=2, replace=False)
                i, j = int(m.sigma[a]), int(m.sigma[b])
                P = m.P
                d = (m.C[a, j] - m.C[a, i] + m.C[b, i] - m.C[b, j]
                     - P[a, b, j, j] - P[a, b, i, i] + P[a, b, i, j] + P[a, b, j, i])
                return float(d), "swap", int(a), int(b)
            a = int(rng.integers(A))
            n = int(rng.choice(np.flatnonzero(~m.occupied)))
            return float(m.C[a, n] - m.C[a, m.sigma[a]]), "move", a, n

        probe = [abs(propose()[0]) for _ in range(min(50, 5 * A))]
        scale = float(np.mean([p for p in probe if p > 1e-12] or [1.0]))
        T0 = scale / -math.log(self.p_start)
        T1 = scale / -math.log(self.p_end)
        # Planned length: by iterations, or re-estimated from the time per step under a budget.
        planned, it, evals = self.iterations, 0, len(probe)
        t_loop = time.perf_counter()
        while it < planned:
            if deadline is not None and (it & 63) == 0:
                now = time.perf_counter()
                if now >= deadline:
                    break
                if it >= 64:
                    per = (now - t_loop) / it
                    planned = it + max(1, int((deadline - now) / max(per, 1e-9)))
            frac = it / max(planned - 1, 1)
            T = T0 * (T1 / T0) ** frac
            d, kind, i, j = propose()
            evals += 1
            if d <= 0 or rng.random() < math.exp(-d / T):
                _apply(m, kind, i, j)
                if m.energy < best_e - 1e-12:
                    best_e, best_sigma = m.energy, m.sigma.copy()
            it += 1
        return assignment_to_x(list(best_sigma), A, N), {"samples_evaluated": evals, "iterations": it,
                                                          "raw_feasible_samples": evals}
