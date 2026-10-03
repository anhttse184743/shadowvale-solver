"""Quantum-inspired evolutionary algorithm (Han & Kim, 2002) on the coordination QUBO.

Runs on an ordinary CPU; "quantum" refers only to the representation:

* each individual is a string of Q-bits, stored as angles θ_i ∈ (0, π/2) with
  P(bit i = 1) = sin²θ_i, all starting at π/4 (every bit 50/50);
* an *observation* samples a bit-string from those probabilities;
* a *rotation gate* nudges θ_i by ±Δθ towards the best solution found so far,
  only where the observed bit disagrees with it (the non-zero rows of Han & Kim's
  lookup table), and θ is clamped away from 0 and π/2 so no bit ever freezes.

Observed strings rarely satisfy the one-hot constraints, so each one is repaired
into an assignment before scoring: agents in a random cyclic order take, among the
nodes their observation switched on, the free node with the lowest marginal energy
(any free node if none was switched on). ``raw_feasible_samples`` counts how many
observations were feasible *before* repair — a metric reported for RQ2.

The incumbent starts at the greedy solution, so QIEA is never worse than greedy.
With ``budget_ms`` it stops at ``time_fraction`` of the budget and returns the best
incumbent; with ``budget_ms=None`` it runs exactly ``generations × population``
samples and is fully reproducible for a given seed.

Ported from the in-game C# prototype (QieaSolver.cs, removed from Unity on
2026-09-30 so that every research solver has a single implementation).
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver
from .greedy import GreedySolver


class QuantumInspiredEvolutionarySolver(Solver):
    variant_id = "qiea"

    # Defaults from `svl-bench tune` (03/10/2026, experiments/tuning): mean gap 0.45 % → 0.30 % on the
    # tuning set (8 generated + 16 recorded game instances, 120 ms), same 30 trials as every solver.
    def __init__(self, population: int = 29, generations: int = 80, delta: float = 0.158,
                 margin: float = 0.271, time_fraction: float = 0.8) -> None:
        self.population = max(2, int(population))
        self.generations = max(1, int(generations))
        self.delta = float(delta)
        self.lo, self.hi = float(margin), math.pi / 2 - float(margin)
        self.time_fraction = float(time_fraction)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        A, N = problem.n_agents, problem.n_nodes
        n = A * N
        Q = problem.Q
        S = Q + Q.T
        np.fill_diagonal(S, 0.0)
        diag = np.diag(Q).copy()
        agent_base = np.arange(A) * N

        best_idx = np.flatnonzero(GreedySolver()._solve(problem, None, rng))
        best_e = _energy(Q, best_idx) + problem.offset
        best_bits = np.zeros(n, dtype=bool)
        best_bits[best_idx] = True

        theta = np.full((self.population, n), math.pi / 4)
        samples = raw_feasible = generations_done = 0
        stopped = False
        for _ in range(self.generations):
            for p in range(self.population):
                if deadline is not None and time.perf_counter() >= deadline:
                    stopped = True
                    break
                observed = (rng.random(n) < np.sin(theta[p]) ** 2).reshape(A, N)
                if np.all(observed.sum(axis=1) == 1) and np.all(observed.sum(axis=0) <= 1):
                    raw_feasible += 1
                idx = agent_base + _repair(observed, diag, S, A, N, rng)
                e = _energy(Q, idx) + problem.offset
                samples += 1
                if e < best_e - 1e-12:
                    best_e, best_idx = e, idx
                    best_bits[:] = False
                    best_bits[idx] = True
                bits = np.zeros(n, dtype=bool)
                bits[idx] = True
                differ = bits != best_bits
                theta[p, differ] += np.where(best_bits[differ], self.delta, -self.delta)
                np.clip(theta[p], self.lo, self.hi, out=theta[p])
            if stopped:
                break
            generations_done += 1

        x = np.zeros(n, dtype=np.int8)
        x[best_idx] = 1
        return x, {"samples_evaluated": samples, "raw_feasible_samples": raw_feasible,
                   "generations": generations_done, "stopped_on_budget": stopped}


def _energy(Q: np.ndarray, idx: np.ndarray) -> float:
    """E(x) for the one-hot x with ones at ``idx`` (ascending). Q is upper-triangular,
    so summing the whole sub-matrix counts every pair exactly once."""
    return float(Q[np.ix_(idx, idx)].sum())


def _repair(observed: np.ndarray, diag: np.ndarray, S: np.ndarray, A: int, N: int,
            rng: np.random.Generator) -> np.ndarray:
    """Observation → feasible assignment (node per agent). A random cyclic starting
    agent removes the fixed first-agent advantage of a plain index-order repair."""
    taken = np.zeros(N, dtype=bool)
    targets = np.empty(A, dtype=np.int64)
    gains = diag.copy()
    start = int(rng.integers(A))
    for turn in range(A):
        a = (start + turn) % A
        seg = gains[a * N:(a + 1) * N].copy()
        seg[taken] = np.inf
        preferred = np.where(observed[a], seg, np.inf)
        node = int(np.argmin(preferred if np.isfinite(preferred).any() else seg))
        targets[a] = node
        taken[node] = True
        gains += S[:, a * N + node]
    return targets
