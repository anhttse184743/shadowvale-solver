"""PBIL (Baluja 1994) on the same bit representation as QIEA — the ablation of the Q-bit.

QIEA keeps, per individual, a probability per bit (sin²θ) and moves it towards the best
solution by a fixed rotation. PBIL keeps one probability vector and moves it towards the best
sample of each generation by a learning rate. Everything else is QIEA's: bits are sampled
independently, repaired into an assignment by the same rule (:func:`qiea._repair`), the
incumbent starts at the greedy solution, probabilities are clamped away from 0 and 1, and the
solver stops at ``time_fraction`` of the budget. If QIEA and PBIL score alike, the Q-bit
rotation is not what makes QIEA accurate; an estimation-of-distribution search is.
"""

from __future__ import annotations

import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver
from .greedy import GreedySolver
from .qiea import _energy, _repair


class PbilSolver(Solver):
    variant_id = "pbil"

    # Defaults from `svl-bench tune` (exp-v2, 28/10/2026, experiments/tuning-v2), 30 trials like every
    # solver: mean gap 0.42 % → 0.23 % on the tuning set (QIEA, tuned the same way in exp-v1: 0.30 %).
    def __init__(self, samples: int = 29, generations: int = 150, learning_rate: float = 0.2035,
                 margin: float = 0.0203, time_fraction: float = 0.8) -> None:
        self.samples = max(2, int(samples))
        self.generations = max(1, int(generations))
        self.lr = float(learning_rate)
        self.lo, self.hi = float(margin), 1.0 - float(margin)
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
        p = np.full(n, 0.5)
        samples = raw_feasible = generations_done = 0
        stopped = False
        for _ in range(self.generations):
            gen_best_e, gen_best_idx = np.inf, None
            for _ in range(self.samples):
                if deadline is not None and time.perf_counter() >= deadline:
                    stopped = True
                    break
                observed = (rng.random(n) < p).reshape(A, N)
                if np.all(observed.sum(axis=1) == 1) and np.all(observed.sum(axis=0) <= 1):
                    raw_feasible += 1
                idx = agent_base + _repair(observed, diag, S, A, N, rng)
                e = _energy(Q, idx) + problem.offset
                samples += 1
                if e < gen_best_e:
                    gen_best_e, gen_best_idx = e, idx
            if gen_best_idx is not None:
                if gen_best_e < best_e - 1e-12:
                    best_e, best_idx = gen_best_e, gen_best_idx
                target = np.zeros(n)
                target[gen_best_idx] = 1.0
                p = (1.0 - self.lr) * p + self.lr * target
                np.clip(p, self.lo, self.hi, out=p)
            if stopped:
                break
            generations_done += 1

        x = np.zeros(n, dtype=np.int8)
        x[best_idx] = 1
        return x, {"samples_evaluated": samples, "raw_feasible_samples": raw_feasible,
                   "generations": generations_done, "stopped_on_budget": stopped}
