"""Classical baseline 3: simulated annealing (dwave-neal).

neal is thermal (classical) annealing on the binary quadratic model — there is
nothing quantum about it. That is exactly why it is the classical counterpart of
``sqa``: same schedule idea, no transverse field, no Trotter replicas.

With ``budget_ms`` it keeps drawing batches of ``batch`` anneals until
``time_fraction`` of the budget has passed; otherwise it does ``num_reads``
reads. The lowest-energy sample wins; the base class repairs it if it violates a
constraint. ``raw_feasible_samples`` counts samples that needed no repair.

Schedule: geometric β from 0.3 (hot enough to cross the one-hot penalty barrier,
λ ≈ 10) to 300 (cold enough to resolve objective differences of ~0.05). Tuned on
20 seeded instances (30/09/2026): 8/20 optimal vs 1/20 with neal's default range.
Single-bit Metropolis moves have to pass through an infeasible state to move an
agent, which is why SA trails SQA/GA/QIEA on this encoding — a result, not a bug.
"""

from __future__ import annotations

import time

import neal
import numpy as np

from ..qubo.formulation import QuboProblem
from ..qubo.to_ising import to_bqm
from .base import Solver, decode


class ClassicalAnnealingSolver(Solver):
    variant_id = "sa"

    def __init__(self, num_sweeps: int = 500, num_reads: int = 64, batch: int = 8,
                 beta_range: tuple[float, float] = (0.3, 300.0), time_fraction: float = 0.8) -> None:
        self.num_sweeps = max(1, int(num_sweeps))
        self.num_reads = max(1, int(num_reads))
        self.batch = max(1, int(batch))
        self.beta_range = beta_range
        self.time_fraction = float(time_fraction)
        self._sampler = neal.SimulatedAnnealingSampler()

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        bqm = to_bqm(problem)
        n = problem.n_vars
        best_x, best_e = None, np.inf
        samples = feasible = 0
        reads = 0
        while True:
            if deadline is not None:
                if reads > 0 and time.perf_counter() >= deadline:
                    break
            elif reads >= self.num_reads:
                break
            k = self.batch if deadline is not None else min(self.batch, self.num_reads - reads)
            result = self._sampler.sample(bqm, num_reads=k, num_sweeps=self.num_sweeps, beta_range=self.beta_range,
                                          seed=int(rng.integers(2**31 - 1)))
            reads += k
            for sample, e in result.data(["sample", "energy"]):
                x = np.array([sample[i] for i in range(n)], dtype=np.int8)
                samples += 1
                feasible += decode(problem, x)[1]
                if e < best_e:
                    best_e, best_x = float(e), x
        return best_x, {"samples_evaluated": samples, "raw_feasible_samples": feasible,
                        "stopped_on_budget": deadline is not None}
