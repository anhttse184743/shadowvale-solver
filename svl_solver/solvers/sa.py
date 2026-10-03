"""Classical baseline 3: simulated annealing (dwave-neal).

neal is thermal (classical) annealing on the binary quadratic model — there is
nothing quantum about it. That is exactly why it is the classical counterpart of
``sqa``: same schedule idea, no transverse field, no Trotter replicas.

With ``budget_ms`` it keeps drawing batches of ``batch`` anneals until
``time_fraction`` of the budget has passed; otherwise it does ``num_reads``
reads. The lowest-energy sample wins; the base class repairs it if it violates a
constraint. ``raw_feasible_samples`` counts samples that needed no repair.

Schedule: geometric β from ``beta_hot / λ`` to ``beta_cold``. The hot end is relative
to the penalty λ: moving an agent means passing through an infeasible state that costs
about λ, and λ grows with the instance (≈10 at 3 agents, 22–36 at 6–8 agents in game).
A fixed hot β tuned on small instances froze SA on game-size ones from the first sweep
(exp(−0.3·27) ≈ 3·10⁻⁴). The cold end stays absolute: it must resolve objective
differences of ~0.05 whatever λ is. ``svl-bench tune`` (03/10/2026, experiments/tuning)
found no configuration that beat these defaults on the repeated final runs (mean gap 8.6 %). Single-bit Metropolis moves still have to cross infeasible states
to move an agent, which is the structural reason SA trails GA/QIEA on this encoding.
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
                 beta_hot: float = 3.0, beta_cold: float = 300.0, time_fraction: float = 0.8) -> None:
        self.num_sweeps = max(1, int(num_sweeps))
        self.num_reads = max(1, int(num_reads))
        self.batch = max(1, int(batch))
        self.beta_hot, self.beta_cold = float(beta_hot), float(beta_cold)
        self.time_fraction = float(time_fraction)
        self._sampler = neal.SimulatedAnnealingSampler()

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        bqm = to_bqm(problem)
        lam = problem.lam if problem.lam > 0 else 1.0
        beta_range = (self.beta_hot / lam, max(self.beta_cold, self.beta_hot / lam * 1.01))
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
            result = self._sampler.sample(bqm, num_reads=k, num_sweeps=self.num_sweeps, beta_range=beta_range,
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
