"""Solvers scheduled for later work packages. They register in the registry
with ``available = False`` so the API can list them and the benchmark runner
can skip them cleanly; calling ``solve`` raises NotImplementedError.

    qpu           real D-Wave QPU feasibility demo on tiny instances → stretch

Everything else is implemented: greedy, brute, ga, sa (classical);
sqa, qiea, qaoa (quantum-inspired / simulated).
"""

from __future__ import annotations

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver


class _Planned(Solver):
    available = False
    due = "TBD"

    def _solve(self, problem: QuboProblem, budget_ms, rng) -> np.ndarray:
        raise NotImplementedError(f"solver '{self.variant_id}' is planned for {self.due}")


class DWaveQpuSolver(_Planned):
    variant_id = "qpu"
    due = "stretch"
