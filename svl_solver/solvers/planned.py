"""Solvers scheduled for later work packages. They register in the registry
with ``available = False`` so the API can list them and the benchmark runner
can skip them cleanly; calling ``solve`` raises NotImplementedError with the
week they are due.

    genetic       classical baseline 2            → P2  (W3–W4)
    sa_classical  classical simulated annealing   → P2  (W3–W4)  dwave-neal (neal is CLASSICAL SA)
    sqa           simulated quantum annealing     → P4  (W8)   path-integral Monte Carlo, NOT neal
    qaoa          QAOA on Qiskit Aer, p = 1..3    → P4  (W9)   instances ≤ ~25 vars
    qpu           real D-Wave QPU feasibility     → stretch
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


class GeneticSolver(_Planned):
    variant_id = "ga"
    due = "P2 (W3–W4)"


class ClassicalAnnealingSolver(_Planned):
    variant_id = "sa"
    due = "P2 (W3–W4)"


class SimulatedQuantumAnnealingSolver(_Planned):
    """Path-integral Monte Carlo: P Trotter replicas coupled by a decaying
    transverse field Γ. Must not reuse dwave-neal, which is classical SA
    (that is the ``sa`` baseline) — otherwise sa and sqa would be one algorithm."""
    variant_id = "sqa"
    due = "P4 (W8)"


class QaoaAerSolver(_Planned):
    """Qiskit Aer statevector/QASM simulation of QAOA, COBYLA outer loop.
    Requires the [quantum] extra. Only instances with ≤ ~25 variables."""
    variant_id = "qaoa"
    due = "P4 (W9)"


class DWaveQpuSolver(_Planned):
    variant_id = "qpu"
    due = "stretch"
