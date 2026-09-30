"""QAOA (Farhi, Goldstone & Gutmann 2014) on a statevector simulator.

The circuit is the textbook one: |+⟩^n, then p layers of the cost phase
e^{−iγ_k C} and the mixer Π_q RX(2β_k), measured in the computational basis.
C is the QUBO energy of each basis state, rescaled to [0, 1] so the angles live
in γ ∈ [0, 2π), β ∈ [0, π). It is simulated exactly with a numpy statevector —
the same amplitudes Qiskit Aer's statevector simulator produces for this circuit
— so no quantum hardware is involved and no speed-up is claimed.

Angles: a coarse grid for p = 1, refined by Nelder–Mead on ⟨C⟩; each further
layer starts from the previous optimum (layer-wise warm start). Then ``shots``
bitstrings are sampled from |ψ|² and the lowest-energy one is returned (the base
class repairs it if needed); ``raw_feasible_samples`` counts shots that were
already valid assignments — how much of the measured distribution respects the
constraints is itself a result for RQ2.

A statevector holds 2^n amplitudes, so instances above ``max_qubits`` variables
are refused with ValueError (the sidecar then falls back to greedy and says so).
Mapping where QAOA is still usable is exactly the "viable regime" question (RQ3).
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver, decode


def basis_energies(Q: np.ndarray) -> np.ndarray:
    """E(z) for every basis state z (bit i of the index = variable i)."""
    n = Q.shape[0]
    idx = np.arange(1 << n, dtype=np.int64)
    bits = [((idx >> i) & 1).astype(np.float64) for i in range(n)]
    e = np.zeros(1 << n)
    for i in range(n):
        if Q[i, i] != 0.0:
            e += Q[i, i] * bits[i]
        for j in range(i + 1, n):
            c = Q[i, j] + Q[j, i]
            if c != 0.0:
                e += c * bits[i] * bits[j]
    return e


def qaoa_state(cost: np.ndarray, n: int, gammas, betas) -> np.ndarray:
    psi = np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * cost)
        c, s = math.cos(beta), -1j * math.sin(beta)
        for q in range(n):
            view = psi.reshape(-1, 2, 1 << q)
            a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = c * a + s * b
            view[:, 1, :] = s * a + c * b
    return psi


class QaoaSolver(Solver):
    variant_id = "qaoa"

    def __init__(self, layers: int = 2, shots: int = 512, grid: int = 12, max_qubits: int = 18,
                 refine_iterations: int = 40, time_fraction: float = 0.8) -> None:
        self.layers = max(1, int(layers))
        self.shots = max(1, int(shots))
        self.grid = max(2, int(grid))
        self.max_qubits = int(max_qubits)
        self.refine_iterations = max(0, int(refine_iterations))
        self.time_fraction = float(time_fraction)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        n = problem.n_vars
        if n > self.max_qubits:
            raise ValueError(f"QAOA needs {n} qubits, statevector limit is {self.max_qubits}")
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        energies = basis_energies(problem.Q)
        lo, hi = energies.min(), energies.max()
        cost = (energies - lo) / max(hi - lo, 1e-12)

        def expectation(g, b) -> float:
            psi = qaoa_state(cost, n, g, b)
            return float(np.dot(np.abs(psi) ** 2, cost))

        def out_of_time() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        # p = 1: grid search.
        best = (np.inf, [0.0], [0.0])
        for g in np.linspace(0, 2 * math.pi, self.grid, endpoint=False):
            for b in np.linspace(0, math.pi, self.grid, endpoint=False):
                v = expectation([g], [b])
                if v < best[0]:
                    best = (v, [g], [b])
                if out_of_time():
                    break
            if out_of_time():
                break
        value, gammas, betas = best
        for layer in range(1, self.layers + 1):
            if layer > 1:
                if out_of_time():
                    break
                gammas, betas = gammas + [gammas[-1]], betas + [betas[-1]]
                value = expectation(gammas, betas)
            value, gammas, betas = self._refine(expectation, value, gammas, betas, out_of_time)
        layers_done = len(gammas)

        probs = np.abs(qaoa_state(cost, n, gammas, betas)) ** 2
        probs /= probs.sum()
        shots = rng.choice(1 << n, size=self.shots, p=probs)
        best_z = int(shots[np.argmin(energies[shots])])
        feasible = 0
        for z in shots:
            x = np.array([(int(z) >> i) & 1 for i in range(n)], dtype=np.int8)
            feasible += decode(problem, x)[1]
        x = np.array([(best_z >> i) & 1 for i in range(n)], dtype=np.int8)
        return x, {"samples_evaluated": self.shots, "raw_feasible_samples": int(feasible),
                   "layers": layers_done, "qubits": n, "expectation": value,
                   "stopped_on_budget": out_of_time()}

    def _refine(self, f, value, gammas, betas, out_of_time):
        """Nelder–Mead on the 2p angles (small, dependency-free)."""
        k = len(gammas)
        x0 = np.array(gammas + betas, dtype=float)
        simplex = [x0] + [x0 + np.eye(2 * k)[i] * .25 for i in range(2 * k)]
        vals = [value] + [f(list(p[:k]), list(p[k:])) for p in simplex[1:]]
        for _ in range(self.refine_iterations):
            if out_of_time():
                break
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            reflect = centroid + (centroid - worst)
            fr = f(list(reflect[:k]), list(reflect[k:]))
            if fr < vals[0]:
                expand = centroid + 2 * (centroid - worst)
                fe = f(list(expand[:k]), list(expand[k:]))
                simplex[-1], vals[-1] = (expand, fe) if fe < fr else (reflect, fr)
            elif fr < vals[-2]:
                simplex[-1], vals[-1] = reflect, fr
            else:
                contract = centroid + .5 * (worst - centroid)
                fc = f(list(contract[:k]), list(contract[k:]))
                if fc < vals[-1]:
                    simplex[-1], vals[-1] = contract, fc
                else:
                    best = simplex[0]
                    simplex = [best] + [best + .5 * (p - best) for p in simplex[1:]]
                    vals = [vals[0]] + [f(list(p[:k]), list(p[k:])) for p in simplex[1:]]
        i = int(np.argmin(vals))
        return vals[i], list(simplex[i][:k]), list(simplex[i][k:])
