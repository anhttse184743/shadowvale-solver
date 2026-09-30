"""Simulated quantum annealing (path-integral Monte Carlo) — quantum-inspired.

Classical Monte Carlo simulation of a transverse-field Ising model, the standard
SQA of Santoro et al. (Science 2002) / Martoňák et al. (PRB 2002). The QUBO is
mapped to spins (x = (1+s)/2) and
copied into P Trotter replicas ("slices"), energies divided by λ. The effective
classical Hamiltonian is

    H = (1/P) Σ_p E_ising(s^p)  −  J⊥(Γ) Σ_p Σ_i s_i^p s_i^{p+1}    (periodic in p)

    J⊥(Γ) = −(T/2) · ln tanh(Γ / (P·T))

Γ, the transverse field, falls linearly from ``gamma_start`` to ``gamma_end``;
as it falls, J⊥ grows and the replicas are forced to agree — a state can move
through a barrier by some replicas crossing first (the "tunnelling" SA lacks).
Each sweep proposes one flip per spin per replica (Metropolis at temperature T).

This is not dwave-neal: neal is classical thermal annealing and is the ``sa``
baseline. Runs on a CPU; no quantum speed-up is claimed.

With ``budget_ms`` the schedule is compressed into ``time_fraction`` of the
budget (sweeps are timed as they go); without it, exactly ``sweeps`` sweeps
(reproducible for a seed). The lowest-energy replica ever seen is returned.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(h, Js) with E_qubo(x) = h·s + ½ s·Js·s + const for x = (1+s)/2; Js symmetric, zero diagonal."""
    Qs = np.triu(Q, 1) + np.triu(Q, 1).T          # symmetric off-diagonal couplings
    h = np.diag(Q) / 2.0 + Qs.sum(axis=1) / 4.0
    Js = Qs / 4.0
    return h, Js


class SimulatedQuantumAnnealingSolver(Solver):
    variant_id = "sqa"

    # Defaults tuned 30/09/2026 under the game's 120 ms budget, on 12 seeded 8-node/3-agent
    # instances and 11 instances recorded in Map 1 (22-24 nodes, 2-4 agents): energies are
    # normalised by the penalty λ so T means the same at every size (normalising by the largest
    # coupling made T effectively hotter as λ grew). P=8, T=0.05, Γ0=1: 12/12 optimal on the small
    # set, mean gap 1.2 % to the best known on the game set.
    def __init__(self, trotter: int = 8, sweeps: int = 600, temperature: float = 0.05,
                 gamma_start: float = 1.0, gamma_end: float = 0.02, time_fraction: float = 0.8,
                 normalise: str = "lambda") -> None:
        self.trotter = max(2, int(trotter))
        self.sweeps = max(1, int(sweeps))
        self.temperature = float(temperature)
        self.gamma_start, self.gamma_end = float(gamma_start), float(gamma_end)
        self.time_fraction = float(time_fraction)
        self.normalise = normalise

    def coupling(self, gamma: float) -> float:
        """J⊥ between neighbouring replicas at transverse field Γ."""
        P, T = self.trotter, self.temperature
        return -0.5 * T * math.log(math.tanh(max(gamma, 1e-12) / (P * T)))

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        h, Js = qubo_to_ising(problem.Q)
        if self.normalise == "lambda" and problem.lam > 0:
            scale = problem.lam                        # T means the same at every instance size
        else:
            scale = max(np.abs(h).max(initial=0.0), np.abs(Js).max(initial=0.0), 1e-12)
        h, Js = h / scale, Js / scale
        P, n, T = self.trotter, problem.n_vars, self.temperature

        s = rng.choice(np.array([-1.0, 1.0]), size=(P, n))
        fields = s @ Js + h                             # local field per replica and spin
        up, down = np.roll(np.arange(P), 1), np.roll(np.arange(P), -1)   # Trotter neighbours (periodic)
        A, N = problem.n_agents, problem.n_nodes
        Q = problem.Q
        best_x, best_e = None, np.inf
        samples = feasible = 0
        planned = self.sweeps
        sweep = 0
        stopped = False
        while sweep < planned:
            if deadline is not None:
                now = time.perf_counter()
                if now >= deadline:
                    stopped = True
                    break
                if sweep == 3:
                    # Re-plan the schedule length from the measured sweep time, so the
                    # whole Γ ramp fits the budget instead of being cut off half-way.
                    per_sweep = (now - t0) / 3
                    planned = max(4, min(self.sweeps, int((deadline - t0) / max(per_sweep, 1e-6))))
            gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * sweep / max(1, planned - 1)
            j_perp = self.coupling(gamma)
            order = rng.permutation(n)
            uniform = rng.random((n, P))
            for k, i in enumerate(order):
                si = s[:, i]
                delta = (-2.0 / P) * si * fields[:, i] + (2.0 * j_perp) * si * (s[up, i] + s[down, i])
                accept = uniform[k] < np.exp(np.minimum(-delta / T, 0.0))   # Metropolis; ΔH ≤ 0 always accepted
                if not accept.any():
                    continue
                flip = np.where(accept, -2.0 * si, 0.0)   # change of s[:, i]
                s[:, i] += flip
                fields += flip[:, None] * Js[i]
            sweep += 1
            # Score every replica on the real QUBO, all at once.
            X = (s > 0).astype(np.float64)
            energies = np.einsum("pi,ij,pj->p", X, Q, X)
            grid = X.reshape(P, A, N)
            ok = (grid.sum(axis=2) == 1).all(axis=1) & (grid.sum(axis=1) <= 1).all(axis=1)
            samples += P
            feasible += int(ok.sum())
            p = int(np.argmin(energies))
            if energies[p] < best_e:
                best_e, best_x = float(energies[p]), X[p].astype(np.int8)
        return best_x, {"samples_evaluated": samples, "raw_feasible_samples": feasible,
                        "sweeps": sweep, "trotter": P, "stopped_on_budget": stopped}
