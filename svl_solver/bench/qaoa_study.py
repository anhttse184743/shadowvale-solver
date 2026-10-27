"""QAOA without the shot and budget confounds (exp-v2, review M5).

exp-v1 reported QAOA as the best of 512 measured bit strings within 120 ms. That mixes three
things: the circuit, the number of shots, and the time budget. This study removes the last two:

* angles are optimised without a time limit (p = 1 grid, then Nelder–Mead, warm start per layer,
  exactly as ``solvers/qaoa.py``), for p = 1, 2, 3;
* from the final statevector it reads P(optimum) and P(feasible) directly, and the probability
  that S shots contain the optimum, 1 − (1 − P_opt)^S, for S = 128, 512, 2048;
* the same numbers for **uniform random sampling** of bit strings (P_opt = k / 2^n) and of
  feasible assignments, so the circuit is credited only with what it adds over chance.

Two ansätze:

* ``x`` — the standard one: |+⟩^n and the transverse-field mixer Σ X_i, penalties in the cost;
* ``xy`` — the constraint-preserving Quantum Alternating Operator Ansatz (Hadfield et al. 2019):
  each agent's N qubits start in the uniform one-hot (W) state and mix with a ring XY mixer,
  which never leaves the one-hot sector. On that sector the mixer is the adjacency matrix of the
  N-cycle, so the evolution is simulated exactly in the N^A one-hot configurations instead of
  2^(A·N) amplitudes. Clashes (two agents on one node) remain a penalty. The circuit still needs
  A·N qubits; only its classical simulation is cheaper, which lets it run on larger instances.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..qubo.formulation import QuboProblem
from ..solvers.assignment import AssignmentModel
from ..solvers.base import decode
from ..solvers.qaoa import QaoaSolver, basis_energies, qaoa_state

SHOTS = (128, 512, 2048)


def _optimise(expectation, layers: int, grid: int = 12, refine: int = 40) -> list[tuple[float, list, list]]:
    """Angles per depth 1..layers: grid at p = 1, Nelder–Mead, warm start for each next layer."""
    nm = QaoaSolver(refine_iterations=refine)
    never = lambda: False  # noqa: E731
    best = (np.inf, [0.0], [0.0])
    for g in np.linspace(0, 2 * math.pi, grid, endpoint=False):
        for b in np.linspace(0, math.pi, grid, endpoint=False):
            v = expectation([g], [b])
            if v < best[0]:
                best = (v, [g], [b])
    value, gammas, betas = best
    out = []
    for layer in range(1, layers + 1):
        if layer > 1:
            gammas, betas = gammas + [gammas[-1]], betas + [betas[-1]]
            value = expectation(gammas, betas)
        value, gammas, betas = nm._refine(expectation, value, gammas, betas, never)
        out.append((value, list(gammas), list(betas)))
    return out


def _hit(p: float, shots: int) -> float:
    return 1.0 - (1.0 - min(max(p, 0.0), 1.0)) ** shots


def study_x(problem: QuboProblem, layers: int = 3) -> list[dict]:
    """Standard QAOA on the full 2^n statevector."""
    n = problem.n_vars
    energies = basis_energies(problem.Q) + problem.offset
    lo, hi = energies.min(), energies.max()
    cost = (energies - lo) / max(hi - lo, 1e-12)
    idx = np.arange(1 << n)
    feasible = np.array([decode(problem, np.array([(int(z) >> i) & 1 for i in range(n)]))[1] for z in idx])
    e_opt = energies[feasible].min()
    is_opt = feasible & (energies <= e_opt + 1e-9)
    t0 = time.perf_counter()
    res = _optimise(lambda g, b: float(np.dot(np.abs(qaoa_state(cost, n, g, b)) ** 2, cost)), layers)
    secs = time.perf_counter() - t0
    out = []
    for p, (value, g, b) in enumerate(res, start=1):
        probs = np.abs(qaoa_state(cost, n, g, b)) ** 2
        probs /= probs.sum()
        out.append(_row("x", problem, p, n, probs[is_opt].sum(), probs[feasible].sum(),
                        int(is_opt.sum()) / (1 << n), int(feasible.sum()) / (1 << n),
                        int(is_opt.sum()), int(feasible.sum()), secs))
    return out


def study_xy(problem: QuboProblem, layers: int = 3, max_states: int = 2_000_000) -> list[dict]:
    """Constraint-preserving QAOA, simulated in the N^A one-hot sector."""
    A, N = problem.n_agents, problem.n_nodes
    if N ** A > max_states:
        raise ValueError(f"{N}^{A} one-hot states exceeds {max_states}")
    m = AssignmentModel(problem)
    shape = (N,) * A
    E = np.full(shape, m.offset)
    for a in range(A):
        E += m.U[a].reshape([N if k == a else 1 for k in range(A)])
        for b in range(a + 1, A):
            E += m.P[a, b].reshape([N if k in (a, b) else 1 for k in range(A)])
    E = E.ravel()
    grids = np.indices(shape).reshape(A, -1)
    distinct = np.ones(E.size, dtype=bool)
    for a in range(A):
        for b in range(a + 1, A):
            distinct &= grids[a] != grids[b]
    e_opt = E[distinct].min()
    is_opt = distinct & (E <= e_opt + 1e-9)
    lo, hi = E.min(), E.max()
    cost = (E - lo) / max(hi - lo, 1e-12)
    ring = np.zeros((N, N))
    for k in range(N):
        ring[k, (k + 1) % N] = ring[(k + 1) % N, k] = 1.0
    w, V = np.linalg.eigh(ring)

    def state(gammas, betas):
        psi = np.full(E.size, 1.0 / math.sqrt(E.size), dtype=np.complex128)
        for gamma, beta in zip(gammas, betas):
            psi = psi * np.exp(-1j * gamma * cost)
            Ub = (V * np.exp(-1j * beta * w)) @ V.T
            t = psi.reshape(shape)
            for a in range(A):
                t = np.moveaxis(np.tensordot(Ub, t, axes=([1], [a])), 0, a)
            psi = t.ravel()
        return psi

    t0 = time.perf_counter()
    res = _optimise(lambda g, b: float(np.dot(np.abs(state(g, b)) ** 2, cost)), layers)
    secs = time.perf_counter() - t0
    n = A * N
    feasible_count = math.perm(N, A)
    out = []
    for p, (value, g, b) in enumerate(res, start=1):
        probs = np.abs(state(g, b)) ** 2
        probs /= probs.sum()
        out.append(_row("xy", problem, p, n, probs[is_opt].sum(), probs[distinct].sum(),
                        int(is_opt.sum()) / 2.0 ** n, feasible_count / 2.0 ** n,
                        int(is_opt.sum()), feasible_count, secs))
    return out


def _row(ansatz, problem, p, n, p_opt, p_feas, uni_opt, uni_feas, k_opt, n_feas, secs) -> dict:
    r = {"ansatz": ansatz, "instance_id": problem.instance_id, "n_agents": problem.n_agents,
         "n_nodes": problem.n_nodes, "n_vars": n, "layers": p, "p_opt": float(p_opt), "p_feasible": float(p_feas),
         "uniform_p_opt": float(uni_opt), "uniform_p_feasible": float(uni_feas),
         "uniform_feasible_p_opt": k_opt / n_feas, "optimisation_s": round(secs, 3)}
    for s in SHOTS:
        r[f"hit_{s}"] = _hit(p_opt, s)
        r[f"uniform_hit_{s}"] = _hit(uni_opt, s)
        r[f"uniform_feasible_hit_{s}"] = _hit(k_opt / n_feas, s)
    return r
