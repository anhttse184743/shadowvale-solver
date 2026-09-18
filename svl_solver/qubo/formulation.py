"""QUBO formulation of squad coordination (research objective O1).

Energy to MINIMISE, over x ∈ {0,1}^(A·N) with x[a,n] = "agent a at node n":

    E(x) = −F(x)
           + λ Σ_a ( Σ_n x[a,n] − 1 )²          (C1) each agent exactly one node
           + λ Σ_n Σ_{a<b} x[a,n]·x[b,n]         (C2) no two agents on one node

where F is the reference objective from model/objective.py, which is already
quadratic in x:

    F(x) =  w_cov   Σ_{a,n} cov[n]·x[a,n]
          − w_red   Σ_{a<b} Σ_{n≠m} overlap[n,m]·x[a,n]·x[b,m]
          + w_flank Σ_{a<b} Σ_{n≠m} flank[n,m]·x[a,n]·x[b,m]
          + w_cover Σ_{a,n} is_cover[n]·x[a,n]
          − w_dist  Σ_{a,n} dist[a,n]·x[a,n]

Expanding (C1) with x² = x:
    ( Σ_n x[a,n] − 1 )²  =  − Σ_n x[a,n]  +  2 Σ_{n<m} x[a,n]·x[a,m]  +  1

so per agent the diagonal gets −λ, same-agent pairs get +2λ, and the constant
+λ goes to ``offset``. On every feasible x both penalty sums vanish, hence

    E(x) + offset == −F(x)          for every feasible x     (tested)
    E(x) + offset  >  min feasible   for every infeasible x  (tested, given λ from penalties.py)

Storage convention: Q is UPPER-TRIANGULAR, E = Σ_i Q[i,i]·x_i + Σ_{i<j} Q[i,j]·x_i·x_j,
which is what dimod / D-Wave / Qiskit-Optimization expect and what Unity's
``QuboRequest.QMatrix`` (flattened row-major) carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.instance import Instance, Weights
from .penalties import compute_lambda
from .variables import var_index


@dataclass
class QuboProblem:
    Q: np.ndarray                 # (n_vars, n_vars) upper-triangular float64
    offset: float
    n_agents: int
    n_nodes: int
    lam: float
    weights: Weights = field(default_factory=Weights)
    instance_id: str = ""

    @property
    def n_vars(self) -> int:
        return self.n_agents * self.n_nodes

    def energy(self, x: np.ndarray) -> float:
        return energy(self.Q, x)

    def energy_with_offset(self, x: np.ndarray) -> float:
        return energy(self.Q, x) + self.offset

    def q_flat(self) -> list[float]:
        """Row-major flattening used by the HTTP API / Unity."""
        return self.Q.astype(float).ravel().tolist()

    @staticmethod
    def from_flat(q_flat: list[float], n_agents: int, n_nodes: int,
                  offset: float = 0.0, lam: float = 0.0,
                  instance_id: str = "") -> "QuboProblem":
        n = n_agents * n_nodes
        Q = np.asarray(q_flat, dtype=float).reshape(n, n)
        # Accept symmetric input too: fold the lower triangle into the upper.
        lower = np.tril(Q, -1)
        if np.any(lower):
            Q = np.triu(Q) + lower.T
        return QuboProblem(Q=Q, offset=float(offset), n_agents=n_agents,
                           n_nodes=n_nodes, lam=float(lam), instance_id=instance_id)


def energy(Q: np.ndarray, x: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float).ravel()
    return float(xv @ Q @ xv)


def build_qubo(inst: Instance, w: Weights | None = None, lam: float | None = None) -> QuboProblem:
    w = w or Weights()
    A, N = inst.n_agents, inst.n_nodes
    n = A * N
    lam = float(compute_lambda(inst, w) if lam is None else lam)

    Q = np.zeros((n, n), dtype=float)

    # ---- linear terms: −(objective linear) − λ (from C1 expansion)
    for a in range(A):
        for node in range(N):
            i = var_index(a, node, N)
            lin_obj = (w.coverage * inst.cov[node]
                       + w.cover * inst.is_cover[node]
                       - w.distance * inst.dist[a, node])
            Q[i, i] = -lin_obj - lam

    # ---- same agent, two different nodes: +2λ (C1 expansion)
    for a in range(A):
        for nn in range(N):
            i = var_index(a, nn, N)
            for m in range(nn + 1, N):
                j = var_index(a, m, N)
                Q[i, j] += 2.0 * lam

    # ---- different agents
    pair_obj = w.redundancy * inst.overlap - w.flanking * inst.flank   # = −(objective pair term), (N,N)
    for a in range(A):
        for b in range(a + 1, A):
            base_i = a * N
            base_j = b * N
            # objective cross terms for n != m
            Q[base_i:base_i + N, base_j:base_j + N] += pair_obj
            # C2: same node → +λ (pair_obj diagonal is 0 because overlap/flank diagonals are 0)
            for node in range(N):
                Q[base_i + node, base_j + node] += lam

    offset = lam * A
    return QuboProblem(Q=Q, offset=offset, n_agents=A, n_nodes=N, lam=lam,
                       weights=w, instance_id=inst.instance_id)
