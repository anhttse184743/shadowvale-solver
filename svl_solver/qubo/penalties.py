"""Choice of the constraint penalty λ.

We need every infeasible bit-vector to have strictly higher energy than the
best feasible one, otherwise a solver could "win" by violating a constraint.
The argument below gives a λ that guarantees this (and is what
tests/test_penalty_dominates.py checks exhaustively on small instances).

Per-variable magnitudes of the objective F (all ≥ 0):

    L  = max_n ( w_cov·cov[n] + w_cover·is_cover[n] )   largest positive linear gain
    D  = w_dist · max dist[a,n]                          largest linear cost
    P+ = w_flank · max flank[n,m]                        largest positive pair gain
    P− = w_red   · max overlap[n,m]                      largest pair cost

Repair any infeasible x in three phases, each step strictly lowering E:

  1. some agent has k ≥ 2 bits → drop one bit of the agent with the most bits.
     Penalty falls by ≥ λ(2k−3); F falls by ≤ L + P+·(A−1)·k.
     Worst case k = 2:                       λ > L + 2(A−1)·P+
  2. all agents ≤ 1 bit, two share a node → move one to a free node
     (exists because occupied < A ≤ N). Penalty falls by ≥ λ;
     F falls by ≤ (L + (A−1)P+) + (D + (A−1)P−).
  3. an agent has no bit → give it the nearest free node. Penalty falls by λ;
     F falls by ≤ D + (A−1)·P−.

    λ = 1 + L + 2(A−1)·P+ + D + (A−1)·P−   satisfies all three.

This is tighter than the generic Lucas-style bound (which multiplies by N)
and keeps the energy landscape reasonable for annealers.
"""

from __future__ import annotations

from ..model.instance import Instance, Weights


def penalty_bound_terms(inst: Instance, w: Weights | None = None) -> dict[str, float]:
    w = w or Weights()
    A = inst.n_agents
    L = float((w.coverage * inst.cov + w.cover * inst.is_cover).max())
    D = float(w.distance * inst.dist.max())
    P_plus = float(w.flanking * inst.flank.max()) if A > 1 else 0.0
    P_minus = float(w.redundancy * inst.overlap.max()) if A > 1 else 0.0
    return {"L": L, "D": D, "P_plus": P_plus, "P_minus": P_minus, "A": float(A)}


def compute_lambda(inst: Instance, w: Weights | None = None, margin: float = 1.0) -> float:
    t = penalty_bound_terms(inst, w)
    A = t["A"]
    return margin + t["L"] + 2.0 * (A - 1) * t["P_plus"] + t["D"] + (A - 1) * t["P_minus"]
