"""Reference objective, written directly on an *assignment* (agent -> node).

This is the ground truth every solver is scored against. It must NOT import
anything from qubo/: the QUBO is a re-encoding of this function, and
tests/test_qubo_matches_objective.py checks that the two agree exactly on
every feasible assignment of small instances.

F(σ) =   w_cov   Σ_a cov[σ(a)]
       − w_red   Σ_{a<b} overlap[σ(a), σ(b)]
       + w_flank Σ_{a<b} flank[σ(a), σ(b)]
       + w_cover Σ_a cover_q[σ(a)]
       − w_dist  Σ_a dist[a, σ(a)]

Higher is better. Solvers minimise the QUBO energy E = −F + penalties.
"""

from __future__ import annotations

from typing import Sequence

from .instance import Instance, Weights


def objective_terms(inst: Instance, assignment: Sequence[int]) -> dict[str, float]:
    """Return each unweighted term separately (useful for telemetry / plots)."""
    A = inst.n_agents
    if len(assignment) != A:
        raise ValueError(f"assignment has {len(assignment)} entries, expected {A}")
    s = [int(v) for v in assignment]

    coverage = float(sum(inst.cov[s[a]] for a in range(A)))
    cover = float(sum(inst.cover_q[s[a]] for a in range(A)))
    distance = float(sum(inst.dist[a, s[a]] for a in range(A)))

    redundancy = 0.0
    flanking = 0.0
    for a in range(A):
        for b in range(a + 1, A):
            redundancy += float(inst.overlap[s[a], s[b]])
            flanking += float(inst.flank[s[a], s[b]])

    return {
        "coverage": coverage,
        "redundancy": redundancy,
        "flanking": flanking,
        "cover": cover,
        "distance": distance,
    }


def objective(inst: Instance, assignment: Sequence[int], w: Weights | None = None) -> float:
    w = w or Weights()
    t = objective_terms(inst, assignment)
    return (
        w.coverage * t["coverage"]
        - w.redundancy * t["redundancy"]
        + w.flanking * t["flanking"]
        + w.cover * t["cover"]
        - w.distance * t["distance"]
    )
