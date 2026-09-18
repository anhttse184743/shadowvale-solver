"""Assignment <-> bit-vector conversion, feasibility checks, and repair.

Variable layout (shared with qubo/variables.py and with the Unity QuboBuilder):
    x[a*N + n] = 1  <=>  agent a is assigned to node n

Feasible  <=>  every agent has exactly one node AND no two agents share a node.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .instance import Instance


def assignment_to_x(assignment: Sequence[int], n_agents: int, n_nodes: int) -> np.ndarray:
    x = np.zeros(n_agents * n_nodes, dtype=np.int8)
    for a, n in enumerate(assignment):
        x[a * n_nodes + int(n)] = 1
    return x


def x_to_assignment(x: np.ndarray, n_agents: int, n_nodes: int) -> list[int]:
    """Decode a bit-vector into an assignment. For a feasible x this is exact.
    For an infeasible x it picks, per agent, the first set bit (or -1 if none);
    call ``repair`` if you need a feasible result."""
    m = np.asarray(x).reshape(n_agents, n_nodes)
    out = []
    for a in range(n_agents):
        idx = np.flatnonzero(m[a])
        out.append(int(idx[0]) if idx.size else -1)
    return out


def is_feasible_assignment(assignment: Sequence[int], n_nodes: int) -> bool:
    s = list(assignment)
    if any(v < 0 or v >= n_nodes for v in s):
        return False
    return len(set(s)) == len(s)


def is_feasible_x(x: np.ndarray, n_agents: int, n_nodes: int) -> bool:
    m = np.asarray(x).reshape(n_agents, n_nodes)
    if not np.all(m.sum(axis=1) == 1):
        return False
    return bool(np.all(m.sum(axis=0) <= 1))


def repair(inst: Instance, x: np.ndarray) -> list[int]:
    """Turn any bit-vector into a feasible assignment, changing as little as
    possible:

    1. agents with several bits keep the node with the lowest travel cost;
    2. conflicts are resolved in agent order, the later agent moves;
    3. agents with no node take the nearest free node.

    Used by the API so the game always receives a usable plan even when a
    heuristic solver returns an infeasible sample.
    """
    A, N = inst.n_agents, inst.n_nodes
    m = np.asarray(x).reshape(A, N)
    chosen: list[int] = []
    taken: set[int] = set()

    for a in range(A):
        idx = [int(i) for i in np.flatnonzero(m[a])]
        idx = [i for i in idx if i not in taken]
        if idx:
            best = min(idx, key=lambda n: (inst.dist[a, n], n))
            chosen.append(best)
            taken.add(best)
        else:
            chosen.append(-1)

    for a in range(A):
        if chosen[a] == -1:
            free = [n for n in range(N) if n not in taken]
            best = min(free, key=lambda n: (inst.dist[a, n], n))
            chosen[a] = best
            taken.add(best)
    return chosen
