"""Instance JSON I/O and the wire format shared with Unity.

``to_qubo_request`` produces exactly the object the Unity
``ShadowVale.Data.Coordination.QuboRequest`` deserialises (snake_case keys),
so instances generated here can be replayed inside the game and vice versa.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..model.instance import Instance
from ..qubo.formulation import QuboProblem, build_qubo


def save_instance(inst: Instance, path: str | Path) -> None:
    Path(path).write_text(json.dumps(inst.to_dict(), indent=2), encoding="utf-8")


def load_instance(path: str | Path) -> Instance:
    return Instance.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def to_qubo_request(inst: Instance, problem: QuboProblem | None = None,
                    latency_budget_ms: int = 120, variant: str = "greedy") -> dict:
    problem = problem or build_qubo(inst)
    return {
        "instance_id": inst.instance_id,
        "seed": inst.seed,
        "agent_count": inst.n_agents,
        "node_count": inst.n_nodes,
        "agent_nodes": list(inst.agent_nodes),
        "cover_nodes": list(inst.cover_nodes),
        "escape_routes": list(inst.escape_routes),
        "q_matrix": problem.q_flat(),
        "offset": problem.offset,
        "weight_coverage": problem.weights.coverage,
        "weight_redundancy": problem.weights.redundancy,
        "weight_flanking": problem.weights.flanking,
        "weight_cover": problem.weights.cover,
        "weight_distance": problem.weights.distance,
        "penalty_conflict": problem.lam,
        "latency_budget_ms": int(latency_budget_ms),
        "variant": variant,
    }
