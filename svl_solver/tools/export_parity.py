"""Exports QUBO parity fixtures for the Unity C# QuboBuilder (M4).

Each case holds the raw per-instance inputs the game computes (2-D positions, player,
cover flags + facing, escape nodes, visibility matrix, agent→node distances), the
derived features (cov, overlap, flank, cover_q), λ, the upper-triangular Q and the
offset. The C# test rebuilds everything from the inputs and must match to 1e-9.

    python -m svl_solver.tools.export_parity <out.json>
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..instances.generator import generate
from ..model.instance import Weights
from ..qubo.formulation import build_qubo

CASES = [
    # (seed, nodes, agents, density, weights)
    (1, 10, 3, 0.3, Weights()),
    (2, 12, 4, 0.5, Weights(coverage=2.0, redundancy=0.7, flanking=1.5, cover=0.8, distance=0.2)),
    (3, 8, 2, 0.0, Weights(coverage=1.0, redundancy=0.0, flanking=0.0, cover=1.0, distance=0.05)),
]


def case(seed, n, a, density, w) -> dict:
    inst = generate(seed=seed, n_nodes=n, n_agents=a, chokepoint_density=density)
    p = build_qubo(inst, w)
    return {
        "instance_id": inst.instance_id,
        "agent_count": inst.n_agents, "node_count": inst.n_nodes,
        "positions": inst.pos.tolist(),
        "player": inst.player_node,
        "is_cover": inst.is_cover.tolist(),
        "cover_facing": inst.cover_facing,
        "escape": inst.escape_routes,
        "visible": inst.visible.astype(int).ravel().tolist(),
        "dist": inst.dist.ravel().tolist(),
        "weights": w.as_dict(),
        "cov": inst.cov.tolist(), "overlap": inst.overlap.ravel().tolist(),
        "flank": inst.flank.ravel().tolist(), "cover_q": inst.cover_q.tolist(),
        "lambda": p.lam, "offset": p.offset,
        "q": p.Q.ravel().tolist(),
        "greedy_energy_sample": float(p.energy_with_offset(np.eye(a, n, dtype=float).ravel())),
    }


def main(argv: list[str]) -> int:
    out = argv[0] if argv else "qubo_parity.json"
    data = {"cases": [case(*c) for c in CASES]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {len(data['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
