"""Coordination instances recorded inside the game (Unity CoordinationRecorder, M4).

Each JSONL line holds the raw inputs of one in-game re-plan — node positions (x, z),
cover flags and facings, escape nodes, the visibility matrix computed by ray casts,
agent→node hop distances, the weights — plus the plan the game applied. From them
this module rebuilds the exact same QUBO the game solved (same formulas as
``model/instance.py``, parity-tested on the C# side), so every solver can be
benchmarked on real game situations, not only on generated graphs (O4/O5).

    records = load_records("coordination-xxxx.jsonl")
    problem = build_qubo(records[0], records[0].weights)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..model.instance import Weights, _cover_quality, _flank_matrix


@dataclass
class GameInstance:
    """Duck-types the parts of ``Instance`` the objective, the QUBO builder and the bench read."""

    instance_id: str
    seed: int
    n_nodes: int
    n_agents: int
    cov: np.ndarray
    overlap: np.ndarray
    flank: np.ndarray
    cover_q: np.ndarray
    dist: np.ndarray
    weights: Weights
    escape_routes: list[int]
    record: dict = field(repr=False, default_factory=dict)

    @property
    def n_vars(self) -> int:
        return self.n_agents * self.n_nodes

    @property
    def recorded_plan(self) -> list[int]:
        return [int(v) for v in self.record.get("plan", [])]


def from_record(rec: dict) -> GameInstance:
    n, a = int(rec["node_count"]), int(rec["agent_count"])
    pos = np.stack([np.asarray(rec["x"], float), np.asarray(rec["z"], float)], axis=1)
    player = int(rec["player_local"])
    visible = np.array([c == "1" for c in rec["visible"]], dtype=bool).reshape(n, n)
    esc = np.zeros(n, dtype=bool)
    esc[list(rec["escape"])] = True
    sees = visible[:, esc].astype(float)
    overlap = sees @ sees.T
    np.fill_diagonal(overlap, 0.0)
    is_cover = np.array([c == "1" for c in rec["is_cover"]], dtype=float)
    return GameInstance(
        instance_id=str(rec["instance_id"]), seed=int(rec.get("seed", 0)), n_nodes=n, n_agents=a,
        cov=sees.sum(axis=1), overlap=overlap, flank=_flank_matrix(pos, player),
        cover_q=_cover_quality(pos, player, is_cover, rec.get("cover_facing")),
        dist=np.asarray(rec["dist"], float).reshape(a, n),
        weights=Weights.from_dict(rec["weights"]),
        escape_routes=sorted(int(e) for e in rec["escape"]), record=rec,
    )


def load_records(paths: str | Path | Iterable[str | Path]) -> list[GameInstance]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[GameInstance] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(from_record(json.loads(line)))
    return out
