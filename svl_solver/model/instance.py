"""The squad-coordination problem instance.

An instance is a *tactical navigation graph* (20-100 nodes, NOT the NavMesh),
the player's node, the escape-route nodes the squad should cover, the cover
nodes agents can use, and where each agent currently stands.

Everything a solver needs is precomputed here as dense numpy arrays so that
the QUBO builder and the reference objective read the *same* numbers:

    cov[n]        how many escape-route nodes are visible from node n
    overlap[n,m]  how many escape-route nodes are visible from BOTH n and m
    flank[n,m]    in [0,1]; 1 when n and m sit on opposite sides of the player
    is_cover[n]   1 if n is a cover node
    cover_q[n]    in [0,1]: is_cover · max(0, cos(facing[n], direction n→player)) — cover
                  only counts when it stands between the node and the player. Without
                  facings (``cover_facing=None``) it equals is_cover.
    dist[a,n]     travel cost (graph hops) for agent a to reach node n

Visibility v1 is a graph-hop radius (``vision_hops``): node i sees node j when
the shortest path between them has at most ``vision_hops`` edges. This is a
deliberate simplification; a ray-cast visibility matrix exported from Unity
can replace it later without touching the formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Weights:
    """Objective weights. All non-negative. See qubo/formulation.py for how
    they enter the energy function."""

    coverage: float = 1.0      # reward per escape node seen from an agent's node
    redundancy: float = 0.5    # penalty per escape node seen by two agents (pushes distinct coverage)
    flanking: float = 1.0      # reward for agent pairs on opposite sides of the player
    cover: float = 0.5         # reward for standing on a cover node
    distance: float = 0.1      # cost per hop of travel

    def as_dict(self) -> dict[str, float]:
        return {
            "coverage": self.coverage,
            "redundancy": self.redundancy,
            "flanking": self.flanking,
            "cover": self.cover,
            "distance": self.distance,
        }

    @staticmethod
    def from_dict(d: dict) -> "Weights":
        return Weights(**{k: float(v) for k, v in d.items()})


@dataclass
class Instance:
    instance_id: str
    seed: int
    n_nodes: int
    edges: list[tuple[int, int]]
    pos: np.ndarray                  # (n_nodes, 2) planar coordinates
    player_node: int
    agent_nodes: list[int]           # current node of each agent (len = n_agents)
    cover_nodes: list[int]
    escape_routes: list[int]
    vision_hops: int = 2
    unreachable_cost: float = 1e3    # dist for nodes not reachable by an agent
    # Per node, the world yaw (degrees; 0 = +y, 90 = +x, as Unity's yaw over x/z) of the side its
    # cover protects from. Ignored for non-cover nodes. None → every cover counts fully.
    cover_facing: list[float] | None = None

    # ---- derived (filled by __post_init__) ----
    graph: nx.Graph = field(init=False, repr=False)
    visible: np.ndarray = field(init=False, repr=False)   # (n, n) bool
    cov: np.ndarray = field(init=False, repr=False)       # (n,)
    overlap: np.ndarray = field(init=False, repr=False)   # (n, n)
    flank: np.ndarray = field(init=False, repr=False)     # (n, n)
    is_cover: np.ndarray = field(init=False, repr=False)  # (n,)
    cover_q: np.ndarray = field(init=False, repr=False)   # (n,)
    dist: np.ndarray = field(init=False, repr=False)      # (n_agents, n)

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float).reshape(self.n_nodes, 2)
        self.agent_nodes = [int(a) for a in self.agent_nodes]
        self.cover_nodes = sorted({int(c) for c in self.cover_nodes})
        self.escape_routes = sorted({int(e) for e in self.escape_routes})
        self.edges = [(int(u), int(v)) for u, v in self.edges]
        if self.cover_facing is not None:
            self.cover_facing = [float(f) for f in self.cover_facing]

        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        g.add_edges_from(self.edges)
        self.graph = g

        self._validate()
        self._precompute()

    # ------------------------------------------------------------------ props
    @property
    def n_agents(self) -> int:
        return len(self.agent_nodes)

    @property
    def n_vars(self) -> int:
        return self.n_agents * self.n_nodes

    # ------------------------------------------------------------- internals
    def _validate(self) -> None:
        n = self.n_nodes
        if n < 2:
            raise ValueError("instance needs at least 2 nodes")
        if not (0 <= self.player_node < n):
            raise ValueError("player_node out of range")
        if self.n_agents == 0:
            raise ValueError("instance needs at least one agent")
        if self.n_agents > n:
            raise ValueError("more agents than nodes: no feasible assignment exists")
        for a in self.agent_nodes:
            if not (0 <= a < n):
                raise ValueError("agent node out of range")
        for c in self.cover_nodes + self.escape_routes:
            if not (0 <= c < n):
                raise ValueError("cover/escape node out of range")
        if self.cover_facing is not None and len(self.cover_facing) != n:
            raise ValueError("cover_facing needs one yaw per node")
        for u, v in self.edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge endpoint out of range")

    def _precompute(self) -> None:
        n = self.n_nodes
        hops = _all_pairs_hops(self.graph, n)

        visible = hops <= self.vision_hops           # includes self (hops == 0)
        self.visible = visible

        esc = np.zeros(n, dtype=bool)
        esc[self.escape_routes] = True
        # sees_esc[n, e] = node n sees escape node e
        sees_esc = visible[:, esc]                    # (n, n_escape)
        self.cov = sees_esc.sum(axis=1).astype(float)
        self.overlap = (sees_esc.astype(float) @ sees_esc.astype(float).T)
        np.fill_diagonal(self.overlap, 0.0)

        self.is_cover = np.zeros(n, dtype=float)
        self.is_cover[self.cover_nodes] = 1.0
        self.cover_q = _cover_quality(self.pos, self.player_node, self.is_cover, self.cover_facing)

        self.flank = _flank_matrix(self.pos, self.player_node)

        d = np.full((self.n_agents, n), self.unreachable_cost, dtype=float)
        for a, start in enumerate(self.agent_nodes):
            row = hops[start]
            reach = np.isfinite(row)
            d[a, reach] = row[reach]
        self.dist = d

    # ----------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "n_nodes": int(self.n_nodes),
            "edges": [[int(u), int(v)] for u, v in self.edges],
            "pos": self.pos.round(6).tolist(),
            "player_node": int(self.player_node),
            "agent_nodes": list(map(int, self.agent_nodes)),
            "cover_nodes": list(map(int, self.cover_nodes)),
            "escape_routes": list(map(int, self.escape_routes)),
            "vision_hops": int(self.vision_hops),
            "unreachable_cost": float(self.unreachable_cost),
            "cover_facing": None if self.cover_facing is None else [round(f, 6) for f in self.cover_facing],
        }

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(
            instance_id=str(d["instance_id"]),
            seed=int(d["seed"]),
            n_nodes=int(d["n_nodes"]),
            edges=[tuple(e) for e in d["edges"]],
            pos=np.asarray(d["pos"], dtype=float),
            player_node=int(d["player_node"]),
            agent_nodes=list(d["agent_nodes"]),
            cover_nodes=list(d.get("cover_nodes", [])),
            escape_routes=list(d.get("escape_routes", [])),
            vision_hops=int(d.get("vision_hops", 2)),
            unreachable_cost=float(d.get("unreachable_cost", 1e3)),
            cover_facing=d.get("cover_facing"),
        )


# --------------------------------------------------------------------- utils
def _all_pairs_hops(g: nx.Graph, n: int) -> np.ndarray:
    hops = np.full((n, n), np.inf)
    for src, lengths in nx.all_pairs_shortest_path_length(g):
        for dst, L in lengths.items():
            hops[src, dst] = L
    return hops


def _flank_matrix(pos: np.ndarray, player: int) -> np.ndarray:
    """flank[n,m] = (1 - cos θ) / 2 where θ is the angle at the player between
    the directions to n and m. 0 when n, m are on the same side, 1 when they
    are exactly opposite. Nodes coinciding with the player have no direction
    and get 0 everywhere."""
    v = pos - pos[player]
    norm = np.linalg.norm(v, axis=1)
    ok = norm > 1e-9
    u = np.zeros_like(v)
    u[ok] = v[ok] / norm[ok, None]
    cos = u @ u.T
    f = (1.0 - cos) / 2.0
    f[~ok, :] = 0.0
    f[:, ~ok] = 0.0
    np.fill_diagonal(f, 0.0)
    return np.clip(f, 0.0, 1.0)


def _cover_quality(pos: np.ndarray, player: int, is_cover: np.ndarray,
                   facing: list[float] | None) -> np.ndarray:
    """is_cover · max(0, cos θ), θ between the cover's facing and the direction to the
    player. A node on the player's own spot has no direction and keeps is_cover."""
    if facing is None:
        return is_cover.copy()
    yaw = np.radians(np.asarray(facing, dtype=float))
    face = np.stack([np.sin(yaw), np.cos(yaw)], axis=1)       # unit, (n, 2)
    to_player = pos[player] - pos
    norm = np.linalg.norm(to_player, axis=1)
    ok = norm > 1e-9
    cos = np.ones(len(pos))
    cos[ok] = (face[ok] * to_player[ok]).sum(axis=1) / norm[ok]
    return is_cover * np.clip(cos, 0.0, 1.0)


def make_positions(coords: Iterable[tuple[float, float]]) -> np.ndarray:
    return np.asarray(list(coords), dtype=float)
