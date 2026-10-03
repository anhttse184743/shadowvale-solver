"""Moving AI benchmark maps → squad-coordination instances on maps from commercial games.

Sturtevant (2012), "Benchmarks for Grid-Based Pathfinding", IEEE Transactions on Computational
Intelligence and AI in Games 4(2):144–148; maps from movingai.com/benchmarks (Dragon Age:
Origins, StarCraft, Warcraft III, Baldur's Gate II — see scripts/fetch_datasets.py).

Only the *maps* are used, not the path-finding scenarios: each map is turned into the same kind
of coordination instance the game builds on Map 1, by the same steps as the Unity code —

1. **tactical graph** (TacticalGraphBaker): passable cells sampled every ``spacing`` cells,
   8-neighbour edges where the straight line between two samples is free;
2. **local subgraph** (LocalSubgraph.Extract): breadth-first from the player up to ``k_nodes``
   nodes, the agents' nodes added on top;
3. **escape nodes** (ContainPlayerTask): the subgraph's boundary nodes (with a neighbour outside),
   never the player's;
4. **cover**: a node within 2 cells of an obstacle (the baker's cover snap); it faces that
   obstacle, the side it shields;
5. **visibility** (VisibilityCache): line of sight on the grid within ``vision`` cells;
6. **dist**: graph hops from each agent's node (TacticalGraph.Hops), 1000 when unreachable.

The output is a record in the format the game writes (CoordinationRecorder), so the instances go
through ``svl-bench game`` exactly like the game's own. Deterministic for a seed.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

from ..model.instance import Weights

PASSABLE = set(".GS")       # ground, grass, swamp; T trees, @/O out of bounds, W water block


def read_map(path: str | Path) -> np.ndarray:
    """Passability grid (rows × cols, True = walkable) of a .map file."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    h = int(next(l for l in lines if l.startswith("height")).split()[1])
    w = int(next(l for l in lines if l.startswith("width")).split()[1])
    start = lines.index("map") + 1
    rows = [l.ljust(w, "@")[:w] for l in lines[start:start + h]]
    return np.array([[c in PASSABLE for c in r] for r in rows], dtype=bool)


def _clear(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when every cell on the straight line a→b (Bresenham) is walkable."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    r, c = r0, c0
    while True:
        if not grid[r, c]:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


class TacticalGraph:
    def __init__(self, grid: np.ndarray, spacing: int = 4):
        self.grid, self.spacing = grid, spacing
        h, w = grid.shape
        self.cells = [(r, c) for r in range(0, h, spacing) for c in range(0, w, spacing) if grid[r, c]]
        index = {cell: i for i, cell in enumerate(self.cells)}
        self.adj: list[list[int]] = [[] for _ in self.cells]
        for i, (r, c) in enumerate(self.cells):
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                j = index.get((r + dr * spacing, c + dc * spacing))
                if j is not None and _clear(grid, (r, c), self.cells[j]):
                    self.adj[i].append(j)
                    self.adj[j].append(i)

    def hops_from(self, src: int) -> np.ndarray:
        d = np.full(len(self.cells), -1, dtype=int)
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def largest_component(self) -> list[int]:
        seen = np.zeros(len(self.cells), dtype=bool)
        best: list[int] = []
        for s in range(len(self.cells)):
            if seen[s]:
                continue
            comp = np.flatnonzero(self.hops_from(s) >= 0)
            seen[comp] = True
            if len(comp) > len(best):
                best = comp.tolist()
        return best


def _cover_facing(grid: np.ndarray, cell: tuple[int, int], radius: int = 2) -> float | None:
    """Yaw (degrees, 0 = +z/rows, 90 = +x/cols, as the game) towards the nearest obstacle within
    ``radius`` cells — the side this spot shields; None when there is none. The game's baker snaps
    cover nodes onto obstacles up to about half a node spacing away; radius 2 at spacing 4 matches."""
    r, c = cell
    h, w = grid.shape
    best, best_d = None, math.inf
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if not (dr or dc):
                continue
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w) and grid[rr, cc]:
                continue
            d = math.hypot(dr, dc)
            if d < best_d:
                best, best_d = (dr, dc), d
    return None if best is None else math.degrees(math.atan2(best[1], best[0])) % 360.0


def instance_record(name: str, graph: TacticalGraph, n_agents: int, seed: int, k_nodes: int = 24,
                    vision: float = 14.0, agent_hops: tuple[int, int] = (2, 6), component: list[int] | None = None,
                    weights: Weights | None = None) -> dict | None:
    """One coordination instance as a game record; None when the map has no room for it at this seed."""
    rng = np.random.default_rng(seed)
    comp = component if component is not None else graph.largest_component()
    if len(comp) < k_nodes + n_agents:
        return None
    for _ in range(50):
        player = int(comp[int(rng.integers(len(comp)))])
        hops = graph.hops_from(player)
        near = [i for i in comp if agent_hops[0] <= hops[i] <= agent_hops[1]]
        if len(near) >= n_agents:
            break
    else:
        return None
    agents = sorted(int(a) for a in rng.choice(near, size=n_agents, replace=False))

    # Local subgraph: breadth-first from the player (neighbours in index order), agents on top.
    reserve = sum(1 for a in agents if a != player)
    limit = max(1, k_nodes - reserve)
    nodes, seen, q = [], {player}, deque([player])
    while q and len(nodes) < limit:
        u = q.popleft()
        nodes.append(u)
        for v in sorted(graph.adj[u]):
            if v not in seen:
                seen.add(v)
                q.append(v)
    for a in agents:
        if a not in nodes:
            nodes.append(a)
    local = {g: i for i, g in enumerate(nodes)}
    n = len(nodes)
    boundary = [i for i, g in enumerate(nodes) if any(v not in local for v in graph.adj[g])]
    escape = [i for i in boundary if i != 0] or [i for i in range(1, n)]

    cells = [graph.cells[g] for g in nodes]
    visible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        visible[i, i] = True
        for j in range(i + 1, n):
            (r0, c0), (r1, c1) = cells[i], cells[j]
            if math.hypot(r1 - r0, c1 - c0) <= vision and _clear(graph.grid, cells[i], cells[j]):
                visible[i, j] = visible[j, i] = True
    facing = [_cover_facing(graph.grid, cell) for cell in cells]
    is_cover = [f is not None for f in facing]
    dist = []
    for a in agents:
        h = graph.hops_from(a)
        dist += [float(h[g]) if h[g] >= 0 else 1000.0 for g in nodes]
    w = weights or Weights()
    return {
        "instance_id": f"mai-{name}-a{n_agents}-s{seed}", "seed": int(seed), "source": "movingai",
        "map": name, "agent_count": n_agents, "node_count": n, "player_local": 0,
        "agent_local": [local[a] for a in agents], "global_nodes": [int(g) for g in nodes], "escape": escape,
        "x": [float(c) for _, c in cells], "z": [float(r) for r, _ in cells],
        "cover_facing": [float(f) if f is not None else 0.0 for f in facing],
        "is_cover": "".join("1" if c else "0" for c in is_cover),
        "visible": "".join("1" if v else "0" for v in visible.ravel()),
        "dist": dist, "weights": w.as_dict(), "plan": [],
    }


def build_set(maps: list[Path], agents: list[int], seeds: list[int], spacing: int = 4, **kw) -> list[dict]:
    out = []
    for path in maps:
        graph = TacticalGraph(read_map(path), spacing=spacing)
        comp = graph.largest_component()
        set_name = path.parent.name
        for a in agents:
            for s in seeds:
                rec = instance_record(f"{set_name}-{path.stem}", graph, a, s, component=comp, **kw)
                if rec is not None:
                    out.append(rec)
    return out
