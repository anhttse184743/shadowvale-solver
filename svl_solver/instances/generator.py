"""Seeded procedural generator of tactical navigation graphs (objective O3).

Same (seed, n_nodes, n_agents, chokepoint_density) → byte-identical instance,
on every machine. Only ``numpy.random.default_rng(seed)`` is used for
randomness and every networkx call is deterministic given node order.

Construction
------------
1. n nodes uniformly in the unit square.
2. k-nearest-neighbour edges (k = 3) → a "room-like" local graph.
3. Euclidean minimum spanning tree edges are added and are never removed →
   the graph is always connected.
4. Every non-MST edge is dropped with probability ``chokepoint_density``.
   Higher density → sparser graph → more chokepoints, which is exactly the
   regime where coordination matters.
5. player: random node. escape routes: sampled among the farthest third of
   nodes (by hops). cover: ~30 % of nodes. agents: distinct nodes at least
   2 hops from the player when possible.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ..model.instance import Instance


def instance_id_for(n_nodes: int, n_agents: int, chokepoint_density: float, seed: int) -> str:
    return f"svl-n{n_nodes}-a{n_agents}-c{chokepoint_density:.2f}-s{seed}"


def generate(seed: int, n_nodes: int, n_agents: int, chokepoint_density: float = 0.3,
             knn: int = 3, vision_hops: int = 2, escape_fraction: float = 0.125,
             cover_fraction: float = 0.3) -> Instance:
    if n_nodes < 4:
        raise ValueError("n_nodes must be >= 4")
    if not (0.0 <= chokepoint_density <= 1.0):
        raise ValueError("chokepoint_density must be in [0, 1]")
    if n_agents < 1 or n_agents > n_nodes - 1:
        raise ValueError("need 1 <= n_agents <= n_nodes - 1")

    rng = np.random.default_rng(seed)
    pos = rng.random((n_nodes, 2))

    # pairwise euclidean distances
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))

    # k-NN edges
    k = min(knn, n_nodes - 1)
    knn_edges: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        order = np.argsort(d[i], kind="stable")
        for j in order[1:k + 1]:
            u, v = (i, int(j)) if i < j else (int(j), i)
            knn_edges.add((u, v))

    # MST edges (always kept → connectivity)
    full = nx.Graph()
    full.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            full.add_edge(i, j, weight=float(d[i, j]))
    mst_edges = {tuple(sorted((int(u), int(v))))
                 for u, v in nx.minimum_spanning_edges(full, algorithm="kruskal", data=False)}

    # thin out non-MST edges → chokepoints
    edges: set[tuple[int, int]] = set(mst_edges)
    for e in sorted(knn_edges - mst_edges):
        if rng.random() >= chokepoint_density:
            edges.add(e)
    edge_list = sorted(edges)

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edge_list)

    player = int(rng.integers(n_nodes))
    hops = nx.single_source_shortest_path_length(g, player)
    hop_arr = np.array([hops[i] for i in range(n_nodes)])

    # escape routes: among the farthest third (ties broken by node id), sample
    n_escape = max(2, int(round(escape_fraction * n_nodes)))
    order_far = sorted(range(n_nodes), key=lambda i: (-hop_arr[i], i))
    far_pool = [i for i in order_far[: max(n_escape, n_nodes // 3)] if i != player]
    escape = sorted(int(v) for v in rng.choice(far_pool, size=min(n_escape, len(far_pool)), replace=False))

    # cover nodes
    n_cover = max(1, int(round(cover_fraction * n_nodes)))
    cover_pool = [i for i in range(n_nodes) if i != player]
    cover = sorted(int(v) for v in rng.choice(cover_pool, size=min(n_cover, len(cover_pool)), replace=False))

    # agents: distinct, ≥ 2 hops from player when enough such nodes exist
    pool = [i for i in range(n_nodes) if i != player and hop_arr[i] >= 2]
    if len(pool) < n_agents:
        pool = [i for i in range(n_nodes) if i != player]
    agents = [int(v) for v in rng.choice(pool, size=n_agents, replace=False)]

    return Instance(
        instance_id=instance_id_for(n_nodes, n_agents, chokepoint_density, seed),
        seed=int(seed),
        n_nodes=int(n_nodes),
        edges=edge_list,
        pos=pos,
        player_node=player,
        agent_nodes=agents,
        cover_nodes=cover,
        escape_routes=escape,
        vision_hops=vision_hops,
    )
