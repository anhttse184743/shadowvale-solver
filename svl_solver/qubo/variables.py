"""Binary variable layout.

    x[a, n] = 1  <=>  agent a is assigned to node n
    flat index  i = a * n_nodes + n

The Unity ``QuboBuilder`` (C#) MUST use the same layout so a Q matrix built on
either side decodes identically. Keep this file trivially small on purpose.
"""

from __future__ import annotations


def var_index(agent: int, node: int, n_nodes: int) -> int:
    return agent * n_nodes + node


def var_agent_node(i: int, n_nodes: int) -> tuple[int, int]:
    return divmod(i, n_nodes)


def n_vars(n_agents: int, n_nodes: int) -> int:
    return n_agents * n_nodes
