import numpy as np
import pytest

from svl_solver.model.instance import Instance


def hand_instance(edges, n_nodes, player, agents, escape, cover, pos=None, vision_hops=1):
    if pos is None:
        pos = [(float(i), 0.0) for i in range(n_nodes)]
    return Instance(
        instance_id="hand", seed=0, n_nodes=n_nodes, edges=edges, pos=np.asarray(pos, dtype=float),
        player_node=player, agent_nodes=agents, cover_nodes=cover, escape_routes=escape,
        vision_hops=vision_hops,
    )


@pytest.fixture
def hand_instance_factory():
    return hand_instance
