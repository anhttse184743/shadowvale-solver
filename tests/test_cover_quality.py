"""cover_q: cover only counts when it stands between the node and the player (decided 30/09/2026)."""

import numpy as np
import pytest

from svl_solver.instances.generator import generate
from svl_solver.model.instance import Instance


def _line(facing):
    # Player at node 0 (origin); cover node 1 one unit east (+x). Facing west (270°) points at him.
    return Instance(instance_id="cq", seed=0, n_nodes=3, edges=[(0, 1), (1, 2)],
                    pos=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), player_node=0,
                    agent_nodes=[2], cover_nodes=[1], escape_routes=[2], cover_facing=facing)


def test_facing_the_player_counts_fully_away_or_sideways_not_at_all():
    assert _line([0, 270, 0]).cover_q[1] == pytest.approx(1.0)
    assert _line([0, 90, 0]).cover_q[1] == pytest.approx(0.0)      # faces away
    assert _line([0, 0, 0]).cover_q[1] == pytest.approx(0.0)       # faces north: sideways
    assert _line([0, 225, 0]).cover_q[1] == pytest.approx(np.cos(np.radians(45)))


def test_non_cover_nodes_score_zero_and_missing_facing_keeps_is_cover():
    inst = _line([0, 270, 270])
    assert inst.cover_q[2] == 0.0
    assert _line(None).cover_q.tolist() == [0.0, 1.0, 0.0]


def test_generator_facing_changes_nothing_else_and_round_trips():
    a = generate(seed=5, n_nodes=20, n_agents=4)
    b = Instance.from_dict(a.to_dict())
    assert b.cover_facing == pytest.approx(a.cover_facing)
    assert np.allclose(a.cover_q, b.cover_q)
    assert all(0.0 <= q <= 1.0 for q in a.cover_q)
    assert all(a.cover_q[n] == 0.0 for n in range(a.n_nodes) if n not in a.cover_nodes)
