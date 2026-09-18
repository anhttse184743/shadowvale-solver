"""Objective O3: the benchmark must be reproducible. Same seed → same instance."""

import json

import networkx as nx

from svl_solver.instances.generator import generate
from svl_solver.instances.io import to_qubo_request
from svl_solver.instances.suite import BUILTIN_SUITES, iter_instances
from svl_solver.model.instance import Instance


def test_same_seed_same_instance():
    a = generate(seed=42, n_nodes=30, n_agents=5, chokepoint_density=0.4)
    b = generate(seed=42, n_nodes=30, n_agents=5, chokepoint_density=0.4)
    assert a.to_dict() == b.to_dict()
    assert json.dumps(to_qubo_request(a), sort_keys=True) == json.dumps(to_qubo_request(b), sort_keys=True)


def test_different_seed_different_instance():
    a = generate(seed=1, n_nodes=30, n_agents=5)
    b = generate(seed=2, n_nodes=30, n_agents=5)
    assert a.to_dict() != b.to_dict()


def test_generated_graph_is_connected_and_valid():
    for seed in range(1, 6):
        for dens in (0.0, 0.5, 1.0):
            inst = generate(seed=seed, n_nodes=25, n_agents=4, chokepoint_density=dens)
            assert nx.is_connected(inst.graph)
            assert len(set(inst.agent_nodes)) == 4
            assert inst.player_node not in inst.agent_nodes
            assert len(inst.escape_routes) >= 2
            assert all(0 <= e < 25 for e in inst.escape_routes)


def test_density_thins_graph():
    dense = generate(seed=7, n_nodes=40, n_agents=4, chokepoint_density=0.0)
    sparse = generate(seed=7, n_nodes=40, n_agents=4, chokepoint_density=1.0)
    assert sparse.graph.number_of_edges() < dense.graph.number_of_edges()
    assert sparse.graph.number_of_edges() == 39      # exactly the spanning tree


def test_json_roundtrip():
    inst = generate(seed=3, n_nodes=12, n_agents=3)
    back = Instance.from_dict(json.loads(json.dumps(inst.to_dict())))
    assert back.to_dict() == inst.to_dict()
    assert (back.cov == inst.cov).all()
    assert (back.dist == inst.dist).all()


def test_builtin_suite_enumerates_expected_count():
    spec = BUILTIN_SUITES["tiny"]
    insts = list(iter_instances(spec))
    assert len(insts) == spec.n_instances == 6
    assert len({i.instance_id for i in insts}) == 6
