"""QIEA (Han & Kim 2002) on the coordination QUBO: escapes a greedy trap, never loses
to greedy, reproducible with a fixed seed, respects the latency budget, and reports
its sampler metrics through the sidecar."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from svl_solver.api.app import create_app
from svl_solver.instances.generator import generate
from svl_solver.instances.io import to_qubo_request
from svl_solver.qubo.formulation import QuboProblem, build_qubo
from svl_solver.solvers.brute_force import BruteForceSolver
from svl_solver.solvers.greedy import GreedySolver
from svl_solver.solvers.qiea import QuantumInspiredEvolutionarySolver


def _greedy_trap() -> QuboProblem:
    # 2 agents × 2 nodes, x = [a0n0, a0n1, a1n0, a1n1].
    # a0: node0 reward 10, node1 reward 9;  a1: node0 reward 9, node1 reward 0.
    # Greedy lets a0 grab node0 first → E = −10. Swapping gives the optimum E = −18.
    # (Test case carried over from the removed Unity QieaSolverTests.)
    q = np.zeros(16)
    q[0], q[5], q[10], q[15] = -110, -109, -109, -100
    q[1] = q[11] = q[2] = q[7] = 200
    return QuboProblem.from_flat(q.tolist(), n_agents=2, n_nodes=2, offset=200)


def test_escapes_greedy_trap_and_finds_optimum():
    p = _greedy_trap()
    assert GreedySolver().solve(p).energy == pytest.approx(-10)
    sol = QuantumInspiredEvolutionarySolver(generations=30).solve(p, seed=42)
    assert sol.feasible
    assert sol.assignment == [1, 0]
    assert sol.energy == pytest.approx(-18)


def test_fixed_generations_are_reproducible_and_report_samples():
    p = _greedy_trap()
    a = QuantumInspiredEvolutionarySolver(population=8, generations=10).solve(p, seed=7)
    b = QuantumInspiredEvolutionarySolver(population=8, generations=10).solve(p, seed=7)
    assert a.assignment == b.assignment
    assert a.meta["samples_evaluated"] == 80
    assert 0 <= a.meta["raw_feasible_samples"] <= 80
    assert a.meta["raw_feasible_samples"] == b.meta["raw_feasible_samples"]
    assert a.meta["stopped_on_budget"] is False


@pytest.mark.parametrize("seed", range(1, 9))
def test_never_worse_than_greedy_and_always_feasible(seed):
    inst = generate(seed=seed, n_nodes=16, n_agents=4, chokepoint_density=0.3)
    p = build_qubo(inst)
    greedy = GreedySolver().solve(p)
    sol = QuantumInspiredEvolutionarySolver(generations=20).solve(p, seed=seed)
    assert sol.feasible
    assert len(set(sol.assignment)) == inst.n_agents
    assert sol.energy <= greedy.energy + 1e-9


def test_reaches_brute_force_optimum_on_small_instances():
    hits = 0
    for seed in range(1, 11):
        inst = generate(seed=seed, n_nodes=8, n_agents=3, chokepoint_density=0.3)
        p = build_qubo(inst)
        best = BruteForceSolver().solve(p)
        sol = QuantumInspiredEvolutionarySolver(generations=60).solve(p, seed=seed)
        hits += sol.energy == pytest.approx(best.energy)
    assert hits >= 8, f"QIEA matched the optimum on only {hits}/10 small instances"


def test_stops_inside_latency_budget():
    inst = generate(seed=3, n_nodes=60, n_agents=8, chokepoint_density=0.3)
    p = build_qubo(inst)
    sol = QuantumInspiredEvolutionarySolver(generations=10_000).solve(p, budget_ms=40, seed=3)
    assert sol.feasible
    assert sol.meta["stopped_on_budget"] is True
    assert sol.latency_ms < 40
    assert sol.meta["samples_evaluated"] > 0


def test_sidecar_serves_qiea_with_sampler_metrics():
    client = TestClient(create_app())
    ids = {v["id"]: v for v in client.get("/variants").json()}
    assert ids["qiea"]["available"] is True and ids["qiea"]["family"] == "quantum_inspired"

    inst = generate(seed=1, n_nodes=12, n_agents=4, chokepoint_density=0.3)
    plan = client.post("/solve", json=to_qubo_request(inst, latency_budget_ms=120, variant="qiea")).json()
    assert plan["solver_variant_id"] == "qiea"
    assert plan["fallback_reason"] is None
    assert len(set(plan["agent_to_target"])) == inst.n_agents
    assert plan["samples_evaluated"] > 0
    assert 0 <= plan["raw_feasible_samples"] <= plan["samples_evaluated"]
