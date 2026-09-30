"""Sidecar contract: /health, /variants, /solve always returns a feasible plan,
unknown variants fall back to greedy, malformed Q is rejected."""

import pytest
from fastapi.testclient import TestClient

from svl_solver.api.app import create_app
from svl_solver.instances.generator import generate
from svl_solver.instances.io import to_qubo_request


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _request(seed=1, n=12, a=4, variant="greedy"):
    inst = generate(seed=seed, n_nodes=n, n_agents=a, chokepoint_density=0.3)
    return inst, to_qubo_request(inst, latency_budget_ms=120, variant=variant)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "greedy" in body["variants_available"]


def test_variants_lists_planned_and_available(client):
    ids = {v["id"]: v for v in client.get("/variants").json()}
    assert ids["greedy"]["available"] is True
    assert ids["qaoa"]["available"] is True and ids["qaoa"]["family"] == "quantum_inspired"
    assert ids["qpu"]["available"] is False


def test_solve_returns_feasible_plan(client):
    inst, req = _request()
    r = client.post("/solve", json=req)
    assert r.status_code == 200, r.text
    plan = r.json()
    t = plan["agent_to_target"]
    assert len(t) == inst.n_agents
    assert len(set(t)) == inst.n_agents
    assert all(0 <= v < inst.n_nodes for v in t)
    assert plan["solver_variant_id"] == "greedy"
    assert plan["hit_latency_budget"] is False
    assert plan["fallback_reason"] is None
    assert plan["instance_id"] == inst.instance_id


def test_unknown_variant_falls_back_to_greedy(client):
    _, req = _request(variant="qpu")
    plan = client.post("/solve", json=req).json()
    assert plan["solver_variant_id"] == "greedy"
    assert "unavailable" in plan["fallback_reason"]


def test_qaoa_beyond_the_statevector_limit_falls_back_with_the_reason(client):
    _, req = _request(n=12, a=4, variant="qaoa")          # 48 qubits
    plan = client.post("/solve", json=req).json()
    assert plan["solver_variant_id"] == "greedy"
    assert "qubits" in plan["fallback_reason"]


def test_brute_variant_on_small_instance_beats_or_equals_greedy(client):
    inst, req = _request(n=8, a=2, variant="brute")
    best = client.post("/solve", json=req).json()
    req["variant"] = "greedy"
    g = client.post("/solve", json=req).json()
    assert best["solver_variant_id"] == "brute"
    assert best["objective_value"] >= g["objective_value"] - 1e-9


def test_bad_matrix_size_rejected(client):
    _, req = _request()
    req["q_matrix"] = req["q_matrix"][:-1]
    assert client.post("/solve", json=req).status_code == 422
