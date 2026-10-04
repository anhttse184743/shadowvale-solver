"""Sidecar contract: /health, /variants, /solve always returns a feasible plan,
unknown variants fall back to greedy, malformed Q is rejected."""

import pytest
from fastapi.testclient import TestClient

from svl_solver.api.app import create_app
from svl_solver.instances.generator import generate
from svl_solver.instances.io import to_qubo_request


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:      # runs startup: the solver worker is warmed up like in play
        yield c


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


def test_concurrent_requests_share_one_lane_and_keep_their_budget():
    # Several squads re-planning at once: solves queue in one lane (the GIL makes parallel solves
    # slower for everyone), each answers within its budget counted from arrival, queued ones with
    # less search or greedy — and the reason says so.
    import concurrent.futures as cf
    import time

    from svl_solver.api.app import _solve_in_lane, warm_up
    from svl_solver.api.schemas import QuboRequest
    from svl_solver.qubo.formulation import QuboProblem

    _, raw = _request(seed=3, n=24, a=6, variant="qiea")
    req = QuboRequest(**raw)
    problem = QuboProblem.from_flat(req.q_matrix, req.agent_count, req.node_count, offset=req.offset,
                                    lam=req.penalty_conflict, instance_id=req.instance_id)

    warm_up()

    def one(_):
        t0 = time.perf_counter()
        sol, hit, reason = _solve_in_lane("qiea", problem, req, t0)
        return (time.perf_counter() - t0) * 1000.0, sol, reason

    with cf.ThreadPoolExecutor(4) as ex:
        results = list(ex.map(one, range(4)))
    for elapsed, sol, _ in results:
        assert len(set(sol.assignment)) == req.agent_count
        assert elapsed <= req.latency_budget_ms + 30, f"{elapsed:.0f} ms"
    assert any(r is None for _, _, r in results)                  # the first one got its whole budget
    assert any(r and "queued" in r for _, _, r in results)        # the last ones were told why
