"""Instances recorded by the game rebuild into the very QUBO the game solved.

The fixture is a real line written by Unity's CoordinationRecorder during a Map 1
test (jetty squad B). Python must get the same λ, offset and energy of the applied
plan as C# did — the cross-language check on real game data, next to the C# parity
test on generated data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from svl_solver.bench.runner import run_suite
from svl_solver.instances.game_records import from_record, load_records
from svl_solver.instances.generator import generate
from svl_solver.instances.suite import SuiteSpec
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.base import assignment_to_x
from svl_solver.solvers.greedy import GreedySolver

FIXTURE = Path(__file__).parent / "fixtures" / "game_record.jsonl"


def test_real_game_record_rebuilds_the_games_qubo():
    inst = load_records(FIXTURE)[0]
    rec = inst.record
    p = build_qubo(inst, inst.weights)
    assert p.lam == pytest.approx(rec["lambda"], rel=1e-9)
    assert p.offset == pytest.approx(rec["offset"], rel=1e-9)
    x = assignment_to_x(inst.recorded_plan, inst.n_agents, inst.n_nodes)
    assert p.energy_with_offset(x) == pytest.approx(rec["energy"], abs=1e-6), \
        "The applied plan has the same energy in Python as it had in C#."
    greedy = GreedySolver().solve(p)
    if rec["variant"] == "greedy":
        assert greedy.assignment == inst.recorded_plan, "Python greedy == C# greedy on the game's own instance."


def _record_of(inst, weights) -> dict:
    """What the C# recorder would write for a generated instance."""
    n, a = inst.n_nodes, inst.n_agents
    return {
        "instance_id": inst.instance_id, "seed": inst.seed, "agent_count": a, "node_count": n,
        "player_local": inst.player_node, "escape": inst.escape_routes,
        "x": inst.pos[:, 0].tolist(), "z": inst.pos[:, 1].tolist(), "cover_facing": inst.cover_facing,
        "is_cover": "".join("1" if c else "0" for c in inst.is_cover),
        "visible": "".join("1" if v else "0" for v in inst.visible.ravel()),
        "dist": inst.dist.ravel().tolist(), "weights": weights.as_dict(), "plan": [],
    }


def test_generated_instance_round_trips_through_the_record_format():
    inst = generate(seed=9, n_nodes=14, n_agents=4, chokepoint_density=0.4)
    game = from_record(json.loads(json.dumps(_record_of(inst, inst_weights := build_qubo(inst).weights))))
    a, b = build_qubo(inst, inst_weights), build_qubo(game, game.weights)
    assert np.allclose(a.Q, b.Q) and a.offset == pytest.approx(b.offset)


def test_bench_runs_on_recorded_instances(tmp_path):
    records = load_records(FIXTURE)
    spec = SuiteSpec(name="game", nodes=[], agents=[], densities=[], seeds=[], budget_ms=60)
    rows = run_suite(spec, ["greedy", "qiea"], repeats=2, out=tmp_path / "game.csv", progress=None, instances=records)
    assert len(rows) == 4
    assert all(r.n_vars == records[0].n_vars for r in rows)
    qiea = min(r.energy for r in rows if r.solver == "qiea")
    greedy = min(r.energy for r in rows if r.solver == "greedy")
    assert qiea <= greedy + 1e-9
