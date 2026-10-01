"""solvers/exact.py: branch and bound proves the optimum where brute force cannot go (O4)."""

import math

import pytest

from svl_solver.bench.runner import _optimum
from svl_solver.instances.game_records import load_records
from svl_solver.instances.generator import generate
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.brute_force import BruteForceSolver
from svl_solver.solvers.exact import BranchAndBoundSolver
from svl_solver.solvers.registry import get_solver

from pathlib import Path


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force(seed):
    inst = generate(seed=seed, n_nodes=11, n_agents=4, chokepoint_density=0.35)
    p = build_qubo(inst)
    exact = BranchAndBoundSolver().solve(p)
    assert exact.meta["proven_optimal"]
    assert exact.feasible
    assert exact.energy == pytest.approx(BruteForceSolver().solve(p).energy, abs=1e-9)
    assert exact.meta["nodes_explored"] < math.perm(11, 4), "the bound prunes"


def test_proves_a_game_size_instance_brute_force_cannot_enumerate():
    inst = generate(seed=3, n_nodes=24, n_agents=6, chokepoint_density=0.3)
    p = build_qubo(inst)
    assert math.perm(24, 6) > 5_000_000
    sol = get_solver("exact").solve(p)
    assert sol.meta["proven_optimal"]
    for other in ("greedy", "qiea"):
        assert sol.energy <= get_solver(other).solve(p, budget_ms=120, seed=1).energy + 1e-9


def test_time_limit_returns_the_best_so_far_unproven():
    p = build_qubo(generate(seed=3, n_nodes=24, n_agents=8, chokepoint_density=0.3))
    sol = BranchAndBoundSolver(time_limit_s=0.2).solve(p)
    assert not sol.meta["proven_optimal"]
    assert sol.feasible
    assert sol.energy <= get_solver("greedy").solve(p).energy + 1e-9, "never worse than its greedy start"


def test_bench_uses_it_for_the_optimum_of_recorded_game_instances():
    inst = load_records(Path(__file__).parent / "fixtures" / "game_record.jsonl")[0]
    p = build_qubo(inst, inst.weights)
    e, f = _optimum(inst, p, max_candidates=1000, cache={})   # forced past brute force
    assert e == pytest.approx(BruteForceSolver().solve(p).energy, abs=1e-9)
