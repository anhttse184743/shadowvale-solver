"""exp-v2 solvers: assignment-space search, PBIL, the set-reduction optimum, McNemar (review of exp-v1)."""

import numpy as np
import pytest

from svl_solver.bench.qaoa_study import study_x, study_xy
from svl_solver.bench.stats import mcnemar_exact
from svl_solver.instances.generator import generate
from svl_solver.qubo.formulation import QuboProblem, build_qubo
from svl_solver.solvers.assignment import AssignmentModel
from svl_solver.solvers.base import assignment_to_x
from svl_solver.solvers.brute_force import BruteForceSolver
from svl_solver.solvers.exact import BranchAndBoundSolver
from svl_solver.solvers.exact_set import hungarian, node_structure
from svl_solver.solvers.registry import get_solver


def test_incremental_energy_matches_the_qubo_after_any_moves():
    rng = np.random.default_rng(0)
    p = build_qubo(generate(seed=2, n_nodes=12, n_agents=4))
    m = AssignmentModel(p)
    m.reset(np.array([0, 1, 2, 3]))
    for _ in range(300):
        if rng.random() < .5:
            m.move(int(rng.integers(4)), int(rng.choice(np.flatnonzero(~m.occupied))))
        else:
            a, b = rng.choice(4, 2, replace=False)
            m.swap(int(a), int(b))
        assert m.energy == pytest.approx(p.energy_with_offset(assignment_to_x(list(m.sigma), 4, 12)), abs=1e-7)


@pytest.mark.parametrize("variant", ["ls", "tabu", "sa1h", "pbil", "exact_set"])
def test_feasible_and_never_worse_than_greedy(variant):
    for seed in range(5):
        p = build_qubo(generate(seed=seed, n_nodes=10, n_agents=3))
        s = get_solver(variant).solve(p, budget_ms=None if variant in ("ls", "exact_set") else 20, seed=seed)
        assert s.feasible
        assert s.energy <= get_solver("greedy").solve(p).energy + 1e-9


def test_the_set_reduction_is_the_optimum():
    for seed in range(8):
        p = build_qubo(generate(seed=seed, n_nodes=9, n_agents=3))
        assert get_solver("exact_set").solve(p).energy == pytest.approx(BruteForceSolver().solve(p).energy, abs=1e-9)
    p = build_qubo(generate(seed=1, n_nodes=24, n_agents=6))
    bb = BranchAndBoundSolver(time_limit_s=120).solve(p)
    assert bb.meta["proven_optimal"]
    assert get_solver("exact_set").solve(p).energy == pytest.approx(bb.energy, abs=1e-9)


def test_the_set_reduction_refuses_agent_specific_pair_terms():
    p = build_qubo(generate(seed=1, n_nodes=6, n_agents=3))
    assert node_structure(p) is not None
    Q = p.Q.copy()
    Q[0, 6 + 1] += 0.5                       # agents 0 and 1 now interact differently from 0 and 2
    q = QuboProblem(Q=Q, offset=p.offset, n_agents=3, n_nodes=6, lam=p.lam)
    assert node_structure(q) is None
    with pytest.raises(ValueError):
        get_solver("exact_set").solve(q)


def test_hungarian_matches_enumeration():
    import itertools
    rng = np.random.default_rng(3)
    for k in range(1, 7):
        c = rng.normal(size=(k, k))
        best = min(sum(c[i, perm[i]] for i in range(k)) for perm in itertools.permutations(range(k)))
        col = hungarian(c)
        assert sorted(col) == list(range(k))
        assert sum(c[i, col[i]] for i in range(k)) == pytest.approx(best)


def test_square_instances_move_by_swaps_only():
    # QAPLIB-like: as many agents as nodes, so no free node to move to.
    rng = np.random.default_rng(5)
    k = 5
    Q = np.triu(rng.uniform(0, 1, size=(k * k, k * k)))
    lam = float(Q.sum())
    for a in range(k):
        for n in range(k):
            Q[a * k + n, a * k + n] -= lam
            for m in range(n + 1, k):
                Q[a * k + n, a * k + m] += 2 * lam
        for b in range(a + 1, k):
            for n in range(k):
                Q[a * k + n, b * k + n] += lam
    p = QuboProblem(Q=Q, offset=lam * k, n_agents=k, n_nodes=k, lam=lam)
    opt = BruteForceSolver().solve(p).energy
    for variant in ("ls", "tabu", "sa1h"):
        s = get_solver(variant).solve(p, budget_ms=30, seed=1)
        assert s.feasible and sorted(s.assignment) == list(range(k))
    assert get_solver("tabu").solve(p, budget_ms=30, seed=1).energy == pytest.approx(opt, abs=1e-9)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_qaoa_study_reads_probabilities_that_add_up():
    p = build_qubo(generate(seed=3, n_nodes=6, n_agents=2))
    for r in study_x(p, layers=1) + study_xy(p, layers=1):
        assert 0 <= r["p_opt"] <= r["p_feasible"] <= 1 + 1e-9
        assert r["uniform_feasible_p_opt"] == pytest.approx(1 / 30) or r["uniform_feasible_p_opt"] > 1 / 30
    xy = study_xy(p, layers=1)[0]
    assert xy["p_feasible"] > .9, "the XY mixer keeps one node per agent; only clashes are left"
