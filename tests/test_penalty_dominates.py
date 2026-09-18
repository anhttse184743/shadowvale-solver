"""λ from qubo/penalties.py must make EVERY infeasible bit-vector strictly
worse than the best feasible one. Checked exhaustively (2^n states)."""

import numpy as np
import pytest

from svl_solver.instances.generator import generate
from svl_solver.model.feasibility import is_feasible_x
from svl_solver.model.instance import Weights
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.brute_force import BruteForceSolver, all_bitstring_energies


@pytest.mark.parametrize("seed,n,a", [(1, 4, 2), (2, 5, 2), (3, 4, 3), (4, 6, 2), (5, 5, 3)])
@pytest.mark.parametrize("w", [Weights(), Weights(coverage=3.0, redundancy=2.0, flanking=2.0, cover=1.0, distance=0.5)])
def test_every_infeasible_state_is_worse_than_best_feasible(seed, n, a, w):
    inst = generate(seed=seed, n_nodes=n, n_agents=a, chokepoint_density=0.3)
    problem = build_qubo(inst, w)
    X, E = all_bitstring_energies(problem)
    feas = np.array([is_feasible_x(x, a, n) for x in X])
    assert feas.any()
    best_feasible = E[feas].min()
    worst_offender = E[~feas].min()
    assert worst_offender > best_feasible + 1e-9, (
        f"infeasible state with E={worst_offender} beats best feasible E={best_feasible}")


@pytest.mark.parametrize("seed,n,a", [(1, 5, 2), (3, 4, 3), (6, 6, 2)])
def test_bitstring_and_assignment_bruteforce_agree(seed, n, a):
    inst = generate(seed=seed, n_nodes=n, n_agents=a)
    problem = build_qubo(inst)
    e_bits = BruteForceSolver(space="bitstrings").solve(problem).energy
    e_assign = BruteForceSolver(space="assignments").solve(problem).energy
    assert abs(e_bits - e_assign) < 1e-9


def test_lambda_is_positive_and_grows_with_agents():
    from svl_solver.qubo.penalties import compute_lambda
    i2 = generate(seed=1, n_nodes=10, n_agents=2)
    i5 = generate(seed=1, n_nodes=10, n_agents=5)
    assert compute_lambda(i2) > 0
    assert compute_lambda(i5) > compute_lambda(i2)
