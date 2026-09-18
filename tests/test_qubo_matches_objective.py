"""Objective O1 correctness: on EVERY feasible assignment of small instances,
QUBO energy (+offset) equals −objective. This is the proof that the Q matrix
encodes the intended function and not something subtly different."""

import itertools

import numpy as np
import pytest

from svl_solver.instances.generator import generate
from svl_solver.model.feasibility import assignment_to_x
from svl_solver.model.instance import Weights
from svl_solver.model.objective import objective
from svl_solver.qubo.formulation import build_qubo


@pytest.mark.parametrize("seed,n,a", [(1, 5, 2), (2, 6, 2), (3, 6, 3), (4, 7, 3), (5, 8, 2)])
@pytest.mark.parametrize("w", [Weights(), Weights(coverage=2.0, redundancy=1.0, flanking=0.3, cover=0.0, distance=0.25)])
def test_energy_equals_negative_objective_on_all_feasible(seed, n, a, w):
    inst = generate(seed=seed, n_nodes=n, n_agents=a, chokepoint_density=0.3)
    problem = build_qubo(inst, w)
    assert np.allclose(problem.Q, np.triu(problem.Q)), "Q must be upper-triangular"

    for perm in itertools.permutations(range(n), a):
        x = assignment_to_x(perm, a, n)
        e = problem.energy_with_offset(x)
        f = objective(inst, perm, w)
        assert abs(e + f) < 1e-9, f"assignment {perm}: E+offset={e}, -F={-f}"


def test_flat_roundtrip_preserves_energy():
    inst = generate(seed=11, n_nodes=9, n_agents=3)
    p = build_qubo(inst)
    from svl_solver.qubo.formulation import QuboProblem
    q2 = QuboProblem.from_flat(p.q_flat(), p.n_agents, p.n_nodes, offset=p.offset, lam=p.lam)
    x = assignment_to_x([0, 4, 7], 3, 9)
    assert abs(q2.energy_with_offset(x) - p.energy_with_offset(x)) < 1e-12


def test_symmetric_input_is_folded_to_upper():
    inst = generate(seed=12, n_nodes=6, n_agents=2)
    p = build_qubo(inst)
    from svl_solver.qubo.formulation import QuboProblem
    sym = (p.Q + p.Q.T) / 2.0
    q2 = QuboProblem.from_flat(sym.ravel().tolist(), 2, 6, offset=p.offset)
    x = assignment_to_x([1, 5], 2, 6)
    assert abs(q2.energy_with_offset(x) - p.energy_with_offset(x)) < 1e-9
