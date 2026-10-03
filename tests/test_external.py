"""External benchmarks: QAPLIB → assignment QUBO, Moving AI maps → coordination instances.

Small hand-made files stand in for the downloaded datasets (which are not in the repository).
"""

import itertools
import math

import numpy as np
import pytest

from svl_solver.external import movingai, qaplib
from svl_solver.instances.game_records import from_record
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.base import assignment_to_x
from svl_solver.solvers.brute_force import all_bitstring_energies
from svl_solver.solvers.registry import get_solver


def _qap(tmp_path, name="tiny4", inverse=False):
    a = np.array([[0, 3, 0, 2], [3, 0, 0, 1], [0, 0, 0, 4], [2, 1, 4, 0]])
    b = np.array([[0, 22, 53, 0], [22, 0, 40, 0], [53, 40, 0, 55], [0, 0, 55, 0]])
    perms = list(itertools.permutations(range(4)))
    costs = [float((a * b[np.ix_(p, p)]).sum()) for p in perms]
    best = min(costs)
    perm = list(perms[int(np.argmin(costs))])
    written = list(np.argsort(perm)) if inverse else perm
    (tmp_path / f"{name}.dat").write_text("4\n\n" + "\n".join(" ".join(map(str, r)) for r in a) + "\n\n"
                                          + "\n".join(" ".join(map(str, r)) for r in b) + "\n")
    (tmp_path / f"{name}.sln").write_text(f" 4 {int(best)}\n " + " ".join(str(int(v) + 1) for v in written) + "\n")
    return best


def test_qaplib_files_load_and_the_published_permutation_reproduces_its_cost(tmp_path):
    best = _qap(tmp_path)
    q = qaplib.load(tmp_path)[0]
    assert q.n == 4 and q.best == best
    assert q.cost(q.best_perm) == pytest.approx(best)


def test_an_inverse_permutation_in_the_solution_file_is_recognised(tmp_path):
    best = _qap(tmp_path, inverse=True)
    q = qaplib.load(tmp_path)[0]
    assert q.best_perm is not None and q.cost(q.best_perm) == pytest.approx(best)


def test_qap_qubo_energy_is_the_qap_cost_on_every_permutation_and_infeasible_never_wins(tmp_path):
    best = _qap(tmp_path)
    q = qaplib.load(tmp_path)[0]
    p = qaplib.to_qubo(q)
    for perm in itertools.permutations(range(4)):
        assert p.energy_with_offset(assignment_to_x(list(perm), 4, 4)) == pytest.approx(q.cost(list(perm)))
    X, E = all_bitstring_energies(p)
    feasible = (X.reshape(-1, 4, 4).sum(axis=2) == 1).all(axis=1) & (X.reshape(-1, 4, 4).sum(axis=1) == 1).all(axis=1)
    assert E[~feasible].min() > best, "the penalty dominates: the QUBO optimum is the QAP optimum"
    assert get_solver("brute").solve(p).energy == pytest.approx(best)


MAP = """type octile
height 12
width 16
map
@@@@@@@@@@@@@@@@
@..............@
@..............@
@....TTTT......@
@....TTTT......@
@..............@
@..............@
@.........WW...@
@.........WW...@
@..............@
@..............@
@@@@@@@@@@@@@@@@
"""


def test_a_map_becomes_a_valid_coordination_instance(tmp_path):
    path = tmp_path / "toy.map"
    path.write_text(MAP)
    grid = movingai.read_map(path)
    assert grid.shape == (12, 16) and not grid[3, 5] and not grid[7, 10] and grid[1, 1]
    g = movingai.TacticalGraph(grid, spacing=2)
    rec = movingai.instance_record("toy", g, n_agents=3, seed=4, k_nodes=12, vision=8, agent_hops=(1, 4))
    assert rec is not None
    n = rec["node_count"]
    assert rec["player_local"] == 0 and 0 not in rec["escape"], "never an escape at the player's own node"
    assert len(rec["agent_local"]) == 3 and len(rec["dist"]) == 3 * n
    assert "1" in rec["is_cover"], "nodes beside the trees and walls are cover"
    vis = np.array([c == "1" for c in rec["visible"]]).reshape(n, n)
    assert (vis == vis.T).all() and vis.diagonal().all()
    inst = from_record(rec)
    p = build_qubo(inst, inst.weights)
    assert get_solver("greedy").solve(p).feasible
    assert movingai.instance_record("toy", g, n_agents=3, seed=4, k_nodes=12, vision=8, agent_hops=(1, 4)) == rec, "deterministic"


def test_line_of_sight_is_blocked_by_obstacles(tmp_path):
    path = tmp_path / "toy.map"
    path.write_text(MAP)
    grid = movingai.read_map(path)
    assert movingai._clear(grid, (1, 1), (1, 14))
    assert not movingai._clear(grid, (3, 1), (4, 14)), "through the trees"
    assert movingai._cover_facing(grid, (2, 6)) is not None
    assert math.isclose(movingai._cover_facing(grid, (2, 6)) % 360, 0.0, abs_tol=46), "faces the trees below it (+z)"
