"""Three tiny instances whose optimum is worked out by hand (comments), then
confirmed both by the brute-force solver on the QUBO and by direct enumeration
of the reference objective. This is deliverable D10 of the week-2 plan."""

import itertools

import pytest

from svl_solver.model.instance import Weights
from svl_solver.model.objective import objective
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.brute_force import BruteForceSolver
from svl_solver.solvers.greedy import GreedySolver


def _enumerate_best(inst, w):
    best, best_f = None, float("-inf")
    for perm in itertools.permutations(range(inst.n_nodes), inst.n_agents):
        f = objective(inst, perm, w)
        if f > best_f:
            best, best_f = list(perm), f
    return best, best_f


# ---------------------------------------------------------------- hand 1
def test_hand1_coverage_vs_distance(hand_instance_factory):
    # Path 0-1-2-3, player at 0, escape route at 3, agent starts at 1, vision 1 hop.
    #   cov: node0=0 (sees 0,1), node1=0 (sees 0,1,2), node2=1 (sees 3), node3=1 (itself)
    #   dist from node1: n0=1, n1=0, n2=1, n3=2
    #   F = 1*cov − 0.1*dist :  n0=−0.1  n1=0  n2=0.9  n3=0.8   → optimum node 2, F = 0.9
    inst = hand_instance_factory(edges=[(0, 1), (1, 2), (2, 3)], n_nodes=4, player=0,
                                 agents=[1], escape=[3], cover=[], vision_hops=1)
    w = Weights(coverage=1.0, redundancy=0.0, flanking=0.0, cover=0.0, distance=0.1)
    assert inst.cov.tolist() == [0, 0, 1, 1]

    sol = BruteForceSolver().solve(build_qubo(inst, w))
    assert sol.feasible and sol.assignment == [2]
    assert sol.objective == pytest.approx(0.9)
    assert _enumerate_best(inst, w) == ([2], pytest.approx(0.9))


# ---------------------------------------------------------------- hand 2
def test_hand2_flanking(hand_instance_factory):
    # Star: player 0 at origin; 1=(1,0) 2=(−1,0) 3=(0,1) 4=(0,−1). No escape/cover.
    #   flank(1,2)=1, flank(3,4)=1, flank(1,3)=flank(1,4)=flank(2,3)=flank(2,4)=0.5, flank(0,·)=0
    #   agents start at 1 and 2; dist weight 0.1 breaks the tie between {1,2} and {3,4}:
    #   a0→1,a1→2: F = 1 − 0.1*(0+0) = 1.0     ← optimum
    #   a0→2,a1→1: F = 1 − 0.1*(2+2) = 0.6
    #   a0→3,a1→4: F = 1 − 0.1*(2+2) = 0.6
    #   a0→1,a1→3: F = 0.5 − 0.1*(0+2) = 0.3
    pos = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    inst = hand_instance_factory(edges=[(0, 1), (0, 2), (0, 3), (0, 4)], n_nodes=5, player=0,
                                 agents=[1, 2], escape=[], cover=[], pos=pos, vision_hops=1)
    w = Weights(coverage=0.0, redundancy=0.0, flanking=1.0, cover=0.0, distance=0.1)
    assert inst.flank[1, 2] == pytest.approx(1.0)
    assert inst.flank[1, 3] == pytest.approx(0.5)
    assert inst.flank[0, 1] == pytest.approx(0.0)

    sol = BruteForceSolver().solve(build_qubo(inst, w))
    assert sol.feasible and sol.assignment == [1, 2]
    assert sol.objective == pytest.approx(1.0)
    assert _enumerate_best(inst, w) == ([1, 2], pytest.approx(1.0))


# ---------------------------------------------------------------- hand 3
def test_hand3_coverage_with_redundancy(hand_instance_factory):
    # player 0; escape routes 4 and 5.
    # edges: 0-1 0-2 0-3 1-4 1-5 2-4 3-5   (vision 1 hop)
    #   cov: n1=2 (sees 4,5)  n2=1 (4)  n3=1 (5)  n4=1 (itself)  n5=1 (itself)  n0=0
    #   overlap(1,2)=1 overlap(1,3)=1 overlap(1,4)=1 overlap(1,5)=1 overlap(2,4)=1 overlap(3,5)=1, others 0
    # weights: cov 1, red 0.75, dist 0.1; agents start at 2 (a0) and 1 (a1).
    #   a0→2,a1→1: F = (1+2) − 0.75 − 0.1*(0+0) = 2.25   ← optimum (unique)
    #   a0→4,a1→1: F = (1+2) − 0.75 − 0.1*(1+0) = 2.15
    #   a0→1,a1→2: F = 2.25 − 0.1*(2+2) = 1.85
    #   a0→3,a1→1: F = 2.25 − 0.1*(2+0) = 2.05
    #   a0→2,a1→3: F = (1+1) − 0   − 0.1*(0+2) = 1.8
    #   a0→2,a1→5: F = (1+1) − 0   − 0.1*(0+1) = 1.9
    inst = hand_instance_factory(edges=[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (3, 5)],
                                 n_nodes=6, player=0, agents=[2, 1], escape=[4, 5], cover=[],
                                 vision_hops=1)
    w = Weights(coverage=1.0, redundancy=0.75, flanking=0.0, cover=0.0, distance=0.1)
    assert inst.cov.tolist() == [0, 2, 1, 1, 1, 1]
    assert inst.overlap[1, 2] == 1 and inst.overlap[2, 3] == 0

    sol = BruteForceSolver().solve(build_qubo(inst, w))
    assert sol.feasible and sol.assignment == [2, 1]
    assert sol.objective == pytest.approx(2.25)
    assert _enumerate_best(inst, w) == ([2, 1], pytest.approx(2.25))


def test_greedy_is_feasible_and_not_better_than_optimum(hand_instance_factory):
    inst = hand_instance_factory(edges=[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (3, 5)],
                                 n_nodes=6, player=0, agents=[2, 1], escape=[4, 5], cover=[],
                                 vision_hops=1)
    w = Weights(coverage=1.0, redundancy=0.75, flanking=0.0, cover=0.0, distance=0.1)
    p = build_qubo(inst, w)
    g = GreedySolver().solve(p)
    b = BruteForceSolver().solve(p)
    assert g.feasible
    assert g.objective <= b.objective + 1e-9
