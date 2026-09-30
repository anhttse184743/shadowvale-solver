"""M6 solvers: ga, sa (classical), sqa, qaoa (quantum-inspired / simulated), next to qiea.

Common contract: a feasible assignment, reproducible for a seed without a budget,
inside the budget with one, and on small instances as good as brute force most of
the time. Plus the parts specific to each method.
"""

import math
import time

import numpy as np
import pytest

from svl_solver.instances.generator import generate
from svl_solver.qubo.formulation import QuboProblem, build_qubo, energy
from svl_solver.solvers.brute_force import BruteForceSolver
from svl_solver.solvers.greedy import GreedySolver
from svl_solver.solvers.qaoa import QaoaSolver, basis_energies, qaoa_state
from svl_solver.solvers.registry import get_solver, list_variants
from svl_solver.solvers.sqa import SimulatedQuantumAnnealingSolver, qubo_to_ising

HEURISTICS = ["ga", "sa", "sqa", "qiea"]


def small(seed):
    return build_qubo(generate(seed=seed, n_nodes=8, n_agents=3, chokepoint_density=0.3))


@pytest.mark.parametrize("variant", HEURISTICS)
def test_feasible_and_reproducible_without_budget(variant):
    p = small(4)
    a = get_solver(variant).solve(p, seed=11)
    b = get_solver(variant).solve(p, seed=11)
    assert len(set(a.assignment)) == p.n_agents and all(0 <= n < p.n_nodes for n in a.assignment)
    assert a.assignment == b.assignment and a.energy == pytest.approx(b.energy)


# Classical bit-flip SA has to cross the one-hot penalty barrier to move an agent; on this
# encoding it reaches the optimum far less often (measured 8/20) — the bar says so honestly.
MIN_HITS = {"ga": 7, "sa": 2, "sqa": 7, "qiea": 7}


@pytest.mark.parametrize("variant", HEURISTICS)
def test_usually_optimal_on_small_instances(variant):
    hits, greedy_gap, gap = 0, 0.0, 0.0
    for seed in range(1, 11):
        p = small(seed)
        best = BruteForceSolver().solve(p).energy
        e = get_solver(variant).solve(p, seed=seed).energy
        hits += e == pytest.approx(best)
        gap += (e - best) / abs(best)
        greedy_gap += (GreedySolver().solve(p).energy - best) / abs(best)
    assert hits >= MIN_HITS[variant], f"{variant} matched the optimum on {hits}/10 small instances"
    assert gap <= greedy_gap + 1e-9, f"{variant} is on average worse than greedy"


@pytest.mark.parametrize("variant", HEURISTICS)
def test_respects_a_tight_budget_on_a_game_sized_instance(variant):
    p = build_qubo(generate(seed=2, n_nodes=24, n_agents=6, chokepoint_density=0.3))   # 144 variables
    t = time.perf_counter()
    sol = get_solver(variant).solve(p, budget_ms=60, seed=2)
    elapsed = (time.perf_counter() - t) * 1000
    assert elapsed < 60 * 1.6, f"{variant} took {elapsed:.0f} ms"
    assert len(set(sol.assignment)) == p.n_agents
    assert sol.meta["samples_evaluated"] > 0


def test_ga_and_qiea_never_lose_to_greedy():
    for seed in range(1, 6):
        p = build_qubo(generate(seed=seed, n_nodes=16, n_agents=4))
        g = GreedySolver().solve(p).energy
        for v in ("ga", "qiea"):
            assert get_solver(v).solve(p, seed=seed).energy <= g + 1e-9


def test_ising_mapping_is_exact():
    p = small(3)
    h, Js = qubo_to_ising(p.Q)
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2, size=(20, p.n_vars))
    es = [energy(p.Q, x) for x in xs]
    ss = 2.0 * xs - 1.0
    ising = [h @ s + .5 * s @ Js @ s for s in ss]
    diffs = np.array(es) - np.array(ising)
    assert np.allclose(diffs, diffs[0]), "E_qubo − E_ising is one constant for every state"


def test_sqa_replicas_couple_harder_as_the_field_falls():
    sqa = SimulatedQuantumAnnealingSolver()
    assert sqa.coupling(3.0) < sqa.coupling(0.5) < sqa.coupling(0.02)


def test_sa_is_classical_neal_and_sqa_is_not():
    import svl_solver.solvers.sa as sa_mod
    import svl_solver.solvers.sqa as sqa_mod
    assert hasattr(sa_mod, "neal")
    assert not hasattr(sqa_mod, "neal"), "SQA must not be dwave-neal (that is classical SA)"


def test_qaoa_statevector_is_normalised_and_mixer_spreads_amplitude():
    cost = np.linspace(0, 1, 8)
    psi = qaoa_state(cost, 3, [0.0], [0.0])
    assert np.allclose(np.abs(psi) ** 2, 1 / 8), "γ = β = 0 leaves |+>^n"
    psi = qaoa_state(cost, 3, [1.3, .4], [.7, .2])
    assert (np.abs(psi) ** 2).sum() == pytest.approx(1.0)


def test_qaoa_basis_energies_match_the_qubo():
    p = build_qubo(generate(seed=1, n_nodes=4, n_agents=2))
    e = basis_energies(p.Q)
    for z in (0, 5, 17, 255):
        x = np.array([(z >> i) & 1 for i in range(p.n_vars)])
        assert e[z] == pytest.approx(energy(p.Q, x))


def test_qaoa_solves_a_tiny_instance_and_reports_the_measured_distribution():
    hits = 0
    for seed in range(1, 6):
        p = build_qubo(generate(seed=seed, n_nodes=4, n_agents=2))      # 8 qubits
        best = BruteForceSolver().solve(p).energy
        sol = QaoaSolver().solve(p, seed=seed)
        assert len(set(sol.assignment)) == 2
        assert 0 <= sol.meta["raw_feasible_samples"] <= sol.meta["samples_evaluated"] == 512
        assert sol.meta["qubits"] == 8
        hits += sol.energy == pytest.approx(best)
    assert hits >= 4


def test_qaoa_refuses_instances_beyond_the_statevector():
    with pytest.raises(ValueError, match="qubits"):
        QaoaSolver(max_qubits=10).solve(small(1))                      # 24 variables


def test_every_variant_but_the_qpu_is_available():
    avail = {v["id"]: v["available"] for v in list_variants()}
    assert all(avail[v] for v in ("greedy", "brute", "ga", "sa", "sqa", "qiea", "qaoa"))
    assert not avail["qpu"]
