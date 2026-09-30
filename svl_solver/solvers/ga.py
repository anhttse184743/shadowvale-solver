"""Classical baseline 2: genetic algorithm over assignments.

A genome is the assignment itself — one node per agent, all distinct — so every
individual is feasible and the search never wastes time on penalty terms:

* population seeded with the greedy solution plus random assignments;
* tournament selection (k = 3), uniform crossover per agent, then a repair that
  moves any agent sharing a node to a random free node;
* mutation: with probability ``mutation`` per agent, move it to a random free
  node, or swap two agents' nodes;
* the ``elite`` best survive unchanged.

With ``budget_ms`` it stops at ``time_fraction`` of the budget; with
``budget_ms=None`` it runs exactly ``generations`` generations (reproducible).
Energies are evaluated on the QUBO like every other solver, so the comparison
with the quantum-inspired variants is on the same objective.
"""

from __future__ import annotations

import time

import numpy as np

from ..qubo.formulation import QuboProblem
from .base import Solver
from .greedy import GreedySolver
from .qiea import _energy


class GeneticSolver(Solver):
    variant_id = "ga"

    def __init__(self, population: int = 24, generations: int = 60, mutation: float = 0.15,
                 elite: int = 2, tournament: int = 3, time_fraction: float = 0.8) -> None:
        self.population = max(4, int(population))
        self.generations = max(1, int(generations))
        self.mutation = float(mutation)
        self.elite = max(1, min(int(elite), self.population - 1))
        self.tournament = max(2, int(tournament))
        self.time_fraction = float(time_fraction)

    def _solve(self, problem: QuboProblem, budget_ms, rng):
        t0 = time.perf_counter()
        deadline = None if budget_ms is None else t0 + budget_ms * self.time_fraction / 1000.0
        A, N = problem.n_agents, problem.n_nodes
        base = np.arange(A) * N

        def energy(genome: np.ndarray) -> float:
            return _energy(problem.Q, base + genome) + problem.offset

        greedy_x = GreedySolver()._solve(problem, None, rng).reshape(A, N)
        pop = [np.argmax(greedy_x, axis=1)]
        while len(pop) < self.population:
            pop.append(rng.choice(N, size=A, replace=False))
        fitness = np.array([energy(g) for g in pop])
        evaluated = len(pop)
        generations_done = 0
        stopped = False

        for _ in range(self.generations):
            order = np.argsort(fitness, kind="stable")
            children = [pop[i].copy() for i in order[: self.elite]]
            while len(children) < self.population:
                if deadline is not None and time.perf_counter() >= deadline:
                    stopped = True
                    break
                p1, p2 = self._pick(fitness, rng), self._pick(fitness, rng)
                mask = rng.random(A) < 0.5
                child = np.where(mask, pop[p1], pop[p2])
                self._mutate(child, N, rng)
                self._repair(child, N, rng)
                children.append(child)
            if stopped:
                break
            pop = children
            fitness = np.array([energy(g) for g in pop])
            evaluated += len(pop) - self.elite
            generations_done += 1

        best = pop[int(np.argmin(fitness))]
        x = np.zeros(A * N, dtype=np.int8)
        x[base + best] = 1
        return x, {"samples_evaluated": evaluated, "raw_feasible_samples": evaluated,
                   "generations": generations_done, "stopped_on_budget": stopped}

    def _pick(self, fitness: np.ndarray, rng: np.random.Generator) -> int:
        contenders = rng.integers(len(fitness), size=self.tournament)
        return int(contenders[np.argmin(fitness[contenders])])

    def _mutate(self, genome: np.ndarray, n_nodes: int, rng: np.random.Generator) -> None:
        for a in range(len(genome)):
            if rng.random() >= self.mutation:
                continue
            if len(genome) > 1 and rng.random() < 0.5:
                b = int(rng.integers(len(genome)))
                genome[a], genome[b] = genome[b], genome[a]
            else:
                genome[a] = int(rng.integers(n_nodes))

    @staticmethod
    def _repair(genome: np.ndarray, n_nodes: int, rng: np.random.Generator) -> None:
        seen: set[int] = set()
        clashes = []
        for a, node in enumerate(genome):
            if int(node) in seen:
                clashes.append(a)
            else:
                seen.add(int(node))
        if not clashes:
            return
        free = [n for n in range(n_nodes) if n not in seen]
        rng.shuffle(free)
        for a, node in zip(clashes, free):
            genome[a] = node
