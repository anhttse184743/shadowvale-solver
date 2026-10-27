"""Reproducible benchmark runner.

    run_suite(spec, solvers=["greedy", "sqa"], repeats=30, out="results/x.csv")

* one CSV row per (instance, solver, repeat), appended as soon as it finishes
  so an overnight run that dies keeps everything done so far;
* ``resume=True`` skips rows already present in the CSV;
* the exact optimum is computed once per instance (cached): by brute force whenever
  N!/(N−A)! ≤ spec.brute_force_max_candidates, else by branch and bound (solvers/exact.py)
  if it proves optimality within EXACT_TIME_LIMIT_S — giving approx_ratio and gap;
* repeat r uses seed = instance.seed * 1000 + r for stochastic solvers, so a
  row can be replayed exactly from its CSV line.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

from ..instances.suite import SuiteSpec, iter_instances
from ..model.instance import Instance
from ..model.objective import objective
from ..qubo.formulation import build_qubo, QuboProblem
from ..solvers.base import Solution
from ..solvers.brute_force import BruteForceSolver
from ..solvers.exact import BranchAndBoundSolver
from ..solvers.exact_set import SetAssignmentExactSolver
from ..solvers.registry import get_solver, REGISTRY
from ..version import code_version
from .metrics import CSV_COLUMNS, approx_ratio, gap


@dataclass
class BenchRow:
    timestamp: str
    suite: str
    instance_id: str
    seed: int
    n_nodes: int
    n_agents: int
    density: float
    n_vars: int
    solver: str
    repeat: int
    budget_ms: int
    feasible_raw: bool
    energy: float
    objective: float
    latency_ms: float
    hit_budget: bool
    opt_energy: float | None
    opt_objective: float | None
    approx_ratio: float | None
    gap: float | None
    note: str = ""
    code_version: str = ""
    evals: int | None = None
    raw_feasible_samples: int | None = None
    opt_method: str = ""


def _density_of(inst: Instance) -> float:
    # instance_id = svl-n{n}-a{a}-c{density}-s{seed}
    try:
        return float(inst.instance_id.split("-c")[1].split("-s")[0])
    except Exception:
        return math.nan


def _drop_partial_tail(path: Path) -> bool:
    """A run killed while writing a row leaves half a line at the end; cut it off so the resumed
    run neither trips over it nor appends its next row onto it. True if something was cut."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        f.seek(0)
        data = f.read()
        f.truncate(data.rfind(b"\n") + 1)
        return True


def _existing_keys(path: Path) -> set[tuple[str, str, int]]:
    """Rows already done (the resume checkpoint); malformed rows do not count and are redone."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                keys.add((r["instance_id"], r["solver"], int(r["repeat"])))
            except (KeyError, TypeError, ValueError):
                continue
    return keys


#: Seconds the branch-and-bound may spend proving the optimum of one instance too big for brute force.
EXACT_TIME_LIMIT_S = 60.0
#: Largest number of node sets the set reduction (solvers/exact_set.py) enumerates for a reference.
EXACT_SET_MAX_SETS = 5_000_000


def _optimum(inst: Instance, problem: QuboProblem, max_candidates: int,
             cache: dict[str, tuple[float | None, float | None, str]]) -> tuple[float | None, float | None, str]:
    """Proven optimum and how it was obtained: brute force when the space is small, else the set
    reduction when Q has the node-only structure (exp-v2), else branch and bound; None when not
    proven in time."""
    if inst.instance_id not in cache:
        sol, method = None, ""
        if math.perm(problem.n_nodes, problem.n_agents) <= max_candidates:
            sol, method = BruteForceSolver(max_candidates=max_candidates).solve(problem), "brute"
        else:
            if EXACT_SET_MAX_SETS > 0:
                try:
                    sol, method = SetAssignmentExactSolver(max_sets=EXACT_SET_MAX_SETS).solve(problem), "exact_set"
                except ValueError:
                    sol = None
            if sol is None and EXACT_TIME_LIMIT_S > 0:
                sol, method = BranchAndBoundSolver(time_limit_s=EXACT_TIME_LIMIT_S).solve(problem), "exact"
                if not sol.meta.get("proven_optimal"):
                    sol, method = None, ""
        cache[inst.instance_id] = ((sol.energy, objective(inst, sol.assignment, problem.weights), method)
                                   if sol is not None else (None, None, ""))
    return cache[inst.instance_id]


def run_suite(spec: SuiteSpec, solvers: Iterable[str], repeats: int = 1,
              out: str | Path = "experiments/results/bench.csv", budget_ms: int | None = None,
              resume: bool = True, progress: Callable[[str], None] | None = print,
              instances: Iterable[Instance] | None = None) -> list[BenchRow]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    solvers = [s.strip() for s in solvers if s.strip()]
    for s in solvers:
        if s not in REGISTRY:
            raise KeyError(f"unknown solver '{s}'")
        if not REGISTRY[s].available:
            raise NotImplementedError(f"solver '{s}' is not implemented yet ({REGISTRY[s].due})")
    budget = int(budget_ms if budget_ms is not None else spec.budget_ms)
    if resume and _drop_partial_tail(out) and progress:
        progress(f"{out}: dropped a half-written last row from an interrupted run")
    done = _existing_keys(out) if resume else set()
    write_header = not out.exists() or os.path.getsize(out) == 0

    rows: list[BenchRow] = []
    version = code_version()
    opt_cache: dict[str, tuple[float, float]] = {}
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        for inst in (instances if instances is not None else iter_instances(spec)):
            # Recorded game instances carry the weights the game used.
            weights = getattr(inst, "weights", None) or spec.weights
            problem = build_qubo(inst, weights)
            dens = _density_of(inst)
            opt_e, opt_f, opt_method = _optimum(inst, problem, spec.brute_force_max_candidates, opt_cache)
            for sname in solvers:
                solver = get_solver(sname)
                for r in range(repeats):
                    if (inst.instance_id, sname, r) in done:
                        continue
                    note = ""
                    try:
                        sol: Solution = solver.solve(problem, budget_ms=budget, seed=inst.seed * 1000 + r)
                    except ValueError as e:              # e.g. brute force too large
                        note = f"skipped: {e}"
                        if progress:
                            progress(f"[{inst.instance_id}] {sname} r{r}: {note}")
                        continue
                    f_sol = objective(inst, sol.assignment, problem.weights)
                    row = BenchRow(
                        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        suite=spec.name, instance_id=inst.instance_id, seed=inst.seed,
                        n_nodes=inst.n_nodes, n_agents=inst.n_agents, density=dens,
                        n_vars=problem.n_vars, solver=sname, repeat=r, budget_ms=budget,
                        feasible_raw=sol.feasible, energy=sol.energy, objective=f_sol,
                        latency_ms=round(sol.latency_ms, 3), hit_budget=sol.hit_budget,
                        opt_energy=opt_e, opt_objective=opt_f,
                        approx_ratio=approx_ratio(f_sol, opt_f), gap=gap(sol.energy, opt_e), note=note,
                        code_version=version,
                        evals=sol.meta.get("samples_evaluated"),
                        raw_feasible_samples=sol.meta.get("raw_feasible_samples"),
                        opt_method=opt_method,
                    )
                    w.writerow(asdict(row))
                    f.flush()
                    rows.append(row)
                    if progress:
                        ar = f"{row.approx_ratio:.3f}" if row.approx_ratio is not None else "n/a"
                        progress(f"[{inst.instance_id}] {sname:6s} r{r}: F={f_sol:8.3f} "
                                 f"ratio={ar} {sol.latency_ms:7.2f} ms")
    return rows
