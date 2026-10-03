"""Benchmarks on standard external datasets (supplementary to the procedurally generated suite).

* QAPLIB (:mod:`svl_solver.external.qaplib`): every solver on the assignment QUBO of each QAP
  instance, gap to the published optimal / best-known cost.
* Moving AI (:mod:`svl_solver.external.movingai`): coordination instances built on maps from
  commercial games; they are game records, so they run through ``svl-bench game``.

Rows use the bench CSV columns, so ``summary`` / ``compare`` / ``plot`` work on them, and resume
like every bench run (rows written as they finish, a half-written last row dropped).
"""

from __future__ import annotations

import csv
import datetime as dt
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..external.qaplib import QapInstance, to_qubo
from ..solvers.registry import get_solver
from ..version import code_version
from .metrics import CSV_COLUMNS
from .runner import BenchRow, _drop_partial_tail, _existing_keys


def run_qaplib(instances: list[QapInstance], solvers: list[str], repeats: int, budget_ms: int, out: str | Path,
               progress: Callable[[str], None] | None = print) -> list[BenchRow]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_partial_tail(out)
    done = _existing_keys(out)
    header = not out.exists() or os.path.getsize(out) == 0
    rows: list[BenchRow] = []
    version = code_version()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if header:
            w.writeheader()
        for q in instances:
            problem = to_qubo(q)
            iid = f"qaplib-{q.name}"
            for s in solvers:
                for r in range(repeats):
                    if (iid, s, r) in done:
                        continue
                    sol = get_solver(s).solve(problem, budget_ms=budget_ms, seed=1000 * (r + 1) + q.n)
                    # A repaired (infeasible) answer is scored on its repaired assignment, like the game uses it.
                    cost = q.cost(sol.assignment)
                    row = BenchRow(
                        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), suite="qaplib",
                        instance_id=iid, seed=q.n, n_nodes=q.n, n_agents=q.n, density=float("nan"), n_vars=q.n * q.n,
                        solver=s, repeat=r, budget_ms=budget_ms, feasible_raw=sol.feasible, energy=cost,
                        objective=-cost, latency_ms=round(sol.latency_ms, 3), hit_budget=sol.hit_budget,
                        opt_energy=q.best, opt_objective=-q.best if q.best is not None else None,
                        approx_ratio=(q.best / cost) if q.best and cost > 0 else None,
                        gap=((cost - q.best) / abs(q.best)) if q.best else None,
                        note="best-known (QAPLIB)", code_version=version,
                    )
                    w.writerow(asdict(row))
                    f.flush()
                    rows.append(row)
            if progress:
                progress(f"{q.name} (n={q.n}) done")
    return rows
