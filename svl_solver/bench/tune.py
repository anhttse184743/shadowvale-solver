"""Hyper-parameter tuning with equal effort for every stochastic solver (RQ2 fairness).

    tune(problems, ["ga", "sa", "sqa", "qiea"], trials=30, budget_ms=120)

Protocol — the same for every solver, so no solver wins the comparison by having been tuned
harder than the others:

* a **tuning set** kept apart from the evaluation set: generated instances with their own seeds
  (900+) and the recorded game instances whose id falls in the tuning split
  (:func:`is_tuning_instance`); ``svl-bench game`` leaves those out unless asked;
* the same number of configurations per solver: the current defaults plus ``trials − 1``
  random draws from that solver's search space (:data:`SPACES`), one run per instance, all
  under the game's budget;
* the score of a configuration is its mean gap to the reference energy of each instance — the
  proven optimum when the exact solver finds it within ``exact_s``, else the best energy any
  run of any solver reached;
* the best ``finalists`` configurations of each solver are re-run ``repeats`` times with fresh
  seeds, and the best mean wins, so a lucky draw cannot.

Reproducible from ``seed``. The report (every trial) goes to CSV, the winners to JSON; the
winners then become the solvers' defaults, with a pointer to the report in their docstrings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..qubo.formulation import QuboProblem
from ..solvers.exact import BranchAndBoundSolver
from ..solvers.greedy import GreedySolver
from ..solvers.registry import REGISTRY
from ..version import code_version

#: name → (kind, low, high) for "log"/"float"/"int"/"int_log", or ("choice", [values]).
SPACES: dict[str, dict[str, tuple]] = {
    "ga": {"population": ("int_log", 8, 96), "mutation": ("float", 0.02, 0.5),
           "elite": ("int", 1, 6), "tournament": ("int", 2, 6)},
    "sa": {"num_sweeps": ("int_log", 30, 3000), "batch": ("choice", [1, 2, 4, 8, 16]),
           "beta_hot": ("log", 0.3, 30.0), "beta_cold": ("log", 10.0, 3000.0)},
    "sqa": {"trotter": ("choice", [2, 4, 8, 16]), "temperature": ("log", 0.005, 0.5),
            "gamma_start": ("log", 0.1, 10.0), "gamma_end": ("log", 0.001, 0.2)},
    "qiea": {"population": ("int_log", 2, 32), "delta": ("log", 0.005, 0.2),
             "margin": ("float", 0.01, 0.3)},
}


def is_tuning_instance(instance_id: str, share: int = 4) -> bool:
    """One recorded instance in ``share`` belongs to the tuning split, decided by its id alone."""
    return int(hashlib.sha1(instance_id.encode("utf-8")).hexdigest(), 16) % share == 0


def sample(space: dict[str, tuple], rng: np.random.Generator) -> dict:
    out = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "choice":
            out[name] = spec[1][int(rng.integers(len(spec[1])))]
        elif kind == "log":
            out[name] = float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
        elif kind == "int_log":
            out[name] = int(round(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))))
        elif kind == "int":
            out[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            out[name] = float(rng.uniform(spec[1], spec[2]))
    return out


@dataclass
class Trial:
    solver: str
    trial: int
    params: dict
    gaps: list[float]
    stage: str = "search"

    @property
    def mean_gap(self) -> float:
        return float(statistics.fmean(self.gaps)) if self.gaps else math.inf


def _gap(e: float, ref: float) -> float:
    return max(0.0, (e - ref) / max(abs(ref), 1e-9))


def tune(problems: list[tuple[str, QuboProblem]], solvers: list[str], trials: int = 30, budget_ms: int = 120,
         finalists: int = 3, repeats: int = 3, seed: int = 2026, exact_s: float = 10.0,
         progress: Callable[[str], None] | None = print) -> tuple[dict[str, dict], list[Trial], dict]:
    """Returns (best params per solver, every trial, reference info)."""
    rng = np.random.default_rng(seed)
    # References: proven optimum where the exact solver gets there, else the best seen anywhere.
    ref: dict[str, float] = {}
    proven: dict[str, bool] = {}
    for pid, p in problems:
        sol = BranchAndBoundSolver(time_limit_s=exact_s).solve(p)
        ref[pid], proven[pid] = sol.energy, bool(sol.meta.get("proven_optimal"))
        ref[pid] = min(ref[pid], GreedySolver().solve(p).energy)
    if progress:
        progress(f"tuning set: {len(problems)} instances, optimum proven on {sum(proven.values())}")

    energies: dict[tuple[str, int, str, int], float] = {}

    def run(solver: str, params: dict, trial: int, rep: int) -> list[float]:
        out = []
        for k, (pid, p) in enumerate(problems):
            s = REGISTRY[solver](**params)
            e = s.solve(p, budget_ms=budget_ms, seed=seed * 7919 + trial * 104729 + rep * 31 + k).energy
            energies[(solver, trial, pid, rep)] = e
            if not proven[pid]:
                ref[pid] = min(ref[pid], e)
            out.append(e)
        return out

    raw: list[tuple[Trial, list[float]]] = []
    for solver in solvers:
        configs = [{}] + [sample(SPACES[solver], rng) for _ in range(max(0, trials - 1))]
        for t, params in enumerate(configs):
            es = run(solver, params, t, 0)
            raw.append((Trial(solver, t, params, []), es))
            if progress and (t % 5 == 0 or t == len(configs) - 1):
                progress(f"  {solver} trial {t + 1}/{len(configs)}")
    # Gaps against the final references (they only improved while trials ran).
    trials_out: list[Trial] = []
    for tr, es in raw:
        tr.gaps = [_gap(e, ref[pid]) for e, (pid, _) in zip(es, problems)]
        trials_out.append(tr)

    best: dict[str, dict] = {}
    for solver in solvers:
        ranked = sorted((t for t in trials_out if t.solver == solver and t.stage == "search"),
                        key=lambda t: t.mean_gap)[:finalists]
        # The defaults always reach the final: a tuned configuration has to beat them on the same
        # repeated runs, not on one lucky pass.
        if all(t.trial != 0 for t in ranked):
            ranked.append(next(t for t in trials_out if t.solver == solver and t.trial == 0 and t.stage == "search"))
        finals = []
        for t in ranked:
            gaps = []
            for rep in range(1, repeats + 1):
                es = run(solver, t.params, t.trial, rep)
                gaps += [_gap(e, ref[pid]) for e, (pid, _) in zip(es, problems)]
            ft = Trial(solver, t.trial, t.params, gaps, stage="final")
            finals.append(ft)
            trials_out.append(ft)
        winner = min(finals, key=lambda t: t.mean_gap)
        default_final = next(t for t in finals if t.trial == 0)
        best[solver] = {"params": winner.params, "mean_gap": winner.mean_gap, "default_mean_gap": default_final.mean_gap,
                        "trial": winner.trial}
        if progress:
            progress(f"{solver}: default gap {100 * default_final.mean_gap:.2f}% -> tuned {100 * winner.mean_gap:.2f}%  {winner.params}")
    info = {"instances": [pid for pid, _ in problems], "proven": proven, "reference": ref,
            "trials": trials, "budget_ms": budget_ms, "finalists": finalists, "repeats": repeats, "seed": seed}
    return best, trials_out, info


def write_report(out_dir: str | Path, best: dict, trials: list[Trial], info: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solver", "stage", "trial", "mean_gap", "params"])
        for t in trials:
            w.writerow([t.solver, t.stage, t.trial, f"{t.mean_gap:.6f}", json.dumps(t.params, sort_keys=True)])
    with open(out / "tuned.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "code_version": code_version(),
                   "protocol": {k: v for k, v in info.items() if k not in ("reference", "proven")},
                   "proven_optimum": sum(info["proven"].values())}, f, indent=1, sort_keys=True)
