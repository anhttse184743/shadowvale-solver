"""exp-v2, review 29/10: how long does the exact set-reduction solver take on the recorded game instances?

One process on the idle machine, the 3,447 evaluated instances (tuning split excluded), each solved
three times (the minimum is the reported time, as for a warm in-game call; the median is kept too).
Each optimum is checked against the one the benchmark proved.

    .venv/Scripts/python.exe scripts/exact_set_latency.py
"""
from __future__ import annotations

import csv
import glob
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from svl_solver.bench.tune import is_tuning_instance
from svl_solver.instances.game_records import load_records
from svl_solver.qubo.formulation import build_qubo
from svl_solver.solvers.exact_set import SetAssignmentExactSolver

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "experiments/results/exp-v1/game-instances"
OUT = ROOT / "experiments/results/exp-v2"


def main() -> None:
    seen, records = set(), []
    for path in sorted(glob.glob(str(SNAP / "coordination-*.jsonl"))):
        for r in load_records(path):
            if r.instance_id not in seen and not is_tuning_instance(r.instance_id):
                seen.add(r.instance_id)
                records.append(r)
    proven = {}
    with open(OUT / "game-tabu-sa1h-pbil.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            proven[row["instance_id"]] = float(row["opt_energy"])

    solver = SetAssignmentExactSolver()
    rows, mismatched = [], 0
    for inst in records:
        problem = build_qubo(inst, inst.weights)
        times, energy = [], None
        for _ in range(3):
            t0 = time.perf_counter()
            sol = solver.solve(problem, budget_ms=None, seed=0)
            times.append(1000 * (time.perf_counter() - t0))
            energy = sol.energy
        if inst.instance_id in proven and abs(energy - proven[inst.instance_id]) > 1e-6:
            mismatched += 1
        rows.append({"instance_id": inst.instance_id, "n_agents": inst.n_agents, "n_nodes": inst.n_nodes,
                     "ms_min": f"{min(times):.3f}", "ms_median": f"{float(np.median(times)):.3f}",
                     "energy": f"{energy:.9f}"})

    with open(OUT / "exact-set-latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"exact_set on {len(rows)} recorded game instances (one process, idle machine; "
             f"min of 3 solves; optimum differs from the benchmark's in {mismatched})",
             f"{'A':>3s} {'inst':>5s} {'p50 ms':>8s} {'p95 ms':>8s} {'max ms':>8s} {'<=96 ms':>8s} {'<=120 ms':>9s}"]
    by_a = defaultdict(list)
    for r in rows:
        by_a[int(r["n_agents"])].append(float(r["ms_min"]))
    for a in sorted(by_a) + [None]:
        t = np.array([x for k, v in by_a.items() for x in v] if a is None else by_a[a])
        lines.append(f"{'all' if a is None else a:>3} {len(t):5d} {np.percentile(t, 50):8.1f} {np.percentile(t, 95):8.1f} "
                     f"{t.max():8.1f} {100 * np.mean(t <= 96):7.1f}% {100 * np.mean(t <= 120):8.1f}%")
    text = "\n".join(lines)
    (OUT / "exact-set-latency.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
