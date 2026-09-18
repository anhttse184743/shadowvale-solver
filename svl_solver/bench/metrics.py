"""Benchmark metrics (research objective O4).

approx_ratio  = F_solver / F_optimum          (maximisation; only when F_opt > 0)
gap           = (E_solver − E_opt) / |E_opt|  (0 = optimal, always defined)
success       = approx_ratio ≥ 0.99 (i.e. within 1 % of optimum)
feasibility   = fraction of raw solver outputs that needed no repair
latency pXX   = percentiles of wall-clock solve time in ms
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

import numpy as np

CSV_COLUMNS = [
    "timestamp", "suite", "instance_id", "seed", "n_nodes", "n_agents", "density", "n_vars",
    "solver", "repeat", "budget_ms", "feasible_raw", "energy", "objective", "latency_ms",
    "hit_budget", "opt_energy", "opt_objective", "approx_ratio", "gap", "note",
]


def approx_ratio(f_sol: float, f_opt: float | None) -> float | None:
    if f_opt is None or not math.isfinite(f_opt) or f_opt <= 1e-12:
        return None
    return f_sol / f_opt


def gap(e_sol: float, e_opt: float | None) -> float | None:
    if e_opt is None or not math.isfinite(e_opt):
        return None
    return (e_sol - e_opt) / max(abs(e_opt), 1e-9)


def _pct(vals: list[float], p: float) -> float | None:
    return float(np.percentile(vals, p)) if vals else None


def summarize(rows: list[dict]) -> list[dict]:
    """Group rows by (solver, n_nodes, n_agents, density) and aggregate."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["solver"], int(r["n_nodes"]), int(r["n_agents"]), float(r["density"]))
        groups[key].append(r)

    out = []
    for (solver, n, a, dens), rs in sorted(groups.items()):
        ratios = [float(r["approx_ratio"]) for r in rs if r.get("approx_ratio") not in (None, "", "None")]
        gaps = [float(r["gap"]) for r in rs if r.get("gap") not in (None, "", "None")]
        lat = [float(r["latency_ms"]) for r in rs]
        feas = [str(r["feasible_raw"]).lower() in ("true", "1") for r in rs]
        hit = [str(r["hit_budget"]).lower() in ("true", "1") for r in rs]
        out.append({
            "solver": solver, "n_nodes": n, "n_agents": a, "density": dens, "runs": len(rs),
            "approx_ratio_mean": mean(ratios) if ratios else None,
            "approx_ratio_min": min(ratios) if ratios else None,
            "gap_mean": mean(gaps) if gaps else None,
            "success_rate": (sum(1 for x in ratios if x >= 0.99) / len(ratios)) if ratios else None,
            "feasibility_rate": sum(feas) / len(feas),
            "timeout_rate": sum(hit) / len(hit),
            "latency_p50_ms": _pct(lat, 50), "latency_p95_ms": _pct(lat, 95), "latency_max_ms": max(lat),
        })
    return out
