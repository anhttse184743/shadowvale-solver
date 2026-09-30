"""Statistical comparison between solver variants (O4, RQ2/RQ3).

    compare(rows, "sqa", "greedy", metric="gap")
        -> {"test": "wilcoxon", "n_pairs": ..., "p_value": ..., "mean_a": ..., "mean_b": ...,
            "mean_diff": ..., "ci95_diff": (lo, hi), ...}

Rows come from the bench CSV. Pairing is by (instance_id, repeat): both solvers on
identical instances and seeds → Wilcoxon signed-rank on the paired differences.
When the pairing is incomplete it falls back to Mann–Whitney U on the two samples.
Both use the normal approximation with tie and continuity corrections (what scipy
calls ``method="asymptotic"``); scipy is not needed. The 95 % confidence interval
of the mean difference is a seeded percentile bootstrap.

For "lower is better" metrics (gap, energy, latency_ms) a negative mean_diff means
variant A is better; for approx_ratio / objective a positive one does.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _values(rows: list[dict], variant: str, metric: str) -> dict[tuple[str, int], float]:
    out = {}
    for r in rows:
        if r.get("solver") != variant:
            continue
        v = r.get(metric)
        if v in (None, "", "None"):
            continue
        out[(str(r["instance_id"]), int(r["repeat"]))] = float(v)
    return out


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _rank(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Average ranks (1-based) and Σ(t³ − t) over tie groups."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ties = 0.0
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    return ranks, ties


def wilcoxon(diffs: Sequence[float]) -> tuple[float, float]:
    """(W+, two-sided p). Zero differences are dropped (Wilcoxon's convention)."""
    d = np.asarray([x for x in diffs if x != 0.0], dtype=float)
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    ranks, ties = _rank(np.abs(d))
    w = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if var <= 0:
        return w, 1.0
    z = (abs(w - mean) - 0.5) / math.sqrt(var)
    return w, min(1.0, 2.0 * _norm_sf(max(z, 0.0)))


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(U of sample a, two-sided p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    ranks, ties = _rank(np.concatenate([a, b]))
    u = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    n = n1 + n2
    mean = n1 * n2 / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return u, 1.0
    z = (abs(u - mean) - 0.5) / math.sqrt(var)
    return u, min(1.0, 2.0 * _norm_sf(max(z, 0.0)))


def bootstrap_ci(diffs: Sequence[float], reps: int = 2000, seed: int = 0, level: float = .95) -> tuple[float, float]:
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def compare(rows: list[dict], variant_a: str, variant_b: str, metric: str = "approx_ratio") -> dict:
    va, vb = _values(rows, variant_a, metric), _values(rows, variant_b, metric)
    shared = sorted(set(va) & set(vb))
    paired = len(shared) > 0 and len(shared) == len(va) == len(vb)
    if paired:
        diffs = [va[k] - vb[k] for k in shared]
        stat, p = wilcoxon(diffs)
        test, ci = "wilcoxon", bootstrap_ci(diffs)
    else:
        a, b = list(va.values()), list(vb.values())
        stat, p = mann_whitney(a, b)
        test = "mann_whitney"
        rng = np.random.default_rng(0)
        boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)] if a and b else []
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (math.nan, math.nan)
    a_vals, b_vals = np.array(list(va.values())), np.array(list(vb.values()))
    return {
        "metric": metric, "a": variant_a, "b": variant_b, "test": test, "n_a": len(va), "n_b": len(vb),
        "n_pairs": len(shared) if paired else 0, "statistic": stat, "p_value": p,
        "mean_a": float(a_vals.mean()) if len(a_vals) else math.nan,
        "mean_b": float(b_vals.mean()) if len(b_vals) else math.nan,
        "median_a": float(np.median(a_vals)) if len(a_vals) else math.nan,
        "median_b": float(np.median(b_vals)) if len(b_vals) else math.nan,
        "mean_diff": (float(a_vals.mean() - b_vals.mean()) if len(a_vals) and len(b_vals) else math.nan),
        "ci95_diff": ci,
    }
