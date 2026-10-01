"""In-game results from the Unity replay harness (RQ4, O5).

The harness (ShadowVale ``Map01/Tests/Replay``) plays Map 1 with exploit bots — always the same
lane, always the same bush, always the same stone trick — once per seed and solver variant, and
writes one JSON object per run. Run *i* uses the same seed for every variant, so variants are
compared on paired runs.

    summarize(rows)                       per (bot, variant): capture rate with a Wilson 95 % CI,
                                          escape time, HP left, coordination score, plan entropy
    compare_variants(rows, "qiea", "greedy", bot="same_route", metric="captured")
                                          paired Wilcoxon on that metric (bench.stats.compare)

Only runs marked ``valid_for_comparison`` (no variant switch, few greedy fallbacks) count unless
``include_invalid`` is set: a sqa run that fell back to greedy all the time measured greedy.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from svl_solver.bench.stats import compare

#: Metrics that make sense per run, with the direction that is better for the *guards*.
METRICS = {
    "captured": "higher",
    "player_hp": "lower",
    "escape_time_mean": "higher",
    "coordination_score": "higher",
    "plan_entropy_bits": "higher",
    "agent_position_entropy_bits": "higher",
    "seconds": "higher",
}


def load_replay(*paths: str | Path) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["captured"] = 1.0 if r.get("captured") in (True, 1, "true") else 0.0
                    rows.append(r)
    return rows


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion — sane at 0 % and 100 %, unlike the normal one."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _usable(rows: list[dict], include_invalid: bool) -> list[dict]:
    return [r for r in rows if include_invalid or r.get("valid_for_comparison", True)]


def summarize(rows: list[dict], include_invalid: bool = False) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in _usable(rows, include_invalid):
        groups[(r["bot"], r["variant"])].append(r)
    out = []
    for (bot, variant), rs in sorted(groups.items()):
        n = len(rs)
        caught = int(sum(r["captured"] for r in rs))
        escaped = [r["escape_time_mean"] for r in rs if r["outcome"] == "escaped" and r.get("escape_time_mean", 0) > 0]
        lo, hi = wilson(caught, n)
        out.append({
            "bot": bot, "variant": variant, "runs": n,
            "capture_rate": caught / n if n else math.nan, "capture_ci95": (lo, hi),
            "undetected": sum(r["outcome"] == "undetected" for r in rs) / n if n else math.nan,
            "escape_time_mean": float(np.mean(escaped)) if escaped else math.nan,
            "player_hp_mean": float(np.mean([r["player_hp"] for r in rs])),
            "coordination_score": float(np.mean([r["coordination_score"] for r in rs])),
            "plan_entropy_bits": float(np.mean([r["plan_entropy_bits"] for r in rs])),
            "fallback_rate": float(np.mean([r.get("fallback_rate", 0) for r in rs])),
            "latency_p95_ms": float(np.mean([r.get("latency_p95_ms", 0) for r in rs])),
        })
    return out


def compare_variants(rows: list[dict], a: str, b: str, bot: str | None = None, metric: str = "captured",
                     include_invalid: bool = False) -> dict:
    """Paired test between two variants on runs with the same bot and seed."""
    use = [r for r in _usable(rows, include_invalid) if bot is None or r["bot"] == bot]
    as_bench = [{"solver": r["variant"], "instance_id": f'{r["bot"]}-{r["seed"]}', "repeat": 0, metric: r.get(metric)}
                for r in use]
    res = compare(as_bench, a, b, metric=metric)
    res["bot"] = bot or "all"
    res["better_for_guards"] = METRICS.get(metric, "higher")
    return res


def format_summary(summary: list[dict]) -> str:
    head = f'{"bot":<12} {"variant":<8} {"runs":>4}  {"capture":>8} {"95% CI":>15}  {"escape s":>8} {"HP left":>7} {"coord":>5} {"H(plan)":>7} {"fallback":>8}'
    lines = [head, "-" * len(head)]
    for s in summary:
        lo, hi = s["capture_ci95"]
        lines.append(
            f'{s["bot"]:<12} {s["variant"]:<8} {s["runs"]:>4}  {s["capture_rate"]:>8.0%} {f"[{lo:.0%}, {hi:.0%}]":>15}  '
            f'{s["escape_time_mean"]:>8.1f} {s["player_hp_mean"]:>7.1f} {s["coordination_score"]:>5.2f} '
            f'{s["plan_entropy_bits"]:>7.2f} {s["fallback_rate"]:>8.0%}')
    return "\n".join(lines)
