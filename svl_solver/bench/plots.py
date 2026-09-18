"""Figures for the report — planned for W9 (needs the [plots] extra).

    latency_vs_nodes(summary_rows, out_png)      one line per solver, p50 with p95 band
    quality_vs_nodes(summary_rows, out_png)      approx_ratio mean ± min
    viable_regime_heatmap(summary_rows, out_png) nodes × agents grid, colour = success_rate
                                                 under the 120 ms budget
"""

from __future__ import annotations


def latency_vs_nodes(summary_rows: list[dict], out_png: str) -> None:
    raise NotImplementedError("scheduled for W9")


def quality_vs_nodes(summary_rows: list[dict], out_png: str) -> None:
    raise NotImplementedError("scheduled for W9")


def viable_regime_heatmap(summary_rows: list[dict], out_png: str) -> None:
    raise NotImplementedError("scheduled for W9")
