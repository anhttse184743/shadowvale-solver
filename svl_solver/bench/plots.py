"""Figures for the report (needs the [plots] extra: matplotlib).

    latency_vs_nodes(summary_rows, out_png)      one line per solver, p50 with p95 band
    quality_vs_nodes(summary_rows, out_png)      mean gap to the optimum (lower is better)
    viable_regime_heatmap(summary_rows, solver, out_png)
                                                 nodes × agents grid, colour = share of runs
                                                 that were optimal-or-near AND inside the budget

``summary_rows`` is the output of ``bench.metrics.summarize``.
"""

from __future__ import annotations

from collections import defaultdict


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on the extra
        raise ImportError("plots need matplotlib: pip install -e .[plots]") from e
    return plt


def _by_solver(rows: list[dict], y: str) -> dict[str, list[tuple[int, float, dict]]]:
    series: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get(y) is not None:
            series[r["solver"]].append((int(r["n_nodes"]) * int(r["n_agents"]), float(r[y]), r))
    return {k: sorted(v, key=lambda t: t[0]) for k, v in series.items()}


def latency_vs_nodes(summary_rows: list[dict], out_png: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for solver, pts in sorted(_by_solver(summary_rows, "latency_p50_ms").items()):
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], marker="o", label=solver)
        ax.fill_between(xs, [p[1] for p in pts], [p[2]["latency_p95_ms"] for p in pts], alpha=.15)
    ax.axhline(120, color="grey", ls="--", lw=1, label="120 ms budget")
    ax.set_xlabel("QUBO variables (agents × nodes)")
    ax.set_ylabel("latency (ms, p50 · p95 band)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def quality_vs_nodes(summary_rows: list[dict], out_png: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for solver, pts in sorted(_by_solver(summary_rows, "gap_mean").items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=solver)
    ax.set_xlabel("QUBO variables (agents × nodes)")
    ax.set_ylabel("mean gap to optimum")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def viable_regime_heatmap(summary_rows: list[dict], solver: str, out_png: str) -> None:
    plt = _plt()
    rows = [r for r in summary_rows if r["solver"] == solver and r.get("success_rate") is not None]
    nodes = sorted({int(r["n_nodes"]) for r in rows})
    agents = sorted({int(r["n_agents"]) for r in rows})
    grid = [[float("nan")] * len(nodes) for _ in agents]
    for r in rows:
        viable = r["success_rate"] * (1.0 - r["timeout_rate"])
        grid[agents.index(int(r["n_agents"]))][nodes.index(int(r["n_nodes"]))] = viable
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(nodes)), nodes)
    ax.set_yticks(range(len(agents)), agents)
    ax.set_xlabel("nodes"); ax.set_ylabel("agents")
    ax.set_title(f"{solver}: optimal and within budget")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
