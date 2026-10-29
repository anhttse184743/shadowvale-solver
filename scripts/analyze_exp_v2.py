"""Tables and figures of the exp-v2 paper from experiments/results/exp-v2 (and the replay JSONL).

    .venv/Scripts/python.exe scripts/analyze_exp_v2.py [--replay <replay-*.jsonl> ...]

Writes experiments/results/exp-v2/analysis.txt (every number quoted in the paper) and the PDF
figures in experiments/results/exp-v2/figures/. Statistics follow the review of exp-v1: repeats of
an instance are averaged before any test (the instance is the unit), Wilcoxon signed-rank over
instances with Holm correction, differences in percentage points with bootstrap CIs; replay captures
use exact McNemar on paired seeds.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from svl_solver.bench.replay import compare_family, load_replay, summarize, wilson
from svl_solver.bench.stats import bootstrap_ci, holm, wilcoxon

OUT = Path("experiments/results/exp-v2")
SOLVERS = ["greedy", "ls", "tabu", "ga", "sa", "sa1h", "pbil", "sqa", "qiea"]
LABEL = {"greedy": "Greedy", "ls": "Greedy+LS", "tabu": "Tabu", "ga": "GA", "sa": "SA (bits)",
         "sa1h": "SA-1h", "pbil": "PBIL", "sqa": "SQA", "qiea": "QIEA", "qaoa": "QAOA", "none": "none"}
lines: list[str] = []
STATS: dict[str, dict[str, dict]] = {}       # table title -> solver -> numbers, for the LaTeX tables
PAPER = Path("paper/iciit2027/tables")


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def read(pattern: str) -> list[dict]:
    """Rows of the matching CSVs. The gap is recomputed from the objective F of the returned
    assignment (repaired when the raw bits were infeasible): the bench's own ``gap`` column uses the
    energy of the raw bits, which for an infeasible answer includes the penalty."""
    rows = []
    for f in sorted(glob.glob(str(OUT / pattern))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    for r in rows:
        f_opt, f_sol = fnum(r.get("opt_objective")), fnum(r.get("objective"))
        if f_opt is not None and f_sol is not None and r.get("suite") != "qaplib":
            g = (f_opt - f_sol) / max(abs(f_opt), 1e-9)
            r["gap"] = str(0.0 if abs(g) < 1e-12 else g)
    return rows


def fnum(v) -> float | None:
    return None if v in (None, "", "None", "nan") else float(v)


def per_instance(rows: list[dict], metric: str = "gap") -> dict[str, dict[str, float]]:
    """solver -> instance -> mean of the metric over repeats."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = fnum(r.get(metric))
        if v is not None:
            acc[r["solver"]][r["instance_id"]].append(v)
    return {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in acc.items()}


def quality_table(rows: list[dict], title: str, solvers=SOLVERS, key: str | None = None) -> None:
    gaps = per_instance(rows)
    store = STATS.setdefault(key or title, {})
    say(f"\n== {title}")
    say(f"{'solver':10s} {'inst':>5s} {'gap%':>7s} {'CI95':>17s} {'atopt%':>7s} {'p50ms':>7s} {'p95ms':>7s} "
        f"{'overBud%':>8s} {'evals':>8s} {'rawfeas%':>8s} {'ansFeas%':>8s}")
    for s in solvers:
        rs = [r for r in rows if r["solver"] == s]
        if not rs:
            continue
        g = np.array(list(gaps.get(s, {}).values()))
        run_g = [fnum(r["gap"]) for r in rs if fnum(r["gap"]) is not None]
        lat = np.array([float(r["latency_ms"]) for r in rs])
        over = np.mean([str(r["hit_budget"]).lower() == "true" for r in rs])
        ev = [fnum(r.get("evals")) for r in rs if fnum(r.get("evals")) is not None]
        # Raw feasibility = share of the solver's own samples that were valid assignments before any
        # repair (samplers report it); the returned answer itself is always checked separately.
        rf = [fnum(r["raw_feasible_samples"]) / fnum(r["evals"]) for r in rs
              if fnum(r.get("raw_feasible_samples")) is not None and fnum(r.get("evals"))]
        if not rf:
            rf = [str(r["feasible_raw"]).lower() == "true" for r in rs]
        ci = bootstrap_ci(list(g)) if len(g) > 1 else (math.nan, math.nan)
        store[s] = {"gap": 100 * g.mean(), "ci": (100 * ci[0], 100 * ci[1]), "inst": len(g),
                    "atopt": 100 * np.mean([x <= 1e-9 for x in run_g]) if run_g else math.nan,
                    "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "over": 100 * over,
                    "evals": float(np.median(ev)) if ev else math.nan, "rawfeas": 100 * float(np.mean(rf))}
        say(f"{LABEL[s]:10s} {len(g):5d} {100 * g.mean():7.3f} [{100 * ci[0]:6.3f},{100 * ci[1]:6.3f}] "
            f"{100 * np.mean([x <= 1e-9 for x in run_g]):7.1f} {np.percentile(lat, 50):7.2f} {np.percentile(lat, 95):7.2f} "
            f"{100 * over:8.1f} {np.median(ev) if ev else float('nan'):8.0f} {100 * np.mean(rf):8.1f} "
            f"{100 * np.mean([str(r['feasible_raw']).lower() == 'true' for r in rs]):8.1f}")


def pairwise(rows: list[dict], pairs: list[tuple[str, str]], title: str) -> None:
    gaps = per_instance(rows)
    res = []
    for a, b in pairs:
        if a not in gaps or b not in gaps:
            continue
        keys = sorted(set(gaps[a]) & set(gaps[b]))
        d = [gaps[a][k] - gaps[b][k] for k in keys]
        _, p = wilcoxon(d)
        res.append((a, b, len(keys), float(np.mean(d)), bootstrap_ci(d), p))
    ph = holm([r[5] for r in res])
    say(f"\n-- {title}: per-instance Wilcoxon, Holm over {len(res)} pairs (diff = A - B in pp of gap)")
    for (a, b, n, m, ci, p), q in zip(res, ph):
        say(f"  {LABEL[a]:>10s} vs {LABEL[b]:<10s} n={n:5d} diff {100 * m:+.3f} pp [{100 * ci[0]:+.3f},{100 * ci[1]:+.3f}] "
            f"p={p:.2g} p_holm={q:.2g}")


def full_suite() -> None:
    rows = read("full-*-*.csv")
    rows = [r for r in rows if "reference" not in r.get("suite", "") and r["budget_ms"] == "120"]
    ref_rows = read("full-reference-2000.csv")
    ref: dict[str, float] = {}
    for r in ref_rows:
        e = float(r["energy"])
        ref[r["instance_id"]] = min(ref.get(r["instance_id"], math.inf), e)
    proven = {}
    for r in rows:
        if fnum(r.get("opt_energy")) is not None:
            proven[r["instance_id"]] = float(r["opt_energy"])
    beaten = 0
    for r in rows:
        iid = r["instance_id"]
        if iid in proven:
            e_ref = proven[iid]
        elif iid in ref:
            e_ref = ref[iid]
            if -float(r["objective"]) < e_ref - 1e-9:
                beaten += 1
        else:
            r["gap"] = ""
            continue
        # On a feasible assignment E = -F, so the reference objective is -E_ref.
        g = (-e_ref - float(r["objective"])) / abs(e_ref)
        r["gap"] = str(0.0 if abs(g) < 1e-12 else g)
    say(f"\nfull suite: {len({r['instance_id'] for r in rows})} instances, proven optimum on {len(proven)}, "
        f"2 s reference on {len(set(ref) - set(proven))}; 120 ms runs beating the 2 s reference: {beaten}")
    quality_table(rows, "full suite (120 ms), gap to proven optimum or independent 2 s reference", key="full")
    pairwise(rows, [("tabu", s) for s in SOLVERS if s != "tabu"] + [("qiea", "pbil"), ("qiea", "ga")], "full suite")
    # By A and N separately, with CIs (review M6).
    gaps = per_instance(rows)
    meta = {r["instance_id"]: (int(r["n_agents"]), int(r["n_nodes"])) for r in rows}
    say("\nfull suite: mean gap % by (A, N) [95% CI]")
    table = {}
    for s in SOLVERS:
        for (A, N) in sorted(set(meta.values())):
            v = [g for i, g in gaps.get(s, {}).items() if meta[i] == (A, N)]
            if v:
                ci = bootstrap_ci(v)
                table[(s, A, N)] = (np.mean(v), ci)
    for A in sorted({a for a, _ in meta.values()}):
        say(f"  A={A}: " + "  ".join(
            f"{LABEL[s]}:" + ",".join(f"{100 * table[(s, A, N)][0]:.2f}" if (s, A, N) in table else "-"
                                       for N in sorted({n for a, n in meta.values() if a == A}))
            for s in SOLVERS if (s, A, 20) in table))
    lat = defaultdict(list)
    for r in rows:
        lat[(r["solver"], int(r["n_agents"]), int(r["n_nodes"]))].append(float(r["latency_ms"]))
    plot_full(table, lat)
    say("full suite: share of runs over the 120 ms budget, by solver: " + ", ".join(
        f"{LABEL[s]} {100 * np.mean([str(r['hit_budget']).lower() == 'true' for r in rows if r['solver'] == s]):.1f}%"
        for s in SOLVERS if any(r["solver"] == s for r in rows)))


def plot_full(table, lat) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    As = sorted({a for _, a, _ in table})
    Ns = sorted({n for _, _, n in table})
    style = {"greedy": ("k", "o"), "ls": ("tab:gray", "s"), "tabu": ("tab:blue", "D"), "ga": ("tab:green", "^"),
             "sa": ("tab:red", "v"), "sa1h": ("tab:pink", "<"), "pbil": ("tab:olive", ">"),
             "sqa": ("tab:purple", "x"), "qiea": ("tab:orange", "*")}
    for kind in ("quality", "latency"):
        fig, axes = plt.subplots(1, len(As), figsize=(7.0, 2.4), sharey=True)
        for ax, A in zip(axes, As):
            for s in SOLVERS:
                xs, ys, lo, hi = [], [], [], []
                for N in Ns:
                    if kind == "quality" and (s, A, N) in table:
                        m, ci = table[(s, A, N)]
                        xs.append(N); ys.append(max(100 * m, 1e-3))
                        lo.append(max(100 * m, 1e-3) - max(100 * ci[0], 1e-3)); hi.append(max(100 * ci[1], 1e-3) - max(100 * m, 1e-3))
                    elif kind == "latency" and lat.get((s, A, N)):
                        v = lat[(s, A, N)]
                        xs.append(N); ys.append(np.percentile(v, 50)); lo.append(0); hi.append(np.percentile(v, 95) - np.percentile(v, 50))
                if xs:
                    c, mk = style[s]
                    ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=mk, ms=3, lw=1, capsize=1.5, label=LABEL[s])
            ax.set_yscale("log")
            ax.set_title(f"A = {A} agents", fontsize=8)
            ax.set_xlabel("nodes N", fontsize=8)
            ax.tick_params(labelsize=7)
            if kind == "latency":
                ax.axhline(120, ls="--", c="k", lw=.8)
        axes[0].set_ylabel("mean gap (%)" if kind == "quality" else "latency p50 (ms)", fontsize=8)
        axes[-1].legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(figs / f"full_{kind}.pdf")
        plt.close(fig)


def external() -> None:
    mv = read("movingai.csv")
    quality_table(mv, "Moving AI (120 ms), gap to proven optimum", key="movingai")
    for b in (120, 1000):
        q = read(f"qaplib-{b}.csv")
        if q:
            quality_table(q, f"QAPLIB n<=30 ({b} ms), gap to published optimum", key=f"qaplib{b}")


def small() -> None:
    rows = read("small.csv")
    if not rows:
        return
    quality_table(rows, "small suite (120 ms), gap to optimum", SOLVERS + ["qaoa"])
    say("QAOA by size (runs, raw feasible %, at optimum %, p50 ms):")
    for n in sorted({int(r["n_vars"]) for r in rows if r["solver"] == "qaoa"}):
        rs = [r for r in rows if r["solver"] == "qaoa" and int(r["n_vars"]) == n]
        g = [fnum(r["gap"]) for r in rs]
        say(f"  n={n}: {len(rs)} runs, raw feasible {100 * np.mean([str(r['feasible_raw']).lower() == 'true' for r in rs]):.0f}%, "
            f"at optimum {100 * np.mean([x is not None and x <= 1e-9 for x in g]):.1f}%, "
            f"p50 {np.median([float(r['latency_ms']) for r in rs]):.0f} ms")


def qaoa_study() -> None:
    rows = read("qaoa-study.csv")
    if not rows:
        return
    say("\n== QAOA study (statevector; means over instances)")
    say(f"{'ansatz':6s} {'n':>3s} {'p':>2s} {'inst':>4s} {'P_opt':>8s} {'P_feas':>7s} {'unifP_opt':>9s} {'unifFeasP_opt':>13s} "
        f"{'hit512':>7s} {'unif512':>7s} {'unifFeas512':>11s}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["ansatz"], int(r["n_vars"]), int(r["layers"]))].append(r)
    for (an, n, p), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        m = lambda k: np.mean([float(r[k]) for r in rs])  # noqa: E731
        say(f"{an:6s} {n:3d} {p:2d} {len(rs):4d} {m('p_opt'):8.4f} {m('p_feasible'):7.3f} {m('uniform_p_opt'):9.2e} "
            f"{m('uniform_feasible_p_opt'):13.4f} {m('hit_512'):7.3f} {m('uniform_hit_512'):7.3f} {m('uniform_feasible_hit_512'):11.3f}")
    say("ratio P_opt / P_opt(uniform over valid assignments), median over instances:")
    for (an, n, p), rs in sorted(groups.items()):
        rat = [float(r["p_opt"]) / float(r["uniform_feasible_p_opt"]) for r in rs]
        say(f"  {an} n={n} p={p}: median {np.median(rat):.2f}  (share > 1: {100 * np.mean([x > 1 for x in rat]):.0f}%)")


def replay(paths: list[str]) -> None:
    if not paths:
        return
    rows = load_replay(*paths)
    say(f"\n== replay: {len(rows)} runs; game versions {sorted({r.get('game_version') for r in rows})}")
    for s in summarize(rows, include_invalid=True):
        lo, hi = s["capture_ci95"]
        say(f"  {s['bot']:10s} {s['variant']:6s} n={s['runs']:3d} capture {100 * s['capture_rate']:5.1f}% [{100 * lo:4.1f},{100 * hi:4.1f}] "
            f"escape {s['escape_time_mean']:5.1f}s coord {s['coordination_score']:.2f} fallback {100 * s['fallback_rate']:.1f}% "
            f"lat95 {s['latency_p95_ms']:.0f}ms")
    for bot in sorted({r["bot"] for r in rows}) + [None]:
        for metric in ("captured", "escape_time_mean", "coordination_score"):
            for r in compare_family(rows, ["greedy", "none"], metric=metric, bot=bot):
                extra = f" discordant {r['discordant']}" if "discordant" in r else ""
                say(f"  {r['bot']:10s} {metric:18s} {r['a']:>6s} {r['mean_a']:.3f} vs {r['b']:<6s} {r['mean_b']:.3f} "
                    f"{r['test']} p={r['p_value']:.3g} p_holm={r['p_holm']:.3g}{extra}")
    plot_replay(rows)
    latex_replay_table(rows)


def plot_replay(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = [v for v in ["none", "greedy", "ls", "tabu", "ga", "sqa", "qiea"] if any(r["variant"] == v for r in rows)]
    bots = sorted({r["bot"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = .8 / len(variants)
    for j, v in enumerate(variants):
        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(bots):
            rs = [r for r in rows if r["bot"] == b and r["variant"] == v]
            if not rs:
                continue
            k = int(sum(r["captured"] for r in rs))
            p = k / len(rs)
            a, c = wilson(k, len(rs))
            xs.append(i + (j - (len(variants) - 1) / 2) * w); ys.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (c - p))
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=1.5, label=LABEL[v], hatch="//" if v == "none" else None,
               edgecolor="k", lw=.4, error_kw={"lw": .6})
    ax.set_xticks(range(len(bots)))
    ax.set_xticklabels([b.replace("_", r"\_") if False else b for b in bots], fontsize=7)
    ax.set_ylabel("capture rate (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25), frameon=False)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "replay_capture.pdf")
    plt.close(fig)


def _f(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def latex_solver_table() -> None:
    """Table: every solver at 120 ms on every suite (gap %), plus game latency, effort, raw feasibility."""
    PAPER.mkdir(parents=True, exist_ok=True)
    g, f, m = STATS.get("game", {}), STATS.get("full", {}), STATS.get("movingai", {})
    q1, q2 = STATS.get("qaplib120", {}), STATS.get("qaplib1000", {})
    out = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
           r" & \multicolumn{5}{c}{Recorded game instances} & Full & Moving & \multicolumn{2}{c}{QAPLIB} \\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}",
           r"Solver & gap & opt. & p50 & evals & raw & gap & AI gap & 120\,ms & 1\,s \\",
           r" & \% & \% & ms & & feas.\,\% & \% & \% & \% & \% \\", r"\midrule"]
    for s in SOLVERS:
        if s not in g:
            continue
        x = g[s]
        evals = "--" if math.isnan(x["evals"]) else format(x["evals"], ",.0f").replace(",", "{,}")
        cells = [LABEL[s], _f(x["gap"], 3 if x["gap"] < .1 else 2), _f(x["atopt"], 1),
                 _f(x["p50"], 1 if x["p50"] >= 1 else 2), evals, _f(x["rawfeas"], 0),
                 _f(f.get(s, {}).get("gap")), _f(m.get(s, {}).get("gap")),
                 _f(q1.get(s, {}).get("gap"), 1), _f(q2.get(s, {}).get("gap"), 1)]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "solvers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def latex_qaoa_table() -> None:
    """QAOA read from the statevector at its deepest layer, against uniform sampling."""
    rows = read("qaoa-study.csv")
    if not rows:
        return
    PAPER.mkdir(parents=True, exist_ok=True)
    p_max = max(int(r["layers"]) for r in rows)
    groups = defaultdict(list)
    for r in rows:
        if int(r["layers"]) == p_max:
            groups[(int(r["n_vars"]), int(r["n_agents"]), int(r["n_nodes"]), r["ansatz"])].append(r)
    out = [r"\begin{tabular}{rrlrrrrr}", r"\toprule",
           r"$n$ & $A{\times}|V|$ & mixer & $P_\text{valid}$ & $P_\text{opt}$ & ratio & hit$_{512}$ & unif. \\", r"\midrule"]
    for (n, a, v, an), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3])):
        m = lambda k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
        ratio = float(np.median([float(r["p_opt"]) / float(r["uniform_feasible_p_opt"]) for r in rs]))
        out.append(f"{n} & {a}$\\times${v} & {'X' if an == 'x' else 'XY'} & {m('p_feasible'):.2f} & {m('p_opt'):.4f} & "
                   f"{ratio:.2f} & {m('hit_512'):.2f} & {m('uniform_feasible_hit_512'):.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "qaoa.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def latex_replay_table(rows) -> None:
    """Capture rate per bot and variant with Wilson 95 % intervals; McNemar against Greedy and none."""
    PAPER.mkdir(parents=True, exist_ok=True)
    variants = [v for v in ["none", "greedy", "ls", "tabu", "ga", "sqa", "qiea"] if any(r["variant"] == v for r in rows)]
    bots = sorted({r["bot"] for r in rows})
    out = [r"\begin{tabular}{l" + "r" * len(variants) + "}", r"\toprule",
           "Bot & " + " & ".join(LABEL[v] for v in variants) + r" \\", r"\midrule"]
    for b in bots + [None]:
        cells = []
        for v in variants:
            rs = [r for r in rows if (b is None or r["bot"] == b) and r["variant"] == v]
            if not rs:
                cells.append("--")
                continue
            k = int(sum(r["captured"] for r in rs))
            lo, hi = wilson(k, len(rs))
            cells.append(f"{100 * k / len(rs):.0f} {{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}")
        name = "all bots" if b is None else b.replace("_", r"\_")
        if b is None:
            out.append(r"\midrule")
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "replay.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs="*", default=[])
    a = ap.parse_args()
    game = read("game-*.csv")
    say(f"game: {len({r['instance_id'] for r in game})} instances; optimum by "
        f"{sorted({(m, sum(1 for r in game if r['opt_method'] == m)) for m in {r['opt_method'] for r in game}})}")
    quality_table(game, "recorded game instances (120 ms), gap to proven optimum", key="game")
    pairwise(game, [("tabu", s) for s in SOLVERS if s != "tabu"] + [("qiea", "pbil"), ("qiea", "ga"), ("ls", "greedy"),
                                                                     ("sa1h", "sa")], "game")
    full_suite()
    external()
    small()
    qaoa_study()
    replay(a.replay)
    latex_solver_table()
    latex_qaoa_table()
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
