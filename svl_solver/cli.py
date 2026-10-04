"""Console entry points.

    svl-bench run  --suite small --solvers greedy,brute --repeats 3 --out experiments/results/small.csv
    svl-bench run  --config experiments/configs/medium.yaml --solvers greedy
    svl-bench summary experiments/results/small.csv
    svl-bench list
    svl-bench gen  --seed 1 --nodes 12 --agents 4 --density 0.3 --out inst.json
    svl-bench game coordination-*.jsonl --solvers greedy,sa,sqa,qiea --repeats 5 --out experiments/results/game.csv
    svl-bench compare experiments/results/game.csv --a sqa --b sa --metric energy
    svl-api        --port 8001
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .bench.metrics import summarize
from .bench.runner import run_suite
from .bench.stats import compare
from .instances.game_records import load_records
from .instances.generator import generate
from .instances.suite import SuiteSpec
from .instances.io import save_instance, to_qubo_request
from .instances.suite import BUILTIN_SUITES, load_suite
from .solvers.registry import list_variants


def _set_exact_limit(a: argparse.Namespace) -> None:
    """--exact-s: seconds the branch-and-bound may take per instance to prove the optimum (0 = never)."""
    from svl_solver.bench import runner

    if getattr(a, "exact_s", None) is not None:
        runner.EXACT_TIME_LIMIT_S = float(a.exact_s)


def _cmd_run(a: argparse.Namespace) -> int:
    _set_exact_limit(a)
    spec = load_suite(a.config) if a.config else BUILTIN_SUITES[a.suite]
    solvers = [s for s in a.solvers.split(",") if s]
    rows = run_suite(spec, solvers, repeats=a.repeats, out=a.out, budget_ms=a.budget_ms,
                     resume=not a.no_resume, progress=None if a.quiet else print)
    print(f"wrote {len(rows)} new rows to {a.out}")
    return 0


def _cmd_summary(a: argparse.Namespace) -> int:
    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for s in summarize(rows):
        ar = f"{s['approx_ratio_mean']:.3f}" if s["approx_ratio_mean"] is not None else "  n/a"
        print(f"{s['solver']:7s} n={s['n_nodes']:3d} a={s['n_agents']:2d} d={s['density']:.2f} "
              f"runs={s['runs']:3d} ratio={ar} feas={s['feasibility_rate']:.2f} "
              f"p50={s['latency_p50_ms']:.2f}ms p95={s['latency_p95_ms']:.2f}ms")
    return 0


def _cmd_list(a: argparse.Namespace) -> int:
    print("suites:")
    for name, s in BUILTIN_SUITES.items():
        print(f"  {name:7s} nodes={s.nodes} agents={s.agents} densities={s.densities} "
              f"seeds={len(s.seeds)} → {s.n_instances} instances")
    print("solvers:")
    for v in list_variants():
        flag = "available" if v["available"] else f"planned ({v['due']})"
        print(f"  {v['id']:7s} {v['family']:17s} {flag}")
    return 0


def _cmd_gen(a: argparse.Namespace) -> int:
    inst = generate(seed=a.seed, n_nodes=a.nodes, n_agents=a.agents, chokepoint_density=a.density)
    if a.request:
        Path(a.out).write_text(json.dumps(to_qubo_request(inst), indent=2), encoding="utf-8")
    else:
        save_instance(inst, a.out)
    print(f"wrote {inst.instance_id} → {a.out}")
    return 0


def _load_game(patterns: list[str]) -> list:
    """Every record in the files (globs allowed), each instance id once; unreadable files are skipped."""
    import glob as _glob
    seen, out = set(), []
    for pat in patterns:
        for path in sorted(_glob.glob(pat)) or [pat]:
            try:
                recs = load_records(path)
            except (OSError, ValueError, KeyError) as e:
                print(f"skipped {path}: {e}")
                continue
            for r in recs:
                if r.instance_id not in seen:
                    seen.add(r.instance_id)
                    out.append(r)
    return out


def _cmd_game(a: argparse.Namespace) -> int:
    from svl_solver.bench.tune import is_tuning_instance

    _set_exact_limit(a)
    records = _load_game(a.records)
    if not a.include_tuning:
        held = sum(1 for r in records if is_tuning_instance(r.instance_id))
        records = [r for r in records if not is_tuning_instance(r.instance_id)]
        print(f"{held} instances of the tuning split left out (--include-tuning to keep them)")
    spec = SuiteSpec(name="game", nodes=[], agents=[], densities=[], seeds=[], budget_ms=a.budget_ms,
                     brute_force_max_candidates=a.brute_max)
    solvers = [s for s in a.solvers.split(",") if s]
    rows = run_suite(spec, solvers, repeats=a.repeats, out=a.out, budget_ms=a.budget_ms,
                     resume=not a.no_resume, progress=None if a.quiet else print, instances=records)
    print(f"{len(records)} recorded game instances -> wrote {len(rows)} new rows to {a.out}")
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.all:
        from svl_solver.bench.stats import compare_all

        variants = [v for v in a.all.split(",") if v]
        print(f"{a.metric}: {len(variants)} solvers, {len(variants) * (len(variants) - 1) // 2} pairs, Holm-adjusted")
        for r in compare_all(rows, variants, metric=a.metric):
            print(f"  {r['a']:>6} vs {r['b']:<6} mean {r['mean_a']:.4f} vs {r['mean_b']:.4f}  {r['test']} "
                  f"p={r['p_value']:.4g}  p_holm={r['p_holm']:.4g}  diff 95% CI [{r['ci95_diff'][0]:.4f}, {r['ci95_diff'][1]:.4f}]")
        return 0
    if not (a.a and a.b):
        print("give --a and --b, or --all solver1,solver2,...")
        return 2
    r = compare(rows, a.a, a.b, metric=a.metric)
    print(f"{r['metric']}: {r['a']} mean={r['mean_a']:.4f} median={r['median_a']:.4f} (n={r['n_a']}) vs "
          f"{r['b']} mean={r['mean_b']:.4f} median={r['median_b']:.4f} (n={r['n_b']})")
    print(f"  {r['test']}: statistic={r['statistic']:.1f} p={r['p_value']:.4g}  "
          f"mean diff={r['mean_diff']:.4f}  95% CI [{r['ci95_diff'][0]:.4f}, {r['ci95_diff'][1]:.4f}]")
    return 0


def _cmd_replay(a: argparse.Namespace) -> int:
    from svl_solver.bench.replay import (METRICS, audit, compare_family, compare_variants, format_summary,
                                         load_replay, summarize)

    rows = load_replay(*a.files)
    every = not a.valid_only
    invalid = sum(1 for r in rows if not r.get("valid_for_comparison", True))
    print(f"{len(rows)} runs from {len(a.files)} file(s); " +
          ((f"{invalid} marked not valid for comparison, " + ("included (every run: the primary analysis)" if every
            else "left out (sensitivity analysis)")) if invalid else "all valid for comparison"))
    for warning in audit(rows):
        print("WARNING: " + warning)
    print(format_summary(summarize(rows, every)))
    bots = [a.bot] if a.bot else sorted({r["bot"] for r in rows}) + [None]
    for metric in a.metrics.split(","):
        better = METRICS.get(metric, "higher")
        if a.a and a.b:
            for bot in bots:
                r = compare_variants(rows, a.a, a.b, bot=bot, metric=metric, include_invalid=every)
                print(f"{r['bot']:<12} {metric}: {a.a} {r['mean_a']:.3f} vs {a.b} {r['mean_b']:.3f}  "
                      f"{r['test']} n={r['n_pairs'] or r['n_a'] + r['n_b']} p={r['p_value']:.4g}  "
                      f"diff 95% CI [{r['ci95_diff'][0]:.3f}, {r['ci95_diff'][1]:.3f}]  ({better} is better for the guards)")
        if a.vs:
            for bot in bots:
                for r in compare_family(rows, a.vs.split(","), metric=metric, bot=bot, include_invalid=every):
                    print(f"{r['bot']:<12} {metric:<16} {r['a']:>6} {r['mean_a']:.3f} vs {r['b']:<6} {r['mean_b']:.3f}  "
                          f"n={r['n_pairs'] or r['n_a'] + r['n_b']} p={r['p_value']:.4g} p_holm={r['p_holm']:.4g}  "
                          f"diff 95% CI [{r['ci95_diff'][0]:.3f}, {r['ci95_diff'][1]:.3f}]  ({better} is better for the guards)")
    return 0


def _cmd_plot(a: argparse.Namespace) -> int:
    from collections import defaultdict
    from pathlib import Path

    from svl_solver.bench import plots

    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # One point per (solver, nodes, agents): the chokepoint densities averaged.
    groups: dict = defaultdict(list)
    for s in summarize(rows):
        groups[(s["solver"], s["n_nodes"], s["n_agents"])].append(s)
    merged = []
    for (solver, n, ag), ss in groups.items():
        m = {"solver": solver, "n_nodes": n, "n_agents": ag}
        for k in ("latency_p50_ms", "latency_p95_ms", "gap_mean", "success_rate", "timeout_rate"):
            vals = [x[k] for x in ss if x.get(k) is not None]
            m[k] = sum(vals) / len(vals) if vals else None
        merged.append(m)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plots.latency_vs_nodes(merged, str(out / "latency.png"))
    plots.quality_vs_nodes(merged, str(out / "quality.png"))
    for solver in sorted({m["solver"] for m in merged}):
        plots.viable_regime_heatmap(merged, solver, str(out / f"viable_{solver}.png"))
    print(f"figures in {out}")
    return 0


def _cmd_qaplib(a: argparse.Namespace) -> int:
    from svl_solver.bench.external import run_qaplib
    from svl_solver.external import qaplib

    insts = qaplib.load(a.data, a.soln, max_n=a.max_n)
    if a.names:
        wanted = set(a.names.split(","))
        insts = [q for q in insts if q.name in wanted]
    print(f"{len(insts)} QAPLIB instances (n <= {a.max_n}) with a published solution")
    rows = run_qaplib(insts, [x for x in a.solvers.split(",") if x], a.repeats, a.budget_ms, a.out,
                      progress=None if a.quiet else print)
    print(f"wrote {len(rows)} new rows to {a.out}")
    return 0


def _cmd_movingai(a: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from svl_solver.external import movingai

    maps = []
    for set_dir in sorted(p for p in Path(a.maps).iterdir() if p.is_dir()):
        files = sorted(set_dir.glob("*.map"))
        if not files:
            continue
        step = max(1, len(files) // a.per_set)
        maps += files[::step][:a.per_set]          # evenly spread over the set, same choice every time
    agents = [int(x) for x in a.agents.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    recs = movingai.build_set(maps, agents, seeds, spacing=a.spacing, k_nodes=a.k_nodes, vision=a.vision)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(maps)} maps -> {len(recs)} coordination instances -> {a.out}")
    return 0


def _cmd_tune(a: argparse.Namespace) -> int:
    import numpy as np
    from svl_solver.bench.tune import is_tuning_instance, tune, write_report
    from svl_solver.instances.generator import generate
    from svl_solver.qubo.formulation import build_qubo

    rng = np.random.default_rng(a.seed)
    problems = []
    if a.game:
        pool = [r for r in _load_game(a.game) if is_tuning_instance(r.instance_id) and r.n_agents >= 4]
        pool.sort(key=lambda r: r.instance_id)
        for i in rng.permutation(len(pool))[:a.game_count]:
            r = pool[int(i)]
            problems.append((r.instance_id, build_qubo(r, r.weights)))
    for i in range(a.generated_count):
        n, ag, d = (20, 40, 60)[i % 3], (4, 6, 8)[(i // 3) % 3], (0.0, 0.3, 0.6)[(i // 9) % 3]
        inst = generate(seed=901 + i, n_nodes=n, n_agents=ag, chokepoint_density=d)
        problems.append((inst.instance_id, build_qubo(inst)))
    solvers = [s for s in a.solvers.split(",") if s]
    best, trials, info = tune(problems, solvers, trials=a.trials, budget_ms=a.budget_ms, finalists=a.finalists,
                              repeats=a.repeats, seed=a.seed, exact_s=a.exact_s)
    write_report(a.out, best, trials, info)
    print(f"report: {a.out}/trials.csv, {a.out}/tuned.json")
    return 0


def bench_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="svl-bench", description="ShadowVale solver benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--suite", choices=sorted(BUILTIN_SUITES), default="small")
    g.add_argument("--config", help="YAML suite file (experiments/configs/*.yaml)")
    r.add_argument("--solvers", default="greedy,brute")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--budget-ms", type=int, default=None)
    r.add_argument("--out", default="experiments/results/bench.csv")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--exact-s", type=float, default=None,
                   help="seconds per instance to prove the optimum beyond brute force (default 60; 0 = never)")
    r.set_defaults(fn=_cmd_run)

    s = sub.add_parser("summary", help="aggregate a results CSV")
    s.add_argument("csv")
    s.set_defaults(fn=_cmd_summary)

    l = sub.add_parser("list", help="list suites and solvers")
    l.set_defaults(fn=_cmd_list)

    ge = sub.add_parser("gen", help="generate one instance as JSON")
    ge.add_argument("--seed", type=int, default=1)
    ge.add_argument("--nodes", type=int, default=20)
    ge.add_argument("--agents", type=int, default=4)
    ge.add_argument("--density", type=float, default=0.3)
    ge.add_argument("--request", action="store_true", help="write a QuboRequest instead of a raw instance")
    ge.add_argument("--out", default="instance.json")
    ge.set_defaults(fn=_cmd_gen)

    gm = sub.add_parser("game", help="benchmark instances recorded inside the game (JSONL)")
    gm.add_argument("records", nargs="+")
    gm.add_argument("--solvers", default="greedy,ga,sa,sqa,qiea")
    gm.add_argument("--repeats", type=int, default=3)
    gm.add_argument("--budget-ms", type=int, default=120)
    gm.add_argument("--brute-max", type=int, default=200_000)
    gm.add_argument("--out", default="experiments/results/game.csv")
    gm.add_argument("--no-resume", action="store_true")
    gm.add_argument("--quiet", action="store_true")
    gm.add_argument("--include-tuning", action="store_true", help="also the instances the solvers were tuned on")
    gm.add_argument("--exact-s", type=float, default=None,
                    help="seconds per instance to prove the optimum beyond brute force (default 60; 0 = never)")
    gm.set_defaults(fn=_cmd_game)

    pl = sub.add_parser("plot", help="figures for the report from a results CSV (needs matplotlib)")
    pl.add_argument("csv")
    pl.add_argument("--out", default="experiments/figures")
    pl.set_defaults(fn=_cmd_plot)

    qp = sub.add_parser("qaplib", help="every solver on QAPLIB instances, gap to the published optimum")
    qp.add_argument("--data", default="datasets/external/qaplib/data")
    qp.add_argument("--soln", default="datasets/external/qaplib/soln")
    qp.add_argument("--max-n", type=int, default=30)
    qp.add_argument("--names", help="only these instances (comma-separated)")
    qp.add_argument("--solvers", default="greedy,ga,sa,sqa,qiea")
    qp.add_argument("--repeats", type=int, default=5)
    qp.add_argument("--budget-ms", type=int, default=120)
    qp.add_argument("--out", default="experiments/results/qaplib.csv")
    qp.add_argument("--quiet", action="store_true")
    qp.set_defaults(fn=_cmd_qaplib)

    ma = sub.add_parser("movingai", help="coordination instances on Moving AI game maps (game-record JSONL)")
    ma.add_argument("--maps", default="datasets/external/movingai", help="folder with one sub-folder of .map files per set")
    ma.add_argument("--per-set", type=int, default=10)
    ma.add_argument("--agents", default="4,6,8")
    ma.add_argument("--seeds", default="1,2,3")
    ma.add_argument("--spacing", type=int, default=4)
    ma.add_argument("--k-nodes", type=int, default=24)
    ma.add_argument("--vision", type=float, default=14.0)
    ma.add_argument("--out", default="datasets/derived/movingai.jsonl")
    ma.set_defaults(fn=_cmd_movingai)

    tn = sub.add_parser("tune", help="equal-effort hyper-parameter search for the stochastic solvers")
    tn.add_argument("--solvers", default="ga,sa,sqa,qiea")
    tn.add_argument("--trials", type=int, default=30)
    tn.add_argument("--budget-ms", type=int, default=120)
    tn.add_argument("--game", nargs="*", default=[], help="recorded game instances (JSONL, globs); only the tuning split is used")
    tn.add_argument("--game-count", type=int, default=16)
    tn.add_argument("--generated-count", type=int, default=8, help="generated in-game-regime instances, seeds 901+")
    tn.add_argument("--finalists", type=int, default=3)
    tn.add_argument("--repeats", type=int, default=3)
    tn.add_argument("--exact-s", type=float, default=10.0)
    tn.add_argument("--seed", type=int, default=2026)
    tn.add_argument("--out", default="experiments/tuning")
    tn.set_defaults(fn=_cmd_tune)

    c = sub.add_parser("compare", help="paired significance test between two solvers in a results CSV")
    c.add_argument("csv")
    c.add_argument("--a")
    c.add_argument("--b")
    c.add_argument("--all", help="every pair of these solvers (comma-separated), Holm-adjusted")
    c.add_argument("--metric", default="gap")
    c.set_defaults(fn=_cmd_compare)

    rp = sub.add_parser("replay", help="in-game results of the Unity replay harness (exploit bots, JSONL)")
    rp.add_argument("files", nargs="+")
    rp.add_argument("--a", help="variant A for a paired comparison")
    rp.add_argument("--b", help="variant B")
    rp.add_argument("--bot", help="only this bot (default: each bot, then all)")
    rp.add_argument("--metrics", default="captured,player_hp,escape_time_mean")
    rp.add_argument("--vs", help="baselines, e.g. greedy,none: every other variant against each, Holm-adjusted")
    rp.add_argument("--valid-only", action="store_true",
                    help="only runs marked valid_for_comparison (sensitivity analysis; default: every run)")
    rp.set_defaults(fn=_cmd_replay)

    a = p.parse_args(argv)
    sys.exit(a.fn(a))


def api_main(argv: list[str] | None = None) -> None:
    from .api.__main__ import main
    main(argv)
