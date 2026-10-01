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


def _cmd_run(a: argparse.Namespace) -> int:
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


def _cmd_game(a: argparse.Namespace) -> int:
    records = load_records(a.records)
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
    r = compare(rows, a.a, a.b, metric=a.metric)
    print(f"{r['metric']}: {r['a']} mean={r['mean_a']:.4f} median={r['median_a']:.4f} (n={r['n_a']}) vs "
          f"{r['b']} mean={r['mean_b']:.4f} median={r['median_b']:.4f} (n={r['n_b']})")
    print(f"  {r['test']}: statistic={r['statistic']:.1f} p={r['p_value']:.4g}  "
          f"mean diff={r['mean_diff']:.4f}  95% CI [{r['ci95_diff'][0]:.4f}, {r['ci95_diff'][1]:.4f}]")
    return 0


def _cmd_replay(a: argparse.Namespace) -> int:
    from svl_solver.bench.replay import METRICS, compare_variants, format_summary, load_replay, summarize

    rows = load_replay(*a.files)
    invalid = sum(1 for r in rows if not r.get("valid_for_comparison", True))
    print(f"{len(rows)} runs from {len(a.files)} file(s)" + (f", {invalid} not valid for comparison" +
          (" (included)" if a.include_invalid else " (left out)") if invalid else ""))
    print(format_summary(summarize(rows, a.include_invalid)))
    if a.a and a.b:
        bots = [a.bot] if a.bot else sorted({r["bot"] for r in rows}) + [None]
        for metric in a.metrics.split(","):
            for bot in bots:
                r = compare_variants(rows, a.a, a.b, bot=bot, metric=metric, include_invalid=a.include_invalid)
                print(f"{r['bot']:<12} {metric}: {a.a} {r['mean_a']:.3f} vs {a.b} {r['mean_b']:.3f}  "
                      f"{r['test']} n={r['n_pairs'] or r['n_a'] + r['n_b']} p={r['p_value']:.4g}  "
                      f"diff 95% CI [{r['ci95_diff'][0]:.3f}, {r['ci95_diff'][1]:.3f}]  ({METRICS.get(metric, 'higher')} is better for the guards)")
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
    gm.set_defaults(fn=_cmd_game)

    c = sub.add_parser("compare", help="paired significance test between two solvers in a results CSV")
    c.add_argument("csv")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--metric", default="gap")
    c.set_defaults(fn=_cmd_compare)

    rp = sub.add_parser("replay", help="in-game results of the Unity replay harness (exploit bots, JSONL)")
    rp.add_argument("files", nargs="+")
    rp.add_argument("--a", help="variant A for a paired comparison")
    rp.add_argument("--b", help="variant B")
    rp.add_argument("--bot", help="only this bot (default: each bot, then all)")
    rp.add_argument("--metrics", default="captured,player_hp,escape_time_mean")
    rp.add_argument("--include-invalid", action="store_true", help="also runs not valid for comparison")
    rp.set_defaults(fn=_cmd_replay)

    a = p.parse_args(argv)
    sys.exit(a.fn(a))


def api_main(argv: list[str] | None = None) -> None:
    from .api.__main__ import main
    main(argv)
