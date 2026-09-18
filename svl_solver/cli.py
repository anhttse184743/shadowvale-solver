"""Console entry points.

    svl-bench run  --suite small --solvers greedy,brute --repeats 3 --out experiments/results/small.csv
    svl-bench run  --config experiments/configs/medium.yaml --solvers greedy
    svl-bench summary experiments/results/small.csv
    svl-bench list
    svl-bench gen  --seed 1 --nodes 12 --agents 4 --density 0.3 --out inst.json
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
from .instances.generator import generate
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

    a = p.parse_args(argv)
    sys.exit(a.fn(a))


def api_main(argv: list[str] | None = None) -> None:
    from .api.__main__ import main
    main(argv)
