# shadowvale-solver

Enemy Coordination Engine for **ShadowVale** (SEP490 RBL capstone): the QUBO
formulation of squad tactics, the classical and quantum-inspired solvers that
minimise it, a reproducible benchmark harness, and the tiny localhost HTTP
sidecar the Unity client calls at runtime.

This is **not** the backend. The backend (ASP.NET Core + PostgreSQL) never
talks to this service. The web dashboard reads benchmark results that were
imported into PostgreSQL from the CSV files produced here.

```
Unity ──POST /solve──▶ svl-solver (127.0.0.1:8001) ──▶ CoordinationPlan
                        └─ same package, offline: svl-bench → experiments/results/*.csv
```

## Setup

```powershell
winget install Python.Python.3.11
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"          # add ".[dev,quantum]" once the QAOA solver lands (W9)
pytest
```

## Commands

| Command | What it does |
|---|---|
| `svl-bench list` | suites and solver variants (and which are still planned) |
| `svl-bench run --suite small --solvers greedy,brute --repeats 3 --out experiments/results/small.csv` | run a suite, append CSV rows, resumable |
| `svl-bench summary experiments/results/small.csv` | approx ratio, feasibility, latency p50/p95 per (solver, size) |
| `svl-bench gen --seed 1 --nodes 20 --agents 4 --request --out req.json` | one instance as a Unity `QuboRequest` |
| `svl-api --port 8001` (or `python -m svl_solver.api`) | start the sidecar |
| `scripts\build_sidecar.ps1` | PyInstaller → `dist\svl-solver.exe` |
| `scripts\run_overnight.ps1 -Suite full -Solvers greedy,sqa` | long resumable run with log |

## Package layout

| Package | Research objective | Content |
|---|---|---|
| `svl_solver/model` | — | `Instance` (nav graph + precomputed cov/overlap/flank/dist), reference `objective`, feasibility + repair |
| `svl_solver/qubo` | **O1** | `build_qubo` (upper-triangular Q + offset), penalty λ with proof sketch, Ising conversion |
| `svl_solver/instances` | **O3** | seeded generator, suites (`small` / `medium` / `full`), JSON I/O and the Unity wire format |
| `svl_solver/solvers` | **O2** | one `Solver` interface; `greedy`, `brute` implemented; `ga`, `sa`, `sqa`, `qiea`, `qaoa`, `qpu` registered as planned |
| `svl_solver/bench` | **O4** | runner (CSV per row, resume, cached optimum when N!/(N−A)! ≤ 200k), metrics, stats/plots (W9) |
| `svl_solver/api` | — | FastAPI: `GET /health`, `GET /variants`, `POST /solve` |

## The model in one screen

Binary variable `x[a,n] = 1` iff agent `a` is sent to node `n` (flat index `a*N + n`).

```
E(x) = − w_cov   Σ cov[n]·x[a,n]                    reward: escape routes seen
       + w_red   Σ_{a<b} overlap[n,m]·x[a,n]·x[b,m]  penalty: two agents see the same route
       − w_flank Σ_{a<b} flank[n,m]·x[a,n]·x[b,m]    reward: opposite sides of the player
       − w_cover Σ is_cover[n]·x[a,n]                 reward: standing in cover
       + w_dist  Σ dist[a,n]·x[a,n]                   cost: travel hops
       + λ Σ_a (Σ_n x[a,n] − 1)²                      each agent exactly one node
       + λ Σ_n Σ_{a<b} x[a,n]·x[b,n]                  no two agents on one node
```

`λ` is derived in `qubo/penalties.py` so that every infeasible bit-vector has
higher energy than the best feasible one; `tests/test_penalty_dominates.py`
checks this exhaustively on small instances, and
`tests/test_qubo_matches_objective.py` checks `E + offset == −objective` on
every feasible assignment.

Visibility v1 is a graph-hop radius (`vision_hops`, default 2). A ray-cast
visibility matrix exported from Unity can replace it without touching the
formulation.

## Two run modes — do not confuse them

| Mode | Constraint | Solvers |
|---|---|---|
| In game (`POST /solve`) | ≤ `latency_budget_ms` (120 ms), always returns a feasible plan, falls back to greedy | greedy, ga, sa, sqa, qiea |
| Offline benchmark (`svl-bench`) | unbounded, runs for hours/days, resumable | all of the above + brute (≤ 200k assignments) + qaoa (≤ ~25 vars) |

## Reproducibility

* `generate(seed, n, a, density)` is byte-identical across machines.
* Benchmark repeat `r` on instance with seed `s` uses solver seed `s*1000 + r`.
* Every CSV row carries `instance_id`, `seed`, `repeat`, `budget_ms`, so any row can be replayed.

## Roadmap (mirrors the project plan)

| Week | Solver work |
|---|---|
| W3 | greedy + brute on `small`; formulation signed off by supervisor |
| W5–W7 | `ga`, `sa`; `small` + `medium` baselines |
| W8 | `sqa` (dwave-neal), `qiea` |
| W9 | `qaoa` (Qiskit Aer) on `small`; `full` overnight; `bench/stats.py`, `bench/plots.py` |
| W10–W13 | more seeds, tune λ / weights, import CSV into backend `solver_runs` |
| W14 | statistics and figures only — no new runs |
