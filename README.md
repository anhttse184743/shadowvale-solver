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
pip install -e ".[dev]"          # + "matplotlib>=3.8" for svl-bench plot
pytest
```

## Commands

| Command | What it does |
|---|---|
| `svl-bench list` | suites and solver variants |
| `svl-bench run --suite small --solvers greedy,brute --repeats 3 --out experiments/results/small.csv` | run a suite, append CSV rows, resumable |
| `svl-bench game <research/coordination-*.jsonl> --solvers greedy,ga,sa,sqa,qiea` | the instances the game actually solved (tuning split left out) |
| `svl-bench summary experiments/results/small.csv` | approx ratio, feasibility, latency p50/p95 per (solver, size) |
| `svl-bench compare <csv> --all greedy,ga,sa,sqa,qiea --metric gap` | paired Wilcoxon for every pair, Holm-adjusted p, bootstrap CI |
| `svl-bench plot <csv> --out experiments/figures` | latency / quality / viable-regime figures (matplotlib) |
| `svl-bench tune --game <jsonl…> --trials 30` | equal-effort hyper-parameter search for ga/sa/sqa/qiea → experiments/tuning |
| `svl-bench replay <replay-*.jsonl> --a qiea --b greedy` | in-game exploit-bot results from the Unity replay harness |
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
| `svl_solver/solvers` | **O2** | one `Solver` interface: classical `greedy`, `ga`, `sa` (dwave-neal); quantum-inspired `sqa` (own path-integral Monte Carlo — not neal, which is classical SA), `qiea` (Han & Kim), `qaoa` (numpy statevector simulator, ≤ 18 qubits); reference `brute`, `exact` (branch and bound); `qpu` = stretch stub |
| `svl_solver/bench` | **O4** | runner (CSV per row, resume; optimum by brute force or branch and bound), metrics, stats (Wilcoxon / Mann-Whitney / bootstrap / Holm), plots, tuning, replay |
| `svl_solver/api` | — | FastAPI: `GET /health`, `GET /variants`, `POST /solve` |

## The model in one screen

Binary variable `x[a,n] = 1` iff agent `a` is sent to node `n` (flat index `a*N + n`).

```
E(x) = − w_cov   Σ cov[n]·x[a,n]                    reward: escape routes seen
       + w_red   Σ_{a<b} overlap[n,m]·x[a,n]·x[b,m]  penalty: two agents see the same route
       − w_flank Σ_{a<b} flank[n,m]·x[a,n]·x[b,m]    reward: opposite sides of the player
       − w_cover Σ cover_q[n]·x[a,n]                  reward: cover facing the player
                                                       (is_cover · max(0, cos(facing, node→player)))
       + w_dist  Σ dist[a,n]·x[a,n]                   cost: travel hops
       + λ Σ_a (Σ_n x[a,n] − 1)²                      each agent exactly one node
       + λ Σ_n Σ_{a<b} x[a,n]·x[b,n]                  no two agents on one node
```

`λ` is derived in `qubo/penalties.py` so that every infeasible bit-vector has
higher energy than the best feasible one; `tests/test_penalty_dominates.py`
checks this exhaustively on small instances, and
`tests/test_qubo_matches_objective.py` checks `E + offset == −objective` on
every feasible assignment.

Generated instances use a graph-hop visibility radius (`vision_hops`, default 2);
instances recorded in the game carry Unity's ray-cast visibility matrix instead —
the formulation is the same.

## Two run modes — do not confuse them

| Mode | Constraint | Solvers |
|---|---|---|
| In game (`POST /solve`) | ≤ `latency_budget_ms` (120 ms), always returns a feasible plan, falls back to greedy | greedy, ga, sa, sqa, qiea |
| Offline benchmark (`svl-bench`) | unbounded, runs for hours/days, resumable | all of the above + brute (≤ 5·10⁶ assignments) + exact (branch and bound, proves 6×24 in seconds) + qaoa (≤ 18 qubits) |

## Reproducibility

* `generate(seed, n, a, density)` is byte-identical across machines.
* Benchmark repeat `r` on instance with seed `s` uses solver seed `s*1000 + r`.
* Every CSV row carries `instance_id`, `seed`, `repeat`, `budget_ms`, so any row can be replayed,
  and `code_version` (package version + git commit, `.dirty` if uncommitted) — which code made it.
* Solver hyper-parameters come from `svl-bench tune` (same number of trials for every solver, on a
  tuning split kept out of the evaluation); the report is in `experiments/tuning/`.
* What to re-run after a change: see `Docs/Experiments/protocol.md` in the game repo.

## Status (10/2026)

All solvers of the proposal are implemented and tested (pytest); the packaged sidecar serves every
variant. Deviations from the proposal, with their reasons: SQA is our own path-integral Monte Carlo
because dwave-neal is classical SA (using it would make `sa` and `sqa` the same algorithm); QAOA runs
on an exact numpy statevector simulator instead of Qiskit Aer (same result, ~1 MB instead of ~500 MB);
the real-QPU demo is a stretch goal and not done.
