# exp-v2 — official run after the review of exp-v1

Run on 28/10/2026, one machine (Intel i5-13450HX, 10 cores / 16 threads, 16 GB, Windows 11,
Python 3.11.9, numpy 2.4.6, dwave-neal 0.6.0). Solver commit `a12809f` (branch/tag `exp-v2`), game
commit `525ded1` (game repo branch `exp-v2`). Every number of the exp-v2 papers is in `analysis.txt`,
produced by `scripts/analyze_exp_v2.py` from the files below.

## What changed from exp-v1 and why (review items)

| Review | Change |
|---|---|
| M3 strong baselines | `ls` (Greedy + steepest local search), `tabu` (robust-tabu-style), `sa1h` (SA over assignments, no penalties in the moves), all over reassign/swap moves with O(1) deltas |
| M3 Q-bit claim | `pbil`: QIEA with the Q-bit rotation replaced by a PBIL probability vector, everything else shared |
| M3 tuning | the three tunable new solvers got the same 30-trial search as GA/SA/SQA/QIEA (`experiments/tuning-v2`) |
| M3 effort | rows record `evals` (samples or moves evaluated) and `raw_feasible_samples` |
| M4 structure | `exact_set`: only the travel cost depends on the agent, so the optimum = best node set + Hungarian assignment; proves every game-size optimum (~0.1 s) and is the reference wherever Q has that structure |
| M5 QAOA | `qaoa-study`: P(optimum) and P(valid) from the statevector, p = 1..3, no time limit, against uniform sampling of bit strings and of valid assignments; standard X mixer and ring-XY mixer (simulated in the one-hot sector, up to 48 variables) |
| M6 references | gap to a proven optimum (game, small, Moving AI, full with ≤ 5·10⁶ node sets) or to an independent 2 s run of tabu and GA (`full-reference-2000.csv`), not to the best of the solvers being ranked |
| M6 figures | full suite by agents and nodes separately, with bootstrap CIs |
| M7 statistics | repeats averaged per instance before Wilcoxon (instance = unit), Holm; replay captures: exact McNemar on paired seeds, Wilson CIs; 100 seeds per cell |
| M7 final build | replay on the final game after recalibrating the bots (`calibration/`) |

## Files

| File | Content |
|---|---|
| `game-*.csv` | 3,447 recorded game instances (exp-v1 snapshot, tuning split excluded), 9 solvers × 3 repeats, 120 ms |
| `full-*.csv`, `full-reference-2000.csv` | generated suite `full` (1,350 instances, ≤ 1,200 variables), 3 repeats; the 2 s reference runs |
| `small.csv` | generated suite `small` with QAOA and brute force, 5 repeats |
| `movingai.csv`, `qaplib-120.csv`, `qaplib-1000.csv` | public datasets |
| `qaoa-study.csv` | QAOA statevector study |
| `calibration/` | bot calibration on greedy, three phases (coarse, fine, fine) |
| `replay-a.jsonl`, `replay-b.jsonl` | exploit-bot replay, seeds 1000–1049 and 1050–1099, two Unity projects at the same commit, one sidecar each |
| `replay-tabu-a.jsonl`, `replay-tabu-b.jsonl` | the same replay for Tabu, run afterwards (same build, seeds and harness); the analysis tests all seven variants as one family |
| `latency-audit-game.csv`, `latency-audit-full.csv`, `full-audit.yaml` | latency re-measured one process at a time on the idle machine (415 game and 90 full-suite instances; the main parts ran as concurrent processes) |
| `exact-set-latency.csv`, `exact-set-latency.txt` | `exact_set` on the 3,447 evaluated game instances, one process on the idle machine (min of 3 solves), every optimum checked against the benchmark's (`scripts/exact_set_latency.py`, added after the review of 29/10) |
| `analysis.txt`, `figures/` | everything quoted in the papers |

## Reproduce

```bash
bash scripts/run_exp_v2.sh game full reference small external qaoa
bash scripts/run_replay_v2.sh <game project> <copy of it at the same commit> 100 none,greedy,ls,ga,sqa,qiea
bash scripts/run_replay_v2.sh <game project> <copy of it at the same commit> 100 tabu tabu-   # added afterwards
.venv/Scripts/python.exe scripts/analyze_exp_v2.py --replay experiments/results/exp-v2/replay-*.jsonl
```

## Known limitations

- Parts ran as up to seven concurrent single-threaded processes; quality at a fixed wall-clock budget
  is slightly lower than on an idle machine for every solver, and the latency columns of those CSVs
  include contention. `latency-audit-*.csv` give the idle-machine latencies.
- The replay covers one camp of Map 1 and three scripted strategies; no human playtest.
