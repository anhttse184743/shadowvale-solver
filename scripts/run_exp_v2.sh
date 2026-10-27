#!/usr/bin/env bash
# exp-v2, level A (offline benchmark) after the review of exp-v1. Each part writes its own CSV,
# row by row, and resumes where it stopped. Parts run as parallel single-threaded processes;
# latency is re-measured afterwards on a quiet machine (part "latency").
#
#   bash scripts/run_exp_v2.sh game full small external qaoa anytime latency
set -u
cd "$(dirname "$0")/.."
B=".venv/Scripts/svl-bench.exe"
OUT="experiments/results/exp-v2"
SNAP="experiments/results/exp-v1/game-instances"   # the same recorded instances as exp-v1
mkdir -p "$OUT"
log() { echo "$(date -Iseconds)  $*" | tee -a "$OUT/official.log"; }
log "=== exp-v2 solver $(git rev-parse --short HEAD)$(git status --porcelain --untracked-files=no | grep -q . && echo .dirty) parts: $*"
ALL="greedy,ls,ga,sa,sqa,qiea,tabu,sa1h,pbil"
pids=()
for part in "$@"; do case "$part" in
  game)
    for grp in "greedy,ls,ga,qiea" "sa,sqa" "tabu,sa1h,pbil"; do
      $B game "$SNAP/coordination-*.jsonl" --solvers "$grp" --repeats 3 --budget-ms 120 --exact-s 5 \
        --out "$OUT/game-${grp//,/-}.csv" --quiet >> "$OUT/game-${grp//,/-}.log" 2>&1 & pids+=($!)
    done ;;
  full)
    for grp in "greedy,ls,tabu,qiea,pbil" "ga,sa1h" "sa,sqa"; do
      $B run --config experiments/configs/full.yaml --solvers "$grp" --repeats 3 --exact-s 0 \
        --out "$OUT/full-${grp//,/-}.csv" --quiet >> "$OUT/full-${grp//,/-}.log" 2>&1 & pids+=($!)
    done ;;
  reference)
    # Independent reference for the full suite: tabu and GA with 2 s, one run each (review M6).
    $B run --config experiments/configs/full.yaml --solvers tabu,ga --repeats 1 --budget-ms 2000 --exact-s 0 \
      --out "$OUT/full-reference-2000.csv" --quiet >> "$OUT/full-reference.log" 2>&1 & pids+=($!) ;;
  small)
    ( $B run --suite small --solvers "$ALL,qaoa,brute" --repeats 5 --out "$OUT/small.csv" --quiet ) >> "$OUT/small.log" 2>&1 & pids+=($!) ;;
  external)
    ( $B game "experiments/results/exp-v1/movingai-instances.jsonl" --include-tuning --solvers "$ALL" --repeats 3 \
        --budget-ms 120 --exact-s 5 --out "$OUT/movingai.csv" --quiet
      for budget in 120 1000; do
        $B qaplib --max-n 30 --solvers "$ALL" --repeats 5 --budget-ms $budget --out "$OUT/qaplib-$budget.csv" --quiet
      done ) >> "$OUT/external.log" 2>&1 & pids+=($!) ;;
  qaoa)
    $B qaoa-study --layers 3 --out "$OUT/qaoa-study.csv" >> "$OUT/qaoa-study.log" 2>&1 & pids+=($!) ;;
  *) log "unknown part $part" ;;
esac; done
for p in "${pids[@]}"; do wait "$p"; done
log "=== done: $*"
