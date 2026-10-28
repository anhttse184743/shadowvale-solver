#!/usr/bin/env bash
# exp-v2, level B: the exploit-bot replay on the final game build, split over two Unity projects
# (the game repo and a copy of it at the same commit) that run at the same time, each with its own
# sidecar so that solves never queue. Seeds 1000..(1000+RUNS-1); the first project takes the lower
# half, the second the upper half; both append to their own JSONL, merged by the analysis.
#
#   bash scripts/run_replay_v2.sh <game project> <copy project> [RUNS=100] [VARIANTS] [TAG]
#
# TAG (e.g. "tabu-") prefixes the output files of a later run, so that it adds a variant on the same
# seeds without touching the logs of the first one.
set -u
cd "$(dirname "$0")/.."
GAME="$1"; COPY="$2"; RUNS="${3:-100}"; VARIANTS="${4:-none,greedy,ls,ga,sqa,qiea}"; TAG="${5:-}"
OUT="$(pwd)/experiments/results/exp-v2"
UNITY="/c/Program Files/Unity/Hub/Editor/6000.3.24f1/Editor/Unity.exe"
mkdir -p "$OUT"
HALF=$((RUNS / 2))
start_sidecar() { ./dist/svl-solver.exe --port "$1" --log-level warning > "$OUT/sidecar-${TAG}$1.log" 2>&1 & echo $!; }
s1=$(start_sidecar 8001); s2=$(start_sidecar 8002)
for p in 8001 8002; do until curl -s -m 2 "http://127.0.0.1:$p/health" > /dev/null; do sleep 1; done; done
echo "$(date -Iseconds) replay $RUNS seeds x [$VARIANTS], game $(git -C "$GAME" rev-parse --short HEAD)" | tee -a "$OUT/official.log"
run() {  # project port seed0 runs tag
  SVL_SOLVER_URL="http://127.0.0.1:$2" SVL_REPLAY_SEED="$3" SVL_REPLAY_RUNS="$4" SVL_REPLAY_VARIANTS="$VARIANTS" \
  SVL_REPLAY_OUT="$OUT/replay-$5.jsonl" \
  "$UNITY" -batchmode -projectPath "$1" -runTests -testPlatform EditMode -assemblyNames "ShadowVale.Map01.Tests" \
    -testFilter 'ShadowVale.Map01.Tests.Replay.ReplayHarnessTests.ExploitBotMatrix' \
    -testResults "$OUT/replay-tests-$5.xml" -logFile "$OUT/replay-unity-$5.log"
}
run "$GAME" 8001 1000 "$HALF" "${TAG}a" & r1=$!
run "$COPY" 8002 $((1000 + HALF)) $((RUNS - HALF)) "${TAG}b" & r2=$!
wait $r1 $r2
kill "$s1" "$s2" 2> /dev/null; powershell -c "Get-Process svl-solver -ErrorAction SilentlyContinue | Stop-Process -Force"
echo "$(date -Iseconds) replay done" | tee -a "$OUT/official.log"
