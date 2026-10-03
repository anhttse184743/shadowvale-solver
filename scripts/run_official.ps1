# The official experiment run (see ShadowVale Docs/Experiments/protocol.md), one part after the
# other on this machine - never in parallel: the solvers' budget is wall-clock time.
#
#   .\scripts\run_official.ps1                 # everything, results in experiments\results\exp-v1
#   .\scripts\run_official.ps1 -Parts game,replay
#
# Needs: both repos on the tagged commit, Unity closed (batch mode locks the project),
# dist\svl-solver.exe built (scripts\build_sidecar.ps1). Takes hours (overnight).
#
# Checkpoints - if the run stops (crash, power, closing the window), start the same command again:
#  * bench parts write each row as soon as it is done and skip rows already in their CSV;
#  * the replay part appends one line per finished run and skips runs already in replay.jsonl;
#  * the recorded game instances are copied into <out>\game-instances on the first run and every
#    later run uses that snapshot, so the set never changes (the replay itself records new ones).
param(
    [string]$Tag = "exp-v1",
    [string[]]$Parts = @("game", "replay", "medium", "small", "full", "movingai", "qaplib"),
    [int]$GameRepeats = 3,
    [int]$SuiteRepeats = 5,
    [int]$FullRepeats = 3,
    [int]$ReplayRuns = 30,
    [string]$Solvers = "greedy,ga,sa,sqa,qiea",
    [string]$GameRepo = "..\ShadowVale",
    [string]$Unity = "C:\Program Files\Unity\Hub\Editor\6000.3.24f1\Editor\Unity.exe",
    [string]$Research = "$env:USERPROFILE\AppData\LocalLow\DefaultCompany\ShadowVale\research"
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$bench = ".\.venv\Scripts\svl-bench.exe"
$out = "experiments\results\$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "official.log"

function Log([string]$m) { "$(Get-Date -Format s)  $m" | Tee-Object -FilePath $log -Append }
function Bench([string[]]$a) { & $bench @a 2>&1 | Where-Object { $_ -notmatch "Warning" } | Tee-Object -FilePath $log -Append }

Log "=== official run $Tag  solver $(git rev-parse --short HEAD)  game $(git -C $GameRepo rev-parse --short HEAD)  parts: $($Parts -join ',')"
if (git status --porcelain --untracked-files=no) { Log "WARNING: solver repo has uncommitted changes (rows will say .dirty)" }
if (git -C $GameRepo status --porcelain --untracked-files=no) { Log "WARNING: game repo has uncommitted changes" }

if ($Parts -contains "game") {
    $snapshot = Join-Path $out "game-instances"
    if (-not (Test-Path $snapshot)) {
        New-Item -ItemType Directory -Force $snapshot | Out-Null
        Copy-Item "$Research\coordination-*.jsonl" $snapshot
        Log "snapshot: $((Get-ChildItem $snapshot -Filter *.jsonl).Count) record files copied to $snapshot"
    } else { Log "snapshot: reusing $snapshot ($((Get-ChildItem $snapshot -Filter *.jsonl).Count) files)" }
    Log "--- A: recorded game instances (evaluation split), $GameRepeats repeats"
    Bench @("game", "$snapshot\coordination-*.jsonl", "--solvers", $Solvers, "--repeats", "$GameRepeats",
            "--budget-ms", "120", "--exact-s", "5", "--out", "$out\game.csv", "--quiet")
}

if ($Parts -contains "replay") {
    Log "--- B: in-game exploit bots, $ReplayRuns runs x 3 bots x [$Solvers]"
    Get-Process svl-solver -ErrorAction SilentlyContinue | Stop-Process -Force   # left over from an interrupted run
    $sidecar = Start-Process -FilePath "dist\svl-solver.exe" -ArgumentList "--port 8001 --log-level warning" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 5
    $env:SVL_REPLAY_RUNS = "$ReplayRuns"; $env:SVL_REPLAY_VARIANTS = $Solvers
    $env:SVL_REPLAY_OUT = (Resolve-Path $out).Path + "\replay.jsonl"; $env:SVL_SOLVER_URL = "http://127.0.0.1:8001"
    & $Unity -batchmode -projectPath (Resolve-Path $GameRepo).Path -runTests -testPlatform EditMode `
        -assemblyNames ShadowVale.Map01.Tests -testFilter ShadowVale.Map01.Tests.Replay.ReplayHarnessTests.ExploitBotMatrix `
        -testResults ((Resolve-Path $out).Path + "\replay-tests.xml") -logFile ((Resolve-Path $out).Path + "\replay-unity.log") | Out-Null
    Stop-Process -Id $sidecar.Id -ErrorAction SilentlyContinue
    Log "replay: $((Get-Content "$out\replay.jsonl" -ErrorAction SilentlyContinue | Measure-Object -Line).Lines) runs written"
}

if ($Parts -contains "medium") {
    Log "--- A: generated suite 'medium' (in-game regime), $SuiteRepeats repeats, optimum where provable in 10 s"
    Bench @("run", "--config", "experiments\configs\medium.yaml", "--solvers", $Solvers, "--repeats", "$SuiteRepeats",
            "--exact-s", "10", "--out", "$out\medium.csv", "--quiet")
}

if ($Parts -contains "small") {
    Log "--- A: generated suite 'small' with QAOA and brute force, $SuiteRepeats repeats"
    Bench @("run", "--suite", "small", "--solvers", "$Solvers,qaoa,brute", "--repeats", "$SuiteRepeats",
            "--out", "$out\small.csv", "--quiet")
}

if ($Parts -contains "full") {
    Log "--- A: generated suite 'full' (20-100 nodes x 4-12 agents), $FullRepeats repeats, objective only"
    Bench @("run", "--config", "experiments\configs\full.yaml", "--solvers", $Solvers, "--repeats", "$FullRepeats",
            "--exact-s", "0", "--out", "$out\full.csv", "--quiet")
}

if ($Parts -contains "movingai") {
    Log "--- External: coordination instances on Moving AI game maps (Sturtevant 2012), $GameRepeats repeats"
    if (-not (Test-Path "datasets\external\movingai")) { & .\.venv\Scripts\python.exe scripts\fetch_datasets.py 2>&1 | Tee-Object -FilePath $log -Append }
    if (-not (Test-Path "$out\movingai-instances.jsonl")) {
        Bench @("movingai", "--per-set", "10", "--agents", "4,6,8", "--seeds", "1,2,3", "--out", "$out\movingai-instances.jsonl")
    }
    Bench @("game", "$out\movingai-instances.jsonl", "--include-tuning", "--solvers", $Solvers, "--repeats", "$GameRepeats",
            "--budget-ms", "120", "--exact-s", "5", "--out", "$out\movingai.csv", "--quiet")
}

if ($Parts -contains "qaplib") {
    if (-not (Test-Path "datasets\external\qaplib")) { & .\.venv\Scripts\python.exe scripts\fetch_datasets.py 2>&1 | Tee-Object -FilePath $log -Append }
    foreach ($budget in @(120, 1000)) {
        Log "--- External: QAPLIB n<=30 (Burkard et al. 1997), budget $budget ms, $SuiteRepeats repeats, gap to the published optimum"
        Bench @("qaplib", "--max-n", "30", "--solvers", $Solvers, "--repeats", "$SuiteRepeats", "--budget-ms", "$budget",
                "--out", "$out\qaplib-$budget.csv", "--quiet")
    }
}

Log "--- analysis"
$all = $Solvers
foreach ($csv in @("game", "medium", "small", "full", "movingai", "qaplib-120", "qaplib-1000")) {
    if (Test-Path "$out\$csv.csv") {
        Log "[$csv] gap to the proven optimum, objective, latency - Holm-adjusted pairs"
        Bench @("compare", "$out\$csv.csv", "--all", $all, "--metric", "gap")
        Bench @("compare", "$out\$csv.csv", "--all", $all, "--metric", "objective")
        Bench @("compare", "$out\$csv.csv", "--all", $all, "--metric", "latency_ms")
        Bench @("plot", "$out\$csv.csv", "--out", "$out\figures-$csv")
    }
}
if (Test-Path "$out\replay.jsonl") { Bench @("replay", "$out\replay.jsonl", "--a", "qiea", "--b", "greedy") }
Log "=== done"
