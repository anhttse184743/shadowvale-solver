# Long benchmark run that survives interruptions (resume is on by default).
#
#   .\scripts\run_overnight.ps1 -Suite full -Solvers greedy,sqa,qiea -Repeats 30
#
# Results: experiments\results\<suite>.csv, log: experiments\results\<suite>.log
param(
    [string]$Suite = "medium",
    [string]$Solvers = "greedy",
    [int]$Repeats = 30,
    [int]$BudgetMs = 120
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
$out = "experiments\results\$Suite.csv"
$log = "experiments\results\$Suite.log"

"=== $(Get-Date -Format s) start suite=$Suite solvers=$Solvers repeats=$Repeats" | Tee-Object -FilePath $log -Append
& $py -m svl_solver.cli 2>&1 | Out-Null   # warm import
& $py -c "from svl_solver.cli import bench_main; bench_main()" run --config "experiments\configs\$Suite.yaml" `
    --solvers $Solvers --repeats $Repeats --budget-ms $BudgetMs --out $out --quiet 2>&1 | Tee-Object -FilePath $log -Append
"=== $(Get-Date -Format s) done" | Tee-Object -FilePath $log -Append

& $py -c "from svl_solver.cli import bench_main; bench_main()" summary $out | Tee-Object -FilePath $log -Append
