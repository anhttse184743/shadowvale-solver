# Package the sidecar API as a single Windows executable so testers do not
# need Python. Output: dist\svl-solver.exe  (copy next to ShadowVale.exe).
#
#   .\scripts\build_sidecar.ps1            # uses .venv
#
# Note: the [quantum] extra (Qiskit) is NOT bundled by default; the shipped
# sidecar carries greedy / sa / sqa / qiea which are the ones that fit the
# 120 ms budget anyway. Pass -WithQuantum to include Qiskit (large exe).
param([switch]$WithQuantum)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) { throw ".venv not found; run: python -m venv .venv; .\.venv\Scripts\pip install -e .[dev]" }
$py = ".\.venv\Scripts\python.exe"

$hidden = @("--hidden-import", "uvicorn.logging", "--hidden-import", "uvicorn.loops.auto",
            "--hidden-import", "uvicorn.protocols.http.auto", "--hidden-import", "uvicorn.lifespan.on",
            "--hidden-import", "neal", "--hidden-import", "dimod")
if ($WithQuantum) { $hidden += @("--collect-all", "qiskit", "--collect-all", "qiskit_aer") }

& $py -m PyInstaller --noconfirm --clean --onefile --name svl-solver `
    --console @hidden `
    -m svl_solver.api

Write-Host "built dist\svl-solver.exe  — run:  dist\svl-solver.exe --port 8001"
