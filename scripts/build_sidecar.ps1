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
            "--collect-all", "neal", "--collect-all", "dimod",  # Cython extensions
            # dimod/cyqmbase has no __init__.py, so collect-all misses its two extensions:
            "--hidden-import", "dimod.cyqmbase.cyqmbase_float32", "--hidden-import", "dimod.cyqmbase.cyqmbase_float64",
            "--collect-submodules", "svl_solver",  # uvicorn imports the app by name
            # Bench-only plotting: the sidecar never draws, keep matplotlib (~17 MB) out of the exe.
            "--exclude-module", "matplotlib", "--exclude-module", "tkinter", "--exclude-module", "PIL")
if ($WithQuantum) { $hidden += @("--collect-all", "qiskit", "--collect-all", "qiskit_aer") }

# PyInstaller logs INFO lines to stderr; with "Stop" PowerShell 5.1 would treat the first one as a failure.
$ErrorActionPreference = "Continue"
& $py -m PyInstaller --noconfirm --clean --onefile --name svl-solver `
    --console @hidden --paths . `
    scripts\sidecar_entry.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

Write-Host "built dist\svl-solver.exe - run: dist\svl-solver.exe --port 8001 ; copy to <game>\Solver\svl-solver.exe"
