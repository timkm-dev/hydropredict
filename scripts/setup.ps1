$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath ".venv")) {
    py -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$preCommit = Join-Path $projectRoot ".venv\Scripts\pre-commit.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"
& $preCommit install

Write-Host "HydroPredict is ready. Activate it with: .\.venv\Scripts\Activate.ps1"

