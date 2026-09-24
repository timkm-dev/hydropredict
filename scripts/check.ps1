$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $projectRoot ".venv\Scripts"
Set-Location -LiteralPath $projectRoot

$checks = @(
    @{ Name = "format"; Command = "ruff.exe"; Arguments = @("format", "--check", ".") },
    @{ Name = "lint"; Command = "ruff.exe"; Arguments = @("check", ".") },
    @{ Name = "types"; Command = "mypy.exe"; Arguments = @() },
    @{ Name = "tests"; Command = "pytest.exe"; Arguments = @() }
)

foreach ($check in $checks) {
    & (Join-Path $bin $check.Command) @($check.Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "The $($check.Name) check failed with exit code $LASTEXITCODE."
    }
}
