$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Vouch: secure local setup" -ForegroundColor Cyan
Write-Host "Credentials are collected by the local setup window and stored only in .env." -ForegroundColor Yellow

$launcher = Join-Path $root "launcher.py"
$commands = @(
    @{ Executable = "py"; Arguments = @("-3.13", $launcher, "--configure") },
    @{ Executable = "py"; Arguments = @("-3.12", $launcher, "--configure") },
    @{ Executable = "python"; Arguments = @($launcher, "--configure") }
)

foreach ($candidate in $commands) {
    if (-not (Get-Command $candidate.Executable -ErrorAction SilentlyContinue)) {
        continue
    }
    $arguments = $candidate.Arguments
    & $candidate.Executable $arguments
    if ($LASTEXITCODE -eq 0) {
        exit 0
    }
}

throw "Python 3.12 or 3.13 could not run launcher.py. Install Python and rerun this script."
