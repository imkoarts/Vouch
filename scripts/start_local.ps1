$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .env)) {
    throw "The .env file is missing. Run CONFIGURE_VOUCH.bat first."
}

$python = $null
foreach ($version in @("3.13", "3.12")) {
    try {
        & py "-$version" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $python = @("py", "-$version")
            break
        }
    }
    catch {}
}
if ($null -eq $python) {
    throw "CPython 3.12 or 3.13 was not found through the Windows py launcher."
}

& $python[0] $python[1] launcher.py configure --non-interactive
& $python[0] $python[1] -m app.cli doctor
if ($LASTEXITCODE -ne 0) {
    throw "Doctor found a blocking error. Fix it before starting the bot."
}

Write-Host "Bot started. Use /discover in Telegram for automatic topic selection." -ForegroundColor Green
Write-Host "Stop with Ctrl+C" -ForegroundColor DarkGray
& $python[0] $python[1] launcher.py run
