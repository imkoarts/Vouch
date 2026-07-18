param(
    [Parameter(Mandatory = $true)]
    [string]$ClientId
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "Codex CLI was not found in PATH."
}
if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
    throw "npx was not found. Install Node.js before configuring X MCP."
}

$secureSecret = Read-Host "X OAuth2 Client Secret" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureSecret)
try {
    $plainSecret = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:CLIENT_ID = $ClientId
    $env:CLIENT_SECRET = $plainSecret
    Write-Host "Starting one-time X OAuth2 login through xurl..."
    npx -y @xdevplatform/xurl auth oauth2
    if ($LASTEXITCODE -ne 0) {
        throw "xurl OAuth2 login failed."
    }
}
finally {
    Remove-Item Env:CLIENT_ID -ErrorAction SilentlyContinue
    Remove-Item Env:CLIENT_SECRET -ErrorAction SilentlyContinue
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $plainSecret = $null
}

Write-Host "OAuth token is stored by xurl under your user profile, not in this project."
Write-Host "Set enabled = true for [mcp_servers.xapi] in .codex/config.toml, then restart Codex."
Write-Host "Use 'codex mcp list' or /mcp to verify the servers."
