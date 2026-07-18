param(
    [Parameter(Mandatory = $false)]
    [string]$SourceArchive = "",

    [Parameter(Mandatory = $false)]
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($SourceArchive)) {
    $SourceArchive = Join-Path $projectRoot "dist\Vouch-v0.19.7-public-source-base.zip"
}

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $projectRoot "dist\vouch-demo-clean-v0.19.7"
}

$archivePath = [System.IO.Path]::GetFullPath($SourceArchive)
$destinationPath = [System.IO.Path]::GetFullPath($Destination)

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Public source archive not found: $archivePath"
}

if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists. Choose a new empty directory: $destinationPath"
}

New-Item -ItemType Directory -Path $destinationPath | Out-Null
Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationPath

$forbiddenDirectoryNames = @(".venv", "data", "drafts", "logs", "__pycache__")
$forbiddenExtensions = @(".db", ".sqlite", ".sqlite3", ".pyc", ".pyo")

$forbiddenItems = Get-ChildItem -LiteralPath $destinationPath -Recurse -Force | Where-Object {
    ($_.PSIsContainer -and $forbiddenDirectoryNames -contains $_.Name) -or
    (-not $_.PSIsContainer -and $forbiddenExtensions -contains $_.Extension.ToLowerInvariant()) -or
    (-not $_.PSIsContainer -and $_.Name -eq ".env")
}

if ($forbiddenItems) {
    $paths = ($forbiddenItems | ForEach-Object { $_.FullName }) -join [Environment]::NewLine
    throw "The clean-demo validation found forbidden runtime state:`n$paths"
}

Write-Host "Clean Vouch demo workspace prepared: $destinationPath"
Write-Host "No credentials, databases, drafts, chat history, logs, or local virtual environment were copied."
