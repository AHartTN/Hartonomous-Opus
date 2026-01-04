# Hartonomous Hypercube - Content Ingestion (Windows)
# Usage: .\scripts\windows\ingest.ps1 <path>
# Examples:
#   .\scripts\windows\ingest.ps1 C:\Documents\notes\
#   .\scripts\windows\ingest.ps1 .\test-data\

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Path
)

. "$PSScriptRoot\env.ps1"

if (-not (Test-Path $Path)) {
    Write-Host "Not found: $Path" -ForegroundColor Red
    exit 1
}

Write-Host "=== Hypercube Content Ingestion ===" -ForegroundColor Cyan
Write-Host "Target: $Path"

$ingester = "$env:HC_BUILD_DIR\cpe_ingest.exe"
if (-not (Test-Path $ingester)) {
    $ingester = "$env:HC_BUILD_DIR\Release\cpe_ingest.exe"
}
if (-not (Test-Path $ingester)) {
    Write-Host "Ingester not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

# libpq uses PGPASSWORD env var for authentication
$env:PGPASSWORD = $env:HC_DB_PASS

& $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST $Path
$exitCode = $LASTEXITCODE

# Clean up password from environment
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue

if ($exitCode -eq 0) {
    Write-Host "`nIngestion complete" -ForegroundColor Green
} else {
    Write-Host "`nIngestion failed" -ForegroundColor Red
    exit 1
}
