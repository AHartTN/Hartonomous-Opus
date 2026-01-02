# Hartonomous Hypercube - Safetensor Model Ingestion (Windows)
# Ingests HuggingFace model packages (vocab, BPE merges, semantic edges)
# Usage: .\scripts\windows\ingest-safetensor.ps1 <model_directory> [-Threshold 0.1]

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelDir,
    [float]$Threshold = 0.1
)

. "$PSScriptRoot\env.ps1"

if (-not (Test-Path $ModelDir)) {
    Write-Host "Not found: $ModelDir" -ForegroundColor Red
    exit 1
}

Write-Host "=== Safetensor Model Ingestion ===" -ForegroundColor Cyan
Write-Host "Model: $ModelDir"
Write-Host "Threshold: $Threshold"

$ingester = "$env:HC_BUILD_DIR\ingest_safetensor.exe"
if (-not (Test-Path $ingester)) {
    $ingester = "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe"
}
if (-not (Test-Path $ingester)) {
    Write-Host "Safetensor ingester not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

& $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -t $Threshold $ModelDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nModel ingestion complete" -ForegroundColor Green
} else {
    Write-Host "`nModel ingestion failed" -ForegroundColor Red
    exit 1
}
