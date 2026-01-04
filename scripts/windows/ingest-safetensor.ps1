# Hartonomous Hypercube - Safetensor Model Ingestion (Windows)
# Ingests HuggingFace model packages using 4D Laplacian Eigenmap projection
# (vocab, BPE merges, semantic edges, 4D coordinates)
# Usage: .\scripts\windows\ingest-safetensor.ps1 <model_directory> [-Threshold 0.1] [-Legacy]

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelDir,
    [float]$Threshold = 0.1,
    [switch]$Legacy  # Use old ingester without 4D projection
)

. "$PSScriptRoot\env.ps1"

if (-not (Test-Path $ModelDir)) {
    Write-Host "Not found: $ModelDir" -ForegroundColor Red
    exit 1
}

Write-Host "=== Safetensor Model Ingestion ===" -ForegroundColor Cyan
Write-Host "Model: $ModelDir"
Write-Host "Threshold: $Threshold"

# Choose ingester: prefer 4D version unless -Legacy specified
if ($Legacy) {
    $ingesterName = "ingest_safetensor.exe"
    Write-Host "Mode: Legacy (no 4D projection)" -ForegroundColor Yellow
} else {
    $ingesterName = "ingest_safetensor_4d.exe"
    Write-Host "Mode: 4D Laplacian Eigenmap projection" -ForegroundColor Green
}

$ingester = "$env:HC_BUILD_DIR\$ingesterName"
if (-not (Test-Path $ingester)) {
    $ingester = "$env:HC_BUILD_DIR\Release\$ingesterName"
}
if (-not (Test-Path $ingester)) {
    # Fall back to legacy if 4D not available
    if (-not $Legacy) {
        Write-Host "4D ingester not found, falling back to legacy..." -ForegroundColor Yellow
        $ingester = "$env:HC_BUILD_DIR\ingest_safetensor.exe"
        if (-not (Test-Path $ingester)) {
            $ingester = "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe"
        }
    }
    if (-not (Test-Path $ingester)) {
        Write-Host "Safetensor ingester not found. Run build.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# libpq uses PGPASSWORD env var for authentication
$env:PGPASSWORD = $env:HC_DB_PASS

& $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -t $Threshold $ModelDir
$exitCode = $LASTEXITCODE

# Clean up password from environment
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue

if ($exitCode -eq 0) {
    Write-Host "`nModel ingestion complete" -ForegroundColor Green
} else {
    Write-Host "`nModel ingestion failed" -ForegroundColor Red
    exit 1
}
