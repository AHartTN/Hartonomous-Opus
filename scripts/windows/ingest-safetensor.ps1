# Hartonomous Hypercube - Safetensor Model Ingestion (Windows)
# Ingests HuggingFace model packages using 4D Laplacian Eigenmap projection
# (vocab, BPE merges, semantic edges, 4D coordinates)
# Usage: .\scripts\windows\ingest-safetensor.ps1 <model_directory> [-Threshold 0.25] [-Legacy]

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelDir,
    [float]$Threshold = 0.25,
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

if ($exitCode -eq 0) {
    Write-Host "`nModel files ingested successfully" -ForegroundColor Green
    
    # NOTE: The C++ ingester uses Laplacian eigenmap which PRESERVES semantic relationships.
    # The Laplacian-projected centroids are the correct semantic coordinates.
    # DO NOT recompute centroids from atom children - that would destroy the semantics.
    
    # Generate k-NN semantic edges if none exist
    Write-Host "`nChecking for k-NN edge generation..."
    $edgeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation" 2>&1
    if ([int]$edgeCount -eq 0) {
        Write-Host "  Generating k-NN semantic edges..."
        $knnResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT generate_knn_edges(10, 'centroid_knn')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Created $knnResult semantic edges" -ForegroundColor Green
        } else {
            Write-Host "  k-NN generation warning: $knnResult" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Already have $edgeCount edges" -ForegroundColor Gray
    }
} else {
    Write-Host "`nModel ingestion failed" -ForegroundColor Red
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    exit 1
}

# NOTE: Don't remove PGPASSWORD - parent scripts may need it

Write-Host "`nIngestion complete" -ForegroundColor Green
