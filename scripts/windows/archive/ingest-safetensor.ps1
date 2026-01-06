# Hartonomous Hypercube - Safetensor Model Ingestion (Windows)
# Ingests HuggingFace model packages:
#   - vocab/BPE merges -> compositions
#   - composition centroids computed from atom children (YOUR coordinate system)
#   - embedding k-NN similarity -> relation edges (model-specific relationships)
#   - router weights (MoE) -> relation edges
# Usage: .\scripts\windows\ingest-safetensor.ps1 <model_directory> [-Threshold 0.25]

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelDir,
    [float]$Threshold = 0.25
)

. "$PSScriptRoot\env.ps1"

if (-not (Test-Path $ModelDir)) {
    Write-Host "Not found: $ModelDir" -ForegroundColor Red
    exit 1
}

Write-Host "=== Safetensor Model Ingestion ===" -ForegroundColor Cyan
Write-Host "Model: $ModelDir"
Write-Host "Threshold: $Threshold"

# Use the main ingester (fixed architecture)
$ingesterName = "ingest_safetensor.exe"
Write-Host "Mode: Relation extraction (k-NN similarity -> relation edges)" -ForegroundColor Green

$ingester = "$env:HC_BUILD_DIR\$ingesterName"
if (-not (Test-Path $ingester)) {
    $ingester = "$env:HC_BUILD_DIR\Release\$ingesterName"
}
if (-not (Test-Path $ingester)) {
    Write-Host "Safetensor ingester not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

# libpq uses PGPASSWORD env var for authentication
$env:PGPASSWORD = $env:HC_DB_PASS

& $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -t $Threshold $ModelDir
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`nModel files ingested successfully" -ForegroundColor Green
    
    # Architecture explanation:
    # - Atoms: Unicode codepoints with YOUR deterministic 4D coordinates (Hilbert -> S³)
    # - Compositions: centroids computed from atom children (done by C++ ingester)
    # - Relations: embedding similarity extracted as k-NN edges (done by C++ ingester)
    #              router weights for MoE models (done by C++ ingester)
    # 
    # Embeddings are MODEL-SPECIFIC addresses - we extract the SIMILARITY GRAPH
    # as relation edges, NOT as composition centroids.
    
    # Generate additional k-NN semantic edges from centroids if needed
    Write-Host "`nChecking for centroid k-NN edge generation..."
    $edgeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation WHERE source_model = 'centroid_knn'" 2>&1
    if ([int]$edgeCount -eq 0) {
        Write-Host "  Generating k-NN edges from YOUR coordinate system centroids..."
        $knnResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT generate_knn_edges(10, 'centroid_knn')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Created $knnResult semantic edges from centroid proximity" -ForegroundColor Green
        } else {
            Write-Host "  k-NN generation warning: $knnResult" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Already have $edgeCount centroid_knn edges" -ForegroundColor Gray
    }
} else {
    Write-Host "`nModel ingestion failed" -ForegroundColor Red
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    exit 1
}

# NOTE: Don't remove PGPASSWORD - parent scripts may need it
Write-Host "`nIngestion complete" -ForegroundColor Green
