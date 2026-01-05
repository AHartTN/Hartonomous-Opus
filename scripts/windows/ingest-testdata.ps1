# Hartonomous Hypercube - Test Data Ingestion (Windows)
# ============================================================================
# Ingests test content from test-data/ directory AND configured model paths.
# SAFE: Idempotent - skips already-ingested content.
#
# For full model ingestion from D:\Models, use: .\ingest-models.ps1
#
# Usage:
#   .\ingest-testdata.ps1             # Ingest text + test models
#   .\ingest-testdata.ps1 -Quick      # Skip model ingestion (faster)
#   .\ingest-testdata.ps1 -AllModels  # Ingest ALL models from D:\Models
# ============================================================================

param(
    [switch]$Quick,      # Skip model ingestion for faster iteration
    [switch]$AllModels   # Ingest all models from D:\Models (not just test-data)
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

$env:PGPASSWORD = $env:HC_DB_PASS
$testDataDir = "$env:HC_PROJECT_ROOT\test-data"

if (-not (Test-Path $testDataDir)) {
    Write-Host "test-data directory not found: $testDataDir" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Hypercube Test Data Ingestion ===" -ForegroundColor Cyan
Write-Host "  Source: $testDataDir"
Write-Host ""

# Find ingestion tool
$ingester = if (Test-Path "$env:HC_BUILD_DIR\ingest.exe") {
    "$env:HC_BUILD_DIR\ingest.exe"
} elseif (Test-Path "$env:HC_BUILD_DIR\Release\ingest.exe") {
    "$env:HC_BUILD_DIR\Release\ingest.exe"
} else { $null }

$dbArgs = @("-d", $env:HC_DB_NAME, "-U", $env:HC_DB_USER, "-h", $env:HC_DB_HOST, "-p", $env:HC_DB_PORT)

# ============================================================================
# 1. TEXT FILES - Universal Ingestion
# ============================================================================
Write-Host "[1/3] Text Files" -ForegroundColor Yellow

$textFiles = Get-ChildItem -Path $testDataDir -Filter "*.txt" -File
if ($textFiles.Count -gt 0 -and $ingester) {
    foreach ($file in $textFiles) {
        Write-Host "      $($file.Name)..." -NoNewline
        $beforeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition"
        
        & $ingester @dbArgs $file.FullName 2>&1 | Out-Null
        
        $afterCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition"
        $newComps = [int]$afterCount - [int]$beforeCount
        
        if ($newComps -gt 0) {
            Write-Host " +$newComps compositions" -ForegroundColor Green
        } else {
            Write-Host " (already ingested)" -ForegroundColor DarkGray
        }
    }
} elseif (-not $ingester) {
    Write-Host "      WARNING: ingest.exe not found - run build.ps1 first" -ForegroundColor Yellow
} else {
    Write-Host "      (no text files found)" -ForegroundColor DarkGray
}

# ============================================================================
# 2. EMBEDDING MODELS
# ============================================================================
Write-Host ""
Write-Host "[2/3] Embedding Models" -ForegroundColor Yellow

if ($Quick) {
    Write-Host "      (skipped with -Quick flag)" -ForegroundColor DarkGray
} else {
    # Re-set PGPASSWORD in case child scripts cleared it
    $env:PGPASSWORD = $env:HC_DB_PASS
    
    $ingestModelsScript = "$PSScriptRoot\ingest-models.ps1"
    if (-not (Test-Path $ingestModelsScript)) {
        Write-Host "      WARNING: ingest-models.ps1 not found" -ForegroundColor Yellow
    } else {
        if ($AllModels) {
            # Ingest from all configured model paths including D:\Models
            Write-Host "      Ingesting ALL models (D:\Models + test-data)..." -ForegroundColor Cyan
            & $ingestModelsScript -Path "D:\Models" -Type "embedding"
        } else {
            # Only test-data models for quick testing
            $testModelDir = "$testDataDir\embedding_models"
            if (Test-Path $testModelDir) {
                Write-Host "      Ingesting test models from: $testModelDir"
                & $ingestModelsScript -Path $testModelDir
            } else {
                Write-Host "      (no test models found in test-data)" -ForegroundColor DarkGray
            }
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      Model ingestion complete" -ForegroundColor Green
        }
    }
}

# Re-set PGPASSWORD in case child scripts cleared it
$env:PGPASSWORD = $env:HC_DB_PASS

# ============================================================================
# 3. SUMMARY
# ============================================================================
Write-Host ""
Write-Host "[3/3] Final State" -ForegroundColor Cyan

$atomCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom").Trim()
$compCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition").Trim()
$relCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation").Trim()
$centroidCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL").Trim()

Write-Host ""
Write-Host "  Atoms:        $atomCount" -ForegroundColor Cyan
Write-Host "  Compositions: $compCount" -ForegroundColor Cyan
Write-Host "  Relations:    $relCount" -ForegroundColor Cyan
Write-Host "  Centroids:    $centroidCount" -ForegroundColor Cyan
Write-Host ""
Write-Host "=== Ingestion Complete ===" -ForegroundColor Green

Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
