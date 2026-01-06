# ============================================================================
# Hartonomous Hypercube - Model Ingestion (Windows)
# ============================================================================
# Ingests HuggingFace models with optimized parallel processing using all CPUs.
# Uses Intel MKL threading and OpenMP for maximum performance.
#
# Usage:
#   .\scripts\windows\ingest.ps1 <model_path>              # Ingest a model
#   .\scripts\windows\ingest.ps1 <model_path> -Threshold 0.3
#   .\scripts\windows\ingest.ps1 <model_path> -Name "mymodel"
#
# Examples:
#   .\scripts\windows\ingest.ps1 "D:\Models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\abc123"
#   .\scripts\windows\ingest.ps1 ".\test-data\embedding_models\minilm"
# ============================================================================

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelPath,
    
    [Parameter(Position=1)]
    [Alias("t")]
    [float]$Threshold = 0.5,
    
    [Alias("n")]
    [string]$Name
)

$ErrorActionPreference = "Stop"

# Load environment
. "$PSScriptRoot\env.ps1"

# ============================================================================
# VALIDATE INPUT
# ============================================================================

if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model path not found: $ModelPath" -ForegroundColor Red
    exit 1
}

# Resolve to absolute path
$ModelPath = (Resolve-Path $ModelPath).Path

# ============================================================================
# AUTO-DETECT MODEL NAME
# ============================================================================

if (-not $Name) {
    # Extract model name from HuggingFace cache path structure
    # e.g., models--sentence-transformers--all-MiniLM-L6-v2/snapshots/abc123
    $pathParts = $ModelPath -split '[/\\]'
    $snapshotIdx = [Array]::IndexOf($pathParts, "snapshots")
    
    if ($snapshotIdx -gt 0) {
        $Name = $pathParts[$snapshotIdx - 1]
    } else {
        $Name = Split-Path -Leaf $ModelPath
    }
    
    # Clean up HuggingFace prefix
    $Name = $Name -replace '^models--', '' -replace '--', '-'
}

# ============================================================================
# THREADING CONFIGURATION
# ============================================================================

$ThreadCount = [Environment]::ProcessorCount

# OpenMP
$env:OMP_NUM_THREADS = $ThreadCount
$env:OMP_DYNAMIC = "FALSE"

# Intel MKL
$env:MKL_NUM_THREADS = $ThreadCount
$env:MKL_DYNAMIC = "FALSE"
$env:MKL_THREADING_LAYER = "INTEL"
$env:KMP_AFFINITY = "granularity=fine,compact,1,0"
$env:KMP_WARNINGS = "0"  # Suppress KMP_AFFINITY override warnings

# Fallback BLAS implementations
$env:OPENBLAS_NUM_THREADS = $ThreadCount
$env:VECLIB_MAXIMUM_THREADS = $ThreadCount

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

$env:PGHOST = $env:HC_DB_HOST
$env:PGPORT = $env:HC_DB_PORT
$env:PGDATABASE = $env:HC_DB_NAME
$env:PGUSER = $env:HC_DB_USER
$env:PGPASSWORD = $env:HC_DB_PASS

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Hartonomous Hypercube - Model Ingestion" -ForegroundColor Cyan
Write-Host " $ThreadCount hardware threads" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model:" -ForegroundColor Yellow
Write-Host "  Name:      $Name"
Write-Host "  Path:      $ModelPath"
Write-Host "  Threshold: $Threshold"
Write-Host ""
Write-Host "Database:" -ForegroundColor Yellow
Write-Host "  $env:PGUSER@$env:PGHOST`:$env:PGPORT/$env:PGDATABASE"
Write-Host ""
Write-Host "Threading:" -ForegroundColor Yellow
Write-Host "  OpenMP:    $env:OMP_NUM_THREADS threads"
Write-Host "  MKL:       $env:MKL_NUM_THREADS threads (dynamic=$env:MKL_DYNAMIC)"
Write-Host ""

# ============================================================================
# FIND INGESTER
# ============================================================================

$IngestExe = "$env:HC_BUILD_DIR\ingest_safetensor.exe"
if (-not (Test-Path $IngestExe)) {
    $IngestExe = "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe"
}
if (-not (Test-Path $IngestExe)) {
    Write-Host "ERROR: ingest_safetensor.exe not found." -ForegroundColor Red
    Write-Host "       Run: .\scripts\windows\build.ps1" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# RUN INGESTION
# ============================================================================

Write-Host "Starting ingestion..." -ForegroundColor Green
Write-Host ""

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

& $IngestExe $ModelPath -t $Threshold -n $Name

$exitCode = $LASTEXITCODE
$stopwatch.Stop()

# ============================================================================
# RESULTS
# ============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host " INGESTION COMPLETE" -ForegroundColor Green
} else {
    Write-Host " INGESTION FAILED (exit code: $exitCode)" -ForegroundColor Red
}
Write-Host " Time: $($stopwatch.Elapsed.ToString('mm\:ss\.fff'))" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

exit $exitCode
