# ============================================================================
# OPTIMIZED MODEL INGESTION - Maximum Performance
# ============================================================================
# This script configures ALL threading and performance optimizations:
# - OpenMP: All CPU threads
# - Intel MKL: All CPU threads with dynamic disabled
# - NUMA-aware thread affinity
# - Proper database connection
#
# Usage: .\scripts\windows\ingest-optimized.ps1 <model_path> [threshold]
# ============================================================================

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelPath,
    
    [Parameter(Position=1)]
    [float]$Threshold = 0.5
)

# Load environment
. "$PSScriptRoot\env.ps1"

# ============================================================================
# INTEL oneAPI RUNTIME - CRITICAL FOR libiomp5md.dll
# ============================================================================
# The threaded MKL and OpenMP require Intel's OpenMP runtime DLL
$IntelOneAPIBase = "D:\Intel\oneAPI"
if (Test-Path $IntelOneAPIBase) {
    # Add Intel compiler and MKL bin directories to PATH for DLLs
    $env:PATH = "$IntelOneAPIBase\compiler\latest\bin;$IntelOneAPIBase\mkl\latest\bin;$env:PATH"
    Write-Host "Intel oneAPI runtime enabled" -ForegroundColor Green
}

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Get actual hardware thread count
$ThreadCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if (-not $ThreadCount -or $ThreadCount -lt 1) {
    $ThreadCount = [Environment]::ProcessorCount
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " OPTIMIZED INGESTION - $ThreadCount hardware threads" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# OpenMP Configuration
# OpenMP Configuration
$env:OMP_NUM_THREADS = $ThreadCount
$env:OMP_DYNAMIC = "FALSE"           # Don't let OpenMP reduce threads
# Note: OMP_PROC_BIND and OMP_PLACES are ignored by Intel OpenMP (uses KMP_AFFINITY instead)

# Intel MKL Configuration (CRITICAL for cblas_sgemm performance)
$env:MKL_NUM_THREADS = $ThreadCount
$env:MKL_DYNAMIC = "FALSE"           # CRITICAL: Force MKL to use all threads
$env:MKL_THREADING_LAYER = "INTEL"   # Use Intel OpenMP (libiomp5)
$env:KMP_AFFINITY = "granularity=fine,compact,1,0"  # Optimal thread binding

# If MKL not available, ensure we still get threading from other BLAS
$env:OPENBLAS_NUM_THREADS = $ThreadCount
$env:VECLIB_MAXIMUM_THREADS = $ThreadCount

# PostgreSQL connection (libpq uses these)
$env:PGHOST = $env:HC_DB_HOST
$env:PGPORT = $env:HC_DB_PORT
$env:PGDATABASE = $env:HC_DB_NAME
$env:PGUSER = $env:HC_DB_USER
$env:PGPASSWORD = $env:HC_DB_PASS

Write-Host ""
Write-Host "THREADING CONFIGURATION:" -ForegroundColor Yellow
Write-Host "  OMP_NUM_THREADS     = $env:OMP_NUM_THREADS"
Write-Host "  OMP_DYNAMIC         = $env:OMP_DYNAMIC"
Write-Host "  MKL_NUM_THREADS     = $env:MKL_NUM_THREADS"
Write-Host "  MKL_DYNAMIC         = $env:MKL_DYNAMIC"
Write-Host ""
Write-Host "DATABASE:" -ForegroundColor Yellow
Write-Host "  Host: $env:PGHOST"
Write-Host "  Port: $env:PGPORT"
Write-Host "  Database: $env:PGDATABASE"
Write-Host "  User: $env:PGUSER"
Write-Host ""
Write-Host "MODEL:" -ForegroundColor Yellow
Write-Host "  Path: $ModelPath"
Write-Host "  Threshold: $Threshold"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Validate model path
if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model path not found: $ModelPath" -ForegroundColor Red
    exit 1
}

# Find the ingester executable
$IngestExe = "$env:HC_BUILD_DIR\ingest_safetensor.exe"
if (-not (Test-Path $IngestExe)) {
    # Try release subdirectory (Visual Studio generator)
    $IngestExe = "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe"
}
if (-not (Test-Path $IngestExe)) {
    Write-Host "ERROR: ingest_safetensor.exe not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting ingestion..." -ForegroundColor Green
Write-Host ""

# Run with timing
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Get model name from path (last directory component)
$ModelName = Split-Path -Leaf (Split-Path -Parent $ModelPath)
if ($ModelName -eq "snapshots") {
    $ModelName = Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $ModelPath))
}
# Clean up HuggingFace prefix if present
$ModelName = $ModelName -replace '^models--', '' -replace '--', '-'

& $IngestExe $ModelPath -t $Threshold -n $ModelName

$exitCode = $LASTEXITCODE
$stopwatch.Stop()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "INGESTION COMPLETE" -ForegroundColor Green
} else {
    Write-Host "INGESTION FAILED (exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "Total time: $($stopwatch.Elapsed.ToString('mm\:ss\.fff'))" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

exit $exitCode
