# ============================================================================
# OPTIMIZED BUILD - Maximum Performance Configuration
# ============================================================================
# This script builds with all optimizations enabled:
# - OpenMP for parallel loops
# - Intel MKL threaded library
# - AVX2 SIMD
# - Link-time optimization
#
# Usage: .\scripts\windows\build-optimized.ps1 [-Clean] [-Verbose]
# ============================================================================

param(
    [switch]$Clean,
    [switch]$Verbose
)

. "$PSScriptRoot\env.ps1"

if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path "$env:HC_BUILD_DIR") {
        Remove-Item -Recurse -Force "$env:HC_BUILD_DIR"
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " OPTIMIZED BUILD - Full Performance Configuration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$BuildDir = "$env:HC_PROJECT_ROOT\cpp\build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

try {
    # ========================================================================
    # CONFIGURE
    # ========================================================================
    Write-Host "`nConfiguring with CMake..." -ForegroundColor Yellow
    
    # Check for Intel MKL
    $MKLRoot = $env:MKLROOT
    if (-not $MKLRoot) {
        # Try common locations
        $MKLPaths = @(
            "D:\Intel\oneAPI\mkl\latest",
            "C:\Program Files (x86)\Intel\oneAPI\mkl\latest",
            "C:\Program Files\Intel\oneAPI\mkl\latest",
            "$env:USERPROFILE\intel\oneAPI\mkl\latest"
        )
        foreach ($path in $MKLPaths) {
            if (Test-Path "$path\include\mkl.h") {
                $MKLRoot = $path
                break
            }
        }
    }
    
    if ($MKLRoot) {
        Write-Host "  Found Intel MKL: $MKLRoot" -ForegroundColor Green
        $env:MKLROOT = $MKLRoot
    } else {
        Write-Host "  Intel MKL not found - will use fallback BLAS" -ForegroundColor Yellow
    }
    
    # Build CMake arguments
    $cmakeArgs = @(
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=cl",
        "-DCMAKE_CXX_COMPILER=cl"
    )
    
    # Add source directory
    $cmakeArgs += ".."
    
    if ($Verbose) {
        Write-Host "  CMake args: $($cmakeArgs -join ' ')" -ForegroundColor Gray
    }
    
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        exit 1
    }
    
    # ========================================================================
    # BUILD
    # ========================================================================
    Write-Host "`nBuilding..." -ForegroundColor Yellow
    
    $jobs = [Environment]::ProcessorCount
    Write-Host "  Using $jobs parallel jobs" -ForegroundColor Gray
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    & cmake --build . --config Release --parallel $jobs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    
    $stopwatch.Stop()
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " BUILD COMPLETE" -ForegroundColor Green
    Write-Host " Build time: $($stopwatch.Elapsed.ToString('mm\:ss\.fff'))" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    
    # Show what was built
    Write-Host "`nBuilt executables:" -ForegroundColor Yellow
    Get-ChildItem -Filter "*.exe" | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $($_.Name) ($size MB)"
    }
    
    # Verify OpenMP was linked
    Write-Host "`nVerifying OpenMP linkage..." -ForegroundColor Yellow
    $dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
    if ($dumpbin) {
        $deps = & dumpbin /dependents "ingest_safetensor.exe" 2>$null | Select-String -Pattern "vcomp|libiomp|libomp"
        if ($deps) {
            Write-Host "  OpenMP runtime found: $($deps.Matches.Value)" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: No OpenMP runtime detected in dependencies" -ForegroundColor Yellow
        }
    }

} finally {
    Pop-Location
}

Write-Host ""
Write-Host "To run optimized ingestion:" -ForegroundColor Cyan
Write-Host "  .\scripts\windows\ingest-optimized.ps1 <model_path>" -ForegroundColor White
