# ============================================================================
# Hartonomous Hypercube - Build C++ Components (Windows)
# ============================================================================
# Optimized build with Intel oneAPI (MKL + OpenMP) and MSVC
#
# Usage:
#   .\scripts\windows\build.ps1                 # Standard build
#   .\scripts\windows\build.ps1 -Clean          # Clean rebuild
#   .\scripts\windows\build.ps1 -Install        # Install PostgreSQL extensions
#   .\scripts\windows\build.ps1 -Verbose        # Show detailed output
# ============================================================================

param(
    [switch]$Clean,
    [switch]$Install,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Load environment
. "$PSScriptRoot\env.ps1"

if ($Clean) {
    & "$PSScriptRoot\clean.ps1"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Hartonomous Hypercube - Build" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Build Type:     $env:HC_BUILD_TYPE"
Write-Host "  Parallel Jobs:  $env:HC_PARALLEL_JOBS"
Write-Host "  Project Root:   $env:HC_PROJECT_ROOT"
Write-Host ""

# ============================================================================
# CHECK INTEL MKL
# ============================================================================

$MKLRoot = $null
$MKLPaths = @(
    "D:\Intel\oneAPI\mkl\latest",
    "$env:MKLROOT",
    "C:\Program Files (x86)\Intel\oneAPI\mkl\latest",
    "C:\Program Files\Intel\oneAPI\mkl\latest"
)

foreach ($path in $MKLPaths) {
    if ($path -and (Test-Path "$path\include\mkl.h")) {
        $MKLRoot = $path
        break
    }
}

if ($MKLRoot) {
    Write-Host "Intel MKL:        $MKLRoot" -ForegroundColor Green
    $env:MKLROOT = $MKLRoot
} else {
    Write-Host "Intel MKL:        Not found (using fallback BLAS)" -ForegroundColor Yellow
}

# ============================================================================
# CMAKE CONFIGURE
# ============================================================================

$BuildDir = "$env:HC_PROJECT_ROOT\cpp\build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

try {
    Write-Host ""
    Write-Host "Configuring with CMake..." -ForegroundColor Yellow
    
    # Check for Ninja and cl.exe
    $hasNinja = Get-Command ninja -ErrorAction SilentlyContinue
    $hasCL = Get-Command cl.exe -ErrorAction SilentlyContinue
    
    $cmakeArgs = @()
    
    if ($hasNinja -and $hasCL) {
        # Ninja with MSVC (fast)
        $cmakeArgs = @(
            "-G", "Ninja",
            "-DCMAKE_C_COMPILER=cl",
            "-DCMAKE_CXX_COMPILER=cl",
            "-DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE"
        )
    } elseif ($hasCL) {
        # Visual Studio generator (slower but works)
        $cmakeArgs = @(
            "-G", "Visual Studio 17 2022",
            "-A", "x64",
            "-DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE"
        )
    } else {
        Write-Host "ERROR: MSVC (cl.exe) not found. Run from Developer Command Prompt." -ForegroundColor Red
        exit 1
    }
    
    $cmakeArgs += ".."
    
    if ($Verbose) {
        Write-Host "  Args: $($cmakeArgs -join ' ')" -ForegroundColor Gray
    }
    
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        exit 1
    }
    
    # ========================================================================
    # BUILD
    # ========================================================================
    
    Write-Host ""
    Write-Host "Building with $env:HC_PARALLEL_JOBS parallel jobs..." -ForegroundColor Yellow
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    & cmake --build . --config $env:HC_BUILD_TYPE --parallel $env:HC_PARALLEL_JOBS
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    
    $stopwatch.Stop()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " BUILD COMPLETE" -ForegroundColor Green
    Write-Host " Time: $($stopwatch.Elapsed.ToString('mm\:ss\.fff'))" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "Executables:" -ForegroundColor Yellow
    Get-ChildItem -Filter "*.exe" -Recurse | 
        Where-Object { $_.Directory.Name -notmatch "CMakeFiles" } |
        ForEach-Object {
            $size = [math]::Round($_.Length / 1MB, 2)
            Write-Host "  $($_.Name) ($size MB)"
        }
    
    Write-Host ""
    Write-Host "Extensions:" -ForegroundColor Yellow
    Get-ChildItem -Filter "*.dll" | 
        Where-Object { $_.Name -match "^(hypercube|semantic|embedding|generative)" } |
        ForEach-Object { Write-Host "  $($_.Name)" }
    
    # Verify OpenMP linkage
    $dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
    if ($dumpbin -and (Test-Path "ingest_safetensor.exe")) {
        $deps = & dumpbin /dependents "ingest_safetensor.exe" 2>$null | Select-String -Pattern "vcomp|libiomp|libomp"
        if ($deps) {
            Write-Host ""
            Write-Host "OpenMP Runtime: $($deps.Matches.Value -join ', ')" -ForegroundColor Green
        }
    }
    
    # ========================================================================
    # INSTALL EXTENSIONS
    # ========================================================================
    
    if ($Install) {
        Write-Host ""
        Write-Host "Installing PostgreSQL Extensions..." -ForegroundColor Yellow
        
        $pgConfig = Get-Command pg_config -ErrorAction SilentlyContinue
        if (-not $pgConfig) {
            Write-Host "pg_config not found. Cannot install extensions." -ForegroundColor Red
            exit 1
        }
        
        $pgLibDir = & pg_config --pkglibdir
        $pgShareDir = & pg_config --sharedir
        $pgExtDir = "$pgShareDir\extension"
        
        Write-Host "  Target: $pgLibDir"
        
        $extensions = @("hypercube", "semantic_ops", "hypercube_ops", "embedding_ops", "generative")
        foreach ($ext in $extensions) {
            if (Test-Path "$ext.dll") {
                Copy-Item "$ext.dll" "$pgLibDir\" -Force
                Write-Host "  Installed: $ext.dll" -ForegroundColor Green
            }
            if (Test-Path "..\sql\$ext--1.0.sql") {
                Copy-Item "..\sql\$ext--1.0.sql" "$pgExtDir\" -Force
            }
            if (Test-Path "..\sql\$ext.control") {
                Copy-Item "..\sql\$ext.control" "$pgExtDir\" -Force
            }
        }
        
        # Copy C library
        if (Test-Path "hypercube_c.dll") {
            Copy-Item "hypercube_c.dll" "$pgLibDir\" -Force
            Write-Host "  Installed: hypercube_c.dll" -ForegroundColor Green
        }
    }

} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  .\scripts\windows\setup-db.ps1       # Setup database schema"
Write-Host "  .\scripts\windows\ingest.ps1 <path>  # Ingest a model"
