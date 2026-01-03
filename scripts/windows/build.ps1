# Hartonomous Hypercube - Build C++ Components (Windows)
# Usage: .\scripts\windows\build.ps1 [-Clean] [-Install]

param(
    [switch]$Clean,
    [switch]$Install
)

. "$PSScriptRoot\env.ps1"

if ($Clean) {
    & "$PSScriptRoot\clean.ps1"
}

Write-Host "=== Building Hypercube C++ ===" -ForegroundColor Cyan
Write-Host "Build type: $env:HC_BUILD_TYPE"
Write-Host "Parallel jobs: $env:HC_PARALLEL_JOBS"

$BuildDir = "$env:HC_PROJECT_ROOT\cpp\build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

try {
    # Configure with CMake
    Write-Host "`nConfiguring..."
    $cmakeArgs = @("-DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE", "..")
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $cmakeArgs = @("-G", "Ninja") + $cmakeArgs
    }
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed" -ForegroundColor Red
        exit 1
    }

    # Build
    Write-Host "`nBuilding..."
    & cmake --build . --config $env:HC_BUILD_TYPE --parallel $env:HC_PARALLEL_JOBS
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "`n=== Build Complete ===" -ForegroundColor Green
    
    # Show built artifacts
    Write-Host "`nExecutables:"
    Get-ChildItem -Filter "*.exe" -Recurse | Where-Object { $_.Directory.Name -notmatch "CMakeFiles" } | Select-Object Name
    
    Write-Host "`nExtensions:"
    Get-ChildItem -Filter "*.dll" | Where-Object { $_.Name -match "^(hypercube|semantic)" } | Select-Object Name
    
    # Install extensions if requested
    if ($Install) {
        Write-Host "`n=== Installing PostgreSQL Extensions ===" -ForegroundColor Cyan
        
        $pgConfig = Get-Command pg_config -ErrorAction SilentlyContinue
        if (-not $pgConfig) {
            Write-Host "pg_config not found. Cannot install extensions." -ForegroundColor Red
            exit 1
        }
        
        $pgLibDir = & pg_config --pkglibdir
        $pgShareDir = & pg_config --sharedir
        $pgExtDir = "$pgShareDir\extension"
        
        Write-Host "Target lib dir: $pgLibDir"
        Write-Host "Target ext dir: $pgExtDir"
        
        # Copy DLLs
        Write-Host "`nCopying extension DLLs..."
        Copy-Item "hypercube.dll" "$pgLibDir\" -Force
        Copy-Item "semantic_ops.dll" "$pgLibDir\" -Force
        Write-Host "  hypercube.dll" -ForegroundColor Green
        Write-Host "  semantic_ops.dll" -ForegroundColor Green
        
        # Copy SQL and control files
        Write-Host "`nCopying extension metadata..."
        Copy-Item "..\sql\hypercube--1.0.sql" "$pgExtDir\" -Force
        Copy-Item "..\hypercube.control" "$pgExtDir\" -Force
        Copy-Item "..\sql\semantic_ops--1.0.sql" "$pgExtDir\" -Force
        Copy-Item "..\sql\semantic_ops.control" "$pgExtDir\" -Force
        Write-Host "  SQL and control files installed" -ForegroundColor Green
        
        Write-Host "`n=== Extensions Installed ===" -ForegroundColor Green
        Write-Host "Run setup-db.ps1 to load into database"
    }
    
} finally {
    Pop-Location
}

Write-Host "`nDone" -ForegroundColor Green
