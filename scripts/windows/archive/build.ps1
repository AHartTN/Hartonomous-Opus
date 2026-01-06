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
    
    # Build argument list - use Visual Studio generator for proper MSVC compilation
    # PostgreSQL extensions require MSVC on Windows (Clang in GNU mode doesn't work with PG headers)
    $cmakeArgs = @("-DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE", "..")
    
    # Prefer Ninja with MSVC toolchain for faster builds
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        # Find cl.exe from VS environment
        $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
        if ($clPath) {
            $cmakeArgs = @(
                "-G", "Ninja",
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl"
            ) + $cmakeArgs
        } else {
            # Fallback to Visual Studio generator (slower but reliable)
            $cmakeArgs = @("-G", "Visual Studio 17 2022", "-A", "x64") + $cmakeArgs
        }
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
        
        # Use CMake install target (installs ALL extensions properly)
        Write-Host "`nRunning CMake install..."
        & cmake --install . --config $env:HC_BUILD_TYPE
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CMake install failed, falling back to manual copy..." -ForegroundColor Yellow
            
            # Manual fallback - ALL extension DLLs
            $dllsToCopy = @(
                "hypercube_c.dll",
                "hypercube.dll",
                "hypercube_ops.dll",
                "semantic_ops.dll",
                "embedding_c.dll",
                "embedding_ops.dll",
                "generative_c.dll",
                "generative.dll"
            )
            
            Write-Host "`nCopying extension DLLs..."
            foreach ($dll in $dllsToCopy) {
                if (Test-Path $dll) {
                    Copy-Item $dll "$pgLibDir\" -Force
                    Write-Host "  $dll" -ForegroundColor Green
                } else {
                    Write-Host "  $dll (not found, skipping)" -ForegroundColor Yellow
                }
            }
            
            # ALL SQL and control files
            $sqlFiles = @(
                @{sql="..\sql\hypercube--1.0.sql"; ctrl="..\hypercube.control"},
                @{sql="..\sql\hypercube_ops--1.0.sql"; ctrl="..\sql\hypercube_ops.control"},
                @{sql="..\sql\semantic_ops--1.0.sql"; ctrl="..\sql\semantic_ops.control"},
                @{sql="..\sql\embedding_ops--1.0.sql"; ctrl="..\sql\embedding_ops.control"},
                @{sql="..\sql\generative--1.0.sql"; ctrl="..\sql\generative.control"}
            )
            
            Write-Host "`nCopying extension metadata..."
            foreach ($ext in $sqlFiles) {
                if (Test-Path $ext.sql) {
                    Copy-Item $ext.sql "$pgExtDir\" -Force
                    Write-Host "  $(Split-Path -Leaf $ext.sql)" -ForegroundColor Green
                }
                if (Test-Path $ext.ctrl) {
                    Copy-Item $ext.ctrl "$pgExtDir\" -Force
                    Write-Host "  $(Split-Path -Leaf $ext.ctrl)" -ForegroundColor Green
                }
            }
        } else {
            Write-Host "  CMake install completed" -ForegroundColor Green
        }
        
        Write-Host "`n=== Extensions Installed ===" -ForegroundColor Green
        Write-Host "Installed extensions: hypercube, hypercube_ops, semantic_ops, embedding_ops, generative"
        Write-Host "Run setup-db.ps1 to load into database"
    }
    
} finally {
    Pop-Location
}

Write-Host "`nDone" -ForegroundColor Green
