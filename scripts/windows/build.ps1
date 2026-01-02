# Hartonomous Hypercube - Build C++ Components (Windows)
# Usage: .\scripts\windows\build.ps1 [-Clean]

param(
    [switch]$Clean
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
Set-Location $BuildDir

# Configure with CMake
Write-Host "`nConfiguring..."
cmake .. -DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "`nBuilding..."
cmake --build . --config $env:HC_BUILD_TYPE --parallel $env:HC_PARALLEL_JOBS
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}

Set-Location $env:HC_PROJECT_ROOT

Write-Host "`nBuild complete" -ForegroundColor Green
Write-Host "Executables in: $BuildDir"
