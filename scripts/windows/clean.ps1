# Hartonomous Hypercube - Clean Build Artifacts (Windows)
# Usage: .\scripts\windows\clean.ps1

. "$PSScriptRoot\env.ps1"

Write-Host "=== Cleaning Build Artifacts ===" -ForegroundColor Cyan

# C++ build directory
if (Test-Path "$env:HC_BUILD_DIR") {
    Write-Host "Removing: $env:HC_BUILD_DIR"
    Remove-Item -Recurse -Force "$env:HC_BUILD_DIR" -ErrorAction SilentlyContinue
}

# CMake cache files
Get-ChildItem -Path "$env:HC_PROJECT_ROOT\cpp" -Filter "CMakeCache.txt" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "$env:HC_PROJECT_ROOT\cpp" -Filter "CMakeFiles" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Compiled objects
Get-ChildItem -Path "$env:HC_PROJECT_ROOT" -Include "*.obj","*.o","*.lib","*.a","*.dll","*.so","*.exe" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Clean complete" -ForegroundColor Green

# Ensure clean always succeeds
$LASTEXITCODE = 0
