# Hartonomous Hypercube - Full Validation (Windows)
# End-to-end validation: build, setup, test
# Usage: .\scripts\windows\validate.ps1 [-Full] [-SkipBuild] [-SkipSetup]

param(
    [switch]$Full,       # Include performance benchmarks
    [switch]$SkipBuild,  # Skip rebuild if already built
    [switch]$SkipSetup   # Skip DB setup if already done
)

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Hartonomous Hypercube - Full Validation                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$ErrorCount = 0

# Step 1: Build (optional)
if (-not $SkipBuild) {
    Write-Host "Step 1: Build" -ForegroundColor Yellow
    & "$PSScriptRoot\build.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed - aborting" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Step 1: Build (skipped)" -ForegroundColor DarkGray
}

# Step 2: Database Setup (optional)
if (-not $SkipSetup) {
    Write-Host "`nStep 2: Database Setup" -ForegroundColor Yellow
    & "$PSScriptRoot\setup-db.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Database setup failed - aborting" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`nStep 2: Database Setup (skipped)" -ForegroundColor DarkGray
}

# Step 3: Run Tests
Write-Host "`nStep 3: Test Suite" -ForegroundColor Yellow
if ($Full) {
    & "$PSScriptRoot\test.ps1"
} else {
    & "$PSScriptRoot\test.ps1" -Quick
}
$ErrorCount = $LASTEXITCODE

# Summary
Write-Host ""
if ($ErrorCount -eq 0) {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  VALIDATION PASSED" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
} else {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host "  VALIDATION FAILED ($ErrorCount errors)" -ForegroundColor Red
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Red
}

exit $ErrorCount
