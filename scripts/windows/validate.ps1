# Hartonomous Hypercube - Validation Script (Windows)
# ============================================================================
# Validates database state and runs test suite.
# SAFE: Does NOT modify data. Only reads and reports.
#
# Usage:
#   .\validate.ps1             # Quick validation + test suite
#   .\validate.ps1 -Full       # Full validation with benchmarks
#   .\validate.ps1 -Quick      # Just show database state
#   .\validate.ps1 -SkipTests  # Validate state but skip test suite
# ============================================================================

param(
    [switch]$Full,       # Include performance benchmarks
    [switch]$Quick,      # Just show database state, no tests
    [switch]$SkipTests   # Skip test suite
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Hartonomous Hypercube - Validation                       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$env:PGPASSWORD = $env:HC_DB_PASS
$validationPassed = $true

try {
    # ========================================================================
    # STEP 1: CONNECTION TEST
    # ========================================================================
    Write-Host "[1/4] Testing PostgreSQL connection..." -NoNewline
    $connResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Cannot connect to PostgreSQL" -ForegroundColor Red
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green
    
    # ========================================================================
    # STEP 2: DATABASE EXISTS
    # ========================================================================
    Write-Host "[2/4] Checking database '$env:HC_DB_NAME'..." -NoNewline
    $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'"
    if ($dbExists -ne "1") {
        Write-Host " NOT FOUND" -ForegroundColor Red
        Write-Host ""
        Write-Host "Database does not exist. Run: .\scripts\windows\setup-db.ps1" -ForegroundColor Yellow
        exit 1
    }
    Write-Host " EXISTS" -ForegroundColor Green
    
    # ========================================================================
    # STEP 3: DATA STATE
    # ========================================================================
    Write-Host "[3/4] Reading database state..." -ForegroundColor Cyan
    Write-Host ""
    
    $atomCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null).Trim()
    $compCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition" 2>$null).Trim()
    $relCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation" 2>$null).Trim()
    $childCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition_child" 2>$null).Trim()
    $centroidCount = (& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL" 2>$null).Trim()
    
    # Format with padding for alignment
    $atomStr = $atomCount.PadLeft(12)
    $compStr = $compCount.PadLeft(12)
    $relStr = $relCount.PadLeft(12)
    $childStr = $childCount.PadLeft(12)
    $centroidStr = $centroidCount.PadLeft(12)
    
    Write-Host "  ┌─────────────────────────────────────────┐"
    Write-Host "  │  TABLE               │  COUNT           │"
    Write-Host "  ├─────────────────────────────────────────┤"
    
    # Atoms (should be ~1.1M for full Unicode)
    if ([int]$atomCount -ge 1100000) {
        Write-Host "  │  Atoms               │$atomStr  │" -ForegroundColor Green
    } elseif ([int]$atomCount -gt 0) {
        Write-Host "  │  Atoms               │$atomStr  │" -ForegroundColor Yellow
    } else {
        Write-Host "  │  Atoms               │$atomStr  │" -ForegroundColor Red
        $validationPassed = $false
    }
    
    # Compositions
    if ([int]$compCount -gt 0) {
        Write-Host "  │  Compositions        │$compStr  │" -ForegroundColor Green
    } else {
        Write-Host "  │  Compositions        │$compStr  │" -ForegroundColor Yellow
    }
    
    # Relations  
    if ([int]$relCount -gt 0) {
        Write-Host "  │  Relations           │$relStr  │" -ForegroundColor Green
    } else {
        Write-Host "  │  Relations           │$relStr  │" -ForegroundColor Yellow
    }
    
    # Composition Children
    Write-Host "  │  Composition Children│$childStr  │" -ForegroundColor Cyan
    
    # Centroids
    if ([int]$centroidCount -eq [int]$compCount -and [int]$compCount -gt 0) {
        Write-Host "  │  With Centroids      │$centroidStr  │" -ForegroundColor Green
    } elseif ([int]$centroidCount -gt 0) {
        Write-Host "  │  With Centroids      │$centroidStr  │" -ForegroundColor Yellow
    } else {
        Write-Host "  │  With Centroids      │$centroidStr  │" -ForegroundColor DarkGray
    }
    
    Write-Host "  └─────────────────────────────────────────┘"
    Write-Host ""
    
    # ========================================================================
    # STEP 4: TEST SUITE (unless skipped)
    # ========================================================================
    if ($Quick) {
        Write-Host "[4/4] Tests skipped (-Quick)" -ForegroundColor DarkGray
    } elseif ($SkipTests) {
        Write-Host "[4/4] Tests skipped (-SkipTests)" -ForegroundColor DarkGray
    } else {
        Write-Host "[4/4] Running test suite..." -ForegroundColor Cyan
        Write-Host ""
        
        if ($Full) {
            & "$PSScriptRoot\test.ps1"
        } else {
            & "$PSScriptRoot\test.ps1" -Quick
        }
        
        if ($LASTEXITCODE -ne 0) {
            $validationPassed = $false
        }
    }
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    Write-Host ""
    if ($validationPassed) {
        Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "  VALIDATION PASSED" -ForegroundColor Green
        Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Red
        Write-Host "  VALIDATION FAILED" -ForegroundColor Red
        Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Red
        exit 1
    }

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
