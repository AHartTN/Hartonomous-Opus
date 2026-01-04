# Hartonomous Hypercube - Full Setup Pipeline (Windows)
# ============================================================================
# Runs the complete setup from clean slate to working LLM-like system:
#   1. Clean build artifacts
#   2. Build/Compile all C/C++
#   3. Install extensions to PostgreSQL
#   4. Drop database (greenfield)
#   5. Create database + schema + extensions
#   6. Seed Unicode atoms
#   7. Ingest embedding model (MiniLM)
#   8. Ingest test content (Moby Dick + images + audio)
#   9. Run full test suite including AI/ML operations
#
# Usage: .\scripts\windows\setup-all.ps1 [-SkipClean] [-SkipBuild] [-SkipTests]
# ============================================================================

param(
    [switch]$SkipClean,
    [switch]$SkipBuild,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║       HARTONOMOUS HYPERCUBE - FULL SETUP PIPELINE                ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  Project:  $env:HC_PROJECT_ROOT"
Write-Host ""

$steps = @(
    @{ Name = "Clean"; Skip = $SkipClean },
    @{ Name = "Build"; Skip = $SkipBuild },
    @{ Name = "Database"; Skip = $false },
    @{ Name = "Ingest"; Skip = $false },
    @{ Name = "Tests"; Skip = $SkipTests }
)

# ============================================================================
# STEP 1: CLEAN
# ============================================================================
if (-not $SkipClean) {
    Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│ STEP 1/5: CLEANING BUILD ARTIFACTS                               │" -ForegroundColor Cyan
    Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
    
    & "$PSScriptRoot\clean.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Clean failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "── Skipping clean (--SkipClean) ──" -ForegroundColor DarkGray
}

# ============================================================================
# STEP 2: BUILD
# ============================================================================
if (-not $SkipBuild) {
    Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│ STEP 2/5: BUILDING C/C++ COMPONENTS                              │" -ForegroundColor Cyan
    Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
    
    & "$PSScriptRoot\build.ps1" -Install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "── Skipping build (--SkipBuild) ──" -ForegroundColor DarkGray
}

# ============================================================================
# STEP 3: DATABASE SETUP (DROP + CREATE + SCHEMA + SEED)
# ============================================================================
Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "│ STEP 3/5: DATABASE SETUP (GREENFIELD)                            │" -ForegroundColor Cyan
Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan

# Drop and recreate for clean slate
& "$PSScriptRoot\setup-db.ps1" -Reset
if ($LASTEXITCODE -ne 0) {
    Write-Host "Database setup failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================================
# STEP 4: INGEST ALL TEST DATA
# ============================================================================
Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "│ STEP 4/5: INGESTING TEST DATA                                    │" -ForegroundColor Cyan
Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan

& "$PSScriptRoot\ingest-testdata.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Ingestion had issues (continuing anyway)" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================================
# STEP 5: RUN FULL TEST SUITE
# ============================================================================
if (-not $SkipTests) {
    Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│ STEP 5/5: RUNNING TEST SUITE                                     │" -ForegroundColor Cyan
    Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
    
    & "$PSScriptRoot\test.ps1"
    $testExitCode = $LASTEXITCODE
    Write-Host ""
} else {
    Write-Host "── Skipping tests (--SkipTests) ──" -ForegroundColor DarkGray
    $testExitCode = 0
}

# ============================================================================
# SUMMARY
# ============================================================================
$duration = (Get-Date) - $startTime

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║                      PIPELINE COMPLETE                           ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Duration: $($duration.TotalSeconds.ToString('F1')) seconds"
Write-Host ""

# Final stats
$env:PGPASSWORD = $env:HC_DB_PASS
try {
    Write-Host "  Final Database State:" -ForegroundColor Green
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c @"
SELECT 
    (SELECT COUNT(*) FROM atom) as "Atoms",
    (SELECT COUNT(*) FROM composition) as "Compositions",
    (SELECT COUNT(*) FROM relation) as "Relations",
    (SELECT COUNT(*) FROM shape) as "Shapes",
    (SELECT MAX(depth) FROM composition) as "Max Depth",
    pg_size_pretty(
        pg_total_relation_size('atom') + 
        pg_total_relation_size('composition') +
        pg_total_relation_size('relation') +
        pg_total_relation_size('shape')
    ) as "Total Size"
"@
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host ""
if ($testExitCode -eq 0) {
    Write-Host "  ✓ All systems operational!" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Some tests failed (exit code: $testExitCode)" -ForegroundColor Yellow
}
Write-Host ""

exit $testExitCode
