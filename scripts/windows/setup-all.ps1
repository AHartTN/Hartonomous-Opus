# Hartonomous Hypercube - Full Setup Pipeline (Windows)
# ============================================================================
# SAFE BY DEFAULT: Only creates/applies schema if missing. Does NOT destroy data.
# Use -Reset flag ONLY for true greenfield setup.
#
# Pipeline:
#   1. Clean build artifacts (optional)
#   2. Build/Compile all C/C++ 
#   3. Install extensions to PostgreSQL
#   4. Create database + apply schema (idempotent - safe to re-run)
#   5. Seed Unicode atoms (if not already seeded)
#   6. Ingest test content (Moby Dick)
#   7. Run test suite
#
# Usage:
#   .\setup-all.ps1                     # Safe: preserves existing data
#   .\setup-all.ps1 -Reset              # DESTRUCTIVE: drops database first
#   .\setup-all.ps1 -SkipClean          # Keep build artifacts
#   .\setup-all.ps1 -SkipBuild          # Skip C++ compilation
#   .\setup-all.ps1 -SkipIngest         # Skip data ingestion
#   .\setup-all.ps1 -SkipTests          # Skip test suite
# ============================================================================

param(
    [switch]$Reset,       # DESTRUCTIVE: Drop and recreate database
    [switch]$SkipClean,
    [switch]$SkipBuild,
    [switch]$SkipIngest,
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
# STEP 3: DATABASE SETUP (idempotent unless -Reset specified)
# ============================================================================
Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
if ($Reset) {
    Write-Host "│ STEP 3/5: DATABASE SETUP (GREENFIELD - DESTRUCTIVE)              │" -ForegroundColor Red
} else {
    Write-Host "│ STEP 3/5: DATABASE SETUP (SAFE - PRESERVING DATA)                │" -ForegroundColor Cyan
}
Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan

if ($Reset) {
    Write-Host ""
    Write-Host "!!! -Reset flag specified. Database will be dropped and recreated !!!" -ForegroundColor Red
    Write-Host ""
    & "$PSScriptRoot\setup-db.ps1" -Reset
} else {
    & "$PSScriptRoot\setup-db.ps1"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Database setup failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ============================================================================
# STEP 4: INGEST TEST DATA
# ============================================================================
if (-not $SkipIngest) {
    Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│ STEP 4/5: INGESTING TEST DATA                                    │" -ForegroundColor Cyan
    Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
    
    & "$PSScriptRoot\ingest-testdata.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Ingestion had issues (continuing anyway)" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host "── Skipping ingest (--SkipIngest) ──" -ForegroundColor DarkGray
}

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
    (SELECT COUNT(*) FROM composition WHERE label IS NOT NULL) as "With Labels",
    (SELECT COUNT(*) FROM relation) as "Relations",
    (SELECT MAX(depth) FROM composition) as "Max Depth",
    pg_size_pretty(
        pg_total_relation_size('atom') + 
        pg_total_relation_size('composition') +
        pg_total_relation_size('relation')
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
