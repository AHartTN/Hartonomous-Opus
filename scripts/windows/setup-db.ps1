# Hartonomous Hypercube - Database Setup (Windows)
# ============================================================================
# Idempotent: Creates/fixes whatever is missing. Run as many times as needed.
#
# Usage:
#   .\setup-db.ps1              # Set up everything that's missing
#   .\setup-db.ps1 -Reset       # DESTRUCTIVE: drops and recreates database
# ============================================================================

param(
    [switch]$Reset      # DESTRUCTIVE: Drop and recreate database
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "=== Hypercube Database Setup ===" -ForegroundColor Cyan
Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  User: $env:HC_DB_USER"
Write-Host ""

$env:PGPASSWORD = $env:HC_DB_PASS

try {
    # ========================================================================
    # CONNECTION TEST
    # ========================================================================
    Write-Host "[1/5] Testing PostgreSQL connection..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Cannot connect: $result" -ForegroundColor Red
        Write-Host "  Check: PostgreSQL running, credentials in scripts/config.env, pg_hba.conf" -ForegroundColor Yellow
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # RESET (DESTRUCTIVE - only if explicitly requested)
    # ========================================================================
    if ($Reset) {
        Write-Host ""
        Write-Host "!!! DESTRUCTIVE: Dropping database $env:HC_DB_NAME !!!" -ForegroundColor Red
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME" 2>&1 | Out-Null
        Write-Host "  Database dropped" -ForegroundColor Yellow
        Write-Host ""
    }

    # ========================================================================
    # DATABASE CREATION
    # ========================================================================
    Write-Host "[2/5] Database..." -NoNewline
    $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'"

    if ($dbExists -ne "1") {
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED to create" -ForegroundColor Red
            exit 1
        }
        Write-Host " created" -ForegroundColor Green
    } else {
        Write-Host " exists" -ForegroundColor Green
    }

    # ========================================================================
    # SCHEMA
    # ========================================================================
    Write-Host "[3/5] Schema..." -NoNewline
    $sqlDir = Join-Path $env:HC_PROJECT_ROOT "sql"
    Push-Location $sqlDir
    try {
        $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -v ON_ERROR_STOP=1 -f "hypercube_schema.sql" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # C++ EXTENSIONS
    # ========================================================================
    Write-Host "[4/5] Extensions..." -NoNewline
    $extensions = @("hypercube", "hypercube_ops", "embedding_ops", "semantic_ops", "generative")
    $loaded = 0
    $failed = @()

    foreach ($ext in $extensions) {
        $extResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $ext;" 2>&1
        if ($LASTEXITCODE -eq 0 -and -not ($extResult -match "ERROR:")) {
            $loaded++
        } else {
            $failed += $ext
        }
    }

    if ($failed.Count -eq 0) {
        Write-Host " $loaded loaded" -ForegroundColor Green
    } else {
        Write-Host " $loaded loaded, missing: $($failed -join ', ')" -ForegroundColor Yellow
        Write-Host "  Run build.ps1 to compile and install extensions" -ForegroundColor Yellow
    }

    # ========================================================================
    # ATOM SEEDING
    # ========================================================================
    Write-Host "[5/5] Atoms..." -NoNewline
    $atomCount = [int](& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null)

    if ($atomCount -ge 1100000) {
        Write-Host " $atomCount (complete)" -ForegroundColor Green
    } else {
        Write-Host " $atomCount (need ~1.1M, seeding...)" -ForegroundColor Yellow

        $seeder = "$env:HC_BIN_DIR\seed_atoms_parallel.exe"
        if (Test-Path $seeder) {
            Write-Host "  Using: $seeder"
            & $seeder -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  Standalone seeder failed, trying extension..." -ForegroundColor Yellow
                $seeder = $null
            }
        }

        if (-not (Test-Path $seeder)) {
            Write-Host "  Using: SELECT seed_atoms();"
            $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT seed_atoms();" 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  Extension seeder failed: $result" -ForegroundColor Red
                Write-Host "  Ensure extensions are installed (run build.ps1)" -ForegroundColor Yellow
                exit 1
            }
        }

        $newCount = [int](& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null)
        if ($newCount -ge 1100000) {
            Write-Host "  Seeded $newCount atoms" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: Only $newCount atoms seeded (expected ~1.1M)" -ForegroundColor Red
            exit 1
        }
    }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    Write-Host ""
    Write-Host "=== Database Ready ===" -ForegroundColor Green

    $stats = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT * FROM db_stats()" 2>$null
    if ($stats) {
        $s = $stats -split '\|'
        Write-Host "  Atoms: $($s[0].Trim())  Compositions: $($s[1].Trim())  Relations: $($s[3].Trim())" -ForegroundColor Cyan
    }
    Write-Host ""

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
