# Hartonomous Hypercube - Database Setup (Windows)
# ============================================================================
# SAFE BY DEFAULT: Creates database and schema if they don't exist.
# Does NOT drop, truncate, or modify existing data unless explicitly requested.
#
# Usage:
#   .\setup-db.ps1              # Safe: creates if missing, skips if exists
#   .\setup-db.ps1 -Reset       # DESTRUCTIVE: drops and recreates database
#   .\setup-db.ps1 -SeedOnly    # Only seeds atoms (skips schema)
#   .\setup-db.ps1 -Force       # Force re-seed atoms even if populated
# ============================================================================

param(
    [switch]$Reset,      # DESTRUCTIVE: Drop and recreate database
    [switch]$SeedOnly,   # Only seed atoms, skip schema
    [switch]$Force       # Force re-seeding even if atoms exist
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "=== Hypercube Database Setup ===" -ForegroundColor Cyan
Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  User: $env:HC_DB_USER"
Write-Host ""

# Temporarily set PGPASSWORD for psql commands
$env:PGPASSWORD = $env:HC_DB_PASS

try {
    # ========================================================================
    # POSTGRESQL SERVICE CHECK
    # ========================================================================
    Write-Host "[1/6] Checking PostgreSQL service..." -NoNewline
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq "Running" } | Select-Object -First 1
    if (-not $pgService) {
        Write-Host " NOT RUNNING" -ForegroundColor Red
        Write-Host ""
        Write-Host "PostgreSQL service is not running." -ForegroundColor Red
        Write-Host ""
        Write-Host "Start PostgreSQL with one of:" -ForegroundColor Yellow
        Write-Host "  1. Services app: Start 'postgresql-x64-XX' service" -ForegroundColor Yellow
        Write-Host "  2. Command line: net start postgresql-x64-XX" -ForegroundColor Yellow
        Write-Host "  3. pg_ctl: pg_ctl -D ""C:\Program Files\PostgreSQL\XX\data"" start" -ForegroundColor Yellow
        Write-Host ""
        # Don't exit - try connection anyway in case it's remote
        Write-Host "Continuing anyway (may be remote PostgreSQL)..." -ForegroundColor DarkGray
    } else {
        Write-Host " Running ($($pgService.Name))" -ForegroundColor Green
    }

    # ========================================================================
    # CONNECTION TEST
    # ========================================================================
    Write-Host "[2/6] Testing PostgreSQL connection..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host ""
        Write-Host "Cannot connect to PostgreSQL:" -ForegroundColor Red
        Write-Host "  $result" -ForegroundColor Red
        Write-Host ""
        Write-Host "Check:" -ForegroundColor Yellow
        Write-Host "  1. PostgreSQL is running (local or remote)" -ForegroundColor Yellow
        Write-Host "  2. Credentials in scripts/config.env are correct" -ForegroundColor Yellow
        Write-Host "     Current: $env:HC_DB_USER @ $env:HC_DB_HOST`:$env:HC_DB_PORT" -ForegroundColor Yellow
        Write-Host "  3. User '$env:HC_DB_USER' exists and has CREATEDB permission" -ForegroundColor Yellow
        Write-Host "  4. pg_hba.conf allows connections from your IP" -ForegroundColor Yellow
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # RESET (DESTRUCTIVE - only if explicitly requested)
    # ========================================================================
    if ($Reset) {
        Write-Host ""
        Write-Host "!!! DESTRUCTIVE OPERATION !!!" -ForegroundColor Red -BackgroundColor DarkRed
        Write-Host ""
        Write-Host "The -Reset flag will DROP the database and ALL data!" -ForegroundColor Red
        Write-Host ""
        
        # Check for existing data
        $existingData = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'" 2>$null
        if ($existingData -eq "1") {
            $compCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition" 2>$null
            $relCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation" 2>$null
            if ([int]$compCount -gt 0 -or [int]$relCount -gt 0) {
                Write-Host "Current database contains:" -ForegroundColor Yellow
                Write-Host "  - $compCount compositions" -ForegroundColor Yellow
                Write-Host "  - $relCount relations" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "This data CANNOT be recovered after reset." -ForegroundColor Red
                Write-Host ""
            }
        }
        
        Write-Host "[RESET] Dropping database $env:HC_DB_NAME..." -ForegroundColor Red
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME" 2>&1 | Out-Null
        Write-Host "[RESET] Database dropped" -ForegroundColor Yellow
    }

    # ========================================================================
    # DATABASE CREATION (idempotent)
    # ========================================================================
    if (-not $SeedOnly) {
        Write-Host "[3/6] Checking database..." -NoNewline
        $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'"
        
        if ($dbExists -ne "1") {
            Write-Host " creating..." -NoNewline
            & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Host " FAILED" -ForegroundColor Red
                exit 1
            }
            Write-Host " CREATED" -ForegroundColor Green
        } else {
            Write-Host " exists" -ForegroundColor Green
        }

        # ====================================================================
        # SCHEMA APPLICATION (single refactored master file)
        # ====================================================================
        Write-Host "[4/6] Applying schema..." -ForegroundColor Cyan

        Write-Host "      hypercube_schema.sql..." -NoNewline
        $sqlDir = Join-Path $env:HC_PROJECT_ROOT "sql"
        Push-Location $sqlDir
        try {
            # Run psql with ON_ERROR_STOP - it will exit non-zero on actual errors, but notices go to stderr
            $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -v ON_ERROR_STOP=1 -f "hypercube_schema.sql"
            if ($LASTEXITCODE -ne 0) {
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host "      psql exited with code $LASTEXITCODE" -ForegroundColor Red
                exit 1
            }
        } finally {
            Pop-Location
        }
        Write-Host " OK" -ForegroundColor Green

        # ====================================================================
        # C++ EXTENSIONS (idempotent - uses CREATE EXTENSION IF NOT EXISTS)
        # Load in dependency order to avoid failures
        # ====================================================================
        Write-Host "[5/6] Loading extensions..." -ForegroundColor Cyan
        # DEPENDENCY ORDER: base → ops → specialized
        $extensions = @(
            "hypercube",        # Base: BLAKE3, Hilbert, coordinates
            "hypercube_ops",    # Depends on: hypercube
            "embedding_ops",    # Depends on: hypercube
            "semantic_ops",     # Depends on: hypercube, embedding_ops
            "generative"        # Depends on: semantic_ops, embedding_ops
        )
        
        foreach ($ext in $extensions) {
            Write-Host "      $ext..." -NoNewline
            $extResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $ext;" 2>&1
            if ($LASTEXITCODE -eq 0 -and -not ($extResult -and ($extResult -match "^ERROR:" -or $extResult -match "^psql:"))) {
                Write-Host " OK" -ForegroundColor Green
            } else {
                Write-Host " not available" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "[3/6] Skipping database check (-SeedOnly)" -ForegroundColor DarkGray
        Write-Host "[4/6] Skipping schema (-SeedOnly)" -ForegroundColor DarkGray
        Write-Host "[5/6] Skipping extensions (-SeedOnly)" -ForegroundColor DarkGray
    }

    # ========================================================================
    # ATOM SEEDING (idempotent - checks count first)
    # ========================================================================
    Write-Host "[6/6] Checking atoms..." -NoNewline
    $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null

    if ([int]$atomCount -ge 1100000 -and -not $Force) {
        Write-Host " $atomCount atoms (already seeded)" -ForegroundColor Green
    } else {
        if ([int]$atomCount -ge 1100000 -and $Force) {
            Write-Host " re-seeding (forced)..." -ForegroundColor Yellow
        } else {
            Write-Host " seeding Unicode atoms..." -ForegroundColor Yellow
        }

        Write-Host ""
        # Use the optimized standalone seeder (built as part of the extension)
        Write-Host "Using optimized parallel atom seeder..." -ForegroundColor Cyan
        $seeder = "$env:HC_BUILD_DIR\seed_atoms_parallel.exe"
        if (-not (Test-Path $seeder)) {
            Write-Host "seed_atoms_parallel.exe not found, trying Release subdirectory..." -ForegroundColor Yellow
            $seeder = "$env:HC_BUILD_DIR\Release\seed_atoms_parallel.exe"
        }
        if (Test-Path $seeder) {
            Write-Host "Running optimized standalone seeder: $seeder"
            & $seeder -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Standalone seeder failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
                Write-Host "Database setup failed!" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Standalone seeder not found, using PostgreSQL extension function..." -ForegroundColor Cyan
            Write-Host "Running SQL: SELECT seed_atoms();"
            $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT seed_atoms();" 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Extension seeder failed: $result" -ForegroundColor Red
                Write-Host "Database setup failed!" -ForegroundColor Red
                exit 1
            }
            Write-Host "Extension seeder completed successfully" -ForegroundColor Green
        }

        $newCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null
        Write-Host "      Seeded $newCount atoms" -ForegroundColor Green
    }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    Write-Host ""
    Write-Host "=== Database Ready ===" -ForegroundColor Green
    Write-Host ""
    
    # Show final counts using db_stats function
    $stats = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT * FROM db_stats()" 2>$null
    if ($stats) {
        $statsArray = $stats -split '\|'
        $finalAtoms = $statsArray[0].Trim()
        $finalComps = $statsArray[1].Trim()
        $finalRels = $statsArray[3].Trim()
    } else {
        $finalAtoms = $finalComps = $finalRels = "0"
    }
    
    Write-Host "  Atoms:        $finalAtoms" -ForegroundColor Cyan
    Write-Host "  Compositions: $finalComps" -ForegroundColor Cyan
    Write-Host "  Relations:    $finalRels" -ForegroundColor Cyan
    Write-Host ""

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
