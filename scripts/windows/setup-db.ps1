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
    # CONNECTION TEST
    # ========================================================================
    Write-Host "[1/5] Testing PostgreSQL connection..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host ""
        Write-Host "Cannot connect to PostgreSQL:" -ForegroundColor Red
        Write-Host "  $result" -ForegroundColor Red
        Write-Host ""
        Write-Host "Check:" -ForegroundColor Yellow
        Write-Host "  1. PostgreSQL is running" -ForegroundColor Yellow
        Write-Host "  2. Credentials in scripts/config.env are correct" -ForegroundColor Yellow
        Write-Host "  3. User '$env:HC_DB_USER' exists and has permissions" -ForegroundColor Yellow
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # RESET (DESTRUCTIVE - only if explicitly requested with confirmation)
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
        Write-Host "[2/5] Checking database..." -NoNewline
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
        # SCHEMA APPLICATION (idempotent - uses CREATE IF NOT EXISTS)
        # ====================================================================
        Write-Host "[3/5] Applying schema..." -ForegroundColor Cyan
        $sqlFiles = Get-ChildItem -Path "$env:HC_PROJECT_ROOT\sql\*.sql" | 
                    Where-Object { $_.Name -notmatch "archive" } | 
                    Sort-Object Name
        
        foreach ($sqlFile in $sqlFiles) {
            Write-Host "      $($sqlFile.Name)..." -NoNewline
            $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -v ON_ERROR_STOP=1 -f $sqlFile.FullName 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host "      Error: $output" -ForegroundColor Red
                exit 1
            }
            Write-Host " OK" -ForegroundColor Green
        }

        # ====================================================================
        # C++ EXTENSIONS (idempotent - uses CREATE EXTENSION IF NOT EXISTS)
        # ====================================================================
        Write-Host "[4/5] Loading extensions..." -ForegroundColor Cyan
        $extensions = @("hypercube", "hypercube_ops", "semantic_ops", "embedding_ops", "generative")
        
        foreach ($ext in $extensions) {
            Write-Host "      $ext..." -NoNewline
            $extResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $ext;" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host " OK" -ForegroundColor Green
            } else {
                Write-Host " not available" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "[2/5] Skipping database check (-SeedOnly)" -ForegroundColor DarkGray
        Write-Host "[3/5] Skipping schema (-SeedOnly)" -ForegroundColor DarkGray
        Write-Host "[4/5] Skipping extensions (-SeedOnly)" -ForegroundColor DarkGray
    }

    # ========================================================================
    # ATOM SEEDING (idempotent - checks count first)
    # ========================================================================
    Write-Host "[5/5] Checking atoms..." -NoNewline
    $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null
    
    if ([int]$atomCount -ge 1100000 -and -not $Force) {
        Write-Host " $atomCount atoms (already seeded)" -ForegroundColor Green
    } else {
        if ([int]$atomCount -ge 1100000 -and $Force) {
            Write-Host " re-seeding (forced)..." -ForegroundColor Yellow
        } else {
            Write-Host " seeding Unicode atoms..." -ForegroundColor Yellow
        }
        
        $seeder = "$env:HC_BUILD_DIR\seed_atoms_parallel.exe"
        if (-not (Test-Path $seeder)) {
            $seeder = "$env:HC_BUILD_DIR\Release\seed_atoms_parallel.exe"
        }
        if (-not (Test-Path $seeder)) {
            Write-Host ""
            Write-Host "ERROR: seed_atoms_parallel.exe not found" -ForegroundColor Red
            Write-Host "Run: .\scripts\windows\build.ps1" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host ""
        & $seeder -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Atom seeding failed" -ForegroundColor Red
            Write-Host "Database setup failed!" -ForegroundColor Red
            exit 1
        }
        
        $newCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom"
        Write-Host "      Seeded $newCount atoms" -ForegroundColor Green
    }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    Write-Host ""
    Write-Host "=== Database Ready ===" -ForegroundColor Green
    Write-Host ""
    
    # Show final counts
    $finalAtoms = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null
    $finalComps = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition" 2>$null
    $finalRels = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation" 2>$null
    
    Write-Host "  Atoms:        $finalAtoms" -ForegroundColor Cyan
    Write-Host "  Compositions: $finalComps" -ForegroundColor Cyan
    Write-Host "  Relations:    $finalRels" -ForegroundColor Cyan
    Write-Host ""

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
