# Hartonomous Hypercube - Database Setup (Windows)
# Creates database, applies schema, seeds Unicode atoms
# Usage: .\scripts\windows\setup-db.ps1 [-Reset]

param(
    [switch]$Reset
)

. "$PSScriptRoot\env.ps1"

Write-Host "=== Hypercube Database Setup ===" -ForegroundColor Cyan
Write-Host "Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"

# Temporarily set PGPASSWORD for psql commands
$env:PGPASSWORD = $env:HC_DB_PASS

try {
    # Check connection (to postgres database which always exists)
    Write-Host "`nTesting connection..."
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Cannot connect to PostgreSQL: $result" -ForegroundColor Red
        Write-Host "Ensure PostgreSQL is running and credentials are correct in scripts\config.env"
        exit 1
    }
    Write-Host "Connected" -ForegroundColor Green

    # Reset if requested
    if ($Reset) {
        Write-Host "`nDropping existing database..."
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME"
    }

    # Check if database exists
    $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'"
    
    if ($dbExists -ne "1") {
        Write-Host "`nCreating database $env:HC_DB_NAME..."
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create database" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "Database exists" -ForegroundColor Green

    # Apply schema
    Write-Host "`nApplying unified schema..."
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -f "$env:HC_PROJECT_ROOT\sql\011_unified_atom.sql"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Schema application failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "Schema applied" -ForegroundColor Green

    # Check atom count
    $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0"
    Write-Host "`nAtom count: $atomCount"

    if ([int]$atomCount -lt 1100000) {
        Write-Host "`nSeeding Unicode atoms (this takes ~30 seconds)..."
        
        $seeder = "$env:HC_BUILD_DIR\seed_atoms_parallel.exe"
        if (-not (Test-Path $seeder)) {
            $seeder = "$env:HC_BUILD_DIR\Release\seed_atoms_parallel.exe"
        }
        if (-not (Test-Path $seeder)) {
            Write-Host "Seeder not found. Run build.ps1 first." -ForegroundColor Red
            exit 1
        }
        
        & $seeder -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Seeding failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "Atoms seeded" -ForegroundColor Green
    } else {
        Write-Host "Atoms already seeded" -ForegroundColor Green
    }

    Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
    
    # Show stats
    Write-Host "`nDatabase Statistics:"
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "SELECT * FROM atom_stats"

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
