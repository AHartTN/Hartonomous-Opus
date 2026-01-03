# Hartonomous Hypercube - Database Setup (Windows)
# Creates database, applies schema, loads extensions, seeds Unicode atoms
# Fully idempotent - safe to run multiple times
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

    # Apply ALL SQL schema files in order
    Write-Host "`nApplying schema files..."
    $sqlFiles = Get-ChildItem -Path "$env:HC_PROJECT_ROOT\sql\*.sql" | Sort-Object Name
    foreach ($sqlFile in $sqlFiles) {
        Write-Host "  $($sqlFile.Name)..." -NoNewline
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -f $sqlFile.FullName -q 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED" -ForegroundColor Red
        } else {
            Write-Host " OK" -ForegroundColor Green
        }
    }
    
    # Load C++ extensions (idempotent - IF NOT EXISTS)
    Write-Host "`nLoading C++ extensions..."
    $extResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS hypercube; CREATE EXTENSION IF NOT EXISTS semantic_ops;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "C++ extensions loaded" -ForegroundColor Green
    } else {
        Write-Host "Warning: C++ extensions not available (run build.ps1 -Install)" -ForegroundColor Yellow
        Write-Host "  $extResult"
    }

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
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "SELECT COUNT(*) as leaf_atoms FROM atom WHERE depth = 0; SELECT COUNT(*) as compositions FROM atom WHERE depth > 0; SELECT MAX(depth) as max_depth FROM atom;"

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
