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
        $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -v ON_ERROR_STOP=1 -f $sqlFile.FullName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED" -ForegroundColor Red
            Write-Host "    Error: $output" -ForegroundColor Red
            Write-Host "Database setup failed!" -ForegroundColor Red
            exit 1
        } else {
            Write-Host " OK" -ForegroundColor Green
        }
    }
    
    # Load C++ extensions (idempotent - IF NOT EXISTS)
    Write-Host "`nLoading C++ extensions..."
    
    # All extensions in dependency order
    $extensions = @(
        "hypercube",        # Base extension (hilbert, blake3, coordinates)
        "hypercube_ops",    # Batch operations  
        "semantic_ops",     # Semantic tree operations
        "embedding_ops",    # SIMD embedding operations
        "generative"        # Generative walk engine
    )
    
    $allLoaded = $true
    foreach ($ext in $extensions) {
        $extResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $ext;" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  $ext" -ForegroundColor Green
        } else {
            Write-Host "  $ext - not available" -ForegroundColor Yellow
            $allLoaded = $false
        }
    }
    
    if ($allLoaded) {
        Write-Host "All C++ extensions loaded" -ForegroundColor Green
    } else {
        Write-Host "Warning: Some C++ extensions not available (run build.ps1 -Install)" -ForegroundColor Yellow
    }

    # Check atom count (atoms table in 4-table schema = leaf atoms only)
    $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom"
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

    # NOTE: Do NOT recompute centroids here. The Laplacian eigenmap projections
    # from ingest-safetensor.ps1 are the correct semantic coordinates.
    # Recomputing from atom children would destroy the semantic relationships.

    # Generate k-NN semantic edges from composition centroids
    Write-Host "`nChecking for k-NN edge generation..."
    $edgeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation" 2>&1
    $compWithCentroid = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL" 2>&1
    
    if ([int]$edgeCount -eq 0 -and [int]$compWithCentroid -gt 0) {
        Write-Host "  Generating k-NN edges for $compWithCentroid compositions..."
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT generate_knn_edges(10, 'centroid_knn')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Created $result semantic edges" -ForegroundColor Green
        } else {
            Write-Host "  k-NN edge generation failed" -ForegroundColor Yellow
        }
    } elseif ([int]$edgeCount -gt 0) {
        Write-Host "  Already have $edgeCount edges" -ForegroundColor Gray
    }

    Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
    
    # Show stats
    Write-Host "`nDatabase Statistics:"
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "SELECT * FROM db_stats();"

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
