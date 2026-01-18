# Hartonomous Hypercube - Database Deployment (Windows)
# ============================================================================
# Deploys schema to remote PostgreSQL database.
# No admin privileges required - pure SQL deployment.
#
# Usage:
#   .\deploy.ps1                    # Deploy to configured database
#   .\deploy.ps1 -Rebuild           # Rebuild consolidated schema first
#   .\deploy.ps1 -CreateDb          # Create database if not exists
#   .\deploy.ps1 -Reset             # DESTRUCTIVE: Drop and recreate
# ============================================================================

param(
    [switch]$Rebuild,    # Rebuild full_schema.sql before deploying
    [switch]$CreateDb,   # Create database if it doesn't exist
    [switch]$Reset       # DESTRUCTIVE: Drop and recreate database
)

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hypercube Database Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Target: $env:HC_DB_USER@$env:HC_DB_HOST`:$env:HC_DB_PORT/$env:HC_DB_NAME"
Write-Host ""

$env:PGPASSWORD = $env:HC_DB_PASS
$sqlDir = Join-Path $env:HC_PROJECT_ROOT "sql"
$deployDir = Join-Path $sqlDir "deploy"
$schemaFile = Join-Path $deployDir "full_schema.sql"

try {
    # ========================================================================
    # REBUILD SCHEMA (if requested or if file doesn't exist)
    # ========================================================================
    if ($Rebuild -or -not (Test-Path $schemaFile)) {
        Write-Host "[1/4] Building consolidated schema..." -ForegroundColor Yellow
        $buildScript = Join-Path $deployDir "build-schema.ps1"
        if (Test-Path $buildScript) {
            & $buildScript
        } else {
            Write-Host "  ERROR: build-schema.ps1 not found" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[1/4] Using existing schema file" -ForegroundColor Green
    }

    # ========================================================================
    # CONNECTION TEST
    # ========================================================================
    Write-Host "[2/4] Testing connection..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Cannot connect to PostgreSQL server" -ForegroundColor Red
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # DATABASE CREATION/RESET
    # ========================================================================
    if ($Reset) {
        Write-Host "[3/4] Resetting database..." -ForegroundColor Red
        $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME" 2>&1
        Write-Host "  Dropped $env:HC_DB_NAME" -ForegroundColor Yellow
        $CreateDb = $true
    }

    if ($CreateDb) {
        Write-Host "[3/4] Creating database..." -NoNewline
        $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'" 2>&1
        if ($dbExists -ne "1") {
            $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME" 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host " FAILED" -ForegroundColor Red
                exit 1
            }
            Write-Host " created" -ForegroundColor Green
        } else {
            Write-Host " exists" -ForegroundColor Green
        }
    } else {
        Write-Host "[3/4] Database check..." -NoNewline
        $dbExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'" 2>&1
        if ($dbExists -ne "1") {
            Write-Host " NOT FOUND" -ForegroundColor Red
            Write-Host "  Run with -CreateDb to create the database" -ForegroundColor Yellow
            exit 1
        }
        Write-Host " OK" -ForegroundColor Green
    }

    # ========================================================================
    # DEPLOY SCHEMA
    # ========================================================================
    Write-Host "[4/4] Deploying schema..." -NoNewline

    # Run schema file with quiet mode to suppress NOTICE messages
    $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -q -v ON_ERROR_STOP=1 -f $schemaFile 2>&1

    # Check for actual errors (not NOTICE/WARNING)
    $hasError = $false
    foreach ($line in $output) {
        if ($line -is [System.Management.Automation.ErrorRecord]) {
            $text = $line.ToString()
            if ($text -match "^(ERROR|FATAL):") {
                $hasError = $true
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host "  $text" -ForegroundColor Red
            }
        }
    }

    if (-not $hasError -and $LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } elseif (-not $hasError) {
        Write-Host " FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    } else {
        exit 1
    }

    # ========================================================================
    # VERIFY
    # ========================================================================
    Write-Host ""
    Write-Host "Verifying deployment..." -ForegroundColor Cyan

    $stats = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT * FROM db_stats()" 2>&1
    if ($stats -and $LASTEXITCODE -eq 0) {
        $s = $stats -split '\|'
        Write-Host "  Atoms:        $($s[0].Trim())" -ForegroundColor Cyan
        Write-Host "  Compositions: $($s[1].Trim())" -ForegroundColor Cyan
        Write-Host "  Relations:    $($s[3].Trim())" -ForegroundColor Cyan
    } else {
        Write-Host "  db_stats() not available yet" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Deployment Complete" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""

} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
