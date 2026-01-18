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

# Don't treat non-terminating errors as fatal - we handle psql errors explicitly
$ErrorActionPreference = "Continue"

# Helper function to run psql and capture output properly
# psql writes notices/warnings to stderr which PowerShell treats as errors
function Invoke-Psql {
    param(
        [string]$Database = $env:HC_DB_NAME,
        [string]$Command,
        [string]$File,
        [switch]$Quiet,
        [switch]$TuplesOnly
    )

    $args = @("-h", $env:HC_DB_HOST, "-p", $env:HC_DB_PORT, "-U", $env:HC_DB_USER, "-d", $Database)

    if ($Quiet) { $args += "-q" }
    if ($TuplesOnly) { $args += "-tA" }

    if ($Command) {
        $args += @("-c", $Command)
    } elseif ($File) {
        $args += @("-v", "ON_ERROR_STOP=1", "-f", $File)
    }

    # Run psql and capture both stdout and stderr
    $output = & psql @args 2>&1

    # Check for actual errors (not just NOTICE/WARNING)
    $hasError = $false
    $errorMsg = ""
    foreach ($line in $output) {
        if ($line -is [System.Management.Automation.ErrorRecord]) {
            $text = $line.ToString()
            # PostgreSQL errors start with "ERROR:" or "FATAL:"
            if ($text -match "^(ERROR|FATAL):") {
                $hasError = $true
                $errorMsg = $text
            }
            # Ignore NOTICE, WARNING, INFO - these are not errors
        }
    }

    if ($hasError -or $LASTEXITCODE -ne 0) {
        return @{ Success = $false; Output = $output; Error = $errorMsg }
    }

    # Return just the stdout content (filter out ErrorRecord objects)
    $stdout = ($output | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n"
    return @{ Success = $true; Output = $stdout.Trim() }
}

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "=== Hypercube Database Setup ===" -ForegroundColor Cyan
Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  User: $env:HC_DB_USER"
Write-Host ""

$env:PGPASSWORD = $env:HC_DB_PASS

# Helper function to run psql without NOTICE spam
function Invoke-PsqlQuiet {
    param([string]$Query, [string]$Database = $env:HC_DB_NAME)
    $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $Database -tAq -c $Query 2>&1
    # Filter out NOTICE messages
    $stdout = ($output | Where-Object {
        -not ($_ -is [System.Management.Automation.ErrorRecord]) -or
        -not ($_.ToString() -match "^(NOTICE|WARNING|INFO):")
    }) -join ""
    return $stdout.Trim()
}

try {
    # ========================================================================
    # CONNECTION TEST
    # ========================================================================
    Write-Host "[1/4] Testing PostgreSQL connection..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Cannot connect: $result" -ForegroundColor Red
        Write-Host "  Check: PostgreSQL running, credentials in scripts/config.env, pg_hba.conf" -ForegroundColor Yellow
        exit 1
    }
    Write-Host " OK" -ForegroundColor Green

    # ========================================================================
    # DATABASE CREATION (or RESET)
    # ========================================================================
    if ($Reset) {
        Write-Host "[2/4] Resetting database..." -ForegroundColor Red
        $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME" 2>&1
        Write-Host "  Dropped $env:HC_DB_NAME" -ForegroundColor Yellow
    }

    Write-Host "[2/4] Database..." -NoNewline
    $dbExists = Invoke-PsqlQuiet -Query "SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'" -Database postgres

    if ($dbExists -ne "1") {
        $null = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED to create" -ForegroundColor Red
            exit 1
        }
        Write-Host " created" -ForegroundColor Green
    } else {
        Write-Host " exists" -ForegroundColor Green
    }

    # ========================================================================
    # SCHEMA DEPLOYMENT (using consolidated file)
    # ========================================================================
    Write-Host "[3/4] Schema..." -NoNewline

    $deployDir = Join-Path $env:HC_PROJECT_ROOT "sql\deploy"
    $schemaFile = Join-Path $deployDir "full_schema.sql"

    # Build consolidated schema if missing
    if (-not (Test-Path $schemaFile)) {
        Write-Host " building..." -NoNewline
        $buildScript = Join-Path $deployDir "build-schema.ps1"
        if (Test-Path $buildScript) {
            $null = & $buildScript 2>&1
        } else {
            Write-Host " FAILED (build script missing)" -ForegroundColor Red
            exit 1
        }
    }

    # Deploy with quiet mode (suppresses NOTICE messages)
    $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -q -v ON_ERROR_STOP=1 -f $schemaFile 2>&1

    # Check for actual errors
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
