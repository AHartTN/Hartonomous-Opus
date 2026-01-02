# Hartonomous Hypercube - Windows Environment Setup
# Source this in PowerShell: . .\scripts\windows\env.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName

# Load config.env if it exists
$ConfigFile = "$ProjectRoot\scripts\config.env"
if (Test-Path $ConfigFile) {
    Get-Content $ConfigFile | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

# Defaults (app-specific, not global PG* vars)
if (-not $env:HC_DB_HOST) { $env:HC_DB_HOST = "localhost" }
if (-not $env:HC_DB_PORT) { $env:HC_DB_PORT = "5432" }
if (-not $env:HC_DB_USER) { $env:HC_DB_USER = "hartonomous" }
if (-not $env:HC_DB_PASS) { $env:HC_DB_PASS = "hartonomous" }
if (-not $env:HC_DB_NAME) { $env:HC_DB_NAME = "hypercube" }
if (-not $env:HC_BUILD_TYPE) { $env:HC_BUILD_TYPE = "Release" }
if (-not $env:HC_PARALLEL_JOBS) { $env:HC_PARALLEL_JOBS = "4" }

$env:HC_PROJECT_ROOT = $ProjectRoot
$env:HC_BUILD_DIR = "$ProjectRoot\cpp\build"

# Build connection string for libpq (used by C++ tools)
$env:HC_CONNINFO = "host=$env:HC_DB_HOST port=$env:HC_DB_PORT dbname=$env:HC_DB_NAME user=$env:HC_DB_USER password=$env:HC_DB_PASS"

function HC-PSQL {
    param([string]$Query)
    $env:PGPASSWORD = $env:HC_DB_PASS
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c $Query
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

function HC-PSQL-File {
    param([string]$File)
    $env:PGPASSWORD = $env:HC_DB_PASS
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -f $File
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host "Hartonomous environment loaded"
Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  Project:  $env:HC_PROJECT_ROOT"
