# ============================================================================
# Hartonomous Hypercube - Windows Environment Setup
# ============================================================================
# Source this in PowerShell: . .\scripts\windows\env.ps1
#
# This script:
#   1. Loads config.env settings
#   2. Initializes Visual Studio Developer Environment
#   3. Adds Intel oneAPI to PATH (for MKL + OpenMP threading)
#   4. Sets up database connection helpers
# ============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName

# ============================================================================
# LOAD CONFIG.ENV
# ============================================================================

$ConfigFile = "$ProjectRoot\scripts\config.env"
if (Test-Path $ConfigFile) {
    Get-Content $ConfigFile | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

# ============================================================================
# VISUAL STUDIO DEVELOPER ENVIRONMENT
# ============================================================================

if (-not $env:VSCMD_VER) {
    # Find Visual Studio installation
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        $vswhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    }

    $vsPath = $null
    if (Test-Path $vswhere) {
        try {
            $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        } catch {
            # vswhere failed, will try fallback paths
        }
    }

    # Fallback: check common locations
    if (-not $vsPath) {
        $searchPaths = @(
            "D:\Microsoft Visual Studio\18\Community",
            "D:\Microsoft Visual Studio\2022\Community",
            "C:\Program Files\Microsoft Visual Studio\2022\Community",
            "C:\Program Files\Microsoft Visual Studio\2022\Professional",
            "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
            "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
            "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
            "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
        )
        foreach ($path in $searchPaths) {
            if (Test-Path "$path\Common7\Tools\VsDevCmd.bat") {
                $vsPath = $path
                break
            }
        }
    }

    if ($vsPath) {
        $vsDevCmdPath = "$vsPath\Common7\Tools\VsDevCmd.bat"
        if (Test-Path $vsDevCmdPath) {
            # Import VS environment variables into PowerShell
            try {
                cmd /c "`"$vsDevCmdPath`" -arch=amd64 -no_logo && set" | ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)$') {
                        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
                    }
                }
                Write-Verbose "Visual Studio environment initialized from: $vsPath"
            } catch {
                Write-Warning "Failed to initialize Visual Studio environment (C++ builds may fail)"
            }
        } else {
            Write-Verbose "Visual Studio found but VsDevCmd.bat missing: $vsPath"
        }
    } else {
        Write-Verbose "Visual Studio not found (optional for database-only operations)"
    }
}

# ============================================================================
# INTEL oneAPI RUNTIME
# ============================================================================
# Required for Intel MKL threading (libiomp5md.dll) and optimized BLAS

$IntelOneAPIPaths = @(
    "D:\Intel\oneAPI",
    "C:\Program Files (x86)\Intel\oneAPI",
    "C:\Program Files\Intel\oneAPI",
    "$env:USERPROFILE\intel\oneAPI"
)

foreach ($oneAPIPath in $IntelOneAPIPaths) {
    if (Test-Path $oneAPIPath) {
        # Set MKLROOT for CMake detection
        $env:MKLROOT = "$oneAPIPath\mkl\latest"

        # Add compiler and MKL bin directories for DLLs
        $compilerBin = "$oneAPIPath\compiler\latest\bin"
        $mklBin = "$oneAPIPath\mkl\latest\bin"

        if ((Test-Path $compilerBin) -and ($env:PATH -notlike "*$compilerBin*")) {
            $env:PATH = "$compilerBin;$mklBin;$env:PATH"
        }
        break
    }
}

# ============================================================================
# DEFAULTS
# ============================================================================
# App-specific variables (don't clash with global PG* vars)

if (-not $env:HC_DB_HOST) { $env:HC_DB_HOST = "localhost" }
if (-not $env:HC_DB_PORT) { $env:HC_DB_PORT = "5432" }
if (-not $env:HC_DB_USER) { $env:HC_DB_USER = "postgres" }
if (-not $env:HC_DB_PASS) { $env:HC_DB_PASS = "postgres" }
if (-not $env:HC_DB_NAME) { $env:HC_DB_NAME = "hypercube" }
if (-not $env:HC_BUILD_TYPE) { $env:HC_BUILD_TYPE = "Release" }
if (-not $env:HC_PARALLEL_JOBS) { 
    $env:HC_PARALLEL_JOBS = [Environment]::ProcessorCount 
}
if (-not $env:HC_INGEST_THRESHOLD) { $env:HC_INGEST_THRESHOLD = "0.5" }

$env:HC_PROJECT_ROOT = $ProjectRoot
$env:HC_BUILD_DIR = "$ProjectRoot\cpp\build"

# Connection string for libpq (used by C++ tools)
$env:HC_CONNINFO = "host=$env:HC_DB_HOST port=$env:HC_DB_PORT dbname=$env:HC_DB_NAME user=$env:HC_DB_USER password=$env:HC_DB_PASS"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function HC-PSQL {
    param([string]$Query)
    $env:PGPASSWORD = $env:HC_DB_PASS
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c $Query
}

function HC-PSQL-File {
    param([string]$File)
    $env:PGPASSWORD = $env:HC_DB_PASS
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -f $File
}

function HC-PSQL-Admin {
    param([string]$Query)
    $env:PGPASSWORD = $env:HC_DB_PASS
    & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c $Query
}

# ============================================================================
# BANNER (once per session)
# ============================================================================

if (-not $env:HC_ENV_LOADED) {
    $env:HC_ENV_LOADED = "1"
    Write-Host "Hartonomous environment loaded" -ForegroundColor Green
    Write-Host "  Database: $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
    Write-Host "  User:     $env:HC_DB_USER"
    Write-Host "  Project:  $env:HC_PROJECT_ROOT"
}

# Reset exit code to prevent inheritance issues
$LASTEXITCODE = 0
