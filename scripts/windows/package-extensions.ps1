# Hartonomous Hypercube - PostgreSQL Extension Packager
# ============================================================================
# Creates a deployment package for PostgreSQL extensions that can be deployed
# to Windows or Linux PostgreSQL servers.
#
# Usage:
#   .\package-extensions.ps1              # Package for deployment
#   .\package-extensions.ps1 -Install     # Install to local PostgreSQL (needs admin)
# ============================================================================

param(
    [switch]$Install    # Install to local PostgreSQL (requires admin privileges)
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

$PackageDir = "$env:HC_PROJECT_ROOT\deploy\pg-extensions"
$BinDir = $env:HC_BIN_DIR
$SqlDir = "$env:HC_PROJECT_ROOT\cpp\sql"
$CppDir = "$env:HC_PROJECT_ROOT\cpp"

Write-Host ""
Write-Host "=== PostgreSQL Extension Packager ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# COLLECT FILES
# ============================================================================

# Clean and create package directory
if (Test-Path $PackageDir) {
    Remove-Item -Recurse -Force $PackageDir
}
New-Item -ItemType Directory -Force $PackageDir | Out-Null
New-Item -ItemType Directory -Force "$PackageDir\lib" | Out-Null
New-Item -ItemType Directory -Force "$PackageDir\extension" | Out-Null

Write-Host "[1/4] Collecting extension DLLs..." -NoNewline

$extensions = @(
    "hypercube",
    "hypercube_c",
    "hypercube_ops",
    "embedding_ops",
    "embedding_c",
    "semantic_ops",
    "generative",
    "generative_c"
)

$dllCount = 0
foreach ($ext in $extensions) {
    $dll = "$BinDir\$ext.dll"
    if (Test-Path $dll) {
        Copy-Item $dll "$PackageDir\lib\"
        $dllCount++
    }
}
Write-Host " $dllCount DLLs" -ForegroundColor Green

Write-Host "[2/4] Collecting control files..." -NoNewline

$controlFiles = @(
    "$CppDir\hypercube.control",
    "$SqlDir\hypercube_ops.control",
    "$SqlDir\embedding_ops.control",
    "$SqlDir\semantic_ops.control",
    "$SqlDir\generative.control"
)

$ctrlCount = 0
foreach ($ctrl in $controlFiles) {
    if (Test-Path $ctrl) {
        Copy-Item $ctrl "$PackageDir\extension\"
        $ctrlCount++
    }
}
Write-Host " $ctrlCount files" -ForegroundColor Green

Write-Host "[3/4] Collecting SQL files..." -NoNewline

$sqlFiles = @(
    "$SqlDir\hypercube--1.0.sql",
    "$SqlDir\hypercube_ops--1.0.sql",
    "$SqlDir\embedding_ops--1.0.sql",
    "$SqlDir\semantic_ops--1.0.sql",
    "$SqlDir\generative--1.0.sql"
)

$sqlCount = 0
foreach ($sql in $sqlFiles) {
    if (Test-Path $sql) {
        Copy-Item $sql "$PackageDir\extension\"
        $sqlCount++
    }
}
Write-Host " $sqlCount files" -ForegroundColor Green

Write-Host "[4/4] Creating install scripts..." -ForegroundColor Yellow

# ============================================================================
# WINDOWS INSTALL SCRIPT
# ============================================================================

$windowsInstall = @'
# Hypercube PostgreSQL Extensions - Windows Installer
# Run as Administrator!

param(
    [string]$PgDir = "C:\Program Files\PostgreSQL\18"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Installing Hypercube extensions to: $PgDir" -ForegroundColor Cyan

# Copy DLLs to lib directory
$libDir = "$PgDir\lib"
Write-Host "Copying DLLs to $libDir..."
Copy-Item "$ScriptDir\lib\*.dll" $libDir -Force

# Copy control and SQL files to extension directory
$extDir = "$PgDir\share\extension"
Write-Host "Copying extension files to $extDir..."
Copy-Item "$ScriptDir\extension\*" $extDir -Force

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Enable extensions in your database with:"
Write-Host "  CREATE EXTENSION hypercube;"
Write-Host "  CREATE EXTENSION hypercube_ops;"
Write-Host "  CREATE EXTENSION embedding_ops;"
Write-Host "  CREATE EXTENSION semantic_ops;"
Write-Host "  CREATE EXTENSION generative;"
'@

$windowsInstall | Out-File -FilePath "$PackageDir\install-windows.ps1" -Encoding UTF8

# ============================================================================
# LINUX INSTALL SCRIPT
# ============================================================================

$linuxInstall = @'
#!/bin/bash
# Hypercube PostgreSQL Extensions - Linux Installer
# Run as root or with sudo!
#
# NOTE: These are Windows DLLs. For Linux, you need to:
#   1. Build on Linux: cd cpp && mkdir build && cd build && cmake .. && make
#   2. The .so files will be in build/lib/
#   3. Copy this script's logic but use .so files instead of .dll

set -e

PG_CONFIG=${PG_CONFIG:-pg_config}
PKGLIBDIR=$($PG_CONFIG --pkglibdir)
SHAREDIR=$($PG_CONFIG --sharedir)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing Hypercube extensions..."
echo "  PKGLIBDIR: $PKGLIBDIR"
echo "  SHAREDIR:  $SHAREDIR/extension"

# Copy shared libraries (change .dll to .so for Linux builds)
echo "Copying shared libraries..."
# cp "$SCRIPT_DIR/lib/"*.so "$PKGLIBDIR/"

# Copy control and SQL files
echo "Copying extension files..."
cp "$SCRIPT_DIR/extension/"* "$SHAREDIR/extension/"

echo ""
echo "Installation complete!"
echo "Enable extensions in your database with:"
echo "  CREATE EXTENSION hypercube;"
echo "  CREATE EXTENSION hypercube_ops;"
echo "  CREATE EXTENSION embedding_ops;"
echo "  CREATE EXTENSION semantic_ops;"
echo "  CREATE EXTENSION generative;"
'@

$linuxInstall | Out-File -FilePath "$PackageDir\install-linux.sh" -Encoding UTF8 -NoNewline

# ============================================================================
# README
# ============================================================================

$readme = @"
# Hypercube PostgreSQL Extensions

## Contents
- lib/           Extension DLLs (Windows) or .so files (Linux)
- extension/     Control files and SQL definitions

## Windows Installation
Run PowerShell as Administrator:
    .\install-windows.ps1 -PgDir "C:\Program Files\PostgreSQL\18"

## Linux Installation
NOTE: The .dll files are for Windows only. For Linux:

1. Build on your Linux server:
   cd /path/to/Hartonomous-Opus/cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)

2. Copy the resulting .so files from build/bin/Release/ to:
   $(pg_config --pkglibdir)/

3. Copy extension/*.control and extension/*.sql to:
   $(pg_config --sharedir)/extension/

4. Or use the provided install-linux.sh after building.

## Enable Extensions
In PostgreSQL:
    CREATE EXTENSION hypercube;
    CREATE EXTENSION hypercube_ops;
    CREATE EXTENSION embedding_ops;
    CREATE EXTENSION semantic_ops;
    CREATE EXTENSION generative;

## Dependencies
- PostGIS (required by hypercube extensions)
"@

$readme | Out-File -FilePath "$PackageDir\README.md" -Encoding UTF8

Write-Host ""
Write-Host "=== Package Created ===" -ForegroundColor Green
Write-Host "  Location: $PackageDir"
Write-Host ""
Write-Host "  Contents:"
Write-Host "    lib/              $dllCount DLLs"
Write-Host "    extension/        $ctrlCount control + $sqlCount SQL files"
Write-Host "    install-windows.ps1"
Write-Host "    install-linux.sh"
Write-Host "    README.md"
Write-Host ""

# ============================================================================
# OPTIONAL: INSTALL TO LOCAL POSTGRESQL
# ============================================================================

if ($Install) {
    Write-Host "Installing to local PostgreSQL..." -ForegroundColor Yellow

    # Get PostgreSQL directories
    $pgConfig = Get-Command pg_config -ErrorAction SilentlyContinue
    if (-not $pgConfig) {
        Write-Host "ERROR: pg_config not found. Add PostgreSQL bin to PATH." -ForegroundColor Red
        exit 1
    }

    $pkgLibDir = & pg_config --pkglibdir
    $shareDir = & pg_config --sharedir

    Write-Host "  PKGLIBDIR: $pkgLibDir"
    Write-Host "  SHAREDIR:  $shareDir\extension"

    try {
        Copy-Item "$PackageDir\lib\*" $pkgLibDir -Force
        Copy-Item "$PackageDir\extension\*" "$shareDir\extension" -Force
        Write-Host "Installation complete!" -ForegroundColor Green
    } catch {
        Write-Host "Installation failed (need admin privileges?): $_" -ForegroundColor Red
        Write-Host "Run PowerShell as Administrator and try again." -ForegroundColor Yellow
    }
}
