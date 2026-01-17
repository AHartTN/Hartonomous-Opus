# ============================================================================
# Hartonomous Hypercube - Windows Build Script
# ============================================================================
# This script sets up the environment and builds the project in one session
# to ensure VS environment variables are properly inherited.
# ============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ProjectRoot = Split-Path -Parent $ProjectRoot

Write-Host "Setting up environment..." -ForegroundColor Green

# Load config.env
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

# Find and setup Visual Studio
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    $vswhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
}

$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
}

# Setup Intel oneAPI
$IntelOneAPIPaths = @(
    "D:\Intel\oneAPI",
    "C:\Program Files (x86)\Intel\oneAPI",
    "C:\Program Files\Intel\oneAPI",
    "$env:USERPROFILE\intel\oneAPI"
)

foreach ($oneAPIPath in $IntelOneAPIPaths) {
    if (Test-Path $oneAPIPath) {
        $env:MKLROOT = "$oneAPIPath\mkl\latest"
        $compilerBin = "$oneAPIPath\compiler\latest\bin"
        $mklBin = "$oneAPIPath\mkl\latest\bin"

        if ((Test-Path $compilerBin) -and ($env:PATH -notlike "*$compilerBin*")) {
            $env:PATH = "$compilerBin;$mklBin;$env:PATH"
        }
        break
    }
}

# Setup VS environment
if ($vsPath -and (Test-Path "$vsPath\Common7\Tools\VsDevCmd.bat")) {
    $vsDevCmd = "$vsPath\Common7\Tools\VsDevCmd.bat"
    Write-Host "Setting up VS DevCmd environment..." -ForegroundColor Yellow

    # Capture VS environment variables
    $vsEnv = cmd /c "`"$vsDevCmd`" -arch=amd64 -no_logo && set" 2>$null
    foreach ($line in $vsEnv) {
        if ($line -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "VS environment configured" -ForegroundColor Green
} else {
    Write-Host "Warning: Could not find VS DevCmd.bat" -ForegroundColor Red
}

# Set project variables
$env:HC_PROJECT_ROOT = $ProjectRoot
$env:HC_BUILD_DIR = "$ProjectRoot\cpp\build"

# Build the project
Write-Host "Building project..." -ForegroundColor Green
Push-Location "$ProjectRoot\cpp"

# Clean build directory
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
New-Item -ItemType Directory -Force "build" | Out-Null

Push-Location "build"

# Configure with CMake
Write-Host "Running CMake..." -ForegroundColor Yellow
& cmake .. -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake failed" -ForegroundColor Red
    Pop-Location
    Pop-Location
    exit 1
}

# Build with cmake --build (works with any generator)
Write-Host "Building..." -ForegroundColor Yellow
& cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    Pop-Location
    Pop-Location
    exit 1
}

Pop-Location
Pop-Location

Write-Host "Build completed successfully!" -ForegroundColor Green
