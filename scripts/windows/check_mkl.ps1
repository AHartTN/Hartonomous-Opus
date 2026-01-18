# ============================================================================
# Hartonomous Hypercube - MKL Environment Diagnostic
# ============================================================================
# Checks MKL installation and environment setup for debugging runtime issues.
# ============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " MKL Environment Diagnostic" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# CHECK ENVIRONMENT VARIABLES
# ============================================================================

Write-Host "Environment Variables:" -ForegroundColor Yellow
$mklroot = if ($env:MKLROOT) { $env:MKLROOT } else { 'NOT SET' }
$mklthreads = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { 'NOT SET' }
Write-Host "  MKLROOT: $mklroot" -ForegroundColor $(if ($env:MKLROOT) { 'Green' } else { 'Red' })
Write-Host "  MKL_NUM_THREADS: $mklthreads" -ForegroundColor $(if ($env:MKL_NUM_THREADS) { 'Green' } else { 'Yellow' })

# Check PATH for MKL directories
$MklInPath = $env:PATH -split ';' | Where-Object { $_ -like "*mkl*" -or $_ -like "*intel*" }
Write-Host "  MKL in PATH: $(if ($MklInPath) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($MklInPath) { 'Green' } else { 'Red' })
if ($MklInPath) {
    Write-Host "    MKL PATH entries:" -ForegroundColor Gray
    $MklInPath | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
}
Write-Host ""

# ============================================================================
# CHECK INTEL ONEAPI INSTALLATION
# ============================================================================

Write-Host "Intel oneAPI Installation:" -ForegroundColor Yellow

$IntelOneAPIPaths = @(
    "D:\Intel\oneAPI",
    "C:\Program Files (x86)\Intel\oneAPI",
    "C:\Program Files\Intel\oneAPI",
    "$env:USERPROFILE\intel\oneAPI"
)

$InstallationFound = $false
foreach ($path in $IntelOneAPIPaths) {
    if (Test-Path $path) {
        Write-Host "  Found installation: $path" -ForegroundColor Green
        $InstallationFound = $true

        # Check MKL subdirectory
        $mklPath = "$path\mkl\latest"
        if (Test-Path $mklPath) {
            Write-Host "    MKL latest: $mklPath" -ForegroundColor Green
        } else {
            Write-Host "    MKL latest: NOT FOUND" -ForegroundColor Red
        }

        # Check bin directory
        $mklBin = "$mklPath\bin"
        if (Test-Path $mklBin) {
            Write-Host "    MKL bin: $mklBin" -ForegroundColor Green

            # List DLLs
            $dlls = Get-ChildItem "$mklBin\*.dll" -ErrorAction SilentlyContinue
            if ($dlls) {
                Write-Host "    MKL DLLs found: $($dlls.Count)" -ForegroundColor Green
                $dlls | Select-Object -First 5 | ForEach-Object {
                    Write-Host "      $($_.Name)" -ForegroundColor Gray
                }
                if ($dlls.Count -gt 5) {
                    Write-Host "      ... and $($dlls.Count - 5) more" -ForegroundColor Gray
                }
            } else {
                Write-Host "    MKL DLLs: NONE FOUND" -ForegroundColor Red
            }
        } else {
            Write-Host "    MKL bin: NOT FOUND" -ForegroundColor Red
        }

        break
    }
}

if (-not $InstallationFound) {
    Write-Host "  No Intel oneAPI installation found in expected locations" -ForegroundColor Red
    Write-Host "  Checked paths:" -ForegroundColor Gray
    $IntelOneAPIPaths | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}
Write-Host ""

# ============================================================================
# CHECK DLL LOADING
# ============================================================================

Write-Host "DLL Loading Test:" -ForegroundColor Yellow

# Check if key MKL DLLs exist in PATH locations
$MklCoreExists = $false
$OpenMPExists = $false

$pathDirs = $env:PATH -split ';'
foreach ($dir in $pathDirs) {
    if (Test-Path "$dir\mkl_core.2.dll") {
        $MklCoreExists = $true
        break
    }
}

foreach ($dir in $pathDirs) {
    if (Test-Path "$dir\libiomp5md.dll") {
        $OpenMPExists = $true
        break
    }
}

Write-Host "  mkl_core.2.dll in PATH: $(if ($MklCoreExists) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($MklCoreExists) { 'Green' } else { 'Red' })
Write-Host "  libiomp5md.dll in PATH: $(if ($OpenMPExists) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($OpenMPExists) { 'Green' } else { 'Red' })

Write-Host ""

# ============================================================================
# DIAGNOSIS
# ============================================================================

Write-Host "Diagnosis:" -ForegroundColor Yellow

$Issues = @()

if (-not $env:MKLROOT) {
    $Issues += "MKLROOT environment variable not set"
}

if (-not $MklInPath) {
    $Issues += "MKL bin directory not in PATH"
}

if (-not $InstallationFound) {
    $Issues += "Intel oneAPI MKL not installed"
}

if (-not $MklCoreExists) {
    $Issues += "mkl_core.2.dll not found in PATH directories"
}

if (-not $OpenMPExists) {
    $Issues += "libiomp5md.dll not found in PATH directories"
}

if ($Issues.Count -eq 0) {
    Write-Host "  No issues detected - MKL should work" -ForegroundColor Green
} else {
    Write-Host "  Issues found:" -ForegroundColor Red
    $Issues | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Run .\scripts\windows\env.ps1 to set up environment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan