# Hartonomous Hypercube - Comprehensive Test Runner (Windows)
# Runs all available tests with proper error handling and reporting

param(
    [switch]$Quick,        # Skip slow tests
    [switch]$Verbose,      # Show detailed output
    [switch]$NoDatabase,   # Skip database-dependent tests
    [string]$TestFilter    # Run only specific test patterns
)

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    Hartonomous Hypercube - Comprehensive Test Runner       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$TestsPassed = 0
$TestsFailed = 0
$TestsSkipped = 0

function Test-Section {
    param([string]$Name, [scriptblock]$TestBlock)
    Write-Host "─── $Name ──────────────────────────────────────" -ForegroundColor Blue
    & $TestBlock
    Write-Host ""
}

function Run-Test {
    param(
        [string]$Name,
        [scriptblock]$TestBlock,
        [switch]$SkipIfNoDB = $false
    )

    if ($SkipIfNoDB -and $NoDatabase) {
        Write-Host "  ○ " -NoNewline -ForegroundColor DarkGray
        Write-Host "$Name" -NoNewline
        Write-Host " → SKIPPED (no database)" -ForegroundColor DarkGray
        $script:TestsSkipped++
        return
    }

    try {
        $result = & $TestBlock
        if ($result) {
            Write-Host "  ✓ " -NoNewline -ForegroundColor Green
            Write-Host "$Name" -NoNewline
            Write-Host " → PASS" -ForegroundColor Gray
            $script:TestsPassed++
        } else {
            Write-Host "  ✗ " -NoNewline -ForegroundColor Red
            Write-Host "$Name" -NoNewline
            Write-Host " → FAIL" -ForegroundColor Yellow
            $script:TestsFailed++
        }
    }
    catch {
        Write-Host "  ✗ " -NoNewline -ForegroundColor Red
        Write-Host "$Name" -NoNewline
        Write-Host " → ERROR: $($_.Exception.Message)" -ForegroundColor Red
        $script:TestsFailed++
    }
}

# Find test executables
$BuildDir = $env:HC_BUILD_DIR
if (Test-Path "$BuildDir\Release") { $BuildDir = "$BuildDir\Release" }

# ═══════════════════════════════════════════════════════════════════════════
Test-Section "C++ Unit Tests" {
    $unitTests = @(
        @{Name="test_hilbert"; Executable="test_hilbert.exe"},
        @{Name="test_coordinates"; Executable="test_coordinates.exe"},
        @{Name="test_blake3"; Executable="test_blake3.exe"},
        @{Name="test_semantic"; Executable="test_semantic.exe"},
        @{Name="test_clustering"; Executable="test_clustering.exe"}
    )

    foreach ($test in $unitTests) {
        Run-Test $test.Name {
            $exe = "$BuildDir\$($test.Executable)"
            if (-not (Test-Path $exe)) { return $false }

            if ($Verbose) {
                $output = & $exe 2>&1
                $exitCode = $LASTEXITCODE
                if ($Verbose) { Write-Host "    Output: $output" -ForegroundColor DarkGray }
                return ($exitCode -eq 0 -and -not ($output -match "FAILED|FAIL"))
            } else {
                $output = & $exe 2>&1
                return ($LASTEXITCODE -eq 0 -and -not ($output -match "FAILED|FAIL"))
            }
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Test-Section "Google Test Suite" {
    Run-Test "hypercube_tests (Google Test)" {
        $exe = "$BuildDir\hypercube_tests.exe"
        if (-not (Test-Path $exe)) { return $false }

        $args = @()
        if ($TestFilter) { $args += "--gtest_filter=$TestFilter" }
        if ($Verbose) { $args += "--gtest_print_time=1" }

        $output = & $exe @args 2>&1
        $exitCode = $LASTEXITCODE

        if ($Verbose) {
            Write-Host "    Exit code: $exitCode" -ForegroundColor DarkGray
            Write-Host "    Output: $output" -ForegroundColor DarkGray
        }

        return ($exitCode -eq 0)
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Test-Section "C++ Integration Tests" {
    $integrationTests = @(
        @{Name="test_integration"; Executable="test_integration.exe"},
        @{Name="test_query_api"; Executable="test_query_api.exe"}
    )

    foreach ($test in $integrationTests) {
        Run-Test $test.Name {
            $exe = "$BuildDir\$($test.Executable)"
            if (-not (Test-Path $exe)) { return $false }

            $output = & $exe 2>&1
            return ($LASTEXITCODE -eq 0)
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Test-Section "Database Tests" {
    # Test database connectivity
    Run-Test "PostgreSQL Connection" -SkipIfNoDB {
        try {
            $env:PGPASSWORD = $env:HC_DB_PASS
            $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>$null
            return ($LASTEXITCODE -eq 0 -and $result -eq "1")
        }
        finally {
            Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
        }
    }

    # Test schema integrity
    Run-Test "Database Schema" -SkipIfNoDB {
        try {
            $env:PGPASSWORD = $env:HC_DB_PASS

            # Check if main tables exist
            $atomExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='atom'" 2>$null
            $compExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='composition'" 2>$null
            $relExists = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='relation'" 2>$null

            return ([int]$atomExists -eq 1 -and [int]$compExists -eq 1 -and [int]$relExists -eq 1)
        }
        finally {
            Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
        }
    }

    # Test core functions
    Run-Test "Core Functions" -SkipIfNoDB {
        try {
            $env:PGPASSWORD = $env:HC_DB_PASS

            # Test atom_is_leaf function
            $isLeaf = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))" 2>$null
            if ($LASTEXITCODE -ne 0 -or $isLeaf -ne "t") { return $false }

            # Test atom_reconstruct_text function
            $reconstruct = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))" 2>$null
            if ($LASTEXITCODE -ne 0 -or $reconstruct -ne "A") { return $false }

            return $true
        }
        finally {
            Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
        }
    }

    # Test atom seeding
    Run-Test "Atom Seeding" -SkipIfNoDB {
        try {
            $env:PGPASSWORD = $env:HC_DB_PASS
            $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom" 2>$null
            return ([int]$atomCount -gt 1100000)
        }
        finally {
            Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Test-Section "Build System Tests" {
    Run-Test "CMake Configuration" {
        return (Test-Path "$env:HC_PROJECT_ROOT\cpp\build\CMakeCache.txt")
    }

    Run-Test "All Targets Built" {
        $requiredExes = @("hc.exe", "hypercube_tests.exe", "test_coordinates.exe", "test_hilbert.exe")
        foreach ($exe in $requiredExes) {
            if (-not (Test-Path "$BuildDir\$exe")) { return $false }
        }
        return $true
    }

    Run-Test "Extensions Built" {
        $extensions = @("hypercube.dll", "embedding_ops.dll", "semantic_ops.dll", "generative.dll")
        foreach ($ext in $extensions) {
            if (-not (Test-Path "$BuildDir\$ext")) { return $false }
        }
        return $true
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# Performance Tests (skip if -Quick specified)
if (-not $Quick) {
    Test-Section "Performance Tests" {
        Run-Test "Laplacian Eigenmap Performance" {
            $exe = "$BuildDir\test_laplacian_4d.exe"
            if (-not (Test-Path $exe)) { return $false }

            $start = Get-Date
            $output = & $exe 2>$null
            $duration = (Get-Date) - $start

            # Should complete in reasonable time (< 30 seconds for small test)
            return ($LASTEXITCODE -eq 0 -and $duration.TotalSeconds -lt 30)
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# Summary
Write-Host "════════════════════════════════════════════════════════════════"
Write-Host "Test Results:" -ForegroundColor White
Write-Host "  ✓ Passed:  $TestsPassed" -ForegroundColor Green
Write-Host "  ✗ Failed:  $TestsFailed" -ForegroundColor Red
Write-Host "  ○ Skipped: $TestsSkipped" -ForegroundColor DarkGray
Write-Host ""

if ($TestsFailed -eq 0) {
    Write-Host "🎉 All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Some tests failed. Check output above for details." -ForegroundColor Red
    exit 1
}
