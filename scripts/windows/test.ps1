# Hartonomous Hypercube - Test Suite (Windows)
# Runs all tests: C++ unit tests, integration tests, SQL tests
# Usage: .\scripts\windows\test.ps1 [-Quick]

param(
    [switch]$Quick  # Skip slow performance tests
)

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Hartonomous Hypercube - Test Suite (Windows)             ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$TestsPassed = 0
$TestsFailed = 0

function Run-Test {
    param(
        [string]$Name,
        [scriptblock]$Test
    )
    
    Write-Host -NoNewline "  Testing: $Name... "
    try {
        $result = & $Test 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PASSED" -ForegroundColor Green
            $script:TestsPassed++
            return $true
        } else {
            Write-Host "FAILED" -ForegroundColor Red
            Write-Host "    $result" -ForegroundColor Yellow
            $script:TestsFailed++
            return $false
        }
    } catch {
        Write-Host "FAILED" -ForegroundColor Red
        Write-Host "    $_" -ForegroundColor Yellow
        $script:TestsFailed++
        return $false
    }
}

# Find test executables
$BuildDir = $env:HC_BUILD_DIR
$ReleaseDir = "$BuildDir\Release"
if (Test-Path $ReleaseDir) { $BuildDir = $ReleaseDir }

# Section 1: C++ Unit Tests
Write-Host "─── C++ Unit Tests ────────────────────────────────────────" -ForegroundColor Blue

$unitTests = @("test_hilbert", "test_coordinates", "test_blake3", "test_semantic")
foreach ($test in $unitTests) {
    $exe = "$BuildDir\$test.exe"
    if (Test-Path $exe) {
        Run-Test $test { & $exe }
    } else {
        Write-Host "  Skipping $test (not built)" -ForegroundColor Yellow
    }
}

# Section 2: Database Connectivity
Write-Host ""
Write-Host "─── Database Connectivity ─────────────────────────────────" -ForegroundColor Blue

$env:PGPASSWORD = $env:HC_DB_PASS
try {
    Run-Test "PostgreSQL connection" {
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "SELECT 1" 2>&1 | Out-Null
    }
    
    Run-Test "PostGIS extension" {
        & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "SELECT PostGIS_Version()" 2>&1 | Out-Null
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# Section 3: Schema Validation
Write-Host ""
Write-Host "─── Schema Validation ─────────────────────────────────────" -ForegroundColor Blue

$env:PGPASSWORD = $env:HC_DB_PASS
try {
    Run-Test "Atom table exists" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'"
        if ($result.Trim() -eq "1") { exit 0 } else { exit 1 }
    }
    
    Run-Test "GIST index exists" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'"
        if ($result.Trim() -eq "1") { exit 0 } else { exit 1 }
    }
    
    Run-Test "Hilbert index exists" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'"
        if ($result.Trim() -eq "1") { exit 0 } else { exit 1 }
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# Section 4: Atom Seeding
Write-Host ""
Write-Host "─── Atom Seeding ──────────────────────────────────────────" -ForegroundColor Blue

$env:PGPASSWORD = $env:HC_DB_PASS
try {
    $atomCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0"
    Write-Host "  Leaf atoms (codepoints): $($atomCount.Trim())"
    
    Run-Test "All Unicode atoms seeded (>1.1M)" {
        if ([int]$atomCount.Trim() -gt 1100000) { exit 0 } else { exit 1 }
    }
    
    Run-Test "SRID = 0 for all atoms" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0"
        if ($result.Trim() -eq "0") { exit 0 } else { exit 1 }
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# Section 5: SQL Function Tests
Write-Host ""
Write-Host "─── SQL Function Tests ────────────────────────────────────" -ForegroundColor Blue

$env:PGPASSWORD = $env:HC_DB_PASS
try {
    Run-Test "atom_is_leaf()" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))"
        if ($result.Trim() -eq "t") { exit 0 } else { exit 1 }
    }
    
    Run-Test "atom_centroid()" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT (atom_centroid((SELECT id FROM atom WHERE codepoint = 65))).x IS NOT NULL"
        if ($result.Trim() -eq "t") { exit 0 } else { exit 1 }
    }
    
    Run-Test "atom_reconstruct_text()" {
        $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))"
        if ($result.Trim() -eq "A") { exit 0 } else { exit 1 }
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# Section 6: C++ Integration Tests
Write-Host ""
Write-Host "─── C++ Integration Tests ─────────────────────────────────" -ForegroundColor Blue

$intTest = "$BuildDir\test_integration.exe"
if (Test-Path $intTest) {
    Run-Test "Integration tests" { & $intTest }
}

$queryTest = "$BuildDir\test_query_api.exe"
if (Test-Path $queryTest) {
    Run-Test "Query API tests" { & $queryTest }
}

# Summary
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════"
if ($TestsFailed -eq 0) {
    Write-Host "  All $TestsPassed tests passed!" -ForegroundColor Green
} else {
    Write-Host "  $TestsFailed tests failed, $TestsPassed passed" -ForegroundColor Red
}
Write-Host "════════════════════════════════════════════════════════════════"
Write-Host ""

exit $TestsFailed
