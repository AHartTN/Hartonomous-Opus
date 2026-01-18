# Hartonomous Hypercube - End-to-End Integration Test Suite (Windows)
# ============================================================================
# COMPLETE pipeline test from clean slate to full functionality validation.
# 
# Pipeline:
#   1. Clean build artifacts
#   2. Build everything (C++ code, extensions)
#   3. Drop database (destructive reset)
#   4. Create database
#   5. Install extensions, schema, UDFs, procedures, views
#   6. Seed database (1.1M+ Unicode atoms)
#   7. Ingest model(s) from test-data
#   8. Ingest test content (moby_dick.txt)
#   9. Run full integration/functionality/feature tests
#
# Usage:
#   .\e2e-test.ps1                    # Full clean slate test
#   .\e2e-test.ps1 -SkipBuild         # Skip build (use existing)
#   .\e2e-test.ps1 -SkipSeed          # Skip atom seeding (if already done)
#   .\e2e-test.ps1 -SkipModels        # Skip model ingestion
#   .\e2e-test.ps1 -Verbose           # Extra output
#   .\e2e-test.ps1 -FailFast          # Stop on first failure
# ============================================================================

param(
    [switch]$SkipBuild,      # Skip C++ build
    [switch]$SkipSeed,       # Skip atom seeding
    [switch]$SkipModels,     # Skip model ingestion
    [switch]$SkipContent,    # Skip text ingestion
    [switch]$Verbose,        # Verbose output
    [switch]$FailFast        # Stop on first failure
)

$ErrorActionPreference = "Stop"
$script:TestsRun = 0
$script:TestsPassed = 0
$script:TestsFailed = 0
$script:Failures = @()
$script:StartTime = Get-Date

# Load environment
. "$PSScriptRoot\env.ps1"
$env:PGPASSWORD = $env:HC_DB_PASS

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-Header {
    param([string]$Text, [string]$Char = "=")
    $line = $Char * 78
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "--- $Text ---" -ForegroundColor Yellow
}

function Write-Step {
    param([int]$Step, [int]$Total, [string]$Text)
    Write-Host ""
    Write-Host "[$Step/$Total] $Text" -ForegroundColor Magenta
}

function Test-Assert {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$Category = "GENERAL"
    )
    $script:TestsRun++
    try {
        $result = & $Test
        if ($result) {
            $script:TestsPassed++
            Write-Host "  [PASS] ${Category}::$Name" -ForegroundColor Green
            return $true
        } else {
            $script:TestsFailed++
            $script:Failures += "${Category}::$Name"
            Write-Host "  [FAIL] ${Category}::$Name" -ForegroundColor Red
            if ($FailFast) { throw "Test failed: ${Category}::$Name" }
            return $false
        }
    } catch {
        $script:TestsFailed++
        $script:Failures += "${Category}::$Name - $($_.Exception.Message)"
        Write-Host "  [FAIL] ${Category}::$Name - $($_.Exception.Message)" -ForegroundColor Red
        if ($FailFast) { throw }
        return $false
    }
}

function Invoke-Psql {
    param([string]$Query, [switch]$Scalar)
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc $Query 2>&1
    if ($Scalar) {
        if ($result -is [string]) {
            return $result.Trim()
        } elseif ($result -is [array]) {
            return ($result | Out-String).Trim()
        } else {
            return [string]$result
        }
    }
    return $result
}

function Test-DatabaseConnected {
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -tAc "SELECT 1" 2>&1
    return $LASTEXITCODE -eq 0
}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

Write-Header "HYPERCUBE END-TO-END INTEGRATION TEST SUITE"
Write-Host ""
Write-Host "  Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  Database:   $env:HC_DB_NAME @ $env:HC_DB_HOST`:$env:HC_DB_PORT"
Write-Host "  Build Type: $env:HC_BUILD_TYPE"
Write-Host ""

# ============================================================================
# PHASE 1: CLEAN BUILD ARTIFACTS
# ============================================================================
Write-Step 1 9 "CLEAN BUILD ARTIFACTS"

if (-not $SkipBuild) {
    & "$PSScriptRoot\clean.ps1"
    Write-Host "  Build artifacts cleaned" -ForegroundColor Green
} else {
    Write-Host "  Skipped (using existing build)" -ForegroundColor DarkGray
}

# ============================================================================
# PHASE 2: BUILD EVERYTHING
# ============================================================================
Write-Step 2 9 "BUILD C++ COMPONENTS"

if (-not $SkipBuild) {
    $buildStart = Get-Date
    
    # Ensure build directory exists
    $BuildDir = "$env:HC_PROJECT_ROOT\cpp\build"
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Push-Location $BuildDir
    
    try {
        Write-Host "  Configuring CMake..."
        $cmakeArgs = @("-DCMAKE_BUILD_TYPE=$env:HC_BUILD_TYPE", "..")
        
        # Prefer Ninja with MSVC
        if (Get-Command ninja -ErrorAction SilentlyContinue) {
            $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
            if ($clPath) {
                $cmakeArgs = @("-G", "Ninja", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl") + $cmakeArgs
            }
        }
        
        & cmake @cmakeArgs 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        Write-Host "  Building (parallel $env:HC_PARALLEL_JOBS)..."
        & cmake --build . --config $env:HC_BUILD_TYPE --parallel $env:HC_PARALLEL_JOBS
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        $buildTime = (Get-Date) - $buildStart
        Write-Host "  Build completed in $($buildTime.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
        
        # Verify key artifacts
        $extensions = @("hypercube.dll", "hypercube_ops.dll", "semantic_ops.dll", "generative.dll")
        $exes = @("seed_atoms_parallel.exe", "ingest.exe", "ingest_safetensor.exe")
        
        foreach ($dll in $extensions) {
            if (Test-Path $dll) {
                Write-Host "    [OK] $dll" -ForegroundColor Green
            } else {
                Write-Host "    [WARN] $dll not found" -ForegroundColor Yellow
            }
        }
        
        foreach ($exe in $exes) {
            if (Test-Path $exe) {
                Write-Host "    [OK] $exe" -ForegroundColor Green
            } else {
                Write-Host "    [WARN] $exe not found" -ForegroundColor Yellow
            }
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  Skipped (-SkipBuild)" -ForegroundColor DarkGray
}

# ============================================================================
# PHASE 3: DROP DATABASE
# ============================================================================
Write-Step 3 9 "DROP DATABASE (RESET)"

if (-not (Test-DatabaseConnected)) {
    throw "Cannot connect to PostgreSQL server"
}

Write-Host "  Dropping database $env:HC_DB_NAME..."
& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $env:HC_DB_NAME" 2>&1 | Out-Null
Write-Host "  Database dropped" -ForegroundColor Green

# ============================================================================
# PHASE 4: CREATE DATABASE
# ============================================================================
Write-Step 4 9 "CREATE DATABASE"

Write-Host "  Creating database $env:HC_DB_NAME..."
& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d postgres -c "CREATE DATABASE $env:HC_DB_NAME" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create database"
}
Write-Host "  Database created" -ForegroundColor Green

# ============================================================================
# PHASE 5: INSTALL EXTENSIONS & SCHEMA
# ============================================================================
Write-Step 5 9 "INSTALL EXTENSIONS, SCHEMA, UDFs"

# Install CMake-built extensions first (they may need to be copied)
Push-Location "$env:HC_PROJECT_ROOT\cpp\build"
try {
    Write-Host "  Installing PostgreSQL extensions via CMake..."
    & cmake --install . --config $env:HC_BUILD_TYPE 2>&1 | Out-Null
} finally {
    Pop-Location
}

# Apply SQL schema files in order
Write-Host "  Applying schema files..."
$sqlFiles = Get-ChildItem -Path "$env:HC_PROJECT_ROOT\sql\*.sql" | 
            Where-Object { $_.Name -notmatch "archive" } | 
            Sort-Object Name

foreach ($sqlFile in $sqlFiles) {
    Write-Host "    $($sqlFile.Name)..." -NoNewline
    $output = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -v ON_ERROR_STOP=1 -f $sqlFile.FullName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "    Error: $output" -ForegroundColor Red
        throw "Schema application failed on $($sqlFile.Name)"
    }
    Write-Host " OK" -ForegroundColor Green
}

# Load extensions
Write-Host "  Loading C++ extensions..."
$extensions = @("hypercube", "hypercube_ops", "semantic_ops", "embedding_ops", "generative")
foreach ($ext in $extensions) {
    Write-Host "    $ext..." -NoNewline
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $ext CASCADE;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " not available" -ForegroundColor Yellow
    }
}

# ============================================================================
# PHASE 6: SEED DATABASE
# ============================================================================
Write-Step 6 9 "SEED DATABASE (Unicode Atoms)"

if (-not $SkipSeed) {
    $atomCount = Invoke-Psql "SELECT atoms FROM db_stats()" -Scalar

    if ([int]$atomCount -lt 1100000) {
        Write-Host "  Seeding 1.1M+ Unicode atoms..."
        $seedStart = Get-Date
        
        # Find seed executable
        $seeder = "$env:HC_BIN_DIR\seed_atoms_parallel.exe"
        if (-not (Test-Path $seeder)) { $seeder = $null }
        
        if ($seeder) {
            & $seeder -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT
            if ($LASTEXITCODE -ne 0) {
                # Fallback to SQL seeding
                Write-Host "  C++ seeder failed, falling back to SQL..." -ForegroundColor Yellow
                Invoke-Psql "SELECT seed_all_atoms();" | Out-Null
            }
        } else {
            # SQL fallback
            Write-Host "  Using SQL seeding (slower)..."
            Invoke-Psql "SELECT seed_all_atoms();" | Out-Null
        }
        
        $finalCount = Invoke-Psql "SELECT atoms FROM db_stats()" -Scalar
        $seedTime = (Get-Date) - $seedStart
        Write-Host "  Seeded $finalCount atoms in $($seedTime.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
    } else {
        Write-Host "  Atoms already seeded: $atomCount" -ForegroundColor Green
    }
} else {
    Write-Host "  Skipped (-SkipSeed)" -ForegroundColor DarkGray
}

# ============================================================================
# PHASE 7: INGEST MODELS
# ============================================================================
Write-Step 7 9 "INGEST EMBEDDING MODELS"

if (-not $SkipModels) {
    $modelPath = "$env:HC_PROJECT_ROOT\test-data\embedding_models"
    
    if (Test-Path $modelPath) {
        Write-Host "  Scanning $modelPath..."
        
        # Find model directories (have config.json)
        $models = Get-ChildItem -Path $modelPath -Recurse -Filter "config.json" -File | 
                  ForEach-Object { $_.Directory }
        
        if ($models.Count -gt 0) {
            $ingestTool = "$env:HC_BIN_DIR\ingest_safetensor.exe"
            if (-not (Test-Path $ingestTool)) { $ingestTool = $null }
            
            foreach ($modelDir in $models) {
                Write-Host "    Ingesting $($modelDir.Name)..." -NoNewline
                $beforeRel = Invoke-Psql "SELECT relations FROM db_stats()" -Scalar

                if ($ingestTool) {
                    & $ingestTool -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT $modelDir.FullName 2>&1 | Out-Null
                }

                $afterRel = Invoke-Psql "SELECT relations FROM db_stats()" -Scalar
                $newRels = [int]$afterRel - [int]$beforeRel
                
                if ($newRels -gt 0) {
                    Write-Host " +$newRels relations" -ForegroundColor Green
                } else {
                    Write-Host " (no new relations)" -ForegroundColor DarkGray
                }
            }
        } else {
            Write-Host "  No models found in $modelPath" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Model path not found: $modelPath" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (-SkipModels)" -ForegroundColor DarkGray
}

# ============================================================================
# PHASE 8: INGEST MOBY DICK
# ============================================================================
Write-Step 8 9 "INGEST TEST CONTENT (Moby Dick)"

if (-not $SkipContent) {
    $mobyPath = "$env:HC_PROJECT_ROOT\test-data\moby_dick.txt"
    
    if (Test-Path $mobyPath) {
        Write-Host "  Ingesting $mobyPath..."
        $beforeComp = Invoke-Psql "SELECT compositions FROM db_stats()" -Scalar

        $ingester = "$env:HC_BIN_DIR\ingest.exe"
        if (-not (Test-Path $ingester)) { $ingester = $null }
        
        if ($ingester) {
            & $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT $mobyPath 2>&1 | Out-Null
        }

        $afterComp = Invoke-Psql "SELECT compositions FROM db_stats()" -Scalar
        $newComps = [int]$afterComp - [int]$beforeComp
        
        Write-Host "  Created $newComps compositions" -ForegroundColor Green
    } else {
        Write-Host "  Moby Dick not found: $mobyPath" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (-SkipContent)" -ForegroundColor DarkGray
}

# ============================================================================
# PHASE 9: RUN FULL TEST SUITE
# ============================================================================
Write-Step 9 9 "FULL INTEGRATION TEST SUITE"

Write-Section "Enterprise SQL Tests"

# Run the enterprise test suite
$testFile = "$env:HC_PROJECT_ROOT\tests\sql\test_enterprise_suite.sql"
if (Test-Path $testFile) {
    $testOutput = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -f $testFile 2>&1
    
    # Parse results
    $passMatches = [regex]::Matches($testOutput, '\[PASS\]')
    $failMatches = [regex]::Matches($testOutput, '\[FAIL\]')
    
    $script:TestsPassed += $passMatches.Count
    $script:TestsFailed += $failMatches.Count
    $script:TestsRun += $passMatches.Count + $failMatches.Count
    
    if ($failMatches.Count -eq 0) {
        Write-Host "  Enterprise SQL Suite: $($passMatches.Count) tests passed" -ForegroundColor Green
    } else {
        Write-Host "  Enterprise SQL Suite: $($passMatches.Count) passed, $($failMatches.Count) failed" -ForegroundColor Red
    }
} else {
    Write-Host "  Enterprise test suite not found" -ForegroundColor Yellow
}

Write-Section "Additional Integration Tests"

# Test: Database connectivity
Test-Assert "database_connected" { 
    $result = Invoke-Psql "SELECT 1" -Scalar
    return $result -eq "1"
} "INTEGRATION"

# Test: Atom count
Test-Assert "atom_count_correct" {
    $count = [int](Invoke-Psql "SELECT atoms FROM db_stats()" -Scalar)
    return $count -ge 1100000
} "INTEGRATION"

# Test: Extensions loaded
Test-Assert "extensions_loaded" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM pg_extension WHERE extname IN ('hypercube','hypercube_ops','semantic_ops','generative','postgis')" -Scalar)
    return $count -ge 4
} "INTEGRATION"

# Test: Core functions exist
Test-Assert "core_functions_exist" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM pg_proc WHERE proname LIKE 'atom_%'" -Scalar)
    return $count -ge 10
} "INTEGRATION"

# Test: atom_by_codepoint works
Test-Assert "atom_by_codepoint_works" {
    $id = Invoke-Psql "SELECT atom_by_codepoint(65)" -Scalar
    return $id.Length -gt 0
} "INTEGRATION"

# Test: atom_text works
Test-Assert "atom_text_works" {
    $text = Invoke-Psql "SELECT atom_text(atom_by_codepoint(65))" -Scalar
    return $text -eq "A"
} "INTEGRATION"

# Test: atom_knn works
Test-Assert "atom_knn_returns_results" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM atom_knn(atom_by_codepoint(65), 10)" -Scalar)
    return $count -eq 10
} "INTEGRATION"

# Test: attention works
Test-Assert "attention_works" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM attention(atom_by_codepoint(65), 10)" -Scalar)
    return $count -eq 10
} "INTEGRATION"

# Test: centroid_distance works (requires geometry objects, not atom IDs)
Test-Assert "centroid_distance_works" {
    $dist = [double](Invoke-Psql "SELECT centroid_distance((SELECT geom FROM atom WHERE id = atom_by_codepoint(65)), (SELECT geom FROM atom WHERE id = atom_by_codepoint(66)))" -Scalar)
    return $dist -ge 0
} "INTEGRATION"

# Test: Hilbert coordinates populated
Test-Assert "hilbert_coordinates_populated" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM atom WHERE hilbert_lo IS NOT NULL LIMIT 1" -Scalar)
    return $count -gt 0
} "INTEGRATION"

# Test: Geometry coordinates valid
Test-Assert "geometry_coordinates_valid" {
    $valid = [int](Invoke-Psql "SELECT COUNT(*) FROM atom WHERE ST_X(geom) >= 0 AND ST_Y(geom) >= 0 LIMIT 10" -Scalar)
    return $valid -gt 0
} "INTEGRATION"

# Test: Unicode coverage
Test-Assert "unicode_full_coverage" {
    $count = [int](Invoke-Psql "SELECT COUNT(DISTINCT codepoint) FROM atom" -Scalar)
    return $count -ge 1100000
} "INTEGRATION"

# Test: Supplementary planes populated
Test-Assert "supplementary_planes" {
    $count = [int](Invoke-Psql "SELECT COUNT(*) FROM atom WHERE codepoint >= 65536" -Scalar)
    return $count -ge 1000000
} "INTEGRATION"

# Test: No orphaned records
Test-Assert "no_orphan_children" {
    $orphans = [int](Invoke-Psql "SELECT COUNT(*) FROM composition_child cc WHERE NOT EXISTS (SELECT 1 FROM composition c WHERE c.id = cc.composition_id)" -Scalar)
    return $orphans -eq 0
} "INTEGRATION"

Write-Section "Performance Tests"

# Test: KNN performance
Test-Assert "knn_performance_100_under_500ms" {
    $start = Get-Date
    Invoke-Psql "SELECT * FROM atom_knn(atom_by_codepoint(65), 100)" | Out-Null
    $elapsed = ((Get-Date) - $start).TotalMilliseconds
    if ($Verbose) { Write-Host "    KNN(100): $($elapsed.ToString('F2'))ms" }
    return $elapsed -lt 500
} "PERFORMANCE"

# Test: Codepoint lookup performance (single session, 100 lookups)
Test-Assert "codepoint_lookup_under_100ms_total" {
    $start = Get-Date
    # Use a single psql call with multiple lookups to avoid connection overhead
    $query = "SELECT atom_by_codepoint(65)"
    for ($i = 1; $i -lt 100; $i++) {
        $query += ", atom_by_codepoint($($i + 65))"
    }
    Invoke-Psql $query | Out-Null
    $elapsed = ((Get-Date) - $start).TotalMilliseconds
    if ($Verbose) { Write-Host "    100 atom_by_codepoint calls: $($elapsed.ToString('F2'))ms" }
    return $elapsed -lt 100
} "PERFORMANCE"

# Test: Range query performance
Test-Assert "range_query_under_100ms" {
    $start = Get-Date
    Invoke-Psql "SELECT COUNT(*) FROM atom WHERE codepoint BETWEEN 65 AND 90" | Out-Null
    $elapsed = ((Get-Date) - $start).TotalMilliseconds
    if ($Verbose) { Write-Host "    Range query: $($elapsed.ToString('F2'))ms" }
    return $elapsed -lt 100
} "PERFORMANCE"

# ============================================================================
# C++ UNIT TESTS (via CTest)
# ============================================================================
Write-Section "C++ Unit Tests"

Push-Location "$env:HC_BUILD_DIR"
try {
    if (Test-Path "CTestTestfile.cmake") {
        Write-Host "  Running CTest..."
        $ctestOutput = & ctest --output-on-failure -C $env:HC_BUILD_TYPE 2>&1 | Out-String
        
        # Parse CTest output - look for "X tests passed, Y tests failed" or "X% tests passed"
        if ($ctestOutput -match "(\d+) tests? passed") {
            $ctestPassed = [int]$Matches[1]
            $script:TestsPassed += $ctestPassed
            $script:TestsRun += $ctestPassed
            Write-Host "  CTest: $ctestPassed tests passed" -ForegroundColor Green
        }
        if ($ctestOutput -match "(\d+) tests? failed") {
            $ctestFailed = [int]$Matches[1]
            $script:TestsFailed += $ctestFailed
            Write-Host "  CTest: $ctestFailed tests failed" -ForegroundColor Red
        }
        
        # Also check for "Total Test time" to confirm CTest ran
        if ($ctestOutput -match "Total Test time") {
            Write-Host "  CTest execution completed" -ForegroundColor Green
        }
    } else {
        Write-Host "  CTest not configured" -ForegroundColor DarkGray
    }
} finally {
    Pop-Location
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
$totalTime = (Get-Date) - $script:StartTime

Write-Header "TEST SUMMARY"

Write-Host ""
Write-Host "  Total Tests:  $($script:TestsRun)"
Write-Host "  Passed:       $($script:TestsPassed)" -ForegroundColor Green
Write-Host "  Failed:       $($script:TestsFailed)" -ForegroundColor $(if ($script:TestsFailed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($script:TestsFailed -gt 0) {
    Write-Host "  Failed Tests:" -ForegroundColor Red
    foreach ($failure in $script:Failures) {
        Write-Host "    - $failure" -ForegroundColor Red
    }
    Write-Host ""
}

$passRate = if ($script:TestsRun -gt 0) { [math]::Round(($script:TestsPassed / $script:TestsRun) * 100, 1) } else { 0 }
Write-Host "  Pass Rate:    $passRate%"
Write-Host "  Duration:     $($totalTime.ToString('hh\:mm\:ss\.fff'))"
Write-Host ""

# Database stats
Write-Host "  Database Statistics:" -ForegroundColor Cyan
$atomCount = Invoke-Psql "SELECT atoms FROM db_stats()" -Scalar
$compCount = Invoke-Psql "SELECT compositions FROM db_stats()" -Scalar
$relCount = Invoke-Psql "SELECT relations FROM db_stats()" -Scalar
Write-Host "    Atoms:        $atomCount"
Write-Host "    Compositions: $compCount"
Write-Host "    Relations:    $relCount"
Write-Host ""

if ($script:TestsFailed -eq 0) {
    Write-Host "  ✓ ALL TESTS PASSED" -ForegroundColor Green -BackgroundColor DarkGreen
} else {
    Write-Host "  ✗ SOME TESTS FAILED" -ForegroundColor White -BackgroundColor DarkRed
}

Write-Host ""
Write-Header "END-TO-END TEST COMPLETE"

exit $(if ($script:TestsFailed -eq 0) { 0 } else { 1 })
