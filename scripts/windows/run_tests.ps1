#!/usr/bin/env pwsh
# =============================================================================
# Hypercube Enterprise Test Suite Runner
# =============================================================================
# Comprehensive test runner that orchestrates C++, SQL, and integration tests
# with proper reporting, timing, and exit codes.
#
# Usage:
#   .\run_tests.ps1              # Run all tests
#   .\run_tests.ps1 -Category cpp    # Run only C++ tests
#   .\run_tests.ps1 -Category sql    # Run only SQL tests
#   .\run_tests.ps1 -Verbose         # Verbose output
#   .\run_tests.ps1 -FailFast        # Stop on first failure
#
# Exit Codes:
#   0 = All tests passed
#   1 = Some tests failed
#   2 = Test infrastructure error
# =============================================================================

param(
    [ValidateSet('all', 'cpp', 'sql', 'integration')]
    [string]$Category = 'all',
    [switch]$Verbose,
    [switch]$FailFast,
    [switch]$NoBuild,
    [string]$DbName = 'hypercube',
    [string]$DbUser = 'hartonomous',
    [string]$DbPass = 'hartonomous',
    [string]$DbHost = 'localhost',
    [int]$DbPort = 5432
)

$ErrorActionPreference = 'Continue'
$script:TestResults = @{
    Passed = 0
    Failed = 0
    Skipped = 0
    StartTime = Get-Date
    Categories = @{}
}

# =============================================================================
# Utility Functions
# =============================================================================

function Write-Header {
    param([string]$Text, [string]$Char = '=')
    $line = $Char * 78
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-SubHeader {
    param([string]$Text)
    Write-Host ""
    Write-Host "--- $Text ---" -ForegroundColor Yellow
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Details = '',
        [double]$Duration = 0
    )
    
    $status = if ($Passed) { "[PASS]" } else { "[FAIL]" }
    $color = if ($Passed) { "Green" } else { "Red" }
    $durationStr = if ($Duration -gt 0) { " (${Duration}ms)" } else { "" }
    
    Write-Host "$status $TestName$durationStr" -ForegroundColor $color
    if ($Details -and (-not $Passed -or $Verbose)) {
        Write-Host "       $Details" -ForegroundColor Gray
    }
    
    if ($Passed) {
        $script:TestResults.Passed++
    } else {
        $script:TestResults.Failed++
    }
}

function Initialize-Category {
    param([string]$Name)
    $script:TestResults.Categories[$Name] = @{
        Passed = 0
        Failed = 0
        StartTime = Get-Date
    }
}

function Complete-Category {
    param([string]$Name)
    $cat = $script:TestResults.Categories[$Name]
    $cat.Duration = ((Get-Date) - $cat.StartTime).TotalSeconds
}

function Get-PostgresConnStr {
    return "host=$DbHost port=$DbPort dbname=$DbName user=$DbUser password=$DbPass"
}

# =============================================================================
# C++ Test Runner (via CTest)
# =============================================================================

function Invoke-CppTests {
    Write-Header "C++ Unit Tests (CTest)"
    Initialize-Category "cpp"
    
    $buildDir = Join-Path $PSScriptRoot "..\..\cpp\build"
    
    if (-not (Test-Path $buildDir)) {
        Write-Host "Build directory not found: $buildDir" -ForegroundColor Red
        return $false
    }
    
    Push-Location $buildDir
    try {
        # Run CTest with verbose output
        $ctestArgs = @('--output-on-failure', '--timeout', '120')
        if ($Verbose) { $ctestArgs += '-V' }
        
        Write-SubHeader "Running CTest"
        $output = & ctest @ctestArgs 2>&1
        $exitCode = $LASTEXITCODE
        
        # Parse CTest output for results
        $passCount = 0
        $failCount = 0
        $testLines = $output | Select-String -Pattern '^\s*\d+/\d+ Test' -AllMatches
        
        foreach ($line in $output) {
            if ($line -match '(\d+) tests passed') {
                $passCount = [int]$Matches[1]
            }
            if ($line -match '(\d+) tests failed') {
                $failCount = [int]$Matches[1]
            }
            if ($Verbose) {
                Write-Host $line -ForegroundColor Gray
            }
        }
        
        # Individual test parsing
        $output | ForEach-Object {
            if ($_ -match '^\s*\d+/\d+ Test\s+#\d+:\s+(\w+)\s+\.+\s+(Passed|Failed)') {
                $testName = $Matches[1]
                $passed = $Matches[2] -eq 'Passed'
                Write-TestResult "CPP::$testName" $passed
                
                $script:TestResults.Categories.cpp.$(if($passed){'Passed'}else{'Failed'})++
            }
        }
        
        if ($exitCode -eq 0) {
            Write-Host "`nAll C++ tests passed!" -ForegroundColor Green
        } else {
            Write-Host "`n$failCount C++ tests failed" -ForegroundColor Red
            if ($FailFast) { throw "C++ tests failed" }
        }
        
        return ($exitCode -eq 0)
    }
    finally {
        Pop-Location
        Complete-Category "cpp"
    }
}

# =============================================================================
# SQL Test Runner
# =============================================================================

function Invoke-SqlTest {
    param(
        [string]$TestFile,
        [string]$TestName
    )
    
    $start = Get-Date
    $connStr = Get-PostgresConnStr
    
    try {
        $env:PGPASSWORD = $DbPass
        $output = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -f $TestFile 2>&1
        $exitCode = $LASTEXITCODE
        $duration = ((Get-Date) - $start).TotalMilliseconds
        
        # Check for FAIL or ERROR in output
        $hasError = $output | Select-String -Pattern 'FAIL|ERROR|EXCEPTION' -Quiet
        $passed = ($exitCode -eq 0) -and (-not $hasError)
        
        # Extract pass/fail counts from output
        $passMatches = ($output | Select-String -Pattern 'PASSED:' -AllMatches).Matches.Count
        $failMatches = ($output | Select-String -Pattern 'FAILED:' -AllMatches).Matches.Count
        
        $details = "Passed: $passMatches, Failed: $failMatches"
        Write-TestResult "SQL::$TestName" $passed $details $duration
        
        if (-not $passed -and $Verbose) {
            Write-Host "Output:" -ForegroundColor Yellow
            $output | Where-Object { $_ -match 'FAIL|ERROR|PASS' } | ForEach-Object {
                Write-Host "  $_" -ForegroundColor Gray
            }
        }
        
        return $passed
    }
    catch {
        Write-TestResult "SQL::$TestName" $false $_.Exception.Message
        return $false
    }
}

function Invoke-SqlTests {
    Write-Header "SQL Test Suite"
    Initialize-Category "sql"
    
    $sqlTestDir = Join-Path $PSScriptRoot "..\..\tests\sql"
    
    if (-not (Test-Path $sqlTestDir)) {
        Write-Host "SQL test directory not found: $sqlTestDir" -ForegroundColor Red
        return $false
    }
    
    # Test database connectivity first
    Write-SubHeader "Database Connectivity"
    try {
        $env:PGPASSWORD = $DbPass
        $result = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -c "SELECT 1 as connected;" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-TestResult "SQL::Connectivity" $false "Cannot connect to database"
            return $false
        }
        Write-TestResult "SQL::Connectivity" $true
    }
    catch {
        Write-TestResult "SQL::Connectivity" $false $_.Exception.Message
        return $false
    }
    
    # Check extensions
    Write-SubHeader "Extension Validation"
    $extCheck = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c "SELECT extname FROM pg_extension ORDER BY extname;" 2>&1
    $extensions = @('hypercube', 'hypercube_ops', 'generative', 'semantic_ops', 'postgis')
    foreach ($ext in $extensions) {
        $found = $extCheck -match $ext
        Write-TestResult "SQL::Extension::$ext" $found
        $script:TestResults.Categories.sql.$(if($found){'Passed'}else{'Failed'})++
    }
    
    # Run test files
    Write-SubHeader "Test Files"
    $testFiles = Get-ChildItem -Path $sqlTestDir -Filter "*.sql" -File | 
                 Where-Object { $_.Name -notmatch '^archive' }
    
    $allPassed = $true
    foreach ($file in $testFiles) {
        $passed = Invoke-SqlTest -TestFile $file.FullName -TestName $file.BaseName
        $script:TestResults.Categories.sql.$(if($passed){'Passed'}else{'Failed'})++
        if (-not $passed) {
            $allPassed = $false
            if ($FailFast) { throw "SQL test failed: $($file.Name)" }
        }
    }
    
    Complete-Category "sql"
    return $allPassed
}

# =============================================================================
# Integration Tests
# =============================================================================

function Invoke-IntegrationTests {
    Write-Header "Integration Tests"
    Initialize-Category "integration"
    
    $env:PGPASSWORD = $DbPass
    
    # Test 1: Atom count validation
    Write-SubHeader "Data Integrity"
    $atomCount = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c "SELECT COUNT(*) FROM atom;" 2>&1
    $atomCount = [int]($atomCount.Trim())
    $passed = $atomCount -gt 0
    Write-TestResult "INT::AtomCount" $passed "Count: $atomCount"
    $script:TestResults.Categories.integration.$(if($passed){'Passed'}else{'Failed'})++
    
    # Test 2: Codepoint coverage
    $cpResult = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c "SELECT MIN(codepoint), MAX(codepoint), COUNT(*) FROM atom WHERE depth=0;" 2>&1
    $parts = $cpResult.Trim() -split '\|'
    if ($parts.Count -eq 3) {
        $min = [int]$parts[0].Trim()
        $max = [int]$parts[1].Trim()
        $count = [int]$parts[2].Trim()
        $passed = ($min -eq 0) -and ($max -eq 1114111) -and ($count -eq 1114112)
        Write-TestResult "INT::CodepointCoverage" $passed "Range: $min-$max, Count: $count (expected 1114112)"
    } else {
        Write-TestResult "INT::CodepointCoverage" $false "Failed to parse result"
    }
    $script:TestResults.Categories.integration.$(if($passed){'Passed'}else{'Failed'})++
    
    # Test 3: Coordinate validation (all atoms have valid 4D coords)
    $coordCheck = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c @"
SELECT COUNT(*) FROM atom 
WHERE coords IS NOT NULL 
  AND ST_X(coords) BETWEEN 0 AND 1
  AND ST_Y(coords) BETWEEN 0 AND 1
  AND ST_Z(coords) BETWEEN 0 AND 1
  AND ST_M(coords) BETWEEN 0 AND 1;
"@ 2>&1
    $validCoords = [int]($coordCheck.Trim())
    $passed = $validCoords -eq $atomCount
    Write-TestResult "INT::CoordinateValidity" $passed "Valid: $validCoords / $atomCount"
    $script:TestResults.Categories.integration.$(if($passed){'Passed'}else{'Failed'})++
    
    # Test 4: HNSW index exists and is valid
    $indexCheck = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c @"
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'atom' AND indexdef LIKE '%hnsw%';
"@ 2>&1
    $hasIndex = $indexCheck -match 'hnsw'
    Write-TestResult "INT::HNSWIndex" $hasIndex
    $script:TestResults.Categories.integration.$(if($hasIndex){'Passed'}else{'Failed'})++
    
    # Test 5: Core functions exist
    Write-SubHeader "Core Functions"
    $coreFunctions = @(
        'hypercube_ingest_text',
        'hypercube_retrieve_text', 
        'hypercube_knn_search',
        'hypercube_text_similarity',
        'hypercube_semantic_search'
    )
    
    foreach ($fn in $coreFunctions) {
        $fnCheck = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c @"
SELECT COUNT(*) FROM pg_proc WHERE proname = '$fn';
"@ 2>&1
        $exists = [int]($fnCheck.Trim()) -gt 0
        Write-TestResult "INT::Function::$fn" $exists
        $script:TestResults.Categories.integration.$(if($exists){'Passed'}else{'Failed'})++
    }
    
    # Test 6: Ingestion roundtrip
    Write-SubHeader "Ingestion Roundtrip"
    $testText = "Hello World Test $(Get-Date -Format 'HHmmss')"
    $ingestResult = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c @"
SELECT hypercube_retrieve_text(hypercube_ingest_text('$testText'));
"@ 2>&1
    $retrieved = $ingestResult.Trim()
    $passed = $retrieved -eq $testText
    Write-TestResult "INT::IngestRoundtrip" $passed "Expected: '$testText', Got: '$retrieved'"
    $script:TestResults.Categories.integration.$(if($passed){'Passed'}else{'Failed'})++
    
    # Test 7: KNN Search
    Write-SubHeader "Query Operations"
    $knnResult = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -c @"
SELECT COUNT(*) FROM hypercube_knn_search(
    (SELECT coords FROM atom WHERE codepoint = 65 LIMIT 1),
    10
);
"@ 2>&1
    $knnCount = [int]($knnResult.Trim())
    $passed = $knnCount -eq 10
    Write-TestResult "INT::KNNSearch" $passed "Returned: $knnCount results (expected 10)"
    $script:TestResults.Categories.integration.$(if($passed){'Passed'}else{'Failed'})++
    
    Complete-Category "integration"
    return ($script:TestResults.Categories.integration.Failed -eq 0)
}

# =============================================================================
# Main Execution
# =============================================================================

Write-Header "HYPERCUBE ENTERPRISE TEST SUITE" "="
Write-Host "Category: $Category"
Write-Host "Database: $DbName@$DbHost`:$DbPort"
Write-Host "Start Time: $($script:TestResults.StartTime)"

$allPassed = $true

try {
    # Build first if needed
    if (-not $NoBuild -and ($Category -eq 'all' -or $Category -eq 'cpp')) {
        Write-SubHeader "Building C++ (Release)"
        $buildDir = Join-Path $PSScriptRoot "..\..\cpp\build"
        if (Test-Path $buildDir) {
            Push-Location $buildDir
            & cmake --build . --config Release --parallel 2>&1 | Out-Null
            Pop-Location
        }
    }
    
    # Run tests by category
    switch ($Category) {
        'all' {
            if (-not (Invoke-CppTests)) { $allPassed = $false }
            if (-not $FailFast -or $allPassed) {
                if (-not (Invoke-SqlTests)) { $allPassed = $false }
            }
            if (-not $FailFast -or $allPassed) {
                if (-not (Invoke-IntegrationTests)) { $allPassed = $false }
            }
        }
        'cpp' {
            if (-not (Invoke-CppTests)) { $allPassed = $false }
        }
        'sql' {
            if (-not (Invoke-SqlTests)) { $allPassed = $false }
        }
        'integration' {
            if (-not (Invoke-IntegrationTests)) { $allPassed = $false }
        }
    }
}
catch {
    Write-Host "`nTest execution aborted: $_" -ForegroundColor Red
    $allPassed = $false
}

# =============================================================================
# Summary Report
# =============================================================================

$totalDuration = ((Get-Date) - $script:TestResults.StartTime).TotalSeconds

Write-Header "TEST SUMMARY"

# Category breakdown
foreach ($cat in $script:TestResults.Categories.Keys | Sort-Object) {
    $c = $script:TestResults.Categories[$cat]
    $catTotal = $c.Passed + $c.Failed
    $catPct = if ($catTotal -gt 0) { [math]::Round(($c.Passed / $catTotal) * 100, 1) } else { 0 }
    $color = if ($c.Failed -eq 0) { "Green" } else { "Red" }
    Write-Host "  $($cat.ToUpper().PadRight(15)) Passed: $($c.Passed.ToString().PadLeft(3))  Failed: $($c.Failed.ToString().PadLeft(3))  ($catPct%)" -ForegroundColor $color
}

Write-Host ""
Write-Host ("=" * 78) -ForegroundColor Cyan

$totalTests = $script:TestResults.Passed + $script:TestResults.Failed
$totalPct = if ($totalTests -gt 0) { [math]::Round(($script:TestResults.Passed / $totalTests) * 100, 1) } else { 0 }

if ($allPassed) {
    Write-Host "  RESULT: ALL TESTS PASSED" -ForegroundColor Green
} else {
    Write-Host "  RESULT: SOME TESTS FAILED" -ForegroundColor Red
}

Write-Host "  Total: $totalTests tests, $($script:TestResults.Passed) passed, $($script:TestResults.Failed) failed ($totalPct%)"
Write-Host "  Duration: $([math]::Round($totalDuration, 2)) seconds"
Write-Host ("=" * 78) -ForegroundColor Cyan

# Exit with appropriate code
exit $(if ($allPassed) { 0 } else { 1 })
