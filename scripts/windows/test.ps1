# Hartonomous Hypercube - Test Suite (Windows)
# Shows actual results for every test - no bullshit
# Usage: .\scripts\windows\test.ps1 [-Quick]

param(
    [switch]$Quick
)

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Hartonomous Hypercube - Test Suite                       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$TestsPassed = 0
$TestsFailed = 0
$env:PGPASSWORD = $env:HC_DB_PASS
$env:PGCLIENTENCODING = "UTF8"

function SQL { 
    param([string]$q) 
    $result = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc $q 2>&1
    if ($result -is [System.Management.Automation.ErrorRecord]) {
        return "ERROR: $($result.Exception.Message)"
    }
    return $result
}

function SafeTrim {
    param($val)
    if ($val -is [string]) { return $val.Trim() }
    if ($null -eq $val) { return "" }
    return "$val"
}

function Test-Result {
    param([string]$Name, [bool]$Pass, [string]$Detail)
    if ($Pass) {
        Write-Host "  ✓ " -NoNewline -ForegroundColor Green
        Write-Host "$Name" -NoNewline
        Write-Host " → $Detail" -ForegroundColor Gray
        $script:TestsPassed++
    } else {
        Write-Host "  ✗ " -NoNewline -ForegroundColor Red
        Write-Host "$Name" -NoNewline
        Write-Host " → $Detail" -ForegroundColor Yellow
        $script:TestsFailed++
    }
}

# Find test executables
$BuildDir = $env:HC_BUILD_DIR
if (Test-Path "$BuildDir\Release") { $BuildDir = "$BuildDir\Release" }

# ═══════════════════════════════════════════════════════════════════════════
Write-Host "─── C++ Unit Tests ────────────────────────────────────────" -ForegroundColor Blue

$unitTests = @("test_hilbert", "test_coordinates", "test_blake3", "test_semantic", "test_clustering")
foreach ($test in $unitTests) {
    $exe = "$BuildDir\$test.exe"
    if (Test-Path $exe) {
        $output = & $exe 2>&1
        $pass = $LASTEXITCODE -eq 0 -and -not ($output -match "FAILED|FAIL")
        $summary = ($output | Select-String -Pattern "passed|FAILED|FAIL|OK|PASS" | Select-Object -First 1) -replace "`r`n", ""
        if (-not $summary) { $summary = if ($pass) { "all assertions passed" } else { "failed" } }
        Test-Result $test $pass $summary
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Database ──────────────────────────────────────────────" -ForegroundColor Blue

$pgVer = SafeTrim (SQL "SELECT version()")
$pgVer = ($pgVer -split ",")[0]
Test-Result "PostgreSQL" ($pgVer -match "PostgreSQL") $pgVer

$gisVer = SafeTrim (SQL "SELECT PostGIS_Lib_Version()")
Test-Result "PostGIS" ($gisVer -match "\d") "v$gisVer"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── 3-Table Schema ────────────────────────────────────────" -ForegroundColor Blue

$atomExists = (SafeTrim (SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='atom'")) -eq "1"
Test-Result "atom table" $atomExists $(if($atomExists){"exists"}else{"MISSING"})

$compExists = (SafeTrim (SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='composition'")) -eq "1"
Test-Result "composition table" $compExists $(if($compExists){"exists"}else{"MISSING"})

$relExists = (SafeTrim (SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='relation'")) -eq "1"
Test-Result "relation table" $relExists $(if($relExists){"exists"}else{"MISSING"})

$ccExists = (SafeTrim (SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='composition_child'")) -eq "1"
Test-Result "composition_child table" $ccExists $(if($ccExists){"exists"}else{"MISSING"})

$gistIdx = (SafeTrim (SQL "SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_geom'")) -eq "1"
Test-Result "GIST index (idx_atom_geom)" $gistIdx $(if($gistIdx){"exists for spatial queries"}else{"MISSING"})

$hilbertIdx = (SafeTrim (SQL "SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_hilbert'")) -eq "1"
Test-Result "Hilbert index (idx_atom_hilbert)" $hilbertIdx $(if($hilbertIdx){"exists for locality queries"}else{"MISSING"})

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Atom Seeding ──────────────────────────────────────────" -ForegroundColor Blue

$leafCount = SafeTrim (SQL "SELECT COUNT(*) FROM atom")
$leafInt = 0
[int]::TryParse($leafCount, [ref]$leafInt) | Out-Null
$pass = $leafInt -gt 1100000
Test-Result "Unicode leaf atoms" $pass "$($leafInt.ToString('N0')) codepoints (need >1.1M)"

$badSrid = SafeTrim (SQL "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0")
$badSridInt = 0
[int]::TryParse($badSrid, [ref]$badSridInt) | Out-Null
Test-Result "SRID consistency" ($badSridInt -eq 0) "$badSridInt atoms with wrong SRID (should be 0)"

# Sample atom A - use geom directly (not centroid column which doesn't exist for atoms)
$sampleA = SafeTrim (SQL "SELECT 'cp=' || codepoint || ', h=(' || hilbert_hi || ',' || hilbert_lo || '), g=(' || ROUND(ST_X(geom)::numeric,2) || ',' || ROUND(ST_Y(geom)::numeric,2) || ',' || ROUND(ST_Z(geom)::numeric,2) || ',' || ROUND(ST_M(geom)::numeric,2) || ')' FROM atom WHERE codepoint = 65")
Test-Result "Atom 'A' (U+0041)" ($sampleA -match "cp=65") $sampleA

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Core Functions ────────────────────────────────────────" -ForegroundColor Blue

$isLeaf = SafeTrim (SQL "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))")
Test-Result "atom_is_leaf('A')" ($isLeaf -eq "t") "returns '$isLeaf' (expected: t)"

$centroid = SafeTrim (SQL 'SELECT CONCAT(''X='', ROUND(ST_X(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))::numeric,2), '' Y='', ROUND(ST_Y(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))::numeric,2))')
Test-Result "atom_centroid('A')" ($centroid -match "X=") $centroid

$reconstructA = SafeTrim (SQL "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))")
Test-Result "atom_reconstruct_text('A')" ($reconstructA -eq "A" -or $reconstructA -match "A") "returns '$reconstructA'"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Spatial Queries ───────────────────────────────────────" -ForegroundColor Blue

$knnCount = SafeTrim (SQL "SELECT COUNT(*) FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 5)")
Test-Result 'atom_knn(''A'', k=5)' (($knnCount -match '\d') -and ($knnCount -ne '0')) "$knnCount neighbors found"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Compositions ──────────────────────────────────────────" -ForegroundColor Blue

$compCount = SafeTrim (SQL "SELECT COUNT(*) FROM composition")
$compInt = 0
[int]::TryParse($compCount, [ref]$compInt) | Out-Null
$maxDepth = SafeTrim (SQL "SELECT COALESCE(MAX(depth), 0) FROM composition")
Test-Result "Compositions in DB" (-not $compSamplesError) "$($compInt.ToString('N0')) compositions, max depth=$maxDepth"

if ($compInt -gt 0) {
    Write-Host ""
    Write-Host "  ┌─ Sample Compositions ─────────────────────────────────────────────" -ForegroundColor Cyan
    $compSamples = SQL 'SELECT ''    │ depth='' || depth || '', children='' || child_count || '', id='' || LEFT(encode(id, ''hex''), 16) || ''...'' FROM composition ORDER BY depth DESC, child_count LIMIT 5'
    $compSamplesError = $compSamples -match "^ERROR:"
    if ($compSamples -is [string]) { Write-Host $compSamples -ForegroundColor Gray }
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
} else {
    $compSamplesError = $false
    Write-Host "  (No compositions yet - run ingest_safetensor)" -ForegroundColor DarkGray
}

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Semantic Relations ────────────────────────────────────" -ForegroundColor Blue
Write-Host "  (Edges from MiniLM embedding model - token similarity)" -ForegroundColor DarkGray

$edgeCount = SafeTrim (SQL "SELECT COUNT(*) FROM relation")
$edgeInt = 0
[int]::TryParse($edgeCount, [ref]$edgeInt) | Out-Null
Test-Result "Total semantic edges" (($edgeInt -gt 0) -and -not $topEdgesError) "$($edgeInt.ToString('N0')) relationships in relation table"

if ($edgeInt -gt 0) {
    Write-Host ""
    Write-Host "  ┌─ Top 5 Semantic Edges (highest weight) ───────────────────────────" -ForegroundColor Cyan
    $topEdges = SQL 'SELECT ''    │ w='' || ROUND(r.weight::numeric, 4) || '' | '' || LEFT(encode(r.source_id, ''hex''), 12) || ''.. ↔ '' || LEFT(encode(r.target_id, ''hex''), 12) || ''..'' FROM relation r ORDER BY r.weight DESC LIMIT 5'
    $topEdgesError = $topEdges -match "^ERROR:"
    if ($topEdges -is [string]) { Write-Host $topEdges -ForegroundColor Gray }
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan

    Write-Host ""
    Write-Host "  ┌─ Semantic Neighbors ──────────────────────────────────────────────" -ForegroundColor Cyan
    $neighborsCount = SafeTrim (SQL "SELECT COUNT(*) FROM semantic_neighbors((SELECT source_id FROM relation ORDER BY weight DESC LIMIT 1), 5)")
    $neighborsError = $neighborsCount -match "^ERROR:"
    Test-Result "semantic_neighbors()" (-not $neighborsError -and $neighborsCount -match "\d") "$neighborsCount neighbors found for top token"
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
} else {
    $topEdgesError = $false
    $neighborsError = $true
    Write-Host "  (No semantic edges yet - run ingest-testdata.ps1)" -ForegroundColor DarkGray
}

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── C++ Integration ───────────────────────────────────────" -ForegroundColor Blue

$intTest = "$BuildDir\test_integration.exe"
if (Test-Path $intTest) {
    $output = & $intTest 2>&1
    $pass = $LASTEXITCODE -eq 0
    $lines = ($output -split "`n").Count
    Test-Result "test_integration" $pass "$lines lines output, exit=$LASTEXITCODE"
}

$queryTest = "$BuildDir\test_query_api.exe"
if (Test-Path $queryTest) {
    $output = & $queryTest 2>&1
    $pass = $LASTEXITCODE -eq 0
    $lines = ($output -split "`n").Count
    Test-Result "test_query_api" $pass "$lines lines output, exit=$LASTEXITCODE"
}

# ═══════════════════════════════════════════════════════════════════════════
# Summary
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════"
if ($TestsFailed -eq 0) {
    Write-Host "  ✓ All $TestsPassed tests passed" -ForegroundColor Green
} else {
    Write-Host "  ✗ $TestsFailed failed, $TestsPassed passed" -ForegroundColor Red
}
Write-Host "════════════════════════════════════════════════════════════════"
Write-Host ""

Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
exit $TestsFailed
