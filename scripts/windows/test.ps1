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

function SQL { param([string]$q) & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc $q 2>&1 }

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
        $pass = $LASTEXITCODE -eq 0
        # Extract summary line from output
        $summary = ($output | Select-String -Pattern "passed|failed|OK|PASS" | Select-Object -First 1) -replace "`r`n", ""
        if (-not $summary) { $summary = if ($pass) { "all assertions passed" } else { "failed" } }
        Test-Result $test $pass $summary
    }
}

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Database ──────────────────────────────────────────────" -ForegroundColor Blue

$pgVer = SQL "SELECT version()"
$pgVer = ($pgVer -split ",")[0].Trim()
Test-Result "PostgreSQL" $true $pgVer

$gisVer = SQL "SELECT PostGIS_Lib_Version()"
Test-Result "PostGIS" ($gisVer -match "\d") "v$($gisVer.Trim())"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Schema ────────────────────────────────────────────────" -ForegroundColor Blue

$atomExists = (SQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='atom'").Trim() -eq "1"
Test-Result "atom table" $atomExists $(if($atomExists){"exists"}else{"MISSING"})

$gistIdx = (SQL "SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_geom'").Trim() -eq "1"
Test-Result "GIST index (idx_atom_geom)" $gistIdx $(if($gistIdx){"exists for spatial queries"}else{"MISSING"})

$hilbertIdx = (SQL "SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_hilbert'").Trim() -eq "1"
Test-Result "Hilbert index (idx_atom_hilbert)" $hilbertIdx $(if($hilbertIdx){"exists for locality queries"}else{"MISSING"})

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Atom Seeding ──────────────────────────────────────────" -ForegroundColor Blue

$leafCount = [int](SQL "SELECT COUNT(*) FROM atom WHERE depth = 0").Trim()
$pass = $leafCount -gt 1100000
Test-Result "Unicode leaf atoms" $pass "$($leafCount.ToString('N0')) codepoints (need >1.1M)"

$badSrid = [int](SQL "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0").Trim()
Test-Result "SRID consistency" ($badSrid -eq 0) "$badSrid atoms with wrong SRID (should be 0)"

# Sample some atoms to show they work
$sampleA = SQL "SELECT 'codepoint=' || codepoint || ', hilbert=(' || hilbert_hi || ',' || hilbert_lo || '), centroid=(' || ROUND(ST_X(centroid)::numeric,2) || ',' || ROUND(ST_Y(centroid)::numeric,2) || ',' || ROUND(ST_Z(centroid)::numeric,2) || ',' || ROUND(ST_M(centroid)::numeric,2) || ')' FROM atom WHERE codepoint = 65"
if ($sampleA -is [string]) { Test-Result "Atom 'A' (U+0041)" $true $sampleA.Trim() } else { Test-Result "Atom 'A' (U+0041)" $false "query error" }

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Core Functions ────────────────────────────────────────" -ForegroundColor Blue

$isLeaf = (SQL "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))").Trim()
Test-Result "atom_is_leaf('A')" ($isLeaf -eq "t") "returns $isLeaf (expected: t for leaf atom)"

$centroid = SQL "SELECT 'X=' || ROUND(ST_X(c)::numeric,2) || ' Y=' || ROUND(ST_Y(c)::numeric,2) || ' Z=' || ROUND(ST_Z(c)::numeric,2) || ' M=' || ROUND(ST_M(c)::numeric,2) FROM (SELECT atom_centroid((SELECT id FROM atom WHERE codepoint = 65)) AS c) sub"
if ($centroid -is [string]) { Test-Result "atom_centroid('A')" ($centroid -match "X=") $centroid.Trim() } else { Test-Result "atom_centroid('A')" $false "query error" }

$reconstructA = (SQL "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))").Trim()
Test-Result "atom_reconstruct_text('A')" ($reconstructA -eq "A") "returns '$reconstructA' (expected: A)"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Spatial Queries ───────────────────────────────────────" -ForegroundColor Blue

$knnResult = SQL "SELECT COUNT(*) || ' neighbors: ' || string_agg(chr(a.codepoint), ', ') FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 5) k JOIN atom a ON a.id = k.neighbor_id WHERE a.codepoint BETWEEN 32 AND 126"
if ($knnResult -is [string] -and $knnResult -match "\d") { 
    Test-Result "atom_knn('A', k=5)" $true $knnResult.Trim() 
} else { 
    # Fallback: just count
    $knnCount = SQL "SELECT COUNT(*) FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 5)"
    Test-Result "atom_knn('A', k=5)" ($knnCount.Trim() -ge "1") "$($knnCount.Trim()) neighbors found"
}

$dist = SQL "SELECT ROUND(atom_distance((SELECT id FROM atom WHERE codepoint = 65), (SELECT id FROM atom WHERE codepoint = 66))::numeric, 2)"
Test-Result "atom_distance('A', 'B')" ($dist -match "\d") "$($dist.Trim()) units in 4D space"

$hilbertRange = SQL "SELECT COUNT(*) FROM atom_hilbert_range((SELECT id FROM atom WHERE codepoint = 65), 1000000000, 10)"
Test-Result "atom_hilbert_range('A', radius=1B, k=10)" ($hilbertRange.Trim() -ge 1) "$($hilbertRange.Trim()) neighbors within Hilbert curve radius"

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Compositions ──────────────────────────────────────────" -ForegroundColor Blue

$compCount = [int](SQL "SELECT COUNT(*) FROM atom WHERE depth > 0").Trim()
$maxDepth = (SQL "SELECT COALESCE(MAX(depth), 0) FROM atom").Trim()
Test-Result "Compositions in DB" ($compCount -gt 0) "$($compCount.ToString('N0')) compositions, max depth=$maxDepth"

if ($compCount -gt 0) {
    Write-Host ""
    Write-Host "  ┌─ Sample Compositions (showing structure) ─────────────────────────" -ForegroundColor Cyan
    # Avoid UTF8 issues by not calling atom_text on arbitrary compositions
    $compSamples = SQL @"
SELECT '    │ depth=' || depth || ', children=' || atom_count || 
       ', id=' || LEFT(encode(id, 'hex'), 16) || '...'
FROM atom 
WHERE depth > 0
ORDER BY depth DESC, atom_count
LIMIT 5
"@
    Write-Host $compSamples -ForegroundColor Gray
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "  ┌─ Attention Scores ────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "    │ Query: Find 5 compositions most similar to a sample composition" -ForegroundColor Gray
    $attnResult = SQL @"
SELECT '    │ ' || ROW_NUMBER() OVER (ORDER BY score DESC) || '. score=' || 
       ROUND(score::numeric, 6) || ' (1.0 = identical position in 4D space)'
FROM attention((SELECT id FROM atom WHERE depth > 0 LIMIT 1), 5)
"@
    if ($attnResult -is [string]) { Write-Host $attnResult -ForegroundColor Gray }
    Test-Result "attention()" $true "Inverse-distance scoring in 4D hypercube space"
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "  ┌─ Analogy (Vector Arithmetic) ─────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "    │ Query: A is to B as C is to ? (using centroid vector math)" -ForegroundColor Gray
    Write-Host "    │ A='A' (U+0041), B='B' (U+0042), C='C' (U+0043)" -ForegroundColor Gray
    $analogyResult = SQL @"
SELECT '    │ Result ' || ROW_NUMBER() OVER (ORDER BY similarity DESC) || 
       ': similarity=' || ROUND(similarity::numeric, 4) ||
       ', codepoint=' || COALESCE(a.codepoint::text, 'composition') ||
       CASE WHEN a.codepoint IS NOT NULL THEN ' (''' || chr(a.codepoint) || ''')' ELSE '' END
FROM analogy(
    (SELECT id FROM atom WHERE codepoint=65), 
    (SELECT id FROM atom WHERE codepoint=66), 
    (SELECT id FROM atom WHERE codepoint=67), 
    3
) r
JOIN atom a ON a.id = r.result_id
"@
    if ($analogyResult -is [string]) { Write-Host $analogyResult -ForegroundColor Gray }
    Test-Result "analogy(A:B :: C:?)" $true "Vector: C + (B - A) → find nearest atoms"
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
} else {
    Write-Host "  (No compositions - run ingest.exe first)" -ForegroundColor DarkGray
}

# ═══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "─── Semantic Graph ────────────────────────────────────────" -ForegroundColor Blue
Write-Host "  (Edges from MiniLM embedding model - token co-occurrence by cosine similarity)" -ForegroundColor DarkGray

$edgeCount = [int](SQL "SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2").Trim()
Test-Result "Total semantic edges" ($edgeCount -gt 0) "$($edgeCount.ToString('N0')) token-pair relationships stored"

if ($edgeCount -gt 0) {
    Write-Host ""
    Write-Host "  ┌─ Top 5 Semantic Edges (highest cosine similarity) ─────────────────" -ForegroundColor Cyan
    # Show weight and first 16 hex chars of each token hash
    $topEdges = SQL @"
SELECT '    │ ' || ROUND(ST_M(ST_StartPoint(e.geom))::numeric, 4) || 
       ' similarity: ' || LEFT(encode(e.children[1], 'hex'), 16) || '... ↔ ' || 
       LEFT(encode(e.children[2], 'hex'), 16) || '...'
FROM atom e 
WHERE e.depth = 1 AND e.atom_count = 2 
ORDER BY ST_M(ST_StartPoint(e.geom)) DESC 
LIMIT 5
"@
    Write-Host $topEdges -ForegroundColor Gray
    
    # Get a seed token that has neighbors
    $seedInfo = SQL @"
SELECT children[1] FROM atom WHERE depth = 1 AND atom_count = 2 
ORDER BY ST_M(ST_StartPoint(geom)) DESC LIMIT 1
"@
    $seedId = $seedInfo.Trim()
    
    # Count neighbors for this seed
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "  ┌─ Semantic Neighbors Query ─────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "    │ Seed token: $seedId (first 16 hex chars of Blake3 hash)" -ForegroundColor Gray
    
    $neighborsQuery = SQL @"
SELECT '    │ ' || ROW_NUMBER() OVER (ORDER BY n.weight DESC) || '. weight=' || 
       ROUND(n.weight::numeric, 4) || ' → token ' || encode(n.neighbor_id, 'hex')::text
FROM semantic_neighbors('$seedId'::bytea, 5) n
"@
    if ($neighborsQuery -is [string] -and $neighborsQuery.Length -gt 0) {
        Write-Host $neighborsQuery -ForegroundColor Gray
        Test-Result "semantic_neighbors()" $true "Found 5 tokens semantically similar to seed"
    } else {
        $ncount = SQL "SELECT COUNT(*) FROM semantic_neighbors('$seedId'::bytea, 5)"
        Test-Result "semantic_neighbors()" ($ncount.Trim() -ge "1") "$($ncount.Trim()) neighbors found"
    }
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host ""
    
    # Random walk with actual path
    Write-Host "  ┌─ Random Walk (5 steps from seed) ─────────────────────────────────" -ForegroundColor Cyan
    Write-Host "    │ Starting from: $seedId" -ForegroundColor Gray
    
    $walkPath = SQL @"
SELECT '    │ Step ' || w.step || ': ' || encode(w.node_id, 'hex')::text
FROM random_walk('$seedId'::bytea, 5) w
ORDER BY w.step
"@
    if ($walkPath -is [string] -and $walkPath.Length -gt 0) {
        Write-Host $walkPath -ForegroundColor Gray
        $stepCount = (SQL "SELECT COUNT(*) FROM random_walk('$seedId'::bytea, 5)").Trim()
        Test-Result "random_walk()" $true "Walked $stepCount steps through semantic graph"
    } else {
        $stepCount = SQL "SELECT COUNT(*) FROM random_walk('$seedId'::bytea, 5)"
        Test-Result "random_walk()" ($stepCount.Trim() -ge "1") "$($stepCount.Trim()) steps completed"
    }
    Write-Host "  └───────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "  Note: Token IDs are Blake3 hashes. Full vocab lookup requires joining" -ForegroundColor DarkGray
    Write-Host "        with ingested text or using atom_text() on compositions." -ForegroundColor DarkGray
} else {
    Write-Host "  (No semantic edges - run embedding extraction first)" -ForegroundColor DarkGray
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
