# Hartonomous Hypercube - Complete Test Data Ingestion (Windows)
# Ingests ALL content from test-data: text, models, images, audio
# Fully idempotent - safe to run multiple times
# Usage: .\scripts\windows\ingest-testdata.ps1

. "$PSScriptRoot\env.ps1"

$env:PGPASSWORD = $env:HC_DB_PASS
$testDataDir = "$env:HC_PROJECT_ROOT\test-data"

if (-not (Test-Path $testDataDir)) {
    Write-Host "test-data directory not found: $testDataDir" -ForegroundColor Red
    exit 1
}

Write-Host "=== Hypercube Test Data Ingestion ===" -ForegroundColor Cyan
Write-Host "Source: $testDataDir"

# Find tools - prefer vocabulary_ingest (proper grammar-based) over cpe_ingest (broken binary pairing)
$vocabIngester = if (Test-Path "$env:HC_BUILD_DIR\Release\vocabulary_ingest.exe") {
    "$env:HC_BUILD_DIR\Release\vocabulary_ingest.exe"
} elseif (Test-Path "$env:HC_BUILD_DIR\vocabulary_ingest.exe") {
    "$env:HC_BUILD_DIR\vocabulary_ingest.exe"
} else { $null }

# Fallback to cpe_ingest if vocabulary_ingest not available
$cpeIngester = if ($vocabIngester) {
    $vocabIngester  # Use vocab ingester as primary
} elseif (Test-Path "$env:HC_BUILD_DIR\Release\cpe_ingest.exe") {
    "$env:HC_BUILD_DIR\Release\cpe_ingest.exe"
} elseif (Test-Path "$env:HC_BUILD_DIR\cpe_ingest.exe") {
    "$env:HC_BUILD_DIR\cpe_ingest.exe"
} else { $null }

$semanticIngester = if (Test-Path "$env:HC_BUILD_DIR\Release\semantic_ingest.exe") { 
    "$env:HC_BUILD_DIR\Release\semantic_ingest.exe" 
} elseif (Test-Path "$env:HC_BUILD_DIR\semantic_ingest.exe") { 
    "$env:HC_BUILD_DIR\semantic_ingest.exe" 
} else { $null }

$embeddingExtractor = if (Test-Path "$env:HC_BUILD_DIR\Release\extract_embeddings.exe") { 
    "$env:HC_BUILD_DIR\Release\extract_embeddings.exe" 
} elseif (Test-Path "$env:HC_BUILD_DIR\extract_embeddings.exe") { 
    "$env:HC_BUILD_DIR\extract_embeddings.exe" 
} else { $null }

$dbArgs = @("-d", $env:HC_DB_NAME, "-U", $env:HC_DB_USER, "-h", $env:HC_DB_HOST, "-p", $env:HC_DB_PORT)

# ============================================================================
# 1. TEXT FILES - CPE Ingestion
# ============================================================================
Write-Host "`n--- 1. Text Files (CPE) ---" -ForegroundColor Yellow

$textFiles = Get-ChildItem -Path $testDataDir -Filter "*.txt" -File
if ($textFiles.Count -gt 0 -and $cpeIngester) {
    foreach ($file in $textFiles) {
        Write-Host "  Ingesting: $($file.Name)..." -NoNewline
        $beforeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        
        & $cpeIngester @dbArgs $file.FullName 2>&1 | Out-Null
        
        $afterCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        $newComps = [int]$afterCount - [int]$beforeCount
        
        if ($newComps -gt 0) {
            Write-Host " +$newComps compositions" -ForegroundColor Green
        } else {
            Write-Host " (already ingested)" -ForegroundColor DarkGray
        }
    }
} elseif (-not $cpeIngester) {
    Write-Host "  WARNING: cpe_ingest.exe not found - run build.ps1 first" -ForegroundColor Yellow
}

# ============================================================================
# 1b. TEXT FILES - Semantic Relationships (Co-occurrence Edges)
# ============================================================================
Write-Host "`n--- 1b. Text Files (Semantic Edges) ---" -ForegroundColor Yellow

if ($textFiles.Count -gt 0 -and $semanticIngester) {
    foreach ($file in $textFiles) {
        Write-Host "  Semantic ingestion: $($file.Name)..." -NoNewline
        $beforeEdges = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2 AND ST_M(ST_StartPoint(geom)) < 100000"
        
        & $semanticIngester @dbArgs --window 5 --threshold 0.01 $file.FullName 2>&1 | Out-Null
        
        $afterEdges = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2 AND ST_M(ST_StartPoint(geom)) < 100000"
        $newEdges = [int]$afterEdges - [int]$beforeEdges
        
        if ($newEdges -gt 0) {
            Write-Host " +$newEdges edges" -ForegroundColor Green
        } else {
            Write-Host " (edges already exist)" -ForegroundColor DarkGray
        }
    }
} elseif (-not $semanticIngester) {
    Write-Host "  WARNING: semantic_ingest.exe not found - run build.ps1 first" -ForegroundColor Yellow
}

# ============================================================================
# 2. EMBEDDING MODEL - Extract Semantic Edges
# ============================================================================
Write-Host "`n--- 2. Embedding Model (Semantic Edges) ---" -ForegroundColor Yellow

$modelSnapshot = Get-ChildItem -Path "$testDataDir\embedding_models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if ($modelSnapshot -and $embeddingExtractor) {
    $vocabFile = Join-Path $modelSnapshot.FullName "vocab.txt"
    $safetensorFile = Join-Path $modelSnapshot.FullName "model.safetensors"
    
    # Extract embeddings (creates semantic edges between similar tokens)
    # Tokens become compositions, pairwise similarity becomes edge weight (M coord)
    if ((Test-Path $safetensorFile) -and (Test-Path $vocabFile)) {
        Write-Host "  Extracting semantic edges from embeddings..." -NoNewline
        Write-Host " (threshold 0.7)"
        & $embeddingExtractor @dbArgs --model $safetensorFile --vocab $vocabFile --threshold 0.7 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Semantic edges created" -ForegroundColor Green
        } else {
            Write-Host "  (extraction failed)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Model files not found" -ForegroundColor Yellow
    }
} elseif (-not $embeddingExtractor) {
    Write-Host "  WARNING: extract_embeddings.exe not found - run build.ps1 first" -ForegroundColor Yellow
} else {
    Write-Host "  Model snapshot not found in test-data" -ForegroundColor Yellow
}

# ============================================================================
# 3. IMAGES - Binary Ingestion as CPE
# ============================================================================
Write-Host "`n--- 3. Images (Binary CPE) ---" -ForegroundColor Yellow

$imageFiles = Get-ChildItem -Path $testDataDir -Filter "*.png" -File
if ($imageFiles.Count -gt 0 -and $cpeIngester) {
    foreach ($file in $imageFiles) {
        Write-Host "  Ingesting: $($file.Name)..." -NoNewline
        $beforeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        
        & $cpeIngester @dbArgs --binary $file.FullName 2>&1 | Out-Null
        
        $afterCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        $newComps = [int]$afterCount - [int]$beforeCount
        
        if ($newComps -gt 0) {
            Write-Host " +$newComps compositions" -ForegroundColor Green
        } else {
            Write-Host " (already ingested or binary mode not supported)" -ForegroundColor DarkGray
        }
    }
} elseif (-not $cpeIngester) {
    Write-Host "  WARNING: cpe_ingest.exe not found - run build.ps1 first" -ForegroundColor Yellow
}

# ============================================================================
# 4. AUDIO - Binary Ingestion as CPE
# ============================================================================
Write-Host "`n--- 4. Audio (Binary CPE) ---" -ForegroundColor Yellow

$audioFiles = Get-ChildItem -Path $testDataDir -Filter "*.wav" -File
if ($audioFiles.Count -gt 0 -and $cpeIngester) {
    foreach ($file in $audioFiles) {
        Write-Host "  Ingesting: $($file.Name)..." -NoNewline
        $beforeCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        
        & $cpeIngester @dbArgs --binary $file.FullName 2>&1 | Out-Null
        
        $afterCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0"
        $newComps = [int]$afterCount - [int]$beforeCount
        
        if ($newComps -gt 0) {
            Write-Host " +$newComps compositions" -ForegroundColor Green
        } else {
            Write-Host " (already ingested or binary mode not supported)" -ForegroundColor DarkGray
        }
    }
} elseif (-not $cpeIngester) {
    Write-Host "  WARNING: cpe_ingest.exe not found - run build.ps1 first" -ForegroundColor Yellow
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host "`n=== Ingestion Complete ===" -ForegroundColor Green
Write-Host "`nDatabase Statistics:"
& psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -c @"
SELECT 
    COUNT(*) FILTER (WHERE depth = 0) as leaf_atoms,
    COUNT(*) FILTER (WHERE depth > 0) as compositions,
    COUNT(*) FILTER (WHERE depth = 1 AND atom_count = 2 AND ST_M(ST_StartPoint(geom)) < 100000) as semantic_edges,
    MAX(depth) as max_depth,
    pg_size_pretty(pg_total_relation_size('atom')) as total_size
FROM atom;
"@

Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
