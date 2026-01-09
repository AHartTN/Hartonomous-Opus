# Hartonomous Hypercube - Full Model Ingestion (Windows)
# ============================================================================
# Scans configured model directories and ingests ALL discoverable models.
# Models are detected by presence of config.json with "model_type" field.
#
# What gets ingested from each model:
#   1. Vocabulary (vocab.txt or tokenizer.json) → compositions with labels
#   2. BPE/WordPiece merges → composition_child hierarchy
#   3. Token embeddings (safetensors) → 4D centroids via Laplacian projection
#   4. k-NN similarity → semantic edges in relation table
#
# Usage:
#   .\ingest-models.ps1                          # Scan default paths
#   .\ingest-models.ps1 -Path "D:\Models"        # Scan specific path
#   .\ingest-models.ps1 -List                    # Just list, don't ingest
#   .\ingest-models.ps1 -Model "all-MiniLM"      # Filter by name
#   .\ingest-models.ps1 -Type "bert"             # Filter by model type
# ============================================================================

param(
    [string[]]$Path,           # Paths to scan (default: $env:HC_MODEL_PATHS or D:\Models)
    [switch]$List,             # Just list discovered models, don't ingest
    [string]$Model,            # Filter by model name (substring match)
    [string]$Type,             # Filter by model_type (bert, llama, etc.)
    [float]$Threshold = 0.25,  # Similarity threshold for edge creation
    [switch]$Force             # Re-ingest even if model appears to be ingested
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Hartonomous Hypercube - Model Ingestion                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# DETERMINE SEARCH PATHS
# ============================================================================

if (-not $Path) {
    if ($env:HC_MODEL_PATHS) {
        $Path = $env:HC_MODEL_PATHS -split ";"
    } else {
        # Default model locations
        $Path = @(
            "D:\Models",
            "$env:USERPROFILE\.cache\huggingface\hub",
            "$env:HC_PROJECT_ROOT\test-data\embedding_models"
        )
    }
}

$validPaths = @()
foreach ($p in $Path) {
    if (Test-Path $p) {
        $validPaths += $p
        Write-Host "  [SCAN] $p" -ForegroundColor Cyan
    } else {
        Write-Host "  [SKIP] $p (not found)" -ForegroundColor DarkGray
    }
}

if ($validPaths.Count -eq 0) {
    Write-Host "No valid paths to scan." -ForegroundColor Red
    exit 1
}

Write-Host ""

# ============================================================================
# DISCOVER MODELS
# ============================================================================

Write-Host "[1/3] Discovering models..." -ForegroundColor Yellow

$models = @()

foreach ($basePath in $validPaths) {
    # Find all config.json files recursively
    $configFiles = Get-ChildItem -Path $basePath -Recurse -Filter "config.json" -File -ErrorAction SilentlyContinue
    
    foreach ($configFile in $configFiles) {
        # Skip .cache and non-snapshot directories unless they're the only option
        $pathStr = $configFile.FullName
        if ($pathStr -match "\.cache" -and $pathStr -notmatch "snapshots") {
            continue
        }
        
        # Prefer snapshot directories
        if ($pathStr -notmatch "snapshots?" -and (Test-Path (Join-Path $configFile.Directory.FullName "snapshots"))) {
            continue
        }
        
        # Parse config.json
        try {
            $config = Get-Content $configFile.FullName -Raw | ConvertFrom-Json
            
            if (-not $config.model_type) {
                continue  # Not a valid model config
            }
            
            # Extract model info
            $modelInfo = @{
                Path = $configFile.Directory.FullName
                ModelType = $config.model_type
                VocabSize = if ($config.vocab_size) { $config.vocab_size } else { 0 }
                HiddenSize = if ($config.hidden_size) { $config.hidden_size } else { 0 }
                NumLayers = if ($config.num_hidden_layers) { $config.num_hidden_layers } else { 0 }
                NumExperts = if ($config.num_local_experts) { $config.num_local_experts } else { 0 }
                IsMultimodal = $null -ne $config.vision_config
                Name = ""
                TokenizerType = ""
                HasSafetensors = $false
                HasVocab = $false
            }
            
            # Extract name from path
            if ($pathStr -match "models--([^/\\]+)--([^/\\]+)") {
                $modelInfo.Name = "$($Matches[1])/$($Matches[2])"
            } else {
                $modelInfo.Name = $configFile.Directory.Name
            }
            
            # Check for safetensors
            $safetensorFiles = Get-ChildItem -Path $modelInfo.Path -Filter "*.safetensors" -File -ErrorAction SilentlyContinue
            $modelInfo.HasSafetensors = $safetensorFiles.Count -gt 0
            
            # Check for vocab
            $modelInfo.HasVocab = (Test-Path (Join-Path $modelInfo.Path "vocab.txt")) -or 
                                  (Test-Path (Join-Path $modelInfo.Path "tokenizer.json"))
            
            # Detect tokenizer type
            $tokenizerJson = Join-Path $modelInfo.Path "tokenizer.json"
            if (Test-Path $tokenizerJson) {
                $tokContent = Get-Content $tokenizerJson -Raw
                if ($tokContent -match '"type"\s*:\s*"BPE"') {
                    $modelInfo.TokenizerType = "BPE"
                } elseif ($tokContent -match '"type"\s*:\s*"WordPiece"') {
                    $modelInfo.TokenizerType = "WordPiece"
                } elseif ($tokContent -match '"type"\s*:\s*"Unigram"') {
                    $modelInfo.TokenizerType = "Unigram"
                } else {
                    $modelInfo.TokenizerType = "Unknown"
                }
            } elseif (Test-Path (Join-Path $modelInfo.Path "tokenizer.model")) {
                $modelInfo.TokenizerType = "SentencePiece"
            }
            
            $models += [PSCustomObject]$modelInfo
            
        } catch {
            # Invalid JSON or other parse error - skip
            continue
        }
    }
}

# Apply filters
if ($Model) {
    $models = $models | Where-Object { $_.Name -like "*$Model*" }
}
if ($Type) {
    $models = $models | Where-Object { $_.ModelType -eq $Type }
}

Write-Host "  Found $($models.Count) model(s)" -ForegroundColor Green
Write-Host ""

# ============================================================================
# LIST MODELS
# ============================================================================

Write-Host "[2/3] Model Summary" -ForegroundColor Yellow
Write-Host ""

$idx = 0
foreach ($m in $models) {
    $idx++
    $status = if ($m.HasSafetensors -and $m.HasVocab) { "✓" } 
              elseif ($m.HasVocab) { "○" } 
              else { "✗" }
    $statusColor = if ($m.HasSafetensors -and $m.HasVocab) { "Green" } 
                   elseif ($m.HasVocab) { "Yellow" } 
                   else { "Red" }
    
    Write-Host "  [$status] " -NoNewline -ForegroundColor $statusColor
    Write-Host "$($m.Name)" -ForegroundColor White
    Write-Host "      Type: $($m.ModelType) | Tokenizer: $($m.TokenizerType) | Vocab: $($m.VocabSize)"
    Write-Host "      Hidden: $($m.HiddenSize) | Layers: $($m.NumLayers)" -NoNewline
    if ($m.NumExperts -gt 0) {
        Write-Host " | Experts: $($m.NumExperts) (MoE)" -NoNewline
    }
    if ($m.IsMultimodal) {
        Write-Host " | Multimodal" -NoNewline
    }
    Write-Host ""
    Write-Host "      Path: $($m.Path)" -ForegroundColor DarkGray
    Write-Host ""
}

if ($List) {
    Write-Host "  (List mode - not ingesting)" -ForegroundColor DarkGray
    exit 0
}

# ============================================================================
# INGEST MODELS
# ============================================================================

Write-Host "[3/3] Ingesting Models" -ForegroundColor Yellow
Write-Host ""

$env:PGPASSWORD = $env:HC_DB_PASS

$successCount = 0
$failCount = 0
$skipCount = 0

# Find the main ingester (ingest_safetensor.exe with hierarchy support)
$universalIngester = if (Test-Path "$env:HC_BUILD_DIR\ingest_safetensor.exe") {
    "$env:HC_BUILD_DIR\ingest_safetensor.exe"
} elseif (Test-Path "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe") {
    "$env:HC_BUILD_DIR\Release\ingest_safetensor.exe"
} else { $null }

foreach ($m in $models) {
    Write-Host "────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Ingesting: $($m.Name)" -ForegroundColor Cyan
    
    # Use universal ingester for ALL models (not just those with vocab)
    if ($m.HasSafetensors -and $universalIngester) {
        # Check if already ingested
        if (-not $Force) {
            $modelName = $m.Name -replace "/", "_" -replace "-", "_"
            $existingCount = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT COUNT(*) FROM relation WHERE source_model = '$($m.Name)'" 2>$null
            if ([int]$existingCount -gt 100) {
                Write-Host "    [SKIP] Already ingested ($existingCount relations)" -ForegroundColor DarkGray
                $skipCount++
                continue
            }
        }
        
        Write-Host "    Mode: Universal (all tensors → 4D projection)"
        & $universalIngester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT -n $m.Name $m.Path
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [OK] Ingested successfully" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "    [FAIL] Ingestion failed" -ForegroundColor Red
            $failCount++
        }
    } elseif ($m.HasVocab) {
        # Legacy vocab-only ingestion
        Write-Host "    Mode: Vocab-only (no safetensors)"
        
        $vocabFile = if (Test-Path (Join-Path $m.Path "tokenizer.json")) {
            Join-Path $m.Path "tokenizer.json"
        } else {
            Join-Path $m.Path "vocab.txt"
        }
        
        # Extract vocab to text and ingest via universal ingester
        $vocabExtract = "$env:HC_BUILD_DIR\vocab_extract.exe"
        if (-not (Test-Path $vocabExtract)) {
            $vocabExtract = "$env:HC_BUILD_DIR\Release\vocab_extract.exe"
        }
        
        if (Test-Path $vocabExtract) {
            $tempVocab = [System.IO.Path]::GetTempFileName()
            & $vocabExtract $vocabFile | Out-File -FilePath $tempVocab -Encoding UTF8
            
            $ingester = "$env:HC_BUILD_DIR\ingest.exe"
            if (-not (Test-Path $ingester)) {
                $ingester = "$env:HC_BUILD_DIR\Release\ingest.exe"
            }
            
            if (Test-Path $ingester) {
                & $ingester -d $env:HC_DB_NAME -U $env:HC_DB_USER -h $env:HC_DB_HOST -p $env:HC_DB_PORT $tempVocab
            }
            
            Remove-Item $tempVocab -ErrorAction SilentlyContinue
        } else {
            Write-Host "    [WARN] vocab_extract.exe not found" -ForegroundColor Yellow
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [OK] Ingested successfully" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "    [FAIL] Ingestion failed (exit code $LASTEXITCODE)" -ForegroundColor Red
            $failCount++
        }
    } else {
        Write-Host "    [SKIP] No safetensors or vocab" -ForegroundColor Yellow
        $skipCount++
    }
}

Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Model Ingestion Complete" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Total:    $($models.Count) models discovered"
Write-Host "  Success:  $successCount" -ForegroundColor Green
Write-Host "  Skipped:  $skipCount" -ForegroundColor Yellow
Write-Host "  Failed:   $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

# Final database state
$env:PGPASSWORD = $env:HC_DB_PASS
$atomResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT atoms FROM db_stats()" 2>$null
$atomCount = if ($atomResult) { $atomResult.Trim() } else { "N/A" }
$compResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT compositions FROM db_stats()" 2>$null
$compCount = if ($compResult) { $compResult.Trim() } else { "N/A" }
$relResult = & psql -h $env:HC_DB_HOST -p $env:HC_DB_PORT -U $env:HC_DB_USER -d $env:HC_DB_NAME -tAc "SELECT relations FROM db_stats()" 2>$null
$relCount = if ($relResult) { $relResult.Trim() } else { "N/A" }
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue

Write-Host "  Final Database State:"
Write-Host "    Atoms:        $atomCount" -ForegroundColor Cyan
Write-Host "    Compositions: $compCount" -ForegroundColor Cyan
Write-Host "    Relations:    $relCount" -ForegroundColor Cyan
Write-Host ""

if ($failCount -gt 0) {
    exit 1
}
