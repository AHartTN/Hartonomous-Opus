#!/bin/bash
# Hartonomous Hypercube - Full Model Ingestion (Linux)
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
#   ./ingest-models.sh                          # Scan default paths
#   ./ingest-models.sh --path "/path/to/models" # Scan specific path
#   ./ingest-models.sh --list                   # Just list, don't ingest
#   ./ingest-models.sh --model "MiniLM"         # Filter by name
#   ./ingest-models.sh --type "bert"            # Filter by model type
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Defaults
SEARCH_PATHS=()
LIST_ONLY=false
MODEL_FILTER=""
TYPE_FILTER=""
THRESHOLD=0.25
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path) SEARCH_PATHS+=("$2"); shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        --model) MODEL_FILTER="$2"; shift 2 ;;
        --type) TYPE_FILTER="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Hartonomous Hypercube - Model Ingestion                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# DETERMINE SEARCH PATHS
# ============================================================================

if [ ${#SEARCH_PATHS[@]} -eq 0 ]; then
    if [ -n "$HC_MODEL_PATHS" ]; then
        IFS=':' read -ra SEARCH_PATHS <<< "$HC_MODEL_PATHS"
    else
        SEARCH_PATHS=(
            "$HOME/.cache/huggingface/hub"
            "$HC_PROJECT_ROOT/test-data/embedding_models"
        )
    fi
fi

VALID_PATHS=()
for p in "${SEARCH_PATHS[@]}"; do
    if [ -d "$p" ]; then
        VALID_PATHS+=("$p")
        echo "  [SCAN] $p"
    else
        echo "  [SKIP] $p (not found)"
    fi
done

if [ ${#VALID_PATHS[@]} -eq 0 ]; then
    echo "No valid paths to scan."
    exit 1
fi

echo ""

# ============================================================================
# DISCOVER MODELS
# ============================================================================

echo "[1/3] Discovering models..."

declare -a MODELS
MODEL_COUNT=0

for base_path in "${VALID_PATHS[@]}"; do
    while IFS= read -r -d '' config_file; do
        path_str="$config_file"
        
        # Skip .cache and non-snapshot directories
        if [[ "$path_str" == *".cache"* ]] && [[ "$path_str" != *"snapshots"* ]]; then
            continue
        fi
        
        # Prefer snapshot directories
        dir_path=$(dirname "$config_file")
        if [[ "$path_str" != *"snapshot"* ]] && [ -d "$dir_path/snapshots" ]; then
            continue
        fi
        
        # Parse config.json for model_type
        model_type=$(grep -oP '"model_type"\s*:\s*"\K[^"]+' "$config_file" 2>/dev/null || echo "")
        if [ -z "$model_type" ]; then
            continue
        fi
        
        # Extract model name from path
        if [[ "$path_str" =~ models--([^/]+)--([^/]+) ]]; then
            model_name="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        else
            model_name=$(basename "$dir_path")
        fi
        
        # Apply filters
        if [ -n "$MODEL_FILTER" ] && [[ "$model_name" != *"$MODEL_FILTER"* ]]; then
            continue
        fi
        if [ -n "$TYPE_FILTER" ] && [ "$model_type" != "$TYPE_FILTER" ]; then
            continue
        fi
        
        # Check for files
        has_safetensors=false
        has_vocab=false
        
        if ls "$dir_path"/*.safetensors 1>/dev/null 2>&1; then
            has_safetensors=true
        fi
        if [ -f "$dir_path/vocab.txt" ] || [ -f "$dir_path/tokenizer.json" ]; then
            has_vocab=true
        fi
        
        # Store model info
        MODELS+=("$dir_path|$model_name|$model_type|$has_safetensors|$has_vocab")
        ((MODEL_COUNT++))
        
    done < <(find "$base_path" -name "config.json" -type f -print0 2>/dev/null)
done

echo "  Found $MODEL_COUNT model(s)"
echo ""

# ============================================================================
# LIST MODELS
# ============================================================================

echo "[2/3] Model Summary"
echo ""

for model_info in "${MODELS[@]}"; do
    IFS='|' read -r path name type has_st has_vocab <<< "$model_info"
    
    if [ "$has_st" = "true" ] && [ "$has_vocab" = "true" ]; then
        status="✓"
        color="\033[0;32m"
    elif [ "$has_vocab" = "true" ]; then
        status="○"
        color="\033[1;33m"
    else
        status="✗"
        color="\033[0;31m"
    fi
    
    echo -e "  [${color}${status}\033[0m] $name"
    echo "      Type: $type | Safetensors: $has_st | Vocab: $has_vocab"
    echo "      Path: $path"
    echo ""
done

if [ "$LIST_ONLY" = true ]; then
    echo "  (List mode - not ingesting)"
    exit 0
fi

# ============================================================================
# INGEST MODELS
# ============================================================================

echo "[3/3] Ingesting Models"
echo ""

export PGPASSWORD="$HC_DB_PASS"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for model_info in "${MODELS[@]}"; do
    IFS='|' read -r path name type has_st has_vocab <<< "$model_info"
    
    echo "────────────────────────────────────────────────────────────────"
    echo "  Ingesting: $name"
    
    if [ "$has_vocab" != "true" ]; then
        echo "    [SKIP] No vocabulary file found"
        ((SKIP_COUNT++))
        continue
    fi
    
    if [ "$has_st" = "true" ]; then
        echo "    Mode: Full (vocab + embeddings → 4D projection)"
        "$SCRIPT_DIR/ingest-safetensor.sh" "$path" --threshold "$THRESHOLD"
    else
        echo "    Mode: Vocab only (no safetensors)"
        
        vocab_extract="$HC_BUILD_DIR/vocab_extract"
        ingester="$HC_BUILD_DIR/ingest"
        
        if [ -x "$vocab_extract" ] && [ -x "$ingester" ]; then
            temp_vocab=$(mktemp)
            "$vocab_extract" "$path/tokenizer.json" > "$temp_vocab" 2>/dev/null || \
            "$vocab_extract" "$path/vocab.txt" > "$temp_vocab" 2>/dev/null
            
            "$ingester" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT" "$temp_vocab"
            rm -f "$temp_vocab"
        else
            echo "    [WARN] vocab_extract or ingest not found"
        fi
    fi
    
    if [ $? -eq 0 ]; then
        echo "    [OK] Ingested successfully"
        ((SUCCESS_COUNT++))
    else
        echo "    [FAIL] Ingestion failed"
        ((FAIL_COUNT++))
    fi
done

unset PGPASSWORD

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Model Ingestion Complete"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Total:    $MODEL_COUNT models discovered"
echo "  Success:  $SUCCESS_COUNT"
echo "  Skipped:  $SKIP_COUNT"
echo "  Failed:   $FAIL_COUNT"
echo ""

# Final database state
export PGPASSWORD="$HC_DB_PASS"
ATOM_COUNT=$(hc_psql -tAc "SELECT atoms FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
COMP_COUNT=$(hc_psql -tAc "SELECT compositions FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
REL_COUNT=$(hc_psql -tAc "SELECT relations FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
unset PGPASSWORD

echo "  Final Database State:"
echo "    Atoms:        $ATOM_COUNT"
echo "    Compositions: $COMP_COUNT"
echo "    Relations:    $REL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
