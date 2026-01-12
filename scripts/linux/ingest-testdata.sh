#!/bin/bash
# Hartonomous Hypercube - Complete Test Data Ingestion (Linux)
# ============================================================================
# INGESTION ORDER (Critical for proper vocabulary-aware tokenization):
#
#   1. Model vocabulary (vocab.txt) - creates token compositions
#   2. Model BPE merges (tokenizer.json) - creates merge edges
#   3. Model weights (model.safetensors) - creates semantic similarity edges
#   4. Text content (Moby Dick, etc.) - uses vocabulary for greedy matching
#
# The AI model package is ingested FIRST so that:
#   - Token compositions exist for greedy matching
#   - BPE merge edges encode subword relationships
#   - Semantic similarity edges encode embedding-space relationships
#
# Then when we ingest Moby Dick, we can:
#   - Greedy match against known vocabulary tokens
#   - Only create new compositions for novel n-grams
#   - Accumulate co-occurrence edges on existing patterns
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

TEST_DATA_DIR="$HC_PROJECT_ROOT/test-data"
MODEL_DIR="$TEST_DATA_DIR/embedding_models"

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "test-data directory not found: $TEST_DATA_DIR"
    exit 1
fi

echo "Hypercube Test Data Ingestion"
echo ""
echo "Source: $TEST_DATA_DIR"
echo ""

# Find tools
INGESTER="$HC_BUILD_DIR/ingest"
EMBEDDING_EXTRACTOR="$HC_BUILD_DIR/extract_embeddings"
SAFETENSOR_INGESTER="$HC_BUILD_DIR/ingest_safetensor"

DB_ARGS="-d $HC_DB_NAME -U $HC_DB_USER -h $HC_DB_HOST"

get_comp_count() {
    hc_psql -tAc "SELECT COUNT(*) FROM composition" 2>/dev/null | tr -d '[:space:]'
}

get_edge_count() {
    hc_psql -tAc "SELECT COUNT(*) FROM relation" 2>/dev/null | tr -d '[:space:]'
}

# ============================================================================
# STEP 1: FIND MODEL PACKAGE
# ============================================================================
echo "Step 1: Locating Model Package"

MODEL_SNAPSHOT=""
VOCAB_FILE=""
TOKENIZER_FILE=""
MODEL_FILE=""

# Find the snapshot directory (HuggingFace cache structure)
if [ -d "$MODEL_DIR" ]; then
    MODEL_SNAPSHOT=$(find "$MODEL_DIR" -type d -name "snapshots" -exec find {} -mindepth 1 -maxdepth 1 -type d \; 2>/dev/null | head -1)
    
    if [ -n "$MODEL_SNAPSHOT" ]; then
        VOCAB_FILE="$MODEL_SNAPSHOT/vocab.txt"
        TOKENIZER_FILE="$MODEL_SNAPSHOT/tokenizer.json"
        MODEL_FILE=$(find "$MODEL_SNAPSHOT" -name "*.safetensors" 2>/dev/null | head -1)
        
        echo "  Model snapshot: $MODEL_SNAPSHOT"
        [ -f "$VOCAB_FILE" ] && echo "  vocab.txt found ($(wc -l < "$VOCAB_FILE") tokens)"
        [ -f "$TOKENIZER_FILE" ] && echo "  tokenizer.json found"
        [ -f "$MODEL_FILE" ] && echo "  $(basename "$MODEL_FILE") found ($(du -h "$MODEL_FILE" | cut -f1))"
    else
        echo "  No model snapshot found in $MODEL_DIR"
    fi
else
    echo "  No embedding_models directory"
fi
echo ""

# ============================================================================
# STEP 2: INGEST VOCABULARY (Token Compositions)
# ============================================================================
echo "Step 2: Ingesting Vocabulary (Token Compositions)"

if [ -f "$VOCAB_FILE" ] && [ -x "$INGESTER" ]; then
    BEFORE_COUNT=$(get_comp_count)
    START_TIME=$(date +%s.%N)
    
    echo "  Ingesting vocab.txt..."
    $INGESTER $DB_ARGS "$VOCAB_FILE" 2>&1 | grep -E "^\[|compositions|OK" | head -10
    
    END_TIME=$(date +%s.%N)
    AFTER_COUNT=$(get_comp_count)
    NEW_COMPS=$((AFTER_COUNT - BEFORE_COUNT))
    DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "?")
    
    echo "  → +$NEW_COMPS compositions in ${DURATION}s"
else
    echo "  Skipping vocabulary (file or ingester not found)"
fi
echo ""

# ============================================================================
# STEP 3: EXTRACT SEMANTIC EDGES FROM MODEL WEIGHTS
# ============================================================================
echo "Step 3: Extracting Semantic Edges (Embedding Similarities)"

if [ -f "$MODEL_FILE" ] && [ -x "$EMBEDDING_EXTRACTOR" ]; then
    BEFORE_EDGES=$(get_edge_count)
    
    echo "  Extracting from $(basename "$MODEL_FILE")..."
    echo "  (Threshold: 0.7 - only storing highly similar pairs)"
    
    $EMBEDDING_EXTRACTOR \
        --model "$MODEL_FILE" \
        ${VOCAB_FILE:+--vocab "$VOCAB_FILE"} \
        --threshold 0.7 \
        $DB_ARGS 2>&1 | grep -E "Pairs|Edges|Sparsity|Complete" | head -10
    
    AFTER_EDGES=$(get_edge_count)
    NEW_EDGES=$((AFTER_EDGES - BEFORE_EDGES))
    
    echo "  → +$NEW_EDGES semantic edges"
elif [ -f "$MODEL_FILE" ] && [ -x "$SAFETENSOR_INGESTER" ]; then
    echo "  Using safetensor ingester..."
    $SAFETENSOR_INGESTER $DB_ARGS -t 0.7 "$(dirname "$MODEL_FILE")" 2>&1 | grep -E "^\[|edges" | head -10
else
    echo "  Skipping model weights (file or extractor not found)"
fi
echo ""

# ============================================================================
# STEP 4: INGEST TEXT CONTENT (Moby Dick, etc.)
# ============================================================================
echo "Step 4: Ingesting Text Content"

if [ -x "$INGESTER" ]; then
    for file in "$TEST_DATA_DIR"/*.txt; do
        [ -f "$file" ] || continue
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)
        
        echo -n "  Ingesting: $filename ($filesize)..."
        
        BEFORE_COUNT=$(get_comp_count)
        START_TIME=$(date +%s.%N)
        
        # Run ingester
        $INGESTER $DB_ARGS "$file" 2>&1 | grep -E "compositions|OK" | tail -1
        
        END_TIME=$(date +%s.%N)
        AFTER_COUNT=$(get_comp_count)
        
        NEW_COMPS=$((AFTER_COUNT - BEFORE_COUNT))
        DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "?")
        echo "    → +$NEW_COMPS compositions (${DURATION}s)"
    done
else
    echo "  ingest tool not found - run build.sh first"
fi
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "Ingestion Complete"
echo ""

hc_psql -c "
SELECT
    (SELECT COUNT(*) FROM atom) as \"Leaf Atoms\",
    (SELECT COUNT(*) FROM composition) as \"Compositions\",
    (SELECT COUNT(*) FROM relation) as \"Semantic Edges\",
    (SELECT MAX(depth) FROM composition) as \"Max Depth\",
    pg_size_pretty(pg_total_relation_size('atom') + pg_total_relation_size('composition') + pg_total_relation_size('relation')) as \"Total Size\";
"
