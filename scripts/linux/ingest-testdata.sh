#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Test Data Ingestion Script (Linux)
# ============================================================================
# Ingests test data into the Hypercube database for testing and validation.
#
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
# Then when we ingest text content, we can:
#   - Greedy match against known vocabulary tokens
#   - Only create new compositions for novel n-grams
#   - Accumulate co-occurrence edges on existing patterns
#
# Usage:
#   ./scripts/linux/ingest-testdata.sh          # Ingest test data with defaults
#   ./scripts/linux/ingest-testdata.sh --clean  # Clear existing data and re-ingest
#   ./scripts/linux/ingest-testdata.sh --help   # Show this help
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/../shared/detect-platform.sh"
source "$SCRIPT_DIR/../shared/logging.sh"

# Default configuration
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean) CLEAN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean       Clear existing data before ingestion"
            echo "  --help, -h    Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Load environment
source "$SCRIPT_DIR/env.sh"

PLATFORM=$(detect_os)
if [ "$PLATFORM" != "linux" ]; then
    log_error "This script is designed for Linux. Detected platform: $PLATFORM"
    exit 1
fi

log_section "Test Data Ingestion ($PLATFORM)"

# Configuration
TEST_DATA_DIR="$HC_PROJECT_ROOT/test-data"
MODEL_DIR="$TEST_DATA_DIR/embedding_models"

echo "  Test Data:  $TEST_DATA_DIR"
echo "  Model Dir:  $MODEL_DIR"
echo "  Clean:      $CLEAN"
echo

# ============================================================================
# PREREQUISITES
# ============================================================================

log_info "Checking prerequisites..."

# Check test data directory
if [ ! -d "$TEST_DATA_DIR" ]; then
    log_error "Test data directory not found: $TEST_DATA_DIR"
    log_info "Please ensure test data is available in the project"
    exit 1
fi

# Check PostgreSQL client
if ! command -v psql &> /dev/null; then
    log_error "psql command not found. Please install PostgreSQL client."
    exit 1
fi

# Test database connection
log_info "Testing database connection..."
if ! hc_psql -c "SELECT 1;" >/dev/null 2>&1; then
    log_error "Cannot connect to database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
    log_info "Please check:"
    log_info "  1. Database is running"
    log_info "  2. Connection parameters are correct"
    log_info "  3. Network connectivity"
    exit 1
fi

log_success "Prerequisites check passed"

# ============================================================================
# TOOL PREPARATION
# ============================================================================

log_subsection "Preparing Ingestion Tools"

INGESTER="$HC_BIN_DIR/ingest"
EMBEDDING_EXTRACTOR="$HC_BIN_DIR/extract_embeddings"
SAFETENSOR_INGESTER="$HC_BIN_DIR/ingest_safetensor"

# Debug logging for path validation
log_info "HC_BIN_DIR: $HC_BIN_DIR"
log_info "HC_BUILD_TYPE: $HC_BUILD_TYPE"
log_info "Expected ingester path: $INGESTER"

REQUIRED_TOOLS=("$INGESTER" "$EMBEDDING_EXTRACTOR" "$SAFETENSOR_INGESTER")

# Verify all required tools exist
for tool in "${REQUIRED_TOOLS[@]}"; do
    if [ ! -x "$tool" ]; then
        log_error "Required tool not found: $(basename "$tool")"
        log_error "Please run ./scripts/build.sh first to build all tools"
        exit 1
    fi
done

log_success "All ingestion tools available"

# Clean existing data if requested
if [ "$CLEAN" = true ]; then
    log_info "Cleaning existing test data..."
    hc_psql -c "
        TRUNCATE TABLE composition CASCADE;
        TRUNCATE TABLE relation CASCADE;
        TRUNCATE TABLE atom CASCADE;
    " 2>/dev/null || log_warning "Some tables may not exist or clean failed"
    log_success "Existing data cleared"
fi

# Database connection arguments
DB_ARGS="-d $HC_DB_NAME -U $HC_DB_USER -h $HC_DB_HOST"

get_comp_count() {
    hc_psql -tAc "SELECT COUNT(*) FROM composition" 2>/dev/null | tr -d '[:space:]'
}

get_edge_count() {
    hc_psql -tAc "SELECT COUNT(*) FROM relation" 2>/dev/null | tr -d '[:space:]'
}

# ============================================================================
# STEP 1: LOCATE MODEL PACKAGE
# ============================================================================

log_subsection "Locating Model Package"

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

        log_info "Model snapshot: $MODEL_SNAPSHOT"
        [ -f "$VOCAB_FILE" ] && log_info "vocab.txt found ($(wc -l < "$VOCAB_FILE") tokens)"
        [ -f "$TOKENIZER_FILE" ] && log_info "tokenizer.json found"
        [ -f "$MODEL_FILE" ] && log_info "$(basename "$MODEL_FILE") found ($(du -h "$MODEL_FILE" | cut -f1))"
    else
        log_warning "No model snapshot found in $MODEL_DIR"
    fi
else
    log_warning "No embedding_models directory"
fi

# ============================================================================
# STEP 2: INGEST VOCABULARY (Token Compositions)
# ============================================================================

log_subsection "Ingesting Vocabulary (Token Compositions)"

if [ -f "$VOCAB_FILE" ] && [ -x "$INGESTER" ]; then
    BEFORE_COUNT=$(get_comp_count)
    START_TIME=$(date +%s.%N)

    log_info "Ingesting vocab.txt..."
    if $INGESTER $DB_ARGS "$VOCAB_FILE" 2>&1 | grep -E "^\[|compositions|OK" | head -10; then
        END_TIME=$(date +%s.%N)
        AFTER_COUNT=$(get_comp_count)
        NEW_COMPS=$((AFTER_COUNT - BEFORE_COUNT))
        DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "?")
        log_success "+$NEW_COMPS compositions ingested in ${DURATION}s"
    else
        log_error "Vocabulary ingestion failed"
        exit 1
    fi
else
    log_warning "Skipping vocabulary (file or ingester not found)"
fi

# ============================================================================
# STEP 3: EXTRACT SEMANTIC EDGES FROM MODEL WEIGHTS
# ============================================================================

log_subsection "Extracting Semantic Edges (Embedding Similarities)"

if [ -f "$MODEL_FILE" ] && [ -x "$EMBEDDING_EXTRACTOR" ]; then
    BEFORE_EDGES=$(get_edge_count)

    log_info "Extracting from $(basename "$MODEL_FILE")... (Threshold: 0.7)"
    if $EMBEDDING_EXTRACTOR \
        --model "$MODEL_FILE" \
        ${VOCAB_FILE:+--vocab "$VOCAB_FILE"} \
        --threshold 0.7 \
        $DB_ARGS 2>&1 | grep -E "Pairs|Edges|Sparsity|Complete" | head -10; then

        AFTER_EDGES=$(get_edge_count)
        NEW_EDGES=$((AFTER_EDGES - BEFORE_EDGES))
        log_success "+$NEW_EDGES semantic edges extracted"
    else
        log_error "Semantic edge extraction failed"
        exit 1
    fi
elif [ -f "$MODEL_FILE" ] && [ -x "$SAFETENSOR_INGESTER" ]; then
    log_info "Using safetensor ingester..."
    if $SAFETENSOR_INGESTER $DB_ARGS -t 0.7 "$(dirname "$MODEL_FILE")" 2>&1 | grep -E "^\[|edges" | head -10; then
        log_success "Safetensor ingestion completed"
    else
        log_error "Safetensor ingestion failed"
        exit 1
    fi
else
    log_warning "Skipping model weights (file or extractor not found)"
fi

# ============================================================================
# STEP 4: INGEST MULTIMODAL CONTENT
# ============================================================================

log_subsection "Ingesting Multimodal Content"

if [ -x "$INGESTER" ]; then
    FILES_FOUND=false
    for file in "$TEST_DATA_DIR"/*; do
        [ -f "$file" ] || continue
        # Skip model directory files (handled separately)
        [[ "$file" == "$MODEL_DIR"* ]] && continue

        FILES_FOUND=true
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)

        log_info "Ingesting: $filename ($filesize)"

        BEFORE_COUNT=$(get_comp_count)
        START_TIME=$(date +%s.%N)

        # Run ingester (supports text, images, audio, etc.)
        if $INGESTER $DB_ARGS "$file" 2>&1 | grep -E "compositions|OK" | tail -1; then
            END_TIME=$(date +%s.%N)
            AFTER_COUNT=$(get_comp_count)
            NEW_COMPS=$((AFTER_COUNT - BEFORE_COUNT))
            DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "?")
            log_success "+$NEW_COMPS compositions ingested (${DURATION}s)"
        else
            log_error "Failed to ingest $filename"
            exit 1
        fi
    done

    if [ "$FILES_FOUND" = false ]; then
        log_warning "No files found in $TEST_DATA_DIR"
    fi
else
    log_error "Ingest tool not found"
    exit 1
fi

# ============================================================================
# VERIFICATION
# ============================================================================

log_subsection "Database Verification"

log_info "Checking ingestion results..."
hc_psql -c "
SELECT
    (SELECT COUNT(*) FROM atom) as \"Leaf Atoms\",
    (SELECT COUNT(*) FROM composition) as \"Compositions\",
    (SELECT COUNT(*) FROM relation) as \"Semantic Edges\",
    (SELECT MAX(depth) FROM composition) as \"Max Depth\",
    pg_size_pretty(pg_total_relation_size('atom') + pg_total_relation_size('composition') + pg_total_relation_size('relation')) as \"Total Size\";
"

log_success "Test data ingestion completed successfully!"

echo
echo "Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo
echo "Next steps:"
echo "  1. Run tests: ./scripts/linux/test-hypercube-full.sh"
echo "  2. Validate data: ./scripts/linux/validate.sh"
echo "  3. Check benchmarks: ./scripts/benchmark.sh"
