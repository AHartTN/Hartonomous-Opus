#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Model Ingestion (Linux)
# ============================================================================
# Ingests HuggingFace models with optimized parallel processing.
#
# Usage:
#   ./scripts/linux/ingest.sh <model_path>
#   ./scripts/linux/ingest.sh <model_path> -t 0.3
#   ./scripts/linux/ingest.sh <model_path> -n "mymodel"
#
# Examples:
#   ./scripts/linux/ingest.sh ~/models/all-MiniLM-L6-v2/snapshots/abc123
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

MODEL_PATH=""
THRESHOLD="${HC_INGEST_THRESHOLD:-0.5}"
MODEL_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold) THRESHOLD="$2"; shift 2 ;;
        -n|--name) MODEL_NAME="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) MODEL_PATH="$1"; shift ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [-t threshold] [-n name]"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

# Resolve to absolute path
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"

# ============================================================================
# AUTO-DETECT MODEL NAME
# ============================================================================

if [ -z "$MODEL_NAME" ]; then
    # Extract from HuggingFace cache path structure
    if [[ "$MODEL_PATH" == *"/snapshots/"* ]]; then
        MODEL_NAME=$(echo "$MODEL_PATH" | sed -E 's|.*/([^/]+)/snapshots/.*|\1|' | sed 's/^models--//' | sed 's/--/-/g')
    else
        MODEL_NAME=$(basename "$MODEL_PATH")
    fi
fi

# ============================================================================
# THREADING CONFIGURATION
# ============================================================================

THREAD_COUNT=$(nproc)

# OpenMP
export OMP_NUM_THREADS=$THREAD_COUNT
export OMP_DYNAMIC=FALSE

# Intel MKL
export MKL_NUM_THREADS=$THREAD_COUNT
export MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=INTEL
export KMP_AFFINITY="granularity=fine,compact,1,0"

# Fallback BLAS
export OPENBLAS_NUM_THREADS=$THREAD_COUNT

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

export PGHOST="$HC_DB_HOST"
export PGPORT="$HC_DB_PORT"
export PGDATABASE="$HC_DB_NAME"
export PGUSER="$HC_DB_USER"
export PGPASSWORD="$HC_DB_PASS"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo ""
echo "============================================================"
echo " Hartonomous Hypercube - Model Ingestion"
echo " $THREAD_COUNT hardware threads"
echo "============================================================"
echo ""
echo "Model:"
echo "  Name:      $MODEL_NAME"
echo "  Path:      $MODEL_PATH"
echo "  Threshold: $THRESHOLD"
echo ""
echo "Database:"
echo "  $PGUSER@$PGHOST:$PGPORT/$PGDATABASE"
echo ""
echo "Threading:"
echo "  OpenMP:    $OMP_NUM_THREADS threads"
echo "  MKL:       $MKL_NUM_THREADS threads (dynamic=$MKL_DYNAMIC)"
echo ""

# ============================================================================
# FIND INGESTER
# ============================================================================

INGEST_EXE="$HC_BIN_DIR/ingest_safetensor"
if [ ! -x "$INGEST_EXE" ]; then
    echo "ERROR: ingest_safetensor not found."
    echo "       Run: ./scripts/linux/build.sh"
    exit 1
fi

# ============================================================================
# RUN INGESTION
# ============================================================================

echo "Starting ingestion..."
echo ""

START_TIME=$(date +%s)

"$INGEST_EXE" "$MODEL_PATH" -t "$THRESHOLD" -n "$MODEL_NAME"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================================
# RESULTS
# ============================================================================

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " INGESTION COMPLETE"
else
    echo " INGESTION FAILED (exit code: $EXIT_CODE)"
fi
echo " Time: ${ELAPSED}s"
echo "============================================================"

exit $EXIT_CODE
