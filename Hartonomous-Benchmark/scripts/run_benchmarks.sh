#!/bin/bash

set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Default values
BUILD_DIR="build"
RESULTS_DIR="results"
FILTER=""
CATEGORY=""
TYPE=""
FORMAT="json"
REPETITIONS=""
MIN_TIME=""
SKIP_BUILD=false
SKIP_PERF=false
SKIP_VALGRIND=false
SKIP_HARDWARE=false
VERBOSE=false

# Function to check if running in WSL
is_wsl() {
    if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
        return 0
    else
        return 1
    fi
}

# Function to check tool availability
check_tool() {
    local tool="$1"
    if command -v "$tool" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run the comprehensive benchmark suite with various filtering and configuration options."
    echo ""
    echo "Options:"
    echo "  -f, --filter PATTERN     Filter benchmarks by name pattern (regex)"
    echo "  -c, --category CATEGORY  Run benchmarks from specific category:"
    echo "                           linear_algebra, signal_processing, rng, ann,"
    echo "                           simd, hybrid, memory, micro"
    echo "  -t, --type TYPE          Filter by data type: float, double, int8, int16, int32, int64"
    echo "  --format FORMAT          Output format: json, csv, console (default: json)"
    echo "  --repetitions N          Number of benchmark repetitions (default: 1)"
    echo "  --min-time SEC           Minimum time per benchmark in seconds (default: 0.1)"
    echo "  --skip-build             Skip build step (assume already built)"
    echo "  --skip-perf              Skip perf profiling"
    echo "  --skip-valgrind          Skip valgrind profiling"
    echo "  --skip-hardware          Skip hardware detection"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -h, --help               Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --category linear_algebra --type double"
    echo "  $0 --filter 'Matrix.*' --format csv"
    echo "  $0 --category memory --skip-valgrind"
    echo "  $0 --repetitions 5 --min-time 1.0"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        --min-time)
            MIN_TIME="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-perf)
            SKIP_PERF=true
            shift
            ;;
        --skip-valgrind)
            SKIP_VALGRIND=true
            shift
            ;;
        --skip-hardware)
            SKIP_HARDWARE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Build if not skipped
if [ "$SKIP_BUILD" = false ]; then
    if [ ! -f "$BUILD_DIR/benchmarks/benchmark_suite" ]; then
        if [ ! -d "$BUILD_DIR" ]; then
            echo "Creating build directory..."
            mkdir "$BUILD_DIR"
        fi

        echo "Configuring and building project..."
        cd "$BUILD_DIR"
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . -- -j$(nproc)
        cd ..
    fi
fi

# Construct benchmark filter based on category
BENCHMARK_FILTER=""
if [ -n "$CATEGORY" ]; then
    case $CATEGORY in
        linear_algebra)
            BENCHMARK_FILTER="MatrixMultiply|LinearSolve|EigenMatrixMultiply|EigenLUSolve|EigenSVD|EigenSparse"
            ;;
        signal_processing)
            BENCHMARK_FILTER="FFT"
            ;;
        rng)
            BENCHMARK_FILTER="RNG"
            ;;
        ann)
            BENCHMARK_FILTER="HNSW"
            ;;
        simd)
            BENCHMARK_FILTER="SIMDIntrinsics|VNNIDotProduct|AVXVectorArithmetic"
            ;;
        hybrid)
            BENCHMARK_FILTER="Hybrid"
            ;;
        memory)
            BENCHMARK_FILTER="Memory"
            ;;
        micro)
            BENCHMARK_FILTER="Vector_Add|Dot_Product|Matrix_Vector"
            ;;
        *)
            echo "Unknown category: $CATEGORY"
            exit 1
            ;;
    esac
fi

# Add type filter
if [ -n "$TYPE" ]; then
    case $TYPE in
        float)
            TYPE_FILTER="Float"
            ;;
        double)
            TYPE_FILTER="Double"
            ;;
        int8)
            TYPE_FILTER="Int8"
            ;;
        int16)
            TYPE_FILTER="Int16"
            ;;
        int32)
            TYPE_FILTER="Int32"
            ;;
        int64)
            TYPE_FILTER="Int64"
            ;;
        *)
            echo "Unknown type: $TYPE"
            exit 1
            ;;
    esac
    if [ -n "$BENCHMARK_FILTER" ]; then
        BENCHMARK_FILTER="$BENCHMARK_FILTER.*$TYPE_FILTER|$TYPE_FILTER.*$BENCHMARK_FILTER"
    else
        BENCHMARK_FILTER="$TYPE_FILTER"
    fi
fi

# Combine with custom filter
if [ -n "$FILTER" ]; then
    if [ -n "$BENCHMARK_FILTER" ]; then
        BENCHMARK_FILTER="$BENCHMARK_FILTER|$FILTER"
    else
        BENCHMARK_FILTER="$FILTER"
    fi
fi

# Construct benchmark command
BENCHMARK_CMD="$BUILD_DIR/benchmarks/benchmark_suite"

if [ -n "$BENCHMARK_FILTER" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_filter=\"$BENCHMARK_FILTER\""
fi

BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_format=$FORMAT"

if [ -n "$REPETITIONS" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_repetitions=$REPETITIONS"
fi

if [ -n "$MIN_TIME" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_min_time=$MIN_TIME"
fi

# Output file
OUTPUT_FILE="$RESULTS_DIR/results.$FORMAT"

if [ "$VERBOSE" = true ]; then
    echo "Benchmark command: $BENCHMARK_CMD"
    echo "Output file: $OUTPUT_FILE"
fi

echo "Running benchmark suite..."
eval "$BENCHMARK_CMD > $OUTPUT_FILE"

# Run perf if not skipped
if [ "$SKIP_PERF" = false ]; then
    if check_tool perf; then
        if is_wsl; then
            echo "Warning: perf may not work properly in WSL. Skipping perf profiling."
        else
            echo "Running perf stat..."
            perf stat $BUILD_DIR/benchmarks/benchmark_suite > "$RESULTS_DIR/perf_stat.txt" 2>&1
        fi
    else
        echo "Warning: perf tool not found. Skipping perf profiling. Install perf-tools or linux-tools-common to enable."
    fi
fi

# Run valgrind if not skipped
if [ "$SKIP_VALGRIND" = false ]; then
    if check_tool valgrind; then
        echo "Running valgrind callgrind..."
        valgrind --tool=callgrind --callgrind-out-file="$RESULTS_DIR/callgrind.out" $BUILD_DIR/benchmarks/benchmark_suite
    else
        echo "Warning: valgrind not found. Skipping valgrind profiling. Install valgrind to enable."
    fi
fi

# Run hardware check if not skipped
if [ "$SKIP_HARDWARE" = false ]; then
    if [ -f "$BUILD_DIR/src/hardware_check" ]; then
        echo "Running hardware check..."
        $BUILD_DIR/src/hardware_check > "$RESULTS_DIR/hardware_info.txt"
    else
        echo "Warning: hardware_check binary not found. Skipping hardware detection. Ensure the project is built."
    fi
fi

echo "Benchmarking complete. Results in $RESULTS_DIR/ directory."