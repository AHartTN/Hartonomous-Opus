#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Unified Benchmark Runner
# ============================================================================
# Runs performance benchmarks across the project with platform detection
# and unified reporting.
#
# Usage:
#   ./scripts/benchmark.sh              # Run comprehensive benchmarks
#   ./scripts/benchmark.sh --quick      # Run quick benchmarks only
#   ./scripts/benchmark.sh --cpp        # C++ benchmarks only
#   ./scripts/benchmark.sh --python     # Python benchmarks only
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"

# Benchmark configuration
PLATFORM=$(detect_os)
QUICK_MODE=false
RUN_CPP=true
RUN_PYTHON=true
OUTPUT_DIR="$PROJECT_ROOT/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true; shift ;;
        --cpp) RUN_PYTHON=false; shift ;;
        --python) RUN_CPP=false; shift ;;
        --output|-o) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --quick       Run quick benchmarks only"
            echo "  --cpp         C++ benchmarks only"
            echo "  --python      Python benchmarks only"
            echo "  --output, -o  Output directory (default: results/)"
            echo "  --help, -h    Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_section "Hartonomous-Opus Benchmarks ($PLATFORM)"

echo "  Mode:        $([ "$QUICK_MODE" = true ] && echo "quick" || echo "comprehensive")"
echo "  Components:  $([ "$RUN_CPP" = true ] && echo "cpp ")$([ "$RUN_PYTHON" = true ] && echo "python ")"
echo "  Output:      $OUTPUT_DIR"
echo "  Timestamp:   $TIMESTAMP"
echo

# ============================================================================
# C++ BENCHMARKS
# ============================================================================

if [ "$RUN_CPP" = true ]; then
    log_subsection "Running C++ Benchmarks"

    # Check if C++ benchmarks are built
    BENCHMARK_EXE="$PROJECT_ROOT/cpp/build/bin/Release/hypercube_bench"
    if [ ! -f "$BENCHMARK_EXE" ] && [ ! -f "${BENCHMARK_EXE}.exe" ]; then
        log_warning "C++ benchmark executable not found"
        log_info "Build benchmarks first: ./scripts/build.sh --release"
    else
        log_info "Running C++ performance benchmarks..."

        cd "$PROJECT_ROOT/cpp/build"

        # Run benchmarks with appropriate format
        if [ "$QUICK_MODE" = true ]; then
            # Quick benchmark mode
            "$BENCHMARK_EXE" --benchmark_filter=".*coordinate.*|.*embedding.*" \
                           --benchmark_format=json \
                           --benchmark_out="$OUTPUT_DIR/cpp_benchmarks_quick_$TIMESTAMP.json"
        else
            # Comprehensive benchmark mode
            "$BENCHMARK_EXE" --benchmark_format=json \
                           --benchmark_out="$OUTPUT_DIR/cpp_benchmarks_full_$TIMESTAMP.json"
        fi

        check_result "C++ benchmarks"
    fi
fi

# ============================================================================
# PYTHON BENCHMARKS
# ============================================================================

if [ "$RUN_PYTHON" = true ]; then
    log_subsection "Running Python Benchmarks"

    if [ -d "$PROJECT_ROOT/Hartonomous-Benchmark" ]; then
        log_info "Running Python benchmark suite..."

        cd "$PROJECT_ROOT/Hartonomous-Benchmark"

        # Run the benchmark script based on platform
        if [ "$PLATFORM" = "windows" ]; then
            if [ "$QUICK_MODE" = true ]; then
                ./scripts/run_benchmarks.bat quick
            else
                ./scripts/run_benchmarks.bat
            fi
        else
            if [ "$QUICK_MODE" = true ]; then
                ./scripts/run_benchmarks.sh quick
            else
                ./scripts/run_benchmarks.sh
            fi
        fi

        check_result "Python benchmarks"
    else
        log_warning "Hartonomous-Benchmark directory not found"
    fi
fi

# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

log_subsection "Generating Benchmark Report"

# Run analysis if available
if [ -f "$PROJECT_ROOT/Hartonomous-Benchmark/scripts/analyze_results.sh" ]; then
    log_info "Analyzing benchmark results..."

    cd "$PROJECT_ROOT/Hartonomous-Benchmark"

    if [ "$PLATFORM" = "windows" ]; then
        ./scripts/analyze_results.bat
    else
        ./scripts/analyze_results.sh
    fi

    check_result "Benchmark analysis"
fi

# Generate summary report
REPORT_FILE="$OUTPUT_DIR/benchmark_summary_$TIMESTAMP.txt"
{
    echo "Hartonomous-Opus Benchmark Summary"
    echo "=================================="
    echo "Date: $(date)"
    echo "Platform: $PLATFORM"
    echo "Mode: $([ "$QUICK_MODE" = true ] && echo "quick" || echo "comprehensive")"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""

    # List generated files
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*$TIMESTAMP*" -type f | while read -r file; do
        echo "  - $(basename "$file")"
    done

    echo ""
    echo "For detailed analysis, see Hartonomous-Benchmark/results/"
} > "$REPORT_FILE"

log_success "Benchmark summary saved to: $REPORT_FILE"

echo
echo "Benchmark results:"
echo "  Summary: $REPORT_FILE"

if [ -d "$PROJECT_ROOT/Hartonomous-Benchmark/results" ]; then
    echo "  Detailed: $PROJECT_ROOT/Hartonomous-Benchmark/results/"
fi

log_success "Benchmark suite completed!"