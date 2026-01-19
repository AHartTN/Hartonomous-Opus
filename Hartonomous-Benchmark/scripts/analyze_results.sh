#!/bin/bash

set -e

# Default values
RESULTS_DIR="results"
OUTPUT_FORMAT="summary"
DETAILED=false
CSV_EXPORT=""
JSON_EXPORT=""
COMPARE_WITH=""
VERBOSE=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Analyze benchmark results with various output formats and comparisons."
    echo ""
    echo "Options:"
    echo "  -d, --detailed           Show detailed analysis including statistics"
    echo "  -f, --format FORMAT      Output format: summary, detailed, csv, json (default: summary)"
    echo "  --csv FILE               Export results to CSV file"
    echo "  --json FILE              Export results to JSON file"
    echo "  --compare FILE           Compare with previous results file"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -h, --help               Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --detailed"
    echo "  $0 --format csv --csv analysis.csv"
    echo "  $0 --compare results/previous.json"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detailed)
            DETAILED=true
            shift
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --csv)
            CSV_EXPORT="$2"
            shift 2
            ;;
        --json)
            JSON_EXPORT="$2"
            shift 2
            ;;
        --compare)
            COMPARE_WITH="$2"
            shift 2
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

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' not found."
    echo "Run ./scripts/run_benchmarks.sh first."
    exit 1
fi

# Check for required tools
if ! command -v jq >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "Warning: Neither jq nor python3 found. JSON parsing will be limited. Install jq for full functionality."
fi

# Function to extract value from JSON using jq or python
extract_json_value() {
    local file="$1"
    local key="$2"
    if command -v jq >/dev/null 2>&1; then
        jq -r "$key" "$file" 2>/dev/null || echo "N/A"
    else
        python3 -c "
import json
import sys
try:
    with open('$file', 'r') as f:
        data = json.load(f)
    keys = '$key'.replace('.[', '[').split('.')
    result = data
    for key in keys:
        if '[' in key and ']' in key:
            k, idx = key.split('[')
            idx = int(idx.rstrip(']'))
            result = result[k][idx]
        else:
            result = result[key]
    print(result)
except:
    print('N/A')
" 2>/dev/null || echo "N/A"
    fi
}

# Function to calculate statistics
calculate_stats() {
    local file="$1"
    if command -v jq >/dev/null 2>&1; then
        echo "=== Benchmark Statistics ==="
        echo "Total benchmarks: $(jq '.benchmarks | length' "$file" 2>/dev/null || echo "N/A")"

        # Calculate average time
        local avg_time=$(jq '[.benchmarks[].real_time] | add / length' "$file" 2>/dev/null || echo "N/A")
        if [ "$avg_time" != "N/A" ]; then
            echo "Average real time: $(printf "%.2f" $avg_time) ns"
        fi

        # Calculate median time
        local median_time=$(jq '[.benchmarks[].real_time] | sort | .[length/2 | floor]' "$file" 2>/dev/null || echo "N/A")
        if [ "$median_time" != "N/A" ]; then
            echo "Median real time: $(printf "%.2f" $median_time) ns"
        fi

        # Calculate standard deviation
        local std_dev=$(jq '
        def mean: add / length;
        def variance: [ .[] as $x | ($x - mean)^2 ] | mean;
        [ .benchmarks[].real_time ] | if length > 0 then variance | sqrt else 0 end
        ' "$file" 2>/dev/null || echo "N/A")
        if [ "$std_dev" != "N/A" ]; then
            echo "Standard deviation: $(printf "%.2f" $std_dev) ns"
        fi

        # Find fastest and slowest
        local min_time=$(jq '[.benchmarks[].real_time] | min' "$file" 2>/dev/null || echo "N/A")
        local max_time=$(jq '[.benchmarks[].real_time] | max' "$file" 2>/dev/null || echo "N/A")
        if [ "$min_time" != "N/A" ]; then
            echo "Fastest benchmark: $(printf "%.2f" $min_time) ns"
            echo "Slowest benchmark: $(printf "%.2f" $max_time) ns"
        fi
    fi
}

# Function to generate CSV
generate_csv() {
    local input_file="$1"
    local output_file="$2"

    if command -v jq >/dev/null 2>&1; then
        echo "name,real_time,cpu_time,time_unit,iterations" > "$output_file"
        jq -r '.benchmarks[] | [.name, .real_time, .cpu_time, .time_unit, .iterations] | @csv' "$input_file" >> "$output_file"
        echo "CSV exported to $output_file"
    else
        echo "Warning: jq not found. CSV export requires jq."
    fi
}

# Function to compare results
compare_results() {
    local file1="$1"
    local file2="$2"

    if command -v jq >/dev/null 2>&1; then
        echo "=== Results Comparison ==="

        # Extract common benchmarks and compare
        jq -r '
        def common_benchmarks:
            (.benchmarks | map(.name)) as $names1 |
            input.benchmarks | map(.name) as $names2 |
            $names1 - ($names1 - $names2) | unique;

        common_benchmarks[] as $bench |
        (.benchmarks[] | select(.name == $bench) | .real_time) as $time1 |
        (input.benchmarks[] | select(.name == $bench) | .real_time) as $time2 |
        if $time1 > 0 and $time2 > 0 then
            "\($bench): \($time1) -> \($time2) (\( (($time2 - $time1) / $time1 * 100) | floor )%)"
        else
            empty end
        ' "$file1" "$file2"
    else
        echo "Warning: jq required for comparison."
    fi
}

# Main analysis
echo "Benchmark Results Analysis"
echo "=========================="

# Hardware info
if [ -f "$RESULTS_DIR/hardware_info.txt" ]; then
    echo ""
    echo "Hardware Information:"
    echo "---------------------"
    cat "$RESULTS_DIR/hardware_info.txt"
fi

# Perf stat
if [ -f "$RESULTS_DIR/perf_stat.txt" ]; then
    echo ""
    echo "Performance Statistics:"
    echo "-----------------------"
    cat "$RESULTS_DIR/perf_stat.txt"
fi

# Results analysis
RESULTS_FILE="$RESULTS_DIR/results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Benchmark Results:"
    echo "------------------"

    case $OUTPUT_FORMAT in
        summary)
            calculate_stats "$RESULTS_FILE"
            echo ""
            echo "Detailed Benchmark Results:"
            echo "---------------------------"
            if command -v jq >/dev/null 2>&1; then
                printf "%-60s %-15s %-15s %-12s %-10s\n" "Benchmark Name" "Real Time" "CPU Time" "Time Unit" "Iterations"
                printf "%-60s %-15s %-15s %-12s %-10s\n" "--------------" "---------" "--------" "---------" "----------"
                jq -r '.benchmarks[] | "\(.name),\(.real_time),\(.cpu_time),\(.time_unit),\(.iterations)"' "$RESULTS_FILE" 2>/dev/null | while IFS=',' read -r name real cpu unit iter; do
                    printf "%-60s %-15s %-15s %-12s %-10s\n" "$name" "$real" "$cpu" "$unit" "$iter"
                done
            else
                echo "Warning: jq not found. Showing raw JSON output. Install jq for detailed analysis and statistics."
                echo "First 50 lines of results:"
                head -50 "$RESULTS_FILE"
            fi
            ;;
        detailed)
            calculate_stats "$RESULTS_FILE"
            echo ""
            echo "All Benchmark Results (Detailed):"
            echo "----------------------------------"
            if command -v jq >/dev/null 2>&1; then
                printf "%-60s %-15s %-15s %-12s %-10s %-15s\n" "Benchmark Name" "Real Time" "CPU Time" "Time Unit" "Iterations" "Items/sec"
                printf "%-60s %-15s %-15s %-12s %-10s %-15s\n" "--------------" "---------" "--------" "---------" "----------" "---------"
                jq -r '.benchmarks[] | "\(.name),\(.real_time),\(.cpu_time),\(.time_unit),\(.iterations),\(.items_per_second // "N/A")"' "$RESULTS_FILE" 2>/dev/null | while IFS=',' read -r name real cpu unit iter items; do
                    printf "%-60s %-15s %-15s %-12s %-10s %-15s\n" "$name" "$real" "$cpu" "$unit" "$iter" "$items"
                done
            else
                cat "$RESULTS_FILE"
            fi
            ;;
        csv)
            if [ -n "$CSV_EXPORT" ]; then
                generate_csv "$RESULTS_FILE" "$CSV_EXPORT"
            else
                generate_csv "$RESULTS_FILE" "$RESULTS_DIR/analysis.csv"
            fi
            ;;
        json)
            if [ -n "$JSON_EXPORT" ]; then
                cp "$RESULTS_FILE" "$JSON_EXPORT"
                echo "JSON exported to $JSON_EXPORT"
            else
                echo "JSON results available at $RESULTS_FILE"
            fi
            ;;
    esac

    # Comparison
    if [ -n "$COMPARE_WITH" ]; then
        if [ -f "$COMPARE_WITH" ]; then
            echo ""
            compare_results "$RESULTS_FILE" "$COMPARE_WITH"
        else
            echo "Warning: Comparison file '$COMPARE_WITH' not found."
        fi
    fi

else
    echo "No benchmark results found. Run ./scripts/run_benchmarks.sh first."
fi

# Additional exports
if [ -n "$CSV_EXPORT" ] && [ "$OUTPUT_FORMAT" != "csv" ]; then
    generate_csv "$RESULTS_FILE" "$CSV_EXPORT"
fi

if [ -n "$JSON_EXPORT" ] && [ "$OUTPUT_FORMAT" != "json" ]; then
    cp "$RESULTS_FILE" "$JSON_EXPORT"
    echo "JSON exported to $JSON_EXPORT"
fi

echo ""
echo "Analysis complete."