param(
    [string]$Filter = "",
    [string]$Category = "",
    [string]$Type = "",
    [string]$Format = "json",
    [string]$Repetitions = "",
    [string]$MinTime = "",
    [switch]$SkipBuild,
    [switch]$SkipPerf,
    [switch]$SkipValgrind,
    [switch]$SkipHardware,
    [switch]$Verbose,
    [switch]$Help
)

# Starting Directory (Where you were when you ran the script)
$StartDir = Get-Location
# Default values
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
$BuildDir = "build"
$ResultsDir = "results"
$ScriptName = $MyInvocation.MyCommand.Name

# Function to display usage
function Show-Usage {
    Write-Host "Usage: $ScriptName [OPTIONS]"
    Write-Host ""
    Write-Host "Run the comprehensive benchmark suite with various filtering and configuration options."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Filter PATTERN        Filter benchmarks by name pattern (regex)"
    Write-Host "  -Category CATEGORY     Run benchmarks from specific category:"
    Write-Host "                         linear_algebra, signal_processing, rng, ann,"
    Write-Host "                         simd, hybrid, memory, micro"
    Write-Host "  -Type TYPE             Filter by data type: float, double, int8, int16, int32, int64"
    Write-Host "  -Format FORMAT         Output format: json, csv, console (default: json)"
    Write-Host "  -Repetitions N         Number of benchmark repetitions (default: 1)"
    Write-Host "  -MinTime SEC           Minimum time per benchmark in seconds (default: 0.1)"
    Write-Host "  -SkipBuild             Skip build step (assume already built)"
    Write-Host "  -SkipPerf              Skip perf profiling"
    Write-Host "  -SkipValgrind          Skip valgrind profiling"
    Write-Host "  -SkipHardware          Skip hardware detection"
    Write-Host "  -Verbose               Enable verbose output"
    Write-Host "  -Help                  Display this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  $ScriptName -Category linear_algebra -Type double"
    Write-Host "  $ScriptName -Filter 'Matrix.*' -Format csv"
    Write-Host "  $ScriptName -Category memory -SkipValgrind"
    Write-Host "  $ScriptName -Repetitions 5 -MinTime 1.0"
    exit 1
}

if ($Help) {
    Show-Usage
}

# Function to check tool availability
function Test-Tool {
    param([string]$Tool)
    try {
        Get-Command $Tool -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Create results directory
if (!(Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

# Build if not skipped
if (!$SkipBuild) {
    $BenchmarkExe = Join-Path $BuildDir "benchmarks/benchmark_suite.exe"
    if (!(Test-Path $BenchmarkExe)) {
        if (!(Test-Path $BuildDir)) {
            Write-Host "Creating build directory..."
            New-Item -ItemType Directory -Path $BuildDir | Out-Null
        }

        Write-Host "Configuring and building project..."
        Push-Location $BuildDir
        try {
            & cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev
            $CpuCount = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
            & cmake --build . --parallel $CpuCount
        } finally {
            Pop-Location
        }
    }
}

# Construct benchmark filter based on category
$BenchmarkFilter = ""
if ($Category) {
    switch ($Category) {
        "linear_algebra" { $BenchmarkFilter = "MatrixMultiply|LinearSolve|EigenMatrixMultiply|EigenLUSolve|EigenSVD|EigenSparse" }
        "signal_processing" { $BenchmarkFilter = "FFT" }
        "rng" { $BenchmarkFilter = "RNG" }
        "ann" { $BenchmarkFilter = "HNSW" }
        "simd" { $BenchmarkFilter = "SIMDIntrinsics|VNNIDotProduct|AVXVectorArithmetic" }
        "hybrid" { $BenchmarkFilter = "Hybrid" }
        "memory" { $BenchmarkFilter = "Memory" }
        "micro" { $BenchmarkFilter = "Vector_Add|Dot_Product|Matrix_Vector" }
        default {
            Write-Host "Unknown category: $Category"
            exit 1
        }
    }
}

# Add type filter
if ($Type) {
    switch ($Type) {
        "float" { $TypeFilter = "Float" }
        "double" { $TypeFilter = "Double" }
        "int8" { $TypeFilter = "Int8" }
        "int16" { $TypeFilter = "Int16" }
        "int32" { $TypeFilter = "Int32" }
        "int64" { $TypeFilter = "Int64" }
        default {
            Write-Host "Unknown type: $Type"
            exit 1
        }
    }
    if ($BenchmarkFilter) {
        $BenchmarkFilter = "$BenchmarkFilter.*$TypeFilter|$TypeFilter.*$BenchmarkFilter"
    } else {
        $BenchmarkFilter = $TypeFilter
    }
}

# Combine with custom filter
if ($Filter) {
    if ($BenchmarkFilter) {
        $BenchmarkFilter = "$BenchmarkFilter|$Filter"
    } else {
        $BenchmarkFilter = $Filter
    }
}

# Construct benchmark command
$BenchmarkExe = Join-Path $BuildDir "benchmarks/benchmark_suite.exe"
$BenchmarkArgs = @("--benchmark_format=$Format")

if ($BenchmarkFilter) {
    $BenchmarkArgs += "--benchmark_filter=`"$BenchmarkFilter`""
}

if ($Repetitions) {
    $BenchmarkArgs += "--benchmark_repetitions=$Repetitions"
}

if ($MinTime) {
    $BenchmarkArgs += "--benchmark_min_time=$MinTime"
}

# Output file
$OutputFile = Join-Path $ResultsDir "results.$Format"

if ($Verbose) {
    Write-Host "Benchmark command: $BenchmarkExe $($BenchmarkArgs -join ' ')"
    Write-Host "Output file: $OutputFile"
}

Write-Host "Running benchmark suite..."
if (Test-Path $BenchmarkExe) {
    & $BenchmarkExe $BenchmarkArgs > $OutputFile
} else {
    Write-Host "Error: Benchmark executable not found at $BenchmarkExe. Build may have failed."
    exit 1
}

# Run perf if not skipped (Windows alternative)
if (!$SkipPerf) {
    # On Windows, perf is not available. Use a simple timing or skip.
    Write-Host "Warning: perf tool not available on Windows. Skipping perf profiling."
}

# Run valgrind if not skipped (not available on Windows)
if (!$SkipValgrind) {
    Write-Host "Warning: valgrind not available on Windows. Skipping valgrind profiling."
}

# Run hardware check if not skipped
if (!$SkipHardware) {
    $HardwareExe = Join-Path $BuildDir "src/hardware_check.exe"
    if (Test-Path $HardwareExe) {
        Write-Host "Running hardware check..."
        & $HardwareExe > (Join-Path $ResultsDir "hardware_info.txt")
    } else {
        Write-Host "Warning: hardware_check binary not found. Skipping hardware detection. Ensure the project is built."
    }
}

Write-Host "Benchmarking complete. Results in $ResultsDir/ directory."

Set-Location $StartDir