# Validation script for cross-platform CMake contract compliance

param(
    [string]$BuildDir = "build"
)

Write-Host "Validating CMake build system compliance..."

$errors = @()

# Check top-level CMakeLists.txt
$cmakeLists = Get-Content "CMakeLists.txt" -Raw

# Forbidden patterns
$forbiddenPatterns = @(
    "HAVE_STD_REGEX.*FORCE",
    "HAVE_GNU_POSIX_REGEX.*FORCE",
    "HAVE_POSIX_REGEX.*FORCE",
    "THREADS_PREFER_PTHREAD_FLAG.*ON",
    "CMAKE_THREAD_LIBS_INIT.*pthreads",
    "lpthreads",
    "lrt\.lib",
    "-lpthread",
    "-lrt"
)

foreach ($pattern in $forbiddenPatterns) {
    if ($cmakeLists -match $pattern) {
        $errors += "Found forbidden pattern '$pattern' in CMakeLists.txt"
    }
}

# Check for Windows-specific POSIX injections
if ($cmakeLists -match "WIN32" -and $cmakeLists -match "lpthreads|pthread") {
    $errors += "Found pthreads reference in WIN32 conditional in CMakeLists.txt"
}

# Check benchmarks CMakeLists.txt
$benchmarksCMake = Get-Content "benchmarks/CMakeLists.txt" -Raw
if ($benchmarksCMake -notmatch "target_link_libraries\(benchmark_suite.*benchmark::benchmark.*Eigen3::Eigen.*hnswlib.*math\)") {
    $errors += "benchmark_suite target_link_libraries does not follow contract"
}

# Check for MKL conditional
if ($benchmarksCMake -notmatch "if\(MKL_FOUND\)") {
    $errors += "Missing MKL_FOUND conditional in benchmarks CMakeLists.txt"
}

# Check build directory for generated files if it exists
if (Test-Path $BuildDir) {
    # Check for forbidden libs in build logs or cache
    $cmakeCache = Get-Content "$BuildDir/CMakeCache.txt" -Raw -ErrorAction SilentlyContinue
    if ($cmakeCache -match "THREADS_PREFER_PTHREAD_FLAG.*ON") {
        $errors += "THREADS_PREFER_PTHREAD_FLAG set to ON in CMakeCache.txt"
    }
}

if ($errors.Count -eq 0) {
    Write-Host "✅ All validations passed. Build system is compliant."
    exit 0
} else {
    Write-Host "❌ Validation failed with $($errors.Count) errors:"
    foreach ($err in $errors) {
        Write-Host "  - $err"
    }
    exit 1
}