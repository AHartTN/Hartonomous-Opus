# BenchmarkSuite

An enterprise-grade comprehensive benchmarking suite for C/C++ applications, focusing on high-performance computing, machine learning, and numerical computing workloads.

## Features

- **Micro-benchmarks** using Google Benchmark framework
- **Hardware detection** and profiling utilities
- **System-level profiling** with perf and valgrind
- **Automated result collection and analysis**
- **CI/CD integration** with GitHub Actions
- **Extensive benchmark coverage** including:
  - Linear algebra operations (matrix multiplication, linear solvers, SVD, sparse matrices)
  - Signal processing (FFT)
  - Random number generation
  - Approximate nearest neighbor search (HNSW)
  - SIMD and AVX vector operations
  - Memory bandwidth benchmarks
  - Hybrid CPU-GPU operations (MKL + SIMD, Eigen + AVX)
  - Micro-benchmarks for vector operations

## Supported Frameworks and Libraries

- **Google Benchmark**: Core benchmarking framework
- **Intel MKL**: Optimized math libraries for Intel processors
- **Eigen**: High-performance linear algebra library
- **HNSWlib**: Approximate nearest neighbor search library
- **SIMD Intrinsics**: Direct use of CPU vector instructions
- **AVX/AVX2/AVX-512**: Advanced Vector Extensions
- **VNNI**: Vector Neural Network Instructions for AI workloads

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+ recommended)
- GCC 9+ or Clang 10+
- CMake 3.10+
- CPU with AVX2 support (recommended for full functionality)

### Dependencies Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libbenchmark-dev \
    linux-tools-common \
    valgrind \
    libeigen3-dev \
    libgtest-dev \
    pkg-config

# Intel MKL (if available)
# Download and install from Intel's website or use conda:
# conda install mkl mkl-include

# For MKL, you may need to set environment variables:
export MKLROOT=/path/to/mkl
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
```

## Setup and Build

1. **Clone and prepare:**
   ```bash
   git clone <repository-url>
   cd BenchmarkSuite
   mkdir build && cd build
   ```

2. **Configure with CMake:**
   ```bash
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_FLAGS="-O3 -march=native -fno-omit-frame-pointer"
   ```

3. **Build:**
   ```bash
   make -j$(nproc)
   ```

## Usage

### Running Benchmarks

#### Full Suite
```bash
./scripts/run_benchmarks.sh
```

#### Filtered Benchmarks
```bash
# Run only matrix-related benchmarks
./build/benchmarks/benchmark_suite --benchmark_filter="Matrix.*"

# Run only SIMD benchmarks
./build/benchmarks/benchmark_suite --benchmark_filter="SIMD|VNNI|AVX"

# Run memory benchmarks
./build/benchmarks/benchmark_suite --benchmark_filter="Memory.*"
```

#### Custom Options
```bash
# Specify output format
./build/benchmarks/benchmark_suite --benchmark_format=json --benchmark_out=results/custom.json

# Set minimum time per benchmark
./build/benchmarks/benchmark_suite --benchmark_min_time=0.1

# Run specific repetitions
./build/benchmarks/benchmark_suite --benchmark_repetitions=5
```

### Analyzing Results

```bash
./scripts/analyze_results.sh
```

This generates:
- Hardware information summary
- Performance statistics
- Benchmark result summaries
- CSV exports for further analysis

## Benchmark Categories

### Linear Algebra Benchmarks
- **MatrixMultiply**: Dense matrix-matrix multiplication
- **LinearSolve**: Linear system solving (LU decomposition)
- **EigenMatrixMultiply**: Eigen library matrix operations
- **EigenLUSolve**: Eigen LU solver
- **EigenSVD**: Eigen singular value decomposition
- **EigenSparse**: Eigen sparse matrix operations

### Signal Processing
- **FFT**: Fast Fourier Transform using complex numbers

### Random Number Generation
- **RNG**: High-performance random number generation

### Approximate Nearest Neighbor Search
- **HNSWIndexBuild**: Building HNSW index
- **HNSWSearch**: Searching in HNSW index
- **HNSWInsertion**: Inserting elements into HNSW index

### SIMD and Vector Operations
- **SIMDIntrinsics**: Direct SIMD instruction usage
- **VNNIDotProduct**: VNNI-accelerated dot products (AI workloads)
- **AVXVectorArithmetic**: AVX vector arithmetic operations

### Hybrid Operations
- **HybridMKL_SIMD**: MKL + SIMD hybrid implementations
- **HybridEigen_AVX**: Eigen + AVX hybrid operations

### Memory Benchmarks
- **Memory_Read_Sequential**: Sequential memory read bandwidth
- **Memory_Write_Sequential**: Sequential memory write bandwidth
- **Memory_Read_Random**: Random memory read access
- **Memory_Write_Random**: Random memory write access
- **Memory_Stream_Copy**: STREAM copy benchmark
- **Memory_Stream_Scale**: STREAM scale benchmark

### Micro-Benchmarks
- **Vector_Add_***: Vector addition with different implementations (Scalar, SIMD, MKL, Hybrid)
- **Dot_Product_***: Dot product operations
- **Matrix_Vector_***: Matrix-vector multiplication

## Configuration

Benchmarks can be configured through the `BenchmarkConfig` structure:

```cpp
BenchmarkConfig config;
config.data_size = 1024;      // Problem size
config.iterations = 1000;     // Number of iterations
config.use_gpu = false;       // GPU acceleration (future)
config.precision = "double";  // Data type precision
```

## Output Formats

- **JSON**: Structured results for automated processing
- **CSV**: Spreadsheet-compatible format
- **Console**: Human-readable summaries

## Hardware Detection

The suite includes comprehensive hardware detection:

```bash
./build/src/hardware_check
```

Provides information about:
- CPU model and capabilities
- Memory configuration
- GPU information (if available)
- SIMD instruction set support

## Project Structure

```
BenchmarkSuite/
├── benchmarks/              # Benchmark implementations
│   ├── benchmark_main.cpp   # Main entry point
│   ├── matrix_benchmark.hpp # Linear algebra benchmarks
│   ├── hnsw_*.hpp          # ANN search benchmarks
│   ├── *_benchmark.hpp     # Specialized benchmarks
│   └── CMakeLists.txt
├── src/                     # Core utilities
│   ├── benchmark/           # Benchmark framework
│   ├── timing/              # Timing utilities
│   ├── results/             # Result aggregation
│   ├── math/                # Math utilities
│   └── hardware.*           # Hardware detection
├── scripts/                 # Automation scripts
│   ├── run_benchmarks.sh    # Benchmark execution
│   └── analyze_results.sh   # Result analysis
├── results/                 # Output directory
├── CMakeLists.txt           # Build configuration
└── README.md               # This file
```

## Contributing

1. Add new benchmarks by implementing the `BenchmarkBase` interface
2. Register benchmarks in `benchmark_main.cpp`
3. Update documentation and scripts as needed
4. Ensure compatibility with existing build system

## License

[Specify your license here]