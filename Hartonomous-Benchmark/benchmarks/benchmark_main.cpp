#include <benchmark/benchmark.h>
#include <iostream>
#include "matrix_benchmark.hpp"
#include "linear_solve_benchmark.hpp"
#include "fft_benchmark.hpp"
#include "rng_benchmark.hpp"
#include "eigen_matrix_multiply_benchmark.hpp"
#include "eigen_lu_solve_benchmark.hpp"
#include "eigen_svd_benchmark.hpp"
#include "eigen_sparse_benchmark.hpp"
#include "hnsw_index_build_benchmark.hpp"
#include "hnsw_search_benchmark.hpp"
#include "hnsw_insertion_benchmark.hpp"
#include "simd_intrinsics_benchmark.hpp"
#include "vnni_dot_product_benchmark.hpp"
#include "avx_vector_arithmetic_benchmark.hpp"
#include "hybrid_mkl_simd_benchmark.hpp"
#include "hybrid_eigen_avx_benchmark.hpp"
#include "memory_bandwidth_benchmark.hpp"
#include "micro_benchmarks.hpp"
#include "../src/results/result_aggregator.hpp"
#include "../src/benchmark/benchmark_registry.hpp"
#include "../src/benchmark/benchmark_factory.hpp"
#include "../src/hardware.hpp" // Include hardware detection

// Legacy Google Benchmark functions
static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state) {
    std::string s(state.range(0), 'a');
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_StringCreation)->RangeMultiplier(2)->Range(1, 1<<10);

// New framework benchmarks
static void BM_MatrixMultiplyFloat(benchmark::State& state) {
  MatrixMultiplyBenchmark<float> bench;
  BenchmarkConfig config;
  config.data_size = state.range(0);
  config.iterations = 1;

  bench.setup(config);

  for (auto _ : state) {
    bench.run();
  }

  auto result = bench.get_result();
  state.SetIterationTime(std::chrono::duration_cast<std::chrono::duration<double>>(result.duration).count());
}
BENCHMARK(BM_MatrixMultiplyFloat)->RangeMultiplier(2)->Range(8, 64)->Unit(benchmark::kMicrosecond);

// Framework demonstration
void run_new_framework_benchmarks() {
    std::cout << "Running new benchmark framework...\n";
    std::cout << "Hardware Detection:\n";
    print_cpu_info();
    // Add GPU and memory info if needed
    HardwareDetector detector;
    auto gpu = detector.detect_gpu_info();
    auto mem = detector.get_memory_info();
    std::cout << "GPU: " << gpu.name << " (" << gpu.memory_mb << " MB)\n";
    std::cout << "Memory: " << mem.total_mb << " MB total, " << mem.available_mb << " MB available\n\n";

    BenchmarkRegistry registry;

    // Register existing benchmarks
    registry.register_typed_benchmark<double, std::string, MatrixMultiplyBenchmark<double>>("MatrixMultiply");
    registry.register_typed_benchmark<double, std::string, LinearSolveBenchmark<double>>("LinearSolve");
    registry.register_typed_benchmark<std::complex<double>, std::string, FFTBenchmark<std::complex<double>>>("FFT");
    registry.register_typed_benchmark<double, std::string, RNGBenchmark<double>>("RNG");
    registry.register_typed_benchmark<double, std::string, EigenMatrixMultiplyBenchmark<double>>("EigenMatrixMultiply");
    registry.register_typed_benchmark<double, std::string, EigenLUSolveBenchmark<double>>("EigenLUSolve");
    registry.register_typed_benchmark<double, std::string, EigenSVDBenchmark<double>>("EigenSVD");
    registry.register_typed_benchmark<double, std::string, EigenSparseBenchmark<double>>("EigenSparse");
    registry.register_typed_benchmark<float, std::string, HNSWIndexBuildBenchmark<float>>("HNSWIndexBuild");
    registry.register_typed_benchmark<float, std::string, HNSWSearchBenchmark<float>>("HNSWSearch");
    registry.register_typed_benchmark<float, std::string, HNSWInsertionBenchmark<float>>("HNSWInsertion");
    registry.register_typed_benchmark<float, std::string, SimdIntrinsicsBenchmark<float>>("SIMDIntrinsicsFloat");
    registry.register_typed_benchmark<double, std::string, SimdIntrinsicsBenchmark<double>>("SIMDIntrinsicsDouble");
    registry.register_typed_benchmark<int8_t, std::string, VnniDotProductBenchmark<int8_t>>("VNNIDotProductInt8");
    registry.register_typed_benchmark<int16_t, std::string, VnniDotProductBenchmark<int16_t>>("VNNIDotProductInt16");
    registry.register_typed_benchmark<float, std::string, VnniDotProductBenchmark<float>>("VNNIDotProductFloat");
    registry.register_typed_benchmark<double, std::string, VnniDotProductBenchmark<double>>("VNNIDotProductDouble");
    registry.register_typed_benchmark<float, std::string, AvxVectorArithmeticBenchmark<float>>("AVXVectorArithmeticFloat");
    registry.register_typed_benchmark<double, std::string, AvxVectorArithmeticBenchmark<double>>("AVXVectorArithmeticDouble");

    registry.register_typed_benchmark<float, std::string, HybridMKL_SIMD_Benchmark<float>>("HybridMKL_SIMD_Float");
    registry.register_typed_benchmark<double, std::string, HybridMKL_SIMD_Benchmark<double>>("HybridMKL_SIMD_Double");
    registry.register_typed_benchmark<int32_t, std::string, HybridMKL_SIMD_Benchmark<int32_t>>("HybridMKL_SIMD_Int32");
    registry.register_typed_benchmark<int64_t, std::string, HybridMKL_SIMD_Benchmark<int64_t>>("HybridMKL_SIMD_Int64");

    registry.register_typed_benchmark<float, std::string, HybridEigen_AVX_Benchmark<float>>("HybridEigen_AVX_Float");
    registry.register_typed_benchmark<double, std::string, HybridEigen_AVX_Benchmark<double>>("HybridEigen_AVX_Double");

    // Register memory bandwidth benchmarks
    registry.register_typed_benchmark<float, std::string, MemoryReadSequentialBenchmark<float>>("Memory_Read_Sequential_Float");
    registry.register_typed_benchmark<double, std::string, MemoryReadSequentialBenchmark<double>>("Memory_Read_Sequential_Double");
    registry.register_typed_benchmark<float, std::string, MemoryWriteSequentialBenchmark<float>>("Memory_Write_Sequential_Float");
    registry.register_typed_benchmark<double, std::string, MemoryWriteSequentialBenchmark<double>>("Memory_Write_Sequential_Double");
    registry.register_typed_benchmark<float, std::string, MemoryReadRandomBenchmark<float>>("Memory_Read_Random_Float");
    registry.register_typed_benchmark<double, std::string, MemoryReadRandomBenchmark<double>>("Memory_Read_Random_Double");
    registry.register_typed_benchmark<float, std::string, MemoryWriteRandomBenchmark<float>>("Memory_Write_Random_Float");
    registry.register_typed_benchmark<double, std::string, MemoryWriteRandomBenchmark<double>>("Memory_Write_Random_Double");
    registry.register_typed_benchmark<float, std::string, MemoryStreamCopyBenchmark<float>>("Memory_Stream_Copy_Float");
    registry.register_typed_benchmark<double, std::string, MemoryStreamCopyBenchmark<double>>("Memory_Stream_Copy_Double");
    registry.register_typed_benchmark<float, std::string, MemoryStreamScaleBenchmark<float>>("Memory_Stream_Scale_Float");
    registry.register_typed_benchmark<double, std::string, MemoryStreamScaleBenchmark<double>>("Memory_Stream_Scale_Double");

    // Register micro-benchmarks
    registry.register_typed_benchmark<float, std::string, VectorAdditionScalarBenchmark<float>>("Vector_Add_Scalar_Float");
    registry.register_typed_benchmark<double, std::string, VectorAdditionScalarBenchmark<double>>("Vector_Add_Scalar_Double");
    registry.register_typed_benchmark<float, std::string, VectorAdditionSIMDBenchmark<float>>("Vector_Add_SIMD_Float");
    registry.register_typed_benchmark<double, std::string, VectorAdditionSIMDBenchmark<double>>("Vector_Add_SIMD_Double");
    registry.register_typed_benchmark<float, std::string, VectorAdditionMKLBenchmark<float>>("Vector_Add_MKL_Float");
    registry.register_typed_benchmark<double, std::string, VectorAdditionMKLBenchmark<double>>("Vector_Add_MKL_Double");
    registry.register_typed_benchmark<float, std::string, VectorAdditionHybridBenchmark<float>>("Vector_Add_Hybrid_Float");
    registry.register_typed_benchmark<double, std::string, VectorAdditionHybridBenchmark<double>>("Vector_Add_Hybrid_Double");

    registry.register_typed_benchmark<float, std::string, DotProductScalarBenchmark<float>>("Dot_Product_Scalar_Float");
    registry.register_typed_benchmark<double, std::string, DotProductScalarBenchmark<double>>("Dot_Product_Scalar_Double");
    registry.register_typed_benchmark<float, std::string, DotProductSIMDBenchmark<float>>("Dot_Product_SIMD_Float");
    registry.register_typed_benchmark<double, std::string, DotProductSIMDBenchmark<double>>("Dot_Product_SIMD_Double");
    registry.register_typed_benchmark<float, std::string, DotProductMKLBenchmark<float>>("Dot_Product_MKL_Float");
    registry.register_typed_benchmark<double, std::string, DotProductMKLBenchmark<double>>("Dot_Product_MKL_Double");
    registry.register_typed_benchmark<float, std::string, DotProductHybridBenchmark<float>>("Dot_Product_Hybrid_Float");
    registry.register_typed_benchmark<double, std::string, DotProductHybridBenchmark<double>>("Dot_Product_Hybrid_Double");

    registry.register_typed_benchmark<float, std::string, MatrixVectorEigenBenchmark<float>>("Matrix_Vector_Eigen_Float");
    registry.register_typed_benchmark<double, std::string, MatrixVectorEigenBenchmark<double>>("Matrix_Vector_Eigen_Double");
    registry.register_typed_benchmark<float, std::string, MatrixVectorEigenAVXBenchmark<float>>("Matrix_Vector_Eigen_AVX_Float");
    registry.register_typed_benchmark<double, std::string, MatrixVectorEigenAVXBenchmark<double>>("Matrix_Vector_Eigen_AVX_Double");

    std::vector<std::string> benchmark_names = {
        "MatrixMultiply", "LinearSolve", "FFT", "RNG", "EigenMatrixMultiply", "EigenLUSolve", "EigenSVD", "EigenSparse",
        "HNSWIndexBuild", "HNSWSearch", "HNSWInsertion", "SIMDIntrinsicsFloat", "SIMDIntrinsicsDouble",
        "VNNIDotProductInt8", "VNNIDotProductInt16", "VNNIDotProductFloat", "VNNIDotProductDouble",
        "AVXVectorArithmeticFloat", "AVXVectorArithmeticDouble",
        "HybridMKL_SIMD_Float", "HybridMKL_SIMD_Double", "HybridMKL_SIMD_Int32", "HybridMKL_SIMD_Int64",
        "HybridEigen_AVX_Float", "HybridEigen_AVX_Double",
        "Memory_Read_Sequential_Float", "Memory_Read_Sequential_Double", "Memory_Write_Sequential_Float", "Memory_Write_Sequential_Double",
        "Memory_Read_Random_Float", "Memory_Read_Random_Double", "Memory_Write_Random_Float", "Memory_Write_Random_Double",
        "Memory_Stream_Copy_Float", "Memory_Stream_Copy_Double", "Memory_Stream_Scale_Float", "Memory_Stream_Scale_Double",
        "Vector_Add_Scalar_Float", "Vector_Add_Scalar_Double", "Vector_Add_SIMD_Float", "Vector_Add_SIMD_Double",
        "Vector_Add_MKL_Float", "Vector_Add_MKL_Double", "Vector_Add_Hybrid_Float", "Vector_Add_Hybrid_Double",
        "Dot_Product_Scalar_Float", "Dot_Product_Scalar_Double", "Dot_Product_SIMD_Float", "Dot_Product_SIMD_Double",
        "Dot_Product_MKL_Float", "Dot_Product_MKL_Double", "Dot_Product_Hybrid_Float", "Dot_Product_Hybrid_Double",
        "Matrix_Vector_Eigen_Float", "Matrix_Vector_Eigen_Double", "Matrix_Vector_Eigen_AVX_Float", "Matrix_Vector_Eigen_AVX_Double"
    };

    ResultAggregator all_results_aggregator;

    for (const auto& name : benchmark_names) {
        auto bench = registry.create_benchmark(name);
        if (bench) {
            BenchmarkConfig config{1000, 256, false, "double"}; // Adjust size as needed
            if (name == "FFT") {
                config.data_size = 512; // Power of 2 for FFT
            } else if (name == "RNG") {
                config.data_size = 1000000; // Large size for RNG
            }
            bench->setup(config);

            ResultAggregator aggregator; // Per benchmark aggregator

            for (int i = 0; i < 5; ++i) {  // Run 5 times for aggregation
                bench->run();
                auto result = bench->get_result();
                aggregator.add_result(result);
                all_results_aggregator.add_result(result); // Also add to global for CSV
            }

            auto aggregated = aggregator.aggregate();
            std::cout << "Aggregated results for " << aggregated.benchmark_name << ":\n";
            std::cout << "  Runs: " << aggregated.total_runs << "\n";
            std::cout << "  Avg duration: " << aggregated.avg_duration.count() << " ns\n";
            std::cout << "  Throughput: " << aggregated.throughput_avg << " ops/s\n\n";
        }
    }

    all_results_aggregator.export_csv("results/new_framework_results.csv");
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // Run legacy benchmarks
    benchmark::RunSpecifiedBenchmarks();

    // Run new framework benchmarks
    run_new_framework_benchmarks();

    return 0;
}