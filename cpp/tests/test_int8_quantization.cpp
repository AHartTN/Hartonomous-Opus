#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <type_traits>

#if HAS_MKL
#include <mkl.h>
#endif

using namespace Eigen;
using namespace std;
using namespace chrono;

template<typename T>
using MatrixT = Matrix<T, Dynamic, Dynamic>;

template<typename T>
MatrixT<T> random_matrix(size_t rows, size_t cols, T min_val, T max_val) {
    MatrixT<T> mat(rows, cols);
    random_device rd;
    mt19937 gen(rd());

    if constexpr (std::is_integral_v<T>) {
        if constexpr (sizeof(T) == 1) {
            // For int8_t/char types, use int as intermediate
            uniform_int_distribution<int> dist(static_cast<int>(min_val), static_cast<int>(max_val));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat(i, j) = static_cast<T>(dist(gen));
                }
            }
        } else {
            uniform_int_distribution<T> dist(min_val, max_val);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat(i, j) = dist(gen);
                }
            }
        }
    } else {
        uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat(i, j) = dist(gen);
            }
        }
    }

    return mat;
}

template<typename T>
double benchmark_gemm(size_t n, size_t iterations = 10) {
    // Create random matrices
    auto A = random_matrix<T>(n, n, -127, 127);
    auto B = random_matrix<T>(n, n, -127, 127);
    MatrixT<T> C(n, n);

    // Warmup
    C = A * B;

    // Benchmark
    vector<double> times;
    for (size_t i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        C = A * B;
        auto end = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
    }

    // Calculate statistics
    double sum = 0.0;
    for (auto t : times) sum += t;
    double mean = sum / times.size();

    double variance = 0.0;
    for (auto t : times) variance += (t - mean) * (t - mean);
    variance /= times.size();
    double stddev = sqrt(variance);

    cout << fixed << setprecision(2);
    cout << "Size: " << n << "x" << n << " | Mean: " << mean << "ms | StdDev: " << stddev << "ms" << endl;

    return mean;
}

int main() {
    cout << "=== Int8 Quantization GEMM Benchmark ===\n\n";

#if HAS_MKL
    cout << "Using Intel MKL for optimized matrix operations\n";
#else
    cout << "Using Eigen (no MKL detected)\n";
#endif

    cout << "Benchmarking matrix multiplication: C = A * B\n\n";

    // Test different matrix sizes
    vector<size_t> sizes = {256, 512, 1024, 2048};

    cout << "Int8 Matrix Multiplication:\n";
    cout << "----------------------------\n";
    for (auto size : sizes) {
        benchmark_gemm<int8_t>(size);
    }

    cout << "\nFloat32 Matrix Multiplication (comparison):\n";
    cout << "-------------------------------------------\n";
    for (auto size : sizes) {
        benchmark_gemm<float>(size);
    }

    cout << "\nNote: Int8 operations should show significant speedup with AVX_VNNI\n";
    cout << "      if available, especially for larger matrices.\n";

    return 0;
}