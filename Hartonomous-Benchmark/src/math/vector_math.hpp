#pragma once

#include <vector>
#include <concepts>
#include <type_traits>
#include <immintrin.h>
#include <mkl.h>
#include <Eigen/Dense>

namespace hartonomous {

// Concepts for type constraints
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

// Templated vector math library
template<Arithmetic T>
class VectorMath {
public:
    // Basic operations
    static void add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static void subtract(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static void multiply(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static void divide(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);

    // Scalar operations
    static void scalar_add(std::vector<T>& v, T scalar);
    static void scalar_multiply(std::vector<T>& v, T scalar);

    // Reduction operations
    static T sum(const std::vector<T>& v);
    static T dot_product(const std::vector<T>& a, const std::vector<T>& b);

    // SIMD-accelerated operations (if available)
    static void add_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static void multiply_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static T dot_product_simd(const std::vector<T>& a, const std::vector<T>& b);

    // MKL-accelerated operations (for floating point)
    static void add_mkl(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static void multiply_mkl(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static T dot_product_mkl(const std::vector<T>& a, const std::vector<T>& b);

    // Hybrid operations (MKL + SIMD)
    static void add_hybrid(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result);
    static T dot_product_hybrid(const std::vector<T>& a, const std::vector<T>& b);

    // Memory bandwidth test
    static size_t memory_bandwidth_read(const std::vector<T>& v, size_t iterations = 1000);
    static size_t memory_bandwidth_write(std::vector<T>& v, size_t iterations = 1000);
};

// Implementation

template<Arithmetic T>
void VectorMath<T>::add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

template<Arithmetic T>
void VectorMath<T>::subtract(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

template<Arithmetic T>
void VectorMath<T>::multiply(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

template<Arithmetic T>
void VectorMath<T>::divide(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    for (size_t i = 0; i < size; ++i) {
        if constexpr (FloatingPoint<T>) {
            result[i] = a[i] / b[i];
        } else {
            result[i] = a[i] / b[i]; // Integer division
        }
    }
}

template<Arithmetic T>
void VectorMath<T>::scalar_add(std::vector<T>& v, T scalar) {
    for (auto& val : v) {
        val += scalar;
    }
}

template<Arithmetic T>
void VectorMath<T>::scalar_multiply(std::vector<T>& v, T scalar) {
    for (auto& val : v) {
        val *= scalar;
    }
}

template<Arithmetic T>
T VectorMath<T>::sum(const std::vector<T>& v) {
    T total = 0;
    for (const auto& val : v) {
        total += val;
    }
    return total;
}

template<Arithmetic T>
T VectorMath<T>::dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    T total = 0;
    size_t size = std::min(a.size(), b.size());
    for (size_t i = 0; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// SIMD implementations
template<Arithmetic T>
void VectorMath<T>::add_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    size_t i = 0;

    if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i], vr);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vr = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(&result[i], vr);
        }
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

template<Arithmetic T>
void VectorMath<T>::multiply_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    size_t size = std::min({a.size(), b.size(), result.size()});
    size_t i = 0;

    if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&result[i], vr);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vr = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(&result[i], vr);
        }
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

template<Arithmetic T>
T VectorMath<T>::dot_product_simd(const std::vector<T>& a, const std::vector<T>& b) {
    T total = 0;
    size_t size = std::min(a.size(), b.size());
    size_t i = 0;

    if constexpr (std::is_same_v<T, float>) {
        __m256 sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        }
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        for (int j = 0; j < 8; ++j) total += temp[j];
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d sum = _mm256_setzero_pd();
        for (; i + 4 <= size; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
        }
        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        for (int j = 0; j < 4; ++j) total += temp[j];
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }

    return total;
}

// MKL implementations (for floating point types)
template<Arithmetic T>
void VectorMath<T>::add_mkl(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    if constexpr (FloatingPoint<T>) {
        size_t size = std::min({a.size(), b.size(), result.size()});
        if constexpr (std::is_same_v<T, float>) {
            vsAdd(size, a.data(), b.data(), result.data());
        } else if constexpr (std::is_same_v<T, double>) {
            vdAdd(size, a.data(), b.data(), result.data());
        }
    } else {
        add(a, b, result); // Fallback for non-floating point
    }
}

template<Arithmetic T>
void VectorMath<T>::multiply_mkl(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    if constexpr (FloatingPoint<T>) {
        size_t size = std::min({a.size(), b.size(), result.size()});
        if constexpr (std::is_same_v<T, float>) {
            vsMul(size, a.data(), b.data(), result.data());
        } else if constexpr (std::is_same_v<T, double>) {
            vdMul(size, a.data(), b.data(), result.data());
        }
    } else {
        multiply(a, b, result); // Fallback for non-floating point
    }
}

template<Arithmetic T>
T VectorMath<T>::dot_product_mkl(const std::vector<T>& a, const std::vector<T>& b) {
    if constexpr (FloatingPoint<T>) {
        size_t size = std::min(a.size(), b.size());
        T result;
        if constexpr (std::is_same_v<T, float>) {
            result = cblas_sdot(size, a.data(), 1, b.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            result = cblas_ddot(size, a.data(), 1, b.data(), 1);
        }
        return result;
    } else {
        return dot_product(a, b); // Fallback for non-floating point
    }
}

// Hybrid implementations
template<Arithmetic T>
void VectorMath<T>::add_hybrid(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
    // Use MKL for large vectors, SIMD for smaller ones
    size_t size = std::min({a.size(), b.size(), result.size()});
    if (size > 10000) {
        add_mkl(a, b, result);
    } else {
        add_simd(a, b, result);
    }
}

template<Arithmetic T>
T VectorMath<T>::dot_product_hybrid(const std::vector<T>& a, const std::vector<T>& b) {
    // Use MKL for large vectors, SIMD for smaller ones
    size_t size = std::min(a.size(), b.size());
    if (size > 10000) {
        return dot_product_mkl(a, b);
    } else {
        return dot_product_simd(a, b);
    }
}

// Memory bandwidth tests
template<Arithmetic T>
size_t VectorMath<T>::memory_bandwidth_read(const std::vector<T>& v, size_t iterations) {
    volatile T sum = 0;
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (const auto& val : v) {
            sum += val;
        }
    }
    return v.size() * sizeof(T) * iterations; // Bytes read
}

template<Arithmetic T>
size_t VectorMath<T>::memory_bandwidth_write(std::vector<T>& v, size_t iterations) {
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (auto& val : v) {
            val += 1;
        }
    }
    return v.size() * sizeof(T) * iterations; // Bytes written
}

} // namespace hartonomous