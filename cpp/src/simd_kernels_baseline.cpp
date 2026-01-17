#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/dispatch.h"
#include <cmath>

// Baseline (scalar) implementations of SIMD kernels

namespace hypercube {

// Define global function pointers for dispatch system
DistanceL2Fn distance_l2_baseline = nullptr;
DistanceIPFn distance_ip_baseline = nullptr;
GemmF32Fn gemm_f32_baseline = nullptr;
DotProductDFn dot_product_d_baseline = nullptr;
DotProductFFn dot_product_f_baseline = nullptr;
ScaleInplaceDFn scale_inplace_d_baseline = nullptr;
SubtractScaledDFn subtract_scaled_d_baseline = nullptr;
NormDFn norm_d_baseline = nullptr;

namespace simd {

float dot_product(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_product_d(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_similarity(const float* a, const float* b, size_t n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return (norm_a > 1e-10f && norm_b > 1e-10f) ? dot / (norm_a * norm_b) : 0.0f;
}

void scale_inplace(double* v, double s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        v[i] *= s;
    }
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

double norm(const double* v, size_t n) {
    return std::sqrt(dot_product_d(v, v, n));
}

void normalize(double* v, size_t n) {
    double nrm = norm(v, n);
    if (nrm > 1e-12) {
        scale_inplace(v, 1.0 / nrm, n);
    }
}

double distance_l2(const float* a, const float* b, size_t n) {
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff);
}

double distance_ip(const float* a, const float* b, size_t n) {
    return static_cast<double>(dot_product(a, b, n));
}

void gemm_f32(float alpha, const float* A, size_t m, size_t k,
              const float* B, size_t n, float beta, float* C) {
    // Naive triple loop implementation
    // C[i][j] = alpha * sum_k(A[i][k] * B[k][j]) + beta * C[i][j]

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

} // namespace simd

// Initialize dispatch function pointers
struct BaselineInit {
    BaselineInit() {
        distance_l2_baseline = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_l2(a, b, n);
        };
        distance_ip_baseline = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_ip(a, b, n);
        };
        gemm_f32_baseline = [](float alpha, const float* A, size_t m, size_t k,
                              const float* B, size_t n, float beta, float* C) -> void {
            simd::gemm_f32(alpha, A, m, k, B, n, beta, C);
        };
        dot_product_d_baseline = [](const double* a, const double* b, size_t n) -> double {
            return simd::dot_product_d(a, b, n);
        };
        dot_product_f_baseline = [](const float* a, const float* b, size_t n) -> float {
            return simd::dot_product(a, b, n);
        };
        scale_inplace_d_baseline = [](double* v, double s, size_t n) -> void {
            simd::scale_inplace(v, s, n);
        };
        subtract_scaled_d_baseline = [](double* a, const double* b, double s, size_t n) -> void {
            simd::subtract_scaled(a, b, s, n);
        };
        norm_d_baseline = [](const double* v, size_t n) -> double {
            return simd::norm(v, n);
        };
    }
} baseline_init;

} // namespace hypercube