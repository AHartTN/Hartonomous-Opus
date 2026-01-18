#pragma once
/**
 * SIMD Kernel Declarations
 *
 * Forward declarations for ISA-specific kernel implementations.
 * These are implemented in simd_kernels_*.cpp files.
 */

#include <cstddef>
#include <cstdint>

namespace hypercube {

// Baseline (scalar) implementations
namespace baseline {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

// SSE4.2 implementations
namespace sse42 {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

// AVX2 implementations
namespace avx2 {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

// AVX2+VNNI implementations
namespace avx2_vnni {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

// AVX-512 implementations
namespace avx512 {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

// AVX-512+VNNI implementations
namespace avx512_vnni {
    float dot_product(const float* a, const float* b, size_t n);
    double dot_product_d(const double* a, const double* b, size_t n);
    float cosine_similarity(const float* a, const float* b, size_t n);
    void scale_inplace(double* v, double s, size_t n);
    void subtract_scaled(double* a, const double* b, double s, size_t n);
    double norm(const double* v, size_t n);
    void normalize(double* v, size_t n);
    double distance_l2(const float* a, const float* b, size_t n);
    double distance_ip(const float* a, const float* b, size_t n);
    void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
}

} // namespace hypercube
