#pragma once

#include <cstdint>
#include <mutex>
#include <memory>

#include "isa_class.h"

namespace hypercube {

// =============================================================================
// Kernel Function Pointer Types
// =============================================================================

/**
 * L2 (Euclidean) distance function
 * Computes ||a - b||_2
 */
using DistanceL2Fn = double (*)(const float* a, const float* b, size_t n);

/**
 * Inner product distance function
 * Computes <a, b> (dot product)
 */
using DistanceIPFn = double (*)(const float* a, const float* b, size_t n);

/**
 * General matrix multiply for float32
 * C = alpha * A * B + beta * C
 * A: m x k, B: k x n, C: m x n (all row-major)
 */
using GemmF32Fn = void (*)(float alpha, const float* A, size_t m, size_t k,
                          const float* B, size_t n, float beta, float* C);

/**
  * Double precision dot product
  * Computes <a, b>
  */
using DotProductDFn = double (*)(const double* a, const double* b, size_t n);

/**
  * Single precision dot product
  * Computes <a, b>
  */
using DotProductFFn = float (*)(const float* a, const float* b, size_t n);

/**
  * In-place scaling of double vector
  * v = s * v
  */
using ScaleInplaceDFn = void (*)(double* v, double s, size_t n);

/**
  * In-place subtract scaled double vector
  * a = a - s * b
  */
using SubtractScaledDFn = void (*)(double* a, const double* b, double s, size_t n);

/**
  * Double precision norm (L2)
  * Computes ||v||_2
  */
using NormDFn = double (*)(const double* v, size_t n);

// =============================================================================
// Kernel Vtable Structure
// =============================================================================

struct KernelVtable {
    DistanceL2Fn distance_l2;
    DistanceIPFn distance_ip;
    GemmF32Fn gemm_f32;
    DotProductDFn dot_product_d;
    DotProductFFn dot_product_f;
    ScaleInplaceDFn scale_inplace_d;
    SubtractScaledDFn subtract_scaled_d;
    NormDFn norm_d;
};

// =============================================================================
// ISA-Specific Kernel Declarations
// =============================================================================

// Baseline scalar implementations
extern DistanceL2Fn distance_l2_baseline;
extern DistanceIPFn distance_ip_baseline;
extern GemmF32Fn gemm_f32_baseline;

// SSE4.2 implementations
extern DistanceL2Fn distance_l2_sse42;
extern DistanceIPFn distance_ip_sse42;
extern GemmF32Fn gemm_f32_sse42;

// AVX2 implementations
extern DistanceL2Fn distance_l2_avx2;
extern DistanceIPFn distance_ip_avx2;
extern GemmF32Fn gemm_f32_avx2;

// AVX2 + VNNI implementations
extern DistanceL2Fn distance_l2_avx2_vnni;
extern DistanceIPFn distance_ip_avx2_vnni;
extern GemmF32Fn gemm_f32_avx2_vnni;

// AVX-512 implementations
extern DistanceL2Fn distance_l2_avx512;
extern DistanceIPFn distance_ip_avx512;
extern GemmF32Fn gemm_f32_avx512;

// AVX-512 + VNNI implementations
extern DistanceL2Fn distance_l2_avx512_vnni;
extern DistanceIPFn distance_ip_avx512_vnni;
extern GemmF32Fn gemm_f32_avx512_vnni;

// AMX implementations
extern DistanceL2Fn distance_l2_amx;
extern DistanceIPFn distance_ip_amx;
extern GemmF32Fn gemm_f32_amx;

// Baseline SIMD implementations
extern DotProductDFn dot_product_d_baseline;
extern DotProductFFn dot_product_f_baseline;
extern ScaleInplaceDFn scale_inplace_d_baseline;
extern SubtractScaledDFn subtract_scaled_d_baseline;
extern NormDFn norm_d_baseline;

// SSE4.2 SIMD implementations
extern DotProductDFn dot_product_d_sse42;
extern DotProductFFn dot_product_f_sse42;
extern ScaleInplaceDFn scale_inplace_d_sse42;
extern SubtractScaledDFn subtract_scaled_d_sse42;
extern NormDFn norm_d_sse42;

// AVX2 SIMD implementations
extern DotProductDFn dot_product_d_avx2;
extern DotProductFFn dot_product_f_avx2;
extern ScaleInplaceDFn scale_inplace_d_avx2;
extern SubtractScaledDFn subtract_scaled_d_avx2;
extern NormDFn norm_d_avx2;

// AVX2 + VNNI SIMD implementations
extern DotProductDFn dot_product_d_avx2_vnni;
extern DotProductFFn dot_product_f_avx2_vnni;
extern ScaleInplaceDFn scale_inplace_d_avx2_vnni;
extern SubtractScaledDFn subtract_scaled_d_avx2_vnni;
extern NormDFn norm_d_avx2_vnni;

// AVX-512 SIMD implementations
extern DotProductDFn dot_product_d_avx512;
extern DotProductFFn dot_product_f_avx512;
extern ScaleInplaceDFn scale_inplace_d_avx512;
extern SubtractScaledDFn subtract_scaled_d_avx512;
extern NormDFn norm_d_avx512;

// AVX-512 + VNNI SIMD implementations
extern DotProductDFn dot_product_d_avx512_vnni;
extern DotProductFFn dot_product_f_avx512_vnni;
extern ScaleInplaceDFn scale_inplace_d_avx512_vnni;
extern SubtractScaledDFn subtract_scaled_d_avx512_vnni;
extern NormDFn norm_d_avx512_vnni;

// AMX SIMD implementations
extern DotProductDFn dot_product_d_amx;
extern DotProductFFn dot_product_f_amx;
extern ScaleInplaceDFn scale_inplace_d_amx;
extern SubtractScaledDFn subtract_scaled_d_amx;
extern NormDFn norm_d_amx;

// =============================================================================
// Dispatch Functions
// =============================================================================

/**
 * Get the active ISA class detected at runtime
 */
IsaClass get_active_isa();

/**
 * Get the kernel vtable for the active ISA
 * Thread-safe with lazy initialization
 */
const KernelVtable& get_kernels();

} // namespace hypercube