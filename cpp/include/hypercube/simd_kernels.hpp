#pragma once
/**
 * Unified SIMD Kernel Declarations with Runtime Dispatch
 *
 * Unified kernel classes that contain multiple ISA implementations
 * with runtime capability checks instead of compile-time guards.
 * All kernels are implemented in simd_kernels_unified.cpp.
 */

#include <cstddef>
#include <cstdint>
#include "cpu_features.hpp"

namespace hypercube {
namespace kernels {

/**
 * Unified Distance L2 (Euclidean Distance) Kernel
 */
struct DistanceL2 {
    static double compute(const float* a, const float* b, size_t n,
                         const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Distance IP (Inner Product) Kernel
 */
struct DistanceIP {
    static double compute(const float* a, const float* b, size_t n,
                         const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Dot Product Double Precision Kernel
 */
struct DotProductD {
    static double compute(const double* a, const double* b, size_t n,
                         const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Dot Product Single Precision Kernel
 */
struct DotProductF {
    static float compute(const float* a, const float* b, size_t n,
                        const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Cosine Similarity Kernel
 */
struct CosineSimilarity {
    static float compute(const float* a, const float* b, size_t n,
                        const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Scale Inplace Double Precision Kernel
 */
struct ScaleInplaceD {
    static void compute(double* v, double s, size_t n,
                       const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Subtract Scaled Double Precision Kernel
 */
struct SubtractScaledD {
    static void compute(double* a, const double* b, double s, size_t n,
                       const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified Norm Double Precision Kernel
 */
struct NormD {
    static double compute(const double* v, size_t n,
                         const cpu_features::RuntimeCpuFeatures& features);
};

/**
 * Unified GEMM F32 Kernel
 */
struct GEMMF32 {
    static void compute(float alpha, const float* A, size_t m, size_t k,
                       const float* B, size_t n, float beta, float* C,
                       const cpu_features::RuntimeCpuFeatures& features);
};

// =============================================================================
// Legacy namespace aliases for backward compatibility
// These will be removed in a future version after updating all callers
// =============================================================================

namespace baseline {
    inline float dot_product(const float* a, const float* b, size_t n) {
        return DotProductF::compute(a, b, n, cpu_features::detect_runtime_cpu_features());
    }
    inline double dot_product_d(const double* a, const double* b, size_t n) {
        return DotProductD::compute(a, b, n, cpu_features::detect_runtime_cpu_features());
    }
    inline float cosine_similarity(const float* a, const float* b, size_t n) {
        return CosineSimilarity::compute(a, b, n, cpu_features::detect_runtime_cpu_features());
    }
    inline void scale_inplace(double* v, double s, size_t n) {
        ScaleInplaceD::compute(v, s, n, cpu_features::detect_runtime_cpu_features());
    }
    inline void subtract_scaled(double* a, const double* b, double s, size_t n) {
        SubtractScaledD::compute(a, b, s, n, cpu_features::detect_runtime_cpu_features());
    }
    inline double norm(const double* v, size_t n) {
        return NormD::compute(v, n, cpu_features::detect_runtime_cpu_features());
    }
    inline double distance_l2(const float* a, const float* b, size_t n) {
        return DistanceL2::compute(a, b, n, cpu_features::detect_runtime_cpu_features());
    }
    inline double distance_ip(const float* a, const float* b, size_t n) {
        return DistanceIP::compute(a, b, n, cpu_features::detect_runtime_cpu_features());
    }
    inline void gemm_f32(float alpha, const float* A, size_t m, size_t k,
                        const float* B, size_t n, float beta, float* C) {
        GEMMF32::compute(alpha, A, m, k, B, n, beta, C, cpu_features::detect_runtime_cpu_features());
    }
}

} // namespace kernels
} // namespace hypercube
