#include "hypercube/kernel_dispatch.h"
#include "hypercube/cpu_features.hpp"
#include <immintrin.h>  // Include all SIMD intrinsics for runtime dispatch
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace hypercube {
namespace kernels {

// Helper function to check capability safety and throw if unsafe
static void check_capability(KernelCapability capability) {
    auto& engine = RuntimeDispatchEngine::instance();
    if (!engine.is_capability_safe(capability)) {
        throw cpu_features::CpuFeatureException(
            RuntimeDispatchEngine::capability_name(capability) + " not safely available"
        );
    }
}

// Distance functions
namespace distance {

double l2(const float* a, const float* b, size_t n) {
    return get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::DISTANCE_L2,
        [](KernelCapability cap, const float* a, const float* b, size_t n) -> double {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::distance_l2_amx(a, b, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512_VNNI:
                    check_capability(KernelCapability::AVX512_VNNI);
                    return impl::distance_l2_avx512(a, b, n); // VNNI variant
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::distance_l2_avx512(a, b, n);
#endif
                case KernelCapability::AVX2_VNNI:
                    check_capability(KernelCapability::AVX2_VNNI);
                    return impl::distance_l2_avx2(a, b, n); // VNNI variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::distance_l2_avx2(a, b, n);
                case KernelCapability::SSE42:
                    return impl::distance_l2_sse42(a, b, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::distance_l2_baseline(a, b, n);
            }
        },
        a, b, n
    );
}

double ip(const float* a, const float* b, size_t n) {
    return get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::DISTANCE_IP,
        [](KernelCapability cap, const float* a, const float* b, size_t n) -> double {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::distance_ip_amx(a, b, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512_VNNI:
                    check_capability(KernelCapability::AVX512_VNNI);
                    return impl::distance_ip_avx512(a, b, n); // VNNI variant
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::distance_ip_avx512(a, b, n);
#endif
                case KernelCapability::AVX2_VNNI:
                    check_capability(KernelCapability::AVX2_VNNI);
                    return impl::distance_ip_avx2(a, b, n); // VNNI variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::distance_ip_avx2(a, b, n);
                case KernelCapability::SSE42:
                    return impl::distance_ip_sse42(a, b, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::distance_ip_baseline(a, b, n);
            }
        },
        a, b, n
    );
}

} // namespace distance

// Matrix operations
namespace matrix {

void gemm_f32(float alpha, const float* A, size_t m, size_t k,
              const float* B, size_t n, float beta, float* C) {
    get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::GEMM_F32,
        [](KernelCapability cap, float alpha, const float* A, size_t m, size_t k,
           const float* B, size_t n, float beta, float* C) {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::gemm_f32_amx(alpha, A, m, k, B, n, beta, C);
#ifdef __AVX512F__
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::gemm_f32_avx512(alpha, A, m, k, B, n, beta, C);
#endif
                case KernelCapability::AVX2_FMA3:
                    check_capability(KernelCapability::AVX2_FMA3);
                    return impl::gemm_f32_avx2(alpha, A, m, k, B, n, beta, C); // FMA3 variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::gemm_f32_avx2(alpha, A, m, k, B, n, beta, C);
                case KernelCapability::SSE42:
                    return impl::gemm_f32_sse42(alpha, A, m, k, B, n, beta, C);
                case KernelCapability::BASELINE:
                default:
                    return impl::gemm_f32_baseline(alpha, A, m, k, B, n, beta, C);
            }
        },
        alpha, A, m, k, B, n, beta, C
    );
}

} // namespace matrix

// Vector operations
namespace vector {

double dot_product(const double* a, const double* b, size_t n) {
    return get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::DOT_PRODUCT_D,
        [](KernelCapability cap, const double* a, const double* b, size_t n) -> double {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::dot_product_d_amx(a, b, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::dot_product_d_avx512(a, b, n);
#endif
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::dot_product_d_avx2(a, b, n);
                case KernelCapability::SSE42:
                    return impl::dot_product_d_sse42(a, b, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::dot_product_d_baseline(a, b, n);
            }
        },
        a, b, n
    );
}

float dot_product(const float* a, const float* b, size_t n) {
    return get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::DOT_PRODUCT_F,
        [](KernelCapability cap, const float* a, const float* b, size_t n) -> float {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::dot_product_f_amx(a, b, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512_VNNI:
                    check_capability(KernelCapability::AVX512_VNNI);
                    return impl::dot_product_f_avx512(a, b, n); // VNNI variant
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::dot_product_f_avx512(a, b, n);
#endif
                case KernelCapability::AVX2_VNNI:
                    check_capability(KernelCapability::AVX2_VNNI);
                    return impl::dot_product_f_avx2(a, b, n); // VNNI variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::dot_product_f_avx2(a, b, n);
                case KernelCapability::SSE42:
                    return impl::dot_product_f_sse42(a, b, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::dot_product_f_baseline(a, b, n);
            }
        },
        a, b, n
    );
}

void scale_inplace(double* v, double s, size_t n) {
    get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::SCALE_INPLACE_D,
        [](KernelCapability cap, double* v, double s, size_t n) {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::scale_inplace_d_amx(v, s, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::scale_inplace_d_avx512(v, s, n);
#endif
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::scale_inplace_d_avx2(v, s, n);
                case KernelCapability::SSE42:
                    return impl::scale_inplace_d_sse42(v, s, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::scale_inplace_d_baseline(v, s, n);
            }
        },
        v, s, n
    );
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
    get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::SUBTRACT_SCALED_D,
        [](KernelCapability cap, double* a, const double* b, double s, size_t n) {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::subtract_scaled_d_amx(a, b, s, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::subtract_scaled_d_avx512(a, b, s, n);
#endif
                case KernelCapability::AVX2_FMA3:
                    check_capability(KernelCapability::AVX2_FMA3);
                    return impl::subtract_scaled_d_avx2(a, b, s, n); // FMA3 variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::subtract_scaled_d_avx2(a, b, s, n);
                case KernelCapability::SSE42:
                    return impl::subtract_scaled_d_sse42(a, b, s, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::subtract_scaled_d_baseline(a, b, s, n);
            }
        },
        a, b, s, n
    );
}

double norm(const double* v, size_t n) {
    return get_fallback_dispatcher().dispatch_with_fallback(
        KernelType::NORM_D,
        [](KernelCapability cap, const double* v, size_t n) -> double {
            switch (cap) {
                case KernelCapability::AMX:
                    check_capability(KernelCapability::AMX);
                    return impl::norm_d_amx(v, n);
#ifdef __AVX512F__
                case KernelCapability::AVX512_VNNI:
                    check_capability(KernelCapability::AVX512_VNNI);
                    return impl::norm_d_avx512(v, n); // VNNI variant
                case KernelCapability::AVX512F:
                    check_capability(KernelCapability::AVX512F);
                    return impl::norm_d_avx512(v, n);
#endif
                case KernelCapability::AVX2_VNNI:
                    check_capability(KernelCapability::AVX2_VNNI);
                    return impl::norm_d_avx2(v, n); // VNNI variant
                case KernelCapability::AVX2:
                    check_capability(KernelCapability::AVX2);
                    return impl::norm_d_avx2(v, n);
                case KernelCapability::SSE42:
                    return impl::norm_d_sse42(v, n);
                case KernelCapability::BASELINE:
                default:
                    return impl::norm_d_baseline(v, n);
            }
        },
        v, n
    );
}

} // namespace vector

// Kernel implementations are now in simd_kernels_unified.cpp

} // namespace kernels
} // namespace hypercube