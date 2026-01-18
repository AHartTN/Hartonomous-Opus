#include "hypercube/dispatch.h"
#include "hypercube/cpu_features.h"
#include "hypercube/simd_kernels.hpp"

#include <mutex>
#include <memory>

namespace hypercube {

// =============================================================================
// Static Variables for Lazy Initialization
// =============================================================================

static std::once_flag dispatch_init_flag;
static std::unique_ptr<KernelVtable> active_kernels;
static IsaClass active_isa = IsaClass::BASELINE;

// =============================================================================
// Private Initialization Function
// =============================================================================

/**
 * Initialize dispatch based on detected ISA capabilities
 * Uses std::call_once for thread-safe initialization
 *
 * Note: We directly reference the namespace functions instead of using
 * global function pointers to avoid static initialization order issues.
 */
static void init_dispatch() {
    auto features = cpu_features::detect_cpu_features();
    active_isa = classify_isa(features);

    // Create vtable based on active ISA
    active_kernels = std::make_unique<KernelVtable>();

    switch (active_isa) {
        case IsaClass::AMX:
            // AMX not yet implemented - fall through to AVX512_VNNI
            [[fallthrough]];

        case IsaClass::AVX512_VNNI:
            active_kernels->distance_l2 = avx512_vnni::distance_l2;
            active_kernels->distance_ip = avx512_vnni::distance_ip;
            active_kernels->gemm_f32 = avx512_vnni::gemm_f32;
            active_kernels->dot_product_d = avx512_vnni::dot_product_d;
            active_kernels->dot_product_f = avx512_vnni::dot_product;
            active_kernels->scale_inplace_d = avx512_vnni::scale_inplace;
            active_kernels->subtract_scaled_d = avx512_vnni::subtract_scaled;
            active_kernels->norm_d = avx512_vnni::norm;
            break;

        case IsaClass::AVX512:
            active_kernels->distance_l2 = avx512::distance_l2;
            active_kernels->distance_ip = avx512::distance_ip;
            active_kernels->gemm_f32 = avx512::gemm_f32;
            active_kernels->dot_product_d = avx512::dot_product_d;
            active_kernels->dot_product_f = avx512::dot_product;
            active_kernels->scale_inplace_d = avx512::scale_inplace;
            active_kernels->subtract_scaled_d = avx512::subtract_scaled;
            active_kernels->norm_d = avx512::norm;
            break;

        case IsaClass::AVX2_VNNI:
            active_kernels->distance_l2 = avx2_vnni::distance_l2;
            active_kernels->distance_ip = avx2_vnni::distance_ip;
            active_kernels->gemm_f32 = avx2_vnni::gemm_f32;
            active_kernels->dot_product_d = avx2_vnni::dot_product_d;
            active_kernels->dot_product_f = avx2_vnni::dot_product;
            active_kernels->scale_inplace_d = avx2_vnni::scale_inplace;
            active_kernels->subtract_scaled_d = avx2_vnni::subtract_scaled;
            active_kernels->norm_d = avx2_vnni::norm;
            break;

        case IsaClass::AVX2:
            active_kernels->distance_l2 = avx2::distance_l2;
            active_kernels->distance_ip = avx2::distance_ip;
            active_kernels->gemm_f32 = avx2::gemm_f32;
            active_kernels->dot_product_d = avx2::dot_product_d;
            active_kernels->dot_product_f = avx2::dot_product;
            active_kernels->scale_inplace_d = avx2::scale_inplace;
            active_kernels->subtract_scaled_d = avx2::subtract_scaled;
            active_kernels->norm_d = avx2::norm;
            break;

        case IsaClass::SSE42:
            active_kernels->distance_l2 = sse42::distance_l2;
            active_kernels->distance_ip = sse42::distance_ip;
            active_kernels->gemm_f32 = sse42::gemm_f32;
            active_kernels->dot_product_d = sse42::dot_product_d;
            active_kernels->dot_product_f = sse42::dot_product;
            active_kernels->scale_inplace_d = sse42::scale_inplace;
            active_kernels->subtract_scaled_d = sse42::subtract_scaled;
            active_kernels->norm_d = sse42::norm;
            break;

        case IsaClass::BASELINE:
        default:
            active_kernels->distance_l2 = baseline::distance_l2;
            active_kernels->distance_ip = baseline::distance_ip;
            active_kernels->gemm_f32 = baseline::gemm_f32;
            active_kernels->dot_product_d = baseline::dot_product_d;
            active_kernels->dot_product_f = baseline::dot_product;
            active_kernels->scale_inplace_d = baseline::scale_inplace;
            active_kernels->subtract_scaled_d = baseline::subtract_scaled;
            active_kernels->norm_d = baseline::norm;
            break;
    }
}

// =============================================================================
// Public API Functions
// =============================================================================

IsaClass get_active_isa() {
    std::call_once(dispatch_init_flag, init_dispatch);
    return active_isa;
}

const KernelVtable& get_kernels() {
    std::call_once(dispatch_init_flag, init_dispatch);
    return *active_kernels;
}

} // namespace hypercube
