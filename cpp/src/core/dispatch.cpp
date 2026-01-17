#include "hypercube/dispatch.h"
#include "hypercube/cpu_features.h"

#include <mutex>
#include <memory>

namespace hypercube {

// =============================================================================
// ISA-Specific Kernel Declarations (extern - implementations defined elsewhere)
// =============================================================================

// Baseline scalar implementations
DistanceL2Fn distance_l2_baseline = nullptr;  // Placeholder - implementation will be added later
DistanceIPFn distance_ip_baseline = nullptr;  // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_baseline = nullptr;        // Placeholder - implementation will be added later

// SSE4.2 implementations
DistanceL2Fn distance_l2_sse42 = nullptr;     // Placeholder - implementation will be added later
DistanceIPFn distance_ip_sse42 = nullptr;     // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_sse42 = nullptr;           // Placeholder - implementation will be added later

// AVX2 implementations
DistanceL2Fn distance_l2_avx2 = nullptr;      // Placeholder - implementation will be added later
DistanceIPFn distance_ip_avx2 = nullptr;      // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_avx2 = nullptr;            // Placeholder - implementation will be added later

// AVX2 + VNNI implementations
DistanceL2Fn distance_l2_avx2_vnni = nullptr; // Placeholder - implementation will be added later
DistanceIPFn distance_ip_avx2_vnni = nullptr; // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_avx2_vnni = nullptr;       // Placeholder - implementation will be added later

// AVX-512 implementations
DistanceL2Fn distance_l2_avx512 = nullptr;    // Placeholder - implementation will be added later
DistanceIPFn distance_ip_avx512 = nullptr;    // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_avx512 = nullptr;          // Placeholder - implementation will be added later

// AVX-512 + VNNI implementations
DistanceL2Fn distance_l2_avx512_vnni = nullptr; // Placeholder - implementation will be added later
DistanceIPFn distance_ip_avx512_vnni = nullptr; // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_avx512_vnni = nullptr;     // Placeholder - implementation will be added later

// AMX implementations
DistanceL2Fn distance_l2_amx = nullptr;       // Placeholder - implementation will be added later
DistanceIPFn distance_ip_amx = nullptr;       // Placeholder - implementation will be added later
GemmF32Fn gemm_f32_amx = nullptr;             // Placeholder - implementation will be added later

// Baseline SIMD implementations
DotProductDFn dot_product_d_baseline = nullptr;      // Placeholder - implementation will be added later
DotProductFFn dot_product_f_baseline = nullptr;      // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_baseline = nullptr;  // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_baseline = nullptr; // Placeholder - implementation will be added later
NormDFn norm_d_baseline = nullptr;                   // Placeholder - implementation will be added later

// SSE4.2 SIMD implementations
DotProductDFn dot_product_d_sse42 = nullptr;         // Placeholder - implementation will be added later
DotProductFFn dot_product_f_sse42 = nullptr;         // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_sse42 = nullptr;     // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_sse42 = nullptr; // Placeholder - implementation will be added later
NormDFn norm_d_sse42 = nullptr;                      // Placeholder - implementation will be added later

// AVX2 SIMD implementations
DotProductDFn dot_product_d_avx2 = nullptr;          // Placeholder - implementation will be added later
DotProductFFn dot_product_f_avx2 = nullptr;          // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_avx2 = nullptr;      // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_avx2 = nullptr;  // Placeholder - implementation will be added later
NormDFn norm_d_avx2 = nullptr;                       // Placeholder - implementation will be added later

// AVX2 + VNNI SIMD implementations
DotProductDFn dot_product_d_avx2_vnni = nullptr;     // Placeholder - implementation will be added later
DotProductFFn dot_product_f_avx2_vnni = nullptr;     // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_avx2_vnni = nullptr; // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_avx2_vnni = nullptr; // Placeholder - implementation will be added later
NormDFn norm_d_avx2_vnni = nullptr;                  // Placeholder - implementation will be added later

// AVX-512 SIMD implementations
DotProductDFn dot_product_d_avx512 = nullptr;        // Placeholder - implementation will be added later
DotProductFFn dot_product_f_avx512 = nullptr;        // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_avx512 = nullptr;    // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_avx512 = nullptr; // Placeholder - implementation will be added later
NormDFn norm_d_avx512 = nullptr;                     // Placeholder - implementation will be added later

// AVX-512 + VNNI SIMD implementations
DotProductDFn dot_product_d_avx512_vnni = nullptr;   // Placeholder - implementation will be added later
DotProductFFn dot_product_f_avx512_vnni = nullptr;   // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_avx512_vnni = nullptr; // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_avx512_vnni = nullptr; // Placeholder - implementation will be added later
NormDFn norm_d_avx512_vnni = nullptr;               // Placeholder - implementation will be added later

// AMX SIMD implementations
DotProductDFn dot_product_d_amx = nullptr;          // Placeholder - implementation will be added later
DotProductFFn dot_product_f_amx = nullptr;          // Placeholder - implementation will be added later
ScaleInplaceDFn scale_inplace_d_amx = nullptr;      // Placeholder - implementation will be added later
SubtractScaledDFn subtract_scaled_d_amx = nullptr;  // Placeholder - implementation will be added later
NormDFn norm_d_amx = nullptr;                       // Placeholder - implementation will be added later

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
 */
static void init_dispatch() {
    auto features = cpu_features::detect_cpu_features();
    active_isa = classify_isa(features);

    // Create vtable based on active ISA
    active_kernels = std::make_unique<KernelVtable>();

    switch (active_isa) {
        case IsaClass::AMX:
            active_kernels->distance_l2 = distance_l2_amx;
            active_kernels->distance_ip = distance_ip_amx;
            active_kernels->gemm_f32 = gemm_f32_amx;
            active_kernels->dot_product_d = dot_product_d_amx;
            active_kernels->dot_product_f = dot_product_f_amx;
            active_kernels->scale_inplace_d = scale_inplace_d_amx;
            active_kernels->subtract_scaled_d = subtract_scaled_d_amx;
            active_kernels->norm_d = norm_d_amx;
            break;

        case IsaClass::AVX512_VNNI:
            active_kernels->distance_l2 = distance_l2_avx512_vnni;
            active_kernels->distance_ip = distance_ip_avx512_vnni;
            active_kernels->gemm_f32 = gemm_f32_avx512_vnni;
            active_kernels->dot_product_d = dot_product_d_avx512_vnni;
            active_kernels->dot_product_f = dot_product_f_avx512_vnni;
            active_kernels->scale_inplace_d = scale_inplace_d_avx512_vnni;
            active_kernels->subtract_scaled_d = subtract_scaled_d_avx512_vnni;
            active_kernels->norm_d = norm_d_avx512_vnni;
            break;

        case IsaClass::AVX512:
            active_kernels->distance_l2 = distance_l2_avx512;
            active_kernels->distance_ip = distance_ip_avx512;
            active_kernels->gemm_f32 = gemm_f32_avx512;
            active_kernels->dot_product_d = dot_product_d_avx512;
            active_kernels->dot_product_f = dot_product_f_avx512;
            active_kernels->scale_inplace_d = scale_inplace_d_avx512;
            active_kernels->subtract_scaled_d = subtract_scaled_d_avx512;
            active_kernels->norm_d = norm_d_avx512;
            break;

        case IsaClass::AVX2_VNNI:
            active_kernels->distance_l2 = distance_l2_avx2_vnni;
            active_kernels->distance_ip = distance_ip_avx2_vnni;
            active_kernels->gemm_f32 = gemm_f32_avx2_vnni;
            active_kernels->dot_product_d = dot_product_d_avx2_vnni;
            active_kernels->dot_product_f = dot_product_f_avx2_vnni;
            active_kernels->scale_inplace_d = scale_inplace_d_avx2_vnni;
            active_kernels->subtract_scaled_d = subtract_scaled_d_avx2_vnni;
            active_kernels->norm_d = norm_d_avx2_vnni;
            break;

        case IsaClass::AVX2:
            active_kernels->distance_l2 = distance_l2_avx2;
            active_kernels->distance_ip = distance_ip_avx2;
            active_kernels->gemm_f32 = gemm_f32_avx2;
            active_kernels->dot_product_d = dot_product_d_avx2;
            active_kernels->dot_product_f = dot_product_f_avx2;
            active_kernels->scale_inplace_d = scale_inplace_d_avx2;
            active_kernels->subtract_scaled_d = subtract_scaled_d_avx2;
            active_kernels->norm_d = norm_d_avx2;
            break;

        case IsaClass::SSE42:
            active_kernels->distance_l2 = distance_l2_sse42;
            active_kernels->distance_ip = distance_ip_sse42;
            active_kernels->gemm_f32 = gemm_f32_sse42;
            active_kernels->dot_product_d = dot_product_d_sse42;
            active_kernels->dot_product_f = dot_product_f_sse42;
            active_kernels->scale_inplace_d = scale_inplace_d_sse42;
            active_kernels->subtract_scaled_d = subtract_scaled_d_sse42;
            active_kernels->norm_d = norm_d_sse42;
            break;

        case IsaClass::BASELINE:
        default:
            active_kernels->distance_l2 = distance_l2_baseline;
            active_kernels->distance_ip = distance_ip_baseline;
            active_kernels->gemm_f32 = gemm_f32_baseline;
            active_kernels->dot_product_d = dot_product_d_baseline;
            active_kernels->dot_product_f = dot_product_f_baseline;
            active_kernels->scale_inplace_d = scale_inplace_d_baseline;
            active_kernels->subtract_scaled_d = subtract_scaled_d_baseline;
            active_kernels->norm_d = norm_d_baseline;
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
