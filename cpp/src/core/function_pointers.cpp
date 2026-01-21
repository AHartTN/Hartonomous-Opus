#include "hypercube/function_pointers.hpp"
#include "hypercube/cpu_features.hpp"
#include "hypercube/simd_intrinsics.hpp"
#include <mutex>

// Include the implementation files
#include "../simd_kernels_distance.cpp"
#include "../simd_kernels_vector.cpp"

namespace hypercube {

// Global function pointer declarations
double (*distance_l2_f32)(const float* a, const float* b, size_t n);
double (*distance_ip_f32)(const float* a, const float* b, size_t n);
void (*gemm_f32)(float alpha, const float* A, size_t m, size_t k,
                 const float* B, size_t n, float beta, float* C);
double (*dot_product_d)(const double* a, const double* b, size_t n);
float (*dot_product_f)(const float* a, const float* b, size_t n);
void (*scale_inplace_d)(double* v, double s, size_t n);
void (*subtract_scaled_d)(double* a, const double* b, double s, size_t n);
double (*norm_d)(const double* v, size_t n);

// Thread-safe initialization flag
static std::once_flag init_flag;

// Selector functions for each operation type
static auto select_distance_l2_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::distance_l2_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::distance_l2_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::distance_l2_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::distance_l2_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::distance_l2_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::distance_l2_sse42;
    } else {
        return kernels::impl::distance_l2_baseline;
    }
}

static auto select_distance_ip_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::distance_ip_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::distance_ip_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::distance_ip_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::distance_ip_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::distance_ip_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::distance_ip_sse42;
    } else {
        return kernels::impl::distance_ip_baseline;
    }
}

static auto select_gemm_f32_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::gemm_f32_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::gemm_f32_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::gemm_f32_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::gemm_f32_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::gemm_f32_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::gemm_f32_sse42;
    } else {
        return kernels::impl::gemm_f32_baseline;
    }
}

static auto select_dot_product_d_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::dot_product_d_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::dot_product_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::dot_product_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::dot_product_d_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::dot_product_d_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::dot_product_d_sse42;
    } else {
        return kernels::impl::dot_product_d_baseline;
    }
}

static auto select_dot_product_f_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::dot_product_f_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::dot_product_f_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::dot_product_f_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::dot_product_f_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::dot_product_f_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::dot_product_f_sse42;
    } else {
        return kernels::impl::dot_product_f_baseline;
    }
}

static auto select_scale_inplace_d_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::scale_inplace_d_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::scale_inplace_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::scale_inplace_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::scale_inplace_d_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::scale_inplace_d_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::scale_inplace_d_sse42;
    } else {
        return kernels::impl::scale_inplace_d_baseline;
    }
}

static auto select_subtract_scaled_d_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::subtract_scaled_d_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::subtract_scaled_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::subtract_scaled_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::subtract_scaled_d_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::subtract_scaled_d_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::subtract_scaled_d_sse42;
    } else {
        return kernels::impl::subtract_scaled_d_baseline;
    }
}

static auto select_norm_d_impl(const cpu_features::RuntimeCpuFeatures& features) {
    // Hierarchy: AMX > AVX512_VNNI > AVX512 > AVX2_VNNI > AVX2 > SSE42 > BASELINE
    if (features.safe_amx_execution && features.amx_tile) {
        return kernels::impl::norm_d_amx;
    } else if (features.safe_avx512_execution && features.avx512f && features.avx512_vnni) {
        return kernels::impl::norm_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx512_execution && features.avx512f) {
        return kernels::impl::norm_d_avx2;  // Use AVX2 since AVX512 may not exist
    } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
        return kernels::impl::norm_d_avx2;  // AVX2_VNNI fallback
    } else if (features.safe_avx_execution && features.avx2) {
        return kernels::impl::norm_d_avx2;
    } else if (features.sse4_2) {
        return kernels::impl::norm_d_sse42;
    } else {
        return kernels::impl::norm_d_baseline;
    }
}

// Initialization function
void initialize_function_pointers() {
    std::call_once(init_flag, []() {
        auto features = cpu_features::detect_runtime_cpu_features();

        // Initialize distance function pointers
        distance_l2_f32 = select_distance_l2_impl(features);
        distance_ip_f32 = select_distance_ip_impl(features);

        // Initialize matrix operation pointers
        gemm_f32 = select_gemm_f32_impl(features);

        // Initialize vector operation pointers
        dot_product_d = select_dot_product_d_impl(features);
        dot_product_f = select_dot_product_f_impl(features);
        scale_inplace_d = select_scale_inplace_d_impl(features);
        subtract_scaled_d = select_subtract_scaled_d_impl(features);
        norm_d = select_norm_d_impl(features);
    });
}

} // namespace hypercube