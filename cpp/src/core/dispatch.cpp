#include "hypercube/dispatch.h"
#include "hypercube/cpu_features.h"
#include "hypercube/kernel_dispatch.h"
#include "hypercube/runtime_dispatch.h"

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
 * Initialize dispatch using runtime capability-based detection
 * Uses std::call_once for thread-safe initialization
 *
 * Uses the new runtime dispatch system with capability-based selection
 * while maintaining backward compatibility with the existing KernelVtable interface.
 */
static void init_dispatch() {
    // Initialize the runtime dispatch engine (this detects CPU features)
    auto& engine = RuntimeDispatchEngine::instance();

    // Determine active ISA based on primary kernel preferences
    // This maintains backward compatibility with code that checks get_active_isa()
    auto l2_preference = engine.get_kernel_preference(KernelType::DISTANCE_L2);
    if (l2_preference.primary == KernelCapability::AMX) {
        active_isa = IsaClass::AMX;
    } else if (l2_preference.primary == KernelCapability::AVX512_VNNI ||
               l2_preference.primary == KernelCapability::AVX512F ||
               l2_preference.primary == KernelCapability::AVX512_BW ||
               l2_preference.primary == KernelCapability::AVX512_BF16) {
        active_isa = IsaClass::AVX512_VNNI;
    } else if (l2_preference.primary == KernelCapability::AVX2_VNNI ||
               l2_preference.primary == KernelCapability::AVX2_FMA3 ||
               l2_preference.primary == KernelCapability::AVX2) {
        active_isa = IsaClass::AVX2_VNNI;
    } else if (l2_preference.primary == KernelCapability::SSE42) {
        active_isa = IsaClass::SSE42;
    } else {
        active_isa = IsaClass::BASELINE;
    }

    // Create vtable that delegates to the new runtime dispatch system
    active_kernels = std::make_unique<KernelVtable>();

    // Use wrapper lambdas that delegate to the new kernel_dispatch functions
    active_kernels->distance_l2 = [](const float* a, const float* b, size_t n) -> double {
        return kernels::distance::l2(a, b, n);
    };

    active_kernels->distance_ip = [](const float* a, const float* b, size_t n) -> double {
        return kernels::distance::ip(a, b, n);
    };

    active_kernels->gemm_f32 = [](float alpha, const float* A, size_t m, size_t k,
                                  const float* B, size_t n, float beta, float* C) {
        kernels::matrix::gemm_f32(alpha, A, m, k, B, n, beta, C);
    };

    active_kernels->dot_product_d = [](const double* a, const double* b, size_t n) -> double {
        return kernels::vector::dot_product(a, b, n);
    };

    active_kernels->dot_product_f = [](const float* a, const float* b, size_t n) -> float {
        return kernels::vector::dot_product(a, b, n);
    };

    active_kernels->scale_inplace_d = [](double* v, double s, size_t n) {
        kernels::vector::scale_inplace(v, s, n);
    };

    active_kernels->subtract_scaled_d = [](double* a, const double* b, double s, size_t n) {
        kernels::vector::subtract_scaled(a, b, s, n);
    };

    active_kernels->norm_d = [](const double* v, size_t n) -> double {
        return kernels::vector::norm(v, n);
    };
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

namespace dispatch {
    void initialize_function_pointers() {
        initialize_function_pointers();
    }
}

} // namespace hypercube
