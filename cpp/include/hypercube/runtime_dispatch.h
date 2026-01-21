/**
 * Runtime Kernel Dispatch System
 *
 * Provides capability-based kernel selection and safe fallback mechanisms
 * for runtime CPU feature detection and dispatch.
 */

#pragma once

#include <functional>
#include <memory>
#include "cpu_features.hpp"

namespace hypercube {

/**
 * Kernel capability levels for runtime dispatch
 */
enum class KernelCapability {
    BASELINE,      // Scalar fallback
    SSE42,         // SSE4.2 optimized
    AVX2,          // AVX2 optimized
    AVX2_FMA3,     // AVX2 with FMA3
    AVX2_VNNI,     // AVX2 with VNNI
    AVX512F,       // AVX-512 Foundation
    AVX512_BW,     // AVX-512 with Byte/Word
    AVX512_VNNI,   // AVX-512 with VNNI
    AVX512_BF16,   // AVX-512 with BFloat16
    AMX,           // AMX tile operations
};

/**
 * Kernel preference structure for dispatch decisions
 */
struct KernelPreference {
    KernelCapability primary;
    KernelCapability fallback;
    KernelCapability emergency; // Always BASELINE

    KernelPreference(KernelCapability p, KernelCapability f, KernelCapability e = KernelCapability::BASELINE)
        : primary(p), fallback(f), emergency(e) {}
};

/**
 * Kernel type enumeration
 */
enum class KernelType {
    DISTANCE_L2,
    DISTANCE_IP,
    GEMM_F32,
    DOT_PRODUCT_D,
    DOT_PRODUCT_F,
    SCALE_INPLACE_D,
    SUBTRACT_SCALED_D,
    NORM_D,
};

/**
 * Kernel level enumeration for fallback hierarchy
 */
enum class KernelLevel {
    PRIMARY,
    FALLBACK,
    EMERGENCY
};

/**
 * Runtime dispatch engine for kernel selection
 */
class RuntimeDispatchEngine {
public:
    static RuntimeDispatchEngine& instance();

    /**
     * Initialize with detected CPU features
     */
    void initialize(const cpu_features::RuntimeCpuFeatures& features);

    /**
     * Get kernel preference for a specific operation
     */
    KernelPreference get_kernel_preference(KernelType type) const;

    /**
     * Check if a capability is safely available
     */
    bool is_capability_safe(KernelCapability capability) const;

    /**
     * Get human-readable capability name
     */
    static std::string capability_name(KernelCapability capability);

public:
    RuntimeDispatchEngine() = default;
    cpu_features::RuntimeCpuFeatures cpu_features_;
    bool initialized_ = false;

    KernelPreference select_distance_l2() const;
    KernelPreference select_distance_ip() const;
    KernelPreference select_gemm_f32() const;
    KernelPreference select_dot_product_d() const;
    KernelPreference select_dot_product_f() const;
    KernelPreference select_scale_inplace_d() const;
    KernelPreference select_subtract_scaled_d() const;
    KernelPreference select_norm_d() const;
};

/**
 * Fallback dispatcher with automatic safety checks
 */
class FallbackDispatcher {
public:
    explicit FallbackDispatcher(RuntimeDispatchEngine& engine) : engine_(engine) {}

    /**
     * Dispatch with automatic fallback on feature exceptions
     * Takes a kernel selector function that returns the appropriate kernel for a capability
     */
    template<typename KernelSelector, typename... Args>
    auto dispatch_with_fallback(KernelType type, KernelSelector selector, Args&&... args) {
        const auto preference = engine_.get_kernel_preference(type);

        // Try primary implementation
        try {
            if (engine_.is_capability_safe(preference.primary)) {
                return selector(preference.primary, std::forward<Args>(args)...);
            }
        } catch (const cpu_features::CpuFeatureException&) {
            // Primary failed, try fallback
        }

        // Try fallback implementation
        try {
            if (engine_.is_capability_safe(preference.fallback)) {
                return selector(preference.fallback, std::forward<Args>(args)...);
            }
        } catch (const cpu_features::CpuFeatureException&) {
            // Fallback failed, use emergency
        }

        // Emergency fallback - should always work
        return selector(preference.emergency, std::forward<Args>(args)...);
    }

private:
    RuntimeDispatchEngine& engine_;
};

/**
 * Convenience function to get the global dispatcher
 */
FallbackDispatcher& get_fallback_dispatcher();

} // namespace hypercube