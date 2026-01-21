/**
 * CPU Feature Detection - Runtime SIMD/AVX capability checks
 *
 * Provides comprehensive runtime detection of CPU features for dynamic dispatch.
 * Supports safe execution validation and fallback mechanisms.
 */

#pragma once

#include <cstdint>
#include <string>
#include <cstring>  // For memcpy in vendor string
#include <stdexcept> // For std::runtime_error

namespace hypercube {
namespace cpu_features {

/**
 * Legacy CPU feature flags (for backward compatibility)
 */
enum class Feature {
    AVX2 = 1 << 0,      // AVX2 instructions
    AVX512F = 1 << 1,   // AVX-512 Foundation
    AVX512DQ = 1 << 2,  // AVX-512 Doubleword and Quadword
    AVX512BW = 1 << 3,  // AVX-512 Byte and Word
    AVX512VL = 1 << 4,  // AVX-512 Vector Length
    FMA3 = 1 << 5,      // Fused Multiply-Add 3
    BMI2 = 1 << 6,      // Bit Manipulation Instructions 2
    AVX_VNNI = 1 << 7,  // AVX-512 Vector Neural Network Instructions
};

/**
 * Comprehensive runtime CPU features structure
 */
struct RuntimeCpuFeatures {
    std::string vendor;
    uint32_t family;
    uint32_t model;

    // SSE family
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;

    // AVX family
    bool avx;
    bool avx2;
    bool fma3;

    // AVX-512 extensions
    bool avx512f;      // Foundation
    bool avx512dq;     // Doubleword/Quadword
    bool avx512bw;     // Byte/Word
    bool avx512vl;     // Vector Length
    bool avx512vnni;   // Vector Neural Network Instructions
    bool avx512bf16;   // BFloat16
    bool avx512fp16;   // Float16 (if available)

    // VNNI variants
    bool avx_vnni;     // 256-bit VNNI
    bool avx512_vnni;  // 512-bit VNNI

    // Advanced features
    bool amx_tile;
    bool amx_int8;
    bool amx_bf16;

    // OS support indicators
    bool os_avx_support;     // XSAVE/XRSTOR for AVX state
    bool os_avx512_support;  // XSAVE/XRSTOR for AVX-512 state
    bool os_amx_support;     // XSAVE/XRSTOR for AMX state

    // Safety flags
    bool safe_avx_execution;
    bool safe_avx512_execution;
    bool safe_amx_execution;

    // Utility functions
    bool has_avx() const { return avx && os_avx_support; }
    bool has_avx2() const { return avx2 && safe_avx_execution; }
    bool has_avx512f() const { return avx512f && safe_avx512_execution; }
    bool has_amx() const { return amx_tile && safe_amx_execution; }
};

/**
 * Exception thrown when CPU features are not available for safe execution
 */
class CpuFeatureException : public std::runtime_error {
public:
    explicit CpuFeatureException(const std::string& feature)
        : std::runtime_error("CPU feature not available: " + feature) {}
};

/**
 * Detect comprehensive CPU features at runtime
 */
RuntimeCpuFeatures detect_runtime_cpu_features();

/**
 * Check if OS supports AVX state saving (XCR0 bits for SSE/AVX)
 */
bool check_os_avx_support();

/**
 * Check if OS supports AVX-512 state saving (XCR0 bits for AVX-512)
 */
bool check_os_avx512_support();

/**
 * Check if OS supports AMX state saving (XCR0 bits for AMX)
 */
bool check_os_amx_support();

/**
 * CPUID result structure
 */
struct CpuidResult {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
};

/**
 * Execute CPUID instruction
 */
inline CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    CpuidResult result{};

#ifdef _MSC_VER
    // MSVC intrinsic
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    result.eax = regs[0];
    result.ebx = regs[1];
    result.ecx = regs[2];
    result.edx = regs[3];
#else
    // GCC/Clang intrinsic
    __asm__ volatile(
        "cpuid"
        : "=a"(result.eax), "=b"(result.ebx), "=c"(result.ecx), "=d"(result.edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif

    return result;
}

/**
 * Check if AVX2 is supported
 */
inline bool has_avx2() {
    // Check for AVX2 support
    auto leaf7 = cpuid(7, 0);
    return (leaf7.ebx & (1 << 5)) != 0;  // AVX2 bit
}

/**
 * Check if AVX-512 Foundation is supported
 */
inline bool has_avx512f() {
    auto leaf7 = cpuid(7, 0);
    return (leaf7.ebx & (1 << 16)) != 0;  // AVX512F bit
}

/**
 * Runtime AVX512 capability check - prevents illegal instruction exceptions
 */
inline bool avx512_supported() {
#ifdef DISABLE_AVX512
    return false;
#else
    return has_avx512f();
#endif
}

/**
 * Check if FMA3 is supported
 */
inline bool has_fma3() {
    auto leaf1 = cpuid(1, 0);
    return (leaf1.ecx & (1 << 12)) != 0;  // FMA3 bit
}

/**
 * Check if AVX_VNNI is supported
 */
inline bool has_avx_vnni() {
    // AVX_VNNI is located in Leaf 7, Subleaf 1, EAX bit 4
    auto leaf7_1 = cpuid(7, 1);
    return (leaf7_1.eax & (1 << 4)) != 0;  // AVX_VNNI bit
}

/**
 * Get CPU vendor string
 */
inline std::string get_cpu_vendor() {
    auto leaf0 = cpuid(0, 0);
    char vendor[13] = {0};
    *reinterpret_cast<uint32_t*>(&vendor[0]) = leaf0.ebx;
    *reinterpret_cast<uint32_t*>(&vendor[4]) = leaf0.edx;
    *reinterpret_cast<uint32_t*>(&vendor[8]) = leaf0.ecx;
    return vendor;
}

/**
 * Get supported feature mask
 */
inline uint32_t get_supported_features() {
    uint32_t features = 0;

    if (has_avx2()) features |= static_cast<uint32_t>(Feature::AVX2);
    if (avx512_supported()) features |= static_cast<uint32_t>(Feature::AVX512F);
    if (has_fma3()) features |= static_cast<uint32_t>(Feature::FMA3);
    if (has_avx_vnni()) features |= static_cast<uint32_t>(Feature::AVX_VNNI);

    // Additional AVX-512 features (only if AVX512F is supported)
    if (features & static_cast<uint32_t>(Feature::AVX512F)) {
        auto leaf7 = cpuid(7, 0);

        if (leaf7.ebx & (1 << 17)) features |= static_cast<uint32_t>(Feature::AVX512DQ);
        if (leaf7.ebx & (1 << 30)) features |= static_cast<uint32_t>(Feature::AVX512BW);
        if (leaf7.ebx & (1 << 31)) features |= static_cast<uint32_t>(Feature::AVX512VL);
    }

    if (cpuid(7, 0).ebx & (1 << 8)) features |= static_cast<uint32_t>(Feature::BMI2);

    return features;
}

/**
 * Check if a specific feature is supported
 */
inline bool has_feature(Feature feature) {
    return (get_supported_features() & static_cast<uint32_t>(feature)) != 0;
}

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
 * Kernel preference structure for fallback hierarchy
 */
struct KernelPreference {
    KernelCapability primary;
    KernelCapability fallback;
    KernelCapability emergency; // Always BASELINE
};

/**
 * Select optimal kernel implementation based on runtime CPU capabilities
 */
KernelPreference select_kernel_implementation(
    const RuntimeCpuFeatures& features,
    const std::string& kernel_type);

/**
 * Hierarchical fallback dispatcher for safe kernel execution
 */
class FallbackDispatcher {
public:
    /**
     * Dispatch with automatic fallback mechanism
     */
    template<typename KernelFn, typename... Args>
    auto dispatch_with_fallback(const std::string& kernel_type, Args&&... args) {
        auto features = detect_runtime_cpu_features();
        auto preference = select_kernel_implementation(features, kernel_type);

        // Try primary implementation
        try {
            if (auto primary = get_kernel_function(kernel_type, preference.primary)) {
                return primary(std::forward<Args>(args)..., features);
            }
        } catch (const CpuFeatureException&) {
            // Primary failed, try fallback
        }

        // Try fallback implementation
        try {
            if (auto fallback = get_kernel_function(kernel_type, preference.fallback)) {
                return fallback(std::forward<Args>(args)..., features);
            }
        } catch (const CpuFeatureException&) {
            // Fallback failed, use emergency
        }

        // Emergency fallback - always BASELINE which should never fail
        if (auto emergency = get_kernel_function(kernel_type, preference.emergency)) {
            return emergency(std::forward<Args>(args)..., features);
        }

        // If we get here, something is very wrong
        throw CpuFeatureException("No viable kernel implementation found for " + kernel_type);
    }

private:
    using KernelFunctionPtr = void*;

    /**
     * Get kernel function pointer for specific capability level
     */
    KernelFunctionPtr get_kernel_function(const std::string& kernel_type, KernelCapability capability);
};

/**
 * Get human-readable CPU information
 */
inline std::string get_cpu_info() {
    std::string info = "CPU Vendor: " + get_cpu_vendor() + "\n";
    info += "Features: ";

    uint32_t features = get_supported_features();
    if (features & static_cast<uint32_t>(Feature::AVX2)) info += "AVX2 ";
    if (features & static_cast<uint32_t>(Feature::AVX512F)) info += "AVX512F ";
    if (features & static_cast<uint32_t>(Feature::AVX512DQ)) info += "AVX512DQ ";
    if (features & static_cast<uint32_t>(Feature::AVX512BW)) info += "AVX512BW ";
    if (features & static_cast<uint32_t>(Feature::AVX512VL)) info += "AVX512VL ";
    if (features & static_cast<uint32_t>(Feature::FMA3)) info += "FMA3 ";
    if (features & static_cast<uint32_t>(Feature::BMI2)) info += "BMI2 ";
    if (features & static_cast<uint32_t>(Feature::AVX_VNNI)) info += "AVX_VNNI ";

    // Add runtime safety note
    if (!avx512_supported() && has_avx512f()) {
        info += "\nNote: AVX512 disabled at compile-time for compatibility";
    }

    return info;
}

} // namespace cpu_features
} // namespace hypercube