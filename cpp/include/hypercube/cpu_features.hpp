/**
 * CPU Feature Detection - Runtime SIMD/AVX capability checks
 *
 * Provides runtime detection of CPU features like AVX2, AVX-512, etc.
 * Allows the application to choose optimal code paths based on actual hardware capabilities.
 */

#pragma once

#include <cstdint>
#include <string>
#include <cstring>  // For memcpy in vendor string

namespace hypercube {
namespace cpu_features {

/**
 * CPU feature flags
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
    auto leaf7 = cpuid(7, 0);
    return (leaf7.ebx & (1 << 11)) != 0;  // AVX_VNNI bit
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