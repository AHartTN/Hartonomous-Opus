#include "hypercube/cpu_features.h"
#include "hypercube/cpu_features.hpp"
#include "hypercube/runtime_dispatch.h"
#include <cstring>  // for memcpy

namespace hypercube {

namespace cpu_features {

inline uint64_t xgetbv(uint32_t xcr) {
#ifdef _MSC_VER
    return _xgetbv(xcr);
#else
    uint32_t low, high;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(xcr));
    return (static_cast<uint64_t>(high) << 32) | low;
#endif
}

// Legacy function for backward compatibility - delegates to new runtime detection
CpuFeatures detect_cpu_features() {
    auto runtime_features = detect_runtime_cpu_features();

    CpuFeatures legacy{};
    legacy.vendor = runtime_features.vendor;
    legacy.sse = runtime_features.sse;
    legacy.sse2 = runtime_features.sse2;
    legacy.sse3 = runtime_features.sse3;
    legacy.ssse3 = runtime_features.ssse3;
    legacy.sse4_1 = runtime_features.sse4_1;
    legacy.sse4_2 = runtime_features.sse4_2;
    legacy.avx = runtime_features.safe_avx_execution;
    legacy.avx2 = runtime_features.safe_avx512_execution;
    legacy.avx512f = runtime_features.safe_avx512_execution;
    legacy.avx512dq = runtime_features.avx512dq && runtime_features.safe_avx512_execution;
    legacy.avx512ifma = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512pf = runtime_features.avx512f && runtime_features.safe_avx512_execution;   // Simplified
    legacy.avx512er = runtime_features.avx512f && runtime_features.safe_avx512_execution;   // Simplified
    legacy.avx512cd = runtime_features.avx512f && runtime_features.safe_avx512_execution;   // Simplified
    legacy.avx512bw = runtime_features.avx512bw && runtime_features.safe_avx512_execution;
    legacy.avx512vl = runtime_features.avx512vl && runtime_features.safe_avx512_execution;
    legacy.avx512_vbmi = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_vbmi2 = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_vnni = runtime_features.avx512vnni && runtime_features.safe_avx512_execution;
    legacy.avx512_bitalg = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_vpopcntdq = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_4vnniw = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_4fmaps = runtime_features.avx512f && runtime_features.safe_avx512_execution; // Simplified
    legacy.avx512_bf16 = runtime_features.avx512bf16 && runtime_features.safe_avx512_execution;
    legacy.fma = runtime_features.fma3;
    legacy.fma4 = false; // Not tracked in runtime features
    legacy.xop = false;  // Not tracked in runtime features
    legacy.f16c = false; // Not tracked in runtime features
    legacy.bmi = false;  // Not tracked in runtime features
    legacy.bmi2 = false; // Not tracked in runtime features
    legacy.lzcnt = false; // Not tracked in runtime features
    legacy.popcnt = false; // Not tracked in runtime features
    legacy.aes = false;   // Not tracked in runtime features
    legacy.pclmulqdq = false; // Not tracked in runtime features
    legacy.rdrand = false;    // Not tracked in runtime features
    legacy.rdseed = false;    // Not tracked in runtime features
    legacy.sha = false;       // Not tracked in runtime features
    legacy.adx = false;       // Not tracked in runtime features
    legacy.avx_vnni = (runtime_features.avx_vnni || runtime_features.avx512vnni) && runtime_features.safe_avx_execution;
    legacy.amx_tile = runtime_features.safe_amx_execution;
    legacy.amx_int8 = runtime_features.safe_amx_execution; // Simplified
    legacy.amx_bf16 = runtime_features.safe_amx_execution; // Simplified

    return legacy;
}

bool check_os_avx_support() {
    // Return cached value from RuntimeDispatchEngine if available
    auto& engine = RuntimeDispatchEngine::instance();
    if (engine.initialized_) {
        return engine.cpu_features_.os_avx_support;
    }

    // Fallback to direct check if engine not initialized
    auto leaf1 = cpuid(1);
    bool osxsave = (leaf1.ecx & (1U << 27)) != 0;
    if (!osxsave) return false;

    uint64_t xcr0 = xgetbv(0);
    // Check that SSE (bit 1) and AVX (bit 2) state saving is enabled
    return (xcr0 & 0x6) == 0x6;
}

bool check_os_avx512_support() {
    // Return cached value from RuntimeDispatchEngine if available
    auto& engine = RuntimeDispatchEngine::instance();
    if (engine.initialized_) {
        return engine.cpu_features_.os_avx512_support;
    }

    // Fallback to direct check if engine not initialized
    auto leaf1 = cpuid(1);
    bool osxsave = (leaf1.ecx & (1U << 27)) != 0;
    if (!osxsave) return false;

    uint64_t xcr0 = xgetbv(0);
    // Check AVX-512 state saving bits (5, 6, 7)
    return (xcr0 & (1ULL << 5)) && (xcr0 & (1ULL << 6)) && (xcr0 & (1ULL << 7));
}

bool check_os_amx_support() {
    // Return cached value from RuntimeDispatchEngine if available
    auto& engine = RuntimeDispatchEngine::instance();
    if (engine.initialized_) {
        return engine.cpu_features_.os_amx_support;
    }

    // Fallback to direct check if engine not initialized
    auto leaf1 = cpuid(1);
    bool osxsave = (leaf1.ecx & (1U << 27)) != 0;
    if (!osxsave) return false;

    uint64_t xcr0 = xgetbv(0);
    // Check AMX state saving bits (11, 12)
    return (xcr0 & (1ULL << 11)) && (xcr0 & (1ULL << 12));
}

KernelPreference select_kernel_implementation(
    const RuntimeCpuFeatures& features,
    const std::string& kernel_type) {

    KernelPreference preference{};
    preference.emergency = KernelCapability::BASELINE; // Always available fallback

    // Distance/L2 kernels prefer AVX-512 with VNNI for best performance
    if (kernel_type == "distance_l2" || kernel_type == "vector_ops") {
        if (features.safe_avx512_execution && features.avx512vnni) {
            preference.primary = KernelCapability::AVX512_VNNI;
            preference.fallback = features.safe_avx_execution && features.avx2 ?
                                KernelCapability::AVX2_VNNI : KernelCapability::AVX2;
        } else if (features.safe_avx512_execution && features.avx512f) {
            preference.primary = KernelCapability::AVX512F;
            preference.fallback = features.safe_avx_execution && features.avx2 ?
                                KernelCapability::AVX2 : KernelCapability::SSE42;
        } else if (features.safe_avx_execution && features.avx2 && features.avx_vnni) {
            preference.primary = KernelCapability::AVX2_VNNI;
            preference.fallback = KernelCapability::AVX2;
        } else if (features.safe_avx_execution && features.avx2) {
            preference.primary = KernelCapability::AVX2;
            preference.fallback = KernelCapability::SSE42;
        } else if (features.sse4_2) {
            preference.primary = KernelCapability::SSE42;
        } else {
            preference.primary = KernelCapability::BASELINE;
        }

    // Matrix multiplication prefers AVX-512 with high bandwidth
    } else if (kernel_type == "matrix_multiply" || kernel_type == "gemm") {
        if (features.safe_avx512_execution && features.avx512bw) {
            preference.primary = KernelCapability::AVX512_BW;
            preference.fallback = KernelCapability::AVX512F;
        } else if (features.safe_avx512_execution && features.avx512f) {
            preference.primary = KernelCapability::AVX512F;
            preference.fallback = features.safe_avx_execution && features.avx2 ?
                                KernelCapability::AVX2 : KernelCapability::SSE42;
        } else if (features.safe_avx_execution && features.avx2 && features.fma3) {
            preference.primary = KernelCapability::AVX2_FMA3;
            preference.fallback = KernelCapability::AVX2;
        } else if (features.safe_avx_execution && features.avx2) {
            preference.primary = KernelCapability::AVX2;
            preference.fallback = KernelCapability::SSE42;
        } else if (features.sse4_2) {
            preference.primary = KernelCapability::SSE42;
        } else {
            preference.primary = KernelCapability::BASELINE;
        }

    // AMX-optimized kernels for certain workloads
    } else if (kernel_type == "amx_optimized") {
        if (features.safe_amx_execution) {
            preference.primary = KernelCapability::AMX;
            preference.fallback = features.safe_avx512_execution && features.avx512f ?
                                KernelCapability::AVX512F : KernelCapability::AVX2;
        } else {
            // Fallback to best available SIMD
            preference = select_kernel_implementation(features, "distance_l2");
        }

    // Default fallback for unknown kernel types
    } else {
        if (features.safe_avx512_execution && features.avx512f) {
            preference.primary = KernelCapability::AVX512F;
            preference.fallback = features.safe_avx_execution && features.avx2 ?
                                KernelCapability::AVX2 : KernelCapability::SSE42;
        } else if (features.safe_avx_execution && features.avx2) {
            preference.primary = KernelCapability::AVX2;
            preference.fallback = KernelCapability::SSE42;
        } else if (features.sse4_2) {
            preference.primary = KernelCapability::SSE42;
        } else {
            preference.primary = KernelCapability::BASELINE;
        }
    }

    // Ensure fallback is set if not already
    if (preference.fallback == KernelCapability::BASELINE && preference.primary != KernelCapability::BASELINE) {
        preference.fallback = KernelCapability::BASELINE;
    }

    return preference;
}

RuntimeCpuFeatures detect_runtime_cpu_features() {
    RuntimeCpuFeatures features{};

    // Get vendor string and basic info
    auto leaf0 = cpuid(0);
    char vendor[13] = {0};
    *reinterpret_cast<uint32_t*>(&vendor[0]) = leaf0.ebx;
    *reinterpret_cast<uint32_t*>(&vendor[4]) = leaf0.edx;
    *reinterpret_cast<uint32_t*>(&vendor[8]) = leaf0.ecx;
    features.vendor = vendor;

    // Get family and model
    auto leaf1 = cpuid(1);
    features.family = ((leaf1.eax >> 8) & 0xF) + ((leaf1.eax >> 20) & 0xFF);
    features.model = ((leaf1.eax >> 4) & 0xF) | ((leaf1.eax >> 12) & 0xF0);

    // Check OS support first
    features.os_avx_support = check_os_avx_support();
    features.os_avx512_support = check_os_avx512_support();
    features.os_amx_support = check_os_amx_support();

    // SSE features
    features.sse = (leaf1.edx & (1U << 25)) != 0;
    features.sse2 = (leaf1.edx & (1U << 26)) != 0;
    features.sse3 = (leaf1.ecx & (1U << 0)) != 0;
    features.ssse3 = (leaf1.ecx & (1U << 9)) != 0;
    features.sse4_1 = (leaf1.ecx & (1U << 19)) != 0;
    features.sse4_2 = (leaf1.ecx & (1U << 20)) != 0;

    // AVX
    bool cpu_avx = (leaf1.ecx & (1U << 28)) != 0;
    features.avx = cpu_avx;

    // AVX2
    auto leaf7 = cpuid(7, 0);
    features.avx2 = cpu_avx && (leaf7.ebx & (1U << 5));

    // FMA3
    features.fma3 = (leaf1.ecx & (1U << 12)) != 0;

    // AVX-512 features
    bool avx512_base = cpu_avx && (leaf7.ebx & (1U << 16));
    features.avx512f = avx512_base;
    features.avx512dq = avx512_base && (leaf7.ebx & (1U << 17));
    features.avx512bw = avx512_base && (leaf7.ebx & (1U << 30));
    features.avx512vl = avx512_base && (leaf7.ebx & (1U << 31));
    features.avx512vnni = avx512_base && (leaf7.ebx & (1U << 11));
    features.avx512bf16 = avx512_base && (leaf7.ebx & (1U << 5));

    // Check for AVX512_FP16 (leaf 7, subleaf 0, EDX bit 23)
    // Note: This might not be available on all compilers yet
    features.avx512fp16 = avx512_base && (leaf7.edx & (1U << 23));

    // VNNI variants
    auto leaf7_1 = cpuid(7, 1);
    features.avx_vnni = cpu_avx && (leaf7_1.eax & (1U << 4));
    features.avx512_vnni = features.avx512vnni;  // Same as avx512vnni

    // AMX features
    features.amx_tile = (leaf7.edx & (1U << 24)) != 0;
    features.amx_int8 = (leaf7.edx & (1U << 25)) != 0;
    features.amx_bf16 = (leaf7.edx & (1U << 22)) != 0;

    // Set safety flags based on OS support
    features.safe_avx_execution = features.avx && features.os_avx_support;
    features.safe_avx512_execution = features.avx512f && features.os_avx512_support;
    features.safe_amx_execution = features.amx_tile && features.os_amx_support;

    return features;
}

// FallbackDispatcher implementation
FallbackDispatcher::KernelFunctionPtr FallbackDispatcher::get_kernel_function(
    const std::string& kernel_type, KernelCapability capability) {

    // This is a simplified implementation - in practice, this would be
    // replaced with a proper kernel registry system
    // For now, return nullptr to indicate no specific function pointer available
    // The actual dispatch logic is handled by the kernel implementations themselves

    (void)kernel_type; // Suppress unused parameter warning
    (void)capability;  // Suppress unused parameter warning

    return nullptr; // Kernel selection is handled at higher level
}

} // namespace cpu_features

} // namespace hypercube
