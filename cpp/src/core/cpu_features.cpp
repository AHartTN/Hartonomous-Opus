#include "hypercube/cpu_features.h"
#include <cstring>  // for memcpy

namespace hypercube {

namespace cpu_features {

struct CpuidResult {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
};

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

inline uint64_t xgetbv(uint32_t xcr) {
#ifdef _MSC_VER
    return _xgetbv(xcr);
#else
    return __builtin_ia32_xgetbv(xcr);
#endif
}

CpuFeatures detect_cpu_features() {
    CpuFeatures features{};

    // Get vendor string
    auto leaf0 = cpuid(0);
    char vendor[13] = {0};
    *reinterpret_cast<uint32_t*>(&vendor[0]) = leaf0.ebx;
    *reinterpret_cast<uint32_t*>(&vendor[4]) = leaf0.edx;
    *reinterpret_cast<uint32_t*>(&vendor[8]) = leaf0.ecx;
    features.vendor = vendor;

    // Check OSXSAVE and get XCR0
    auto leaf1 = cpuid(1);
    bool osxsave = (leaf1.ecx & (1U << 27)) != 0;
    uint64_t xcr0 = 0;
    if (osxsave) {
        xcr0 = xgetbv(0);
    }

    // SSE features
    features.sse = (leaf1.edx & (1U << 25)) != 0;
    features.sse2 = (leaf1.edx & (1U << 26)) != 0;
    features.sse3 = (leaf1.ecx & (1U << 0)) != 0;
    features.ssse3 = (leaf1.ecx & (1U << 9)) != 0;
    features.sse4_1 = (leaf1.ecx & (1U << 19)) != 0;
    features.sse4_2 = (leaf1.ecx & (1U << 20)) != 0;

    // AVX
    bool cpu_avx = (leaf1.ecx & (1U << 28)) != 0;
    features.avx = cpu_avx && osxsave && (xcr0 & (1ULL << 2));

    // AVX2
    auto leaf7 = cpuid(7, 0);
    features.avx2 = cpu_avx && (leaf7.ebx & (1U << 5)) && osxsave && (xcr0 & (1ULL << 2));

    // AVX-512
    bool avx512_base = cpu_avx && (leaf7.ebx & (1U << 16));
    bool avx512_os_support = osxsave && (xcr0 & (1ULL << 5)) && (xcr0 & (1ULL << 6)) && (xcr0 & (1ULL << 7));
    features.avx512f = avx512_base && avx512_os_support;
    features.avx512dq = features.avx512f && (leaf7.ebx & (1U << 17));
    features.avx512ifma = features.avx512f && (leaf7.ebx & (1U << 21));
    features.avx512pf = features.avx512f && (leaf7.ebx & (1U << 26));
    features.avx512er = features.avx512f && (leaf7.ebx & (1U << 27));
    features.avx512cd = features.avx512f && (leaf7.ebx & (1U << 28));
    features.avx512bw = features.avx512f && (leaf7.ebx & (1U << 30));
    features.avx512vl = features.avx512f && (leaf7.ebx & (1U << 31));
    features.avx512_vbmi = features.avx512f && (leaf7.ebx & (1U << 1));
    features.avx512_vbmi2 = features.avx512f && (leaf7.ebx & (1U << 6));
    features.avx512_vnni = features.avx512f && (leaf7.ebx & (1U << 11));
    features.avx512_bitalg = features.avx512f && (leaf7.ebx & (1U << 12));
    features.avx512_vpopcntdq = features.avx512f && (leaf7.ebx & (1U << 14));
    features.avx512_4vnniw = features.avx512f && (leaf7.ecx & (1U << 2));
    features.avx512_4fmaps = features.avx512f && (leaf7.ecx & (1U << 3));
    features.avx512_bf16 = features.avx512f && (leaf7.ebx & (1U << 5));

    // VNNI
    auto leaf7_1 = cpuid(7, 1);
    features.avx_vnni = cpu_avx && (leaf7_1.eax & (1U << 4)) && osxsave && (xcr0 & (1ULL << 2));

    // AMX
    bool amx_os_support = osxsave && (xcr0 & (1ULL << 11)) && (xcr0 & (1ULL << 12));
    features.amx_tile = (leaf7.edx & (1U << 24)) && amx_os_support;
    features.amx_int8 = (leaf7.edx & (1U << 25)) && amx_os_support;
    features.amx_bf16 = (leaf7.edx & (1U << 22)) && amx_os_support;

    // Misc
    features.fma = (leaf1.ecx & (1U << 12)) != 0;
    auto leaf_ext = cpuid(0x80000001);
    features.fma4 = (leaf_ext.ecx & (1U << 16)) != 0;
    features.xop = (leaf_ext.ecx & (1U << 11)) != 0;
    features.f16c = (leaf1.ecx & (1U << 29)) != 0;
    features.bmi = (leaf7.ebx & (1U << 3)) != 0;
    features.bmi2 = (leaf7.ebx & (1U << 8)) != 0;
    features.lzcnt = (leaf_ext.ecx & (1U << 5)) != 0;
    features.popcnt = (leaf1.ecx & (1U << 23)) != 0;
    features.aes = (leaf1.ecx & (1U << 25)) != 0;
    features.pclmulqdq = (leaf1.ecx & (1U << 1)) != 0;
    features.rdrand = (leaf1.ecx & (1U << 30)) != 0;
    features.rdseed = (leaf7.ebx & (1U << 18)) != 0;
    features.sha = (leaf7.ebx & (1U << 29)) != 0;
    features.adx = (leaf7.ebx & (1U << 19)) != 0;

    return features;
}

} // namespace cpu_features

} // namespace hypercube
