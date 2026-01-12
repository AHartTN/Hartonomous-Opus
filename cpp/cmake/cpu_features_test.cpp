#include <iostream>
#include <cstdint>

// CPUID function for feature detection
inline void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    eax = regs[0];
    ebx = regs[1];
    ecx = regs[2];
    edx = regs[3];
#else
    __asm__ volatile(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif
}

int main() {
    uint32_t eax, ebx, ecx, edx;

    // Initialize all features to false
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_avx512bw = false;
    bool has_avx512vl = false;
    bool has_fma3 = false;
    bool has_avx_vnni = false;
    bool has_bmi2 = false;

    // Check if CPUID is supported (bit 21 of EDX from leaf 1)
    cpuid(1, 0, eax, ebx, ecx, edx);
    bool cpuid_supported = (edx & (1 << 21)) != 0;

    if (cpuid_supported) {
        // Check AVX2 (leaf 7, subleaf 0, bit 5 of EBX)
        cpuid(7, 0, eax, ebx, ecx, edx);
        has_avx2 = (ebx & (1 << 5)) != 0;

        // Check AVX-512 Foundation (leaf 7, subleaf 0, bit 16 of EBX)
        has_avx512f = (ebx & (1 << 16)) != 0;

        // Check additional AVX-512 features (only if AVX512F is supported)
        if (has_avx512f) {
            has_avx512dq = (ebx & (1 << 17)) != 0;
            has_avx512bw = (ebx & (1 << 30)) != 0;
            has_avx512vl = (ebx & (1 << 31)) != 0;
        }

        // Check FMA3 (leaf 1, subleaf 0, bit 12 of ECX) - already retrieved above
        has_fma3 = (ecx & (1 << 12)) != 0;

        // Check AVX-VNNI (leaf 7, subleaf 0, bit 11 of EBX) - already retrieved above
        has_avx_vnni = (ebx & (1 << 11)) != 0;

        // Check BMI2 (leaf 7, subleaf 0, bit 8 of EBX) - already retrieved above
        has_bmi2 = (ebx & (1 << 8)) != 0;
    }

    std::cout << "AVX2:" << (has_avx2 ? "1" : "0") << std::endl;
    std::cout << "AVX512F:" << (has_avx512f ? "1" : "0") << std::endl;
    std::cout << "AVX512DQ:" << (has_avx512dq ? "1" : "0") << std::endl;
    std::cout << "AVX512BW:" << (has_avx512bw ? "1" : "0") << std::endl;
    std::cout << "AVX512VL:" << (has_avx512vl ? "1" : "0") << std::endl;
    std::cout << "FMA3:" << (has_fma3 ? "1" : "0") << std::endl;
    std::cout << "AVX_VNNI:" << (has_avx_vnni ? "1" : "0") << std::endl;
    std::cout << "BMI2:" << (has_bmi2 ? "1" : "0") << std::endl;

    return 0;
}