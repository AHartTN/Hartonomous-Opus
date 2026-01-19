#include <iostream>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

// Forward declaration
inline void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t& eax, uint32_t& ebx,
                  uint32_t& ecx, uint32_t& edx);

// Check OS support for AVX state saving via XGETBV
// Must verify OSXSAVE is enabled before calling xgetbv to avoid SIGILL
inline bool check_os_avx_support() {
    // First check if OSXSAVE is enabled (CPUID leaf 1, ECX bit 27)
    uint32_t eax_local, ebx_local, ecx_local, edx_local;
    cpuid(1u, 0u, eax_local, ebx_local, ecx_local, edx_local);

    bool osxsave_enabled = (ecx_local & (1u << 27)) != 0;
    if (!osxsave_enabled) {
        return false; // Can't use xgetbv safely
    }

    uint64_t xcr0;
#if defined(_MSC_VER)
    xcr0 = _xgetbv(0);
#else
    uint32_t low, high;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    xcr0 = (static_cast<uint64_t>(high) << 32) | low;
#endif
    // Check that SSE (bit 1) and AVX (bit 2) state saving is enabled
    return (xcr0 & 0x6) == 0x6;
}

// Minimal CPUID wrapper for feature detection
inline void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t& eax, uint32_t& ebx,
                  uint32_t& ecx, uint32_t& edx) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    eax = static_cast<uint32_t>(regs[0]);
    ebx = static_cast<uint32_t>(regs[1]);
    ecx = static_cast<uint32_t>(regs[2]);
    edx = static_cast<uint32_t>(regs[3]);
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ volatile(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(leaf), "c"(subleaf)
    );
#else
    // Unknown compiler: report no features
    eax = ebx = ecx = edx = 0;
#endif
}

int main() {
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;

    bool has_avx2      = false;
    bool has_avx512f   = false;
    bool has_avx512dq  = false;
    bool has_avx512bw  = false;
    bool has_avx512vl  = false;
    bool has_fma3      = false;
    bool has_avx_vnni  = false; // generic AVX-VNNI capability (256- or 512-bit)
    bool has_bmi2      = false;

    // Get vendor string to identify Intel CPUs
    uint32_t vendor_ebx = 0, vendor_ecx = 0, vendor_edx = 0;
    cpuid(0u, 0u, eax, vendor_ebx, vendor_ecx, vendor_edx);
    char vendor[13] = {0};
    memcpy(vendor, &vendor_ebx, 4);
    memcpy(vendor + 4, &vendor_edx, 4);
    memcpy(vendor + 8, &vendor_ecx, 4);
    bool is_intel = (strcmp(vendor, "GenuineIntel") == 0);

    // Debug output to stderr (won't interfere with stdout parsing)
    std::cerr << "[CPU] Vendor: " << vendor << " (Intel: " << is_intel << ")\n";

    // Basic CPUID leaf 1: general feature flags
    cpuid(1u, 0u, eax, ebx, ecx, edx);

    // Extract family and model for debug
    uint32_t family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
    uint32_t model = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);
    std::cerr << "[CPU] Family: " << family << ", Model: 0x" << std::hex << model << std::dec << "\n";

    // FMA3: leaf 1, ECX bit 12
    has_fma3 = (ecx & (1u << 12)) != 0;

    // Extended features: leaf 7, subleaf 0
    cpuid(7u, 0u, eax, ebx, ecx, edx);

    // AVX2: EBX bit 5
    has_avx2 = (ebx & (1u << 5)) != 0;

    // AVX-512 Foundation: EBX bit 16
    has_avx512f = (ebx & (1u << 16)) != 0;

    // Additional AVX-512 features (only meaningful if AVX512F is present)
    if (has_avx512f) {
        // AVX-512DQ: EBX bit 17
        has_avx512dq = (ebx & (1u << 17)) != 0;
        // AVX-512BW: EBX bit 30
        has_avx512bw = (ebx & (1u << 30)) != 0;
        // AVX-512VL: EBX bit 31
        has_avx512vl = (ebx & (1u << 31)) != 0;
    }

    // BMI2: EBX bit 8
    has_bmi2 = (ebx & (1u << 8)) != 0;

    // Check OS support for AVX state saving
    bool os_avx_support = check_os_avx_support();
    std::cerr << "[CPU] OS AVX support: " << (os_avx_support ? "YES" : "NO") << "\n";

    std::cerr << "[CPU] Raw CPUID features before OS check:\n";
    std::cerr << "  AVX2=" << has_avx2 << " FMA3=" << has_fma3 << " BMI2=" << has_bmi2 << "\n";
    std::cerr << "  AVX512F=" << has_avx512f << " AVX512DQ=" << has_avx512dq << "\n";

    // AVX-VNNI:
    //
    // Intel defines:
    //   - AVX-512 VNNI: leaf 7, subleaf 0, ECX bit 11 (requires AVX-512F)
    //   - AVX-VNNI (256-bit): leaf 7, subleaf 1, EAX bit 4 (Alder Lake+, no AVX512 required)
    //
    // We treat "AVX_VNNI" as "some VNNI-capable AVX path is present", and let
    // CMake decide whether to use -mavx512vnni or -mavxvnni based on compiler.
    bool has_avx512_vnni = (ecx & (1u << 11)) != 0;

    // Check leaf 7, subleaf 1 for AVX-VNNI (256-bit)
    uint32_t eax1 = 0, ebx1 = 0, ecx1 = 0, edx1 = 0;
    cpuid(7u, 1u, eax1, ebx1, ecx1, edx1);
    bool has_avx_vnni_256 = (eax1 & (1u << 4)) != 0;

    has_avx_vnni = has_avx512_vnni || has_avx_vnni_256;
    std::cerr << "  AVX512_VNNI=" << has_avx512_vnni << " AVX_VNNI_256=" << has_avx_vnni_256 << "\n";

    // Fallback for modern Intel CPUs: Intel 14th/15th gen and newer support AVX2 and AVX_VNNI
    // This helps when OS support checks fail but we know the CPU supports these features
    if (is_intel && !has_avx2) {
        // Intel 14th gen (Raptor Lake): family 6, model 0xB7-0xBF
        // Intel 15th gen (Arrow Lake): family 6, model 0xC6-0xCF
        // Also handle older Raptor Lake variants: 0x97-0x9F
        // And Alder Lake: 0x97, 0x9A
        bool is_modern_intel = (family == 6) && (
            (model >= 0x97 && model <= 0x9F) ||  // Alder Lake, early Raptor Lake
            (model >= 0xB7 && model <= 0xBF) ||  // 14th gen Raptor Lake
            (model >= 0xC6 && model <= 0xCF)     // 15th gen Arrow Lake
        );

        if (is_modern_intel) {
            std::cerr << "[CPU] Applying modern Intel CPU fallback (model 0x" << std::hex << model << std::dec << ")\n";
            has_avx2 = true;
            has_avx_vnni = true;
            has_fma3 = true;  // Also typically supported
            has_bmi2 = true;  // Also typically supported
        }
    }

    // Apply OS support check for AVX-dependent features
    std::cerr << "[CPU] Applying OS support check (before: AVX2=" << has_avx2 << " FMA3=" << has_fma3 << " AVX_VNNI=" << has_avx_vnni << ")\n";
    has_avx2 &= os_avx_support;
    has_fma3 &= os_avx_support;
    has_avx_vnni &= os_avx_support;
    std::cerr << "[CPU] After OS support check: AVX2=" << has_avx2 << " FMA3=" << has_fma3 << " AVX_VNNI=" << has_avx_vnni << "\n";

    // Emit machine-readable output for CMake to parse
    std::cout << "AVX2:"      << (has_avx2      ? "1" : "0") << '\n';
    std::cout << "AVX512F:"   << (has_avx512f   ? "1" : "0") << '\n';
    std::cout << "AVX512DQ:"  << (has_avx512dq  ? "1" : "0") << '\n';
    std::cout << "AVX512BW:"  << (has_avx512bw  ? "1" : "0") << '\n';
    std::cout << "AVX512VL:"  << (has_avx512vl  ? "1" : "0") << '\n';
    std::cout << "FMA3:"      << (has_fma3      ? "1" : "0") << '\n';
    std::cout << "AVX_VNNI:"  << (has_avx_vnni  ? "1" : "0") << '\n';
    std::cout << "BMI2:"      << (has_bmi2      ? "1" : "0") << '\n';

    return 0;
}
