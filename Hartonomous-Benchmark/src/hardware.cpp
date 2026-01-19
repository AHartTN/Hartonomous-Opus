#include <cstdio>

struct CpuFeatures {
    bool mmx = false;
    bool sse = false;
    bool sse2 = false;
    bool sse3 = false;
    bool ssse3 = false;
    bool sse41 = false;
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
    bool vnni = false;
};

struct GpuInfo {
    const char* name = "Unknown";
    size_t memory_mb = 0;
    bool cuda_supported = false;
};

struct MemoryInfo {
    size_t total_mb = 0;
    size_t available_mb = 0;
};

class HardwareDetector {
public:
    CpuFeatures detect_cpu_features() {
        CpuFeatures features;

        // Use compile-time detection via predefined macros
        // This avoids runtime CPUID calls that may trigger antivirus
#ifdef __MMX__
        features.mmx = true;
#endif
#ifdef __SSE__
        features.sse = true;
#endif
#ifdef __SSE2__
        features.sse2 = true;
#endif
#ifdef __SSE3__
        features.sse3 = true;
#endif
#ifdef __SSSE3__
        features.ssse3 = true;
#endif
#ifdef __SSE4_1__
        features.sse41 = true;
#endif
#ifdef __SSE4_2__
        features.sse42 = true;
#endif
#ifdef __AVX__
        features.avx = true;
#endif
#ifdef __AVX2__
        features.avx2 = true;
#endif
#ifdef __AVX512F__
        features.avx512f = true;
#endif
#ifdef __AVX512VNNI__
        features.vnni = true;
#endif

        return features;
    }

    GpuInfo detect_gpu_info() {
        GpuInfo info;
        // Basic GPU detection - in real implementation, use CUDA/OpenCL APIs
        info.name = "Nvidia GeForce RTX 4060";
        info.memory_mb = 8192; // 8GB
        info.cuda_supported = true;
        return info;
    }

    MemoryInfo get_memory_info() {
        MemoryInfo info;
        // Basic memory detection - in real implementation, use sysinfo APIs
        info.total_mb = 32768; // 32GB
        info.available_mb = 24576; // 24GB
        return info;
    }
};

void print_cpu_info() {
    HardwareDetector detector;
    auto features = detector.detect_cpu_features();
    printf("CPU Features:\n");
    if (features.mmx) printf(" - MMX\n");
    if (features.sse) printf(" - SSE\n");
    if (features.sse2) printf(" - SSE2\n");
    if (features.sse3) printf(" - SSE3\n");
    if (features.ssse3) printf(" - SSSE3\n");
    if (features.sse41) printf(" - SSE4.1\n");
    if (features.sse42) printf(" - SSE4.2\n");
    if (features.avx) printf(" - AVX\n");
    if (features.avx2) printf(" - AVX2\n");
    if (features.avx512f) printf(" - AVX-512F\n");
    if (features.vnni) printf(" - VNNI\n");
}

int main() {
    print_cpu_info();
    return 0;
}