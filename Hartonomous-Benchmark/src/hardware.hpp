#pragma once

#include <iostream>
#include <cpuid.h>
#include <vector>
#include <string>

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
    std::string name;
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
        unsigned int eax, ebx, ecx, edx;

        // Basic features
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        features.mmx = edx & (1 << 23);
        features.sse = edx & (1 << 25);
        features.sse2 = edx & (1 << 26);
        features.sse3 = ecx & 1;
        features.ssse3 = ecx & (1 << 9);
        features.sse41 = ecx & (1 << 19);
        features.sse42 = ecx & (1 << 20);
        features.avx = ecx & (1 << 28);

        // Extended features
        __get_cpuid(7, &eax, &ebx, &ecx, &edx);
        features.avx2 = ebx & (1 << 5);
        features.avx512f = ebx & (1 << 16);
        features.vnni = ecx & (1 << 11); // AVX512_VNNI

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

inline void print_cpu_info() {
    HardwareDetector detector;
    auto features = detector.detect_cpu_features();
    std::cout << "CPU Features:" << std::endl;
    if (features.mmx) std::cout << " - MMX" << std::endl;
    if (features.sse) std::cout << " - SSE" << std::endl;
    if (features.sse2) std::cout << " - SSE2" << std::endl;
    if (features.sse3) std::cout << " - SSE3" << std::endl;
    if (features.ssse3) std::cout << " - SSSE3" << std::endl;
    if (features.sse41) std::cout << " - SSE4.1" << std::endl;
    if (features.sse42) std::cout << " - SSE4.2" << std::endl;
    if (features.avx) std::cout << " - AVX" << std::endl;
    if (features.avx2) std::cout << " - AVX2" << std::endl;
    if (features.avx512f) std::cout << " - AVX-512F" << std::endl;
    if (features.vnni) std::cout << " - VNNI" << std::endl;
}