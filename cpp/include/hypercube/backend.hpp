// =============================================================================
// hypercube/backend.hpp - Modular Backend Detection & Selection
// =============================================================================
// 
// This header provides compile-time and runtime detection of available
// computational backends, selecting the optimal implementation for the
// current hardware.
//
// Backend Priority (eigensolvers):
//   1. Intel MKL (DSYEVR) - Fastest on Intel CPUs, AVX-512 optimized
//   2. Eigen (SelfAdjointEigenSolver) - Good performance, portable
//   3. Custom Jacobi - Fallback, always available
//
// SIMD Priority:
//   1. AVX-512 (512-bit, 16 floats)
//   2. AVX2 + FMA (256-bit, 8 floats)
//   3. SSE4.2 (128-bit, 4 floats)
//   4. Scalar fallback
//
// Usage:
//   #include "hypercube/backend.hpp"
//   
//   // Get backend info at runtime
//   auto info = hypercube::Backend::detect();
//   std::cout << info.summary() << std::endl;
//   
//   // Check specific capabilities
//   if (info.has_mkl) { /* use MKL path */ }
//   if (info.simd_level >= SIMDLevel::AVX2) { /* use AVX2 intrinsics */ }
//
// =============================================================================

#pragma once

#include <string>
#include <cstdint>
#include <sstream>
#include <array>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace hypercube {

// =============================================================================
// SIMD Level Enumeration
// =============================================================================

enum class SIMDLevel : int {
    Scalar = 0,
    SSE2 = 1,
    SSE4_2 = 2,
    AVX = 3,
    AVX2 = 4,
    AVX512 = 5
};

inline const char* simd_level_name(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::Scalar: return "Scalar";
        case SIMDLevel::SSE2: return "SSE2";
        case SIMDLevel::SSE4_2: return "SSE4.2";
        case SIMDLevel::AVX: return "AVX";
        case SIMDLevel::AVX2: return "AVX2";
        case SIMDLevel::AVX512: return "AVX-512";
        default: return "Unknown";
    }
}

inline int simd_width(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::AVX512: return 16;  // 512 bits = 16 floats
        case SIMDLevel::AVX2:
        case SIMDLevel::AVX: return 8;      // 256 bits = 8 floats
        case SIMDLevel::SSE4_2:
        case SIMDLevel::SSE2: return 4;     // 128 bits = 4 floats
        default: return 1;
    }
}

// =============================================================================
// Eigensolver Backend Enumeration
// =============================================================================

enum class EigensolverBackend : int {
    Jacobi = 0,      // Custom implementation, always available
    Eigen = 1,       // Eigen::SelfAdjointEigenSolver
    MKL = 2          // Intel MKL DSYEVR
};

inline const char* eigensolver_name(EigensolverBackend backend) {
    switch (backend) {
        case EigensolverBackend::Jacobi: return "Jacobi (custom)";
        case EigensolverBackend::Eigen: return "Eigen SelfAdjointEigenSolver";
        case EigensolverBackend::MKL: return "Intel MKL DSYEVR";
        default: return "Unknown";
    }
}

// =============================================================================
// k-NN Backend Enumeration
// =============================================================================

enum class KNNBackend : int {
    BruteForce = 0,  // O(n²) naive
    HNSWLIB = 1,     // Hierarchical Navigable Small World
    FAISS = 2        // Facebook AI Similarity Search
};

inline const char* knn_backend_name(KNNBackend backend) {
    switch (backend) {
        case KNNBackend::BruteForce: return "Brute-force O(n²)";
        case KNNBackend::HNSWLIB: return "HNSWLIB O(n log n)";
        case KNNBackend::FAISS: return "FAISS";
        default: return "Unknown";
    }
}

// =============================================================================
// CPU Info Structure
// =============================================================================

struct CPUInfo {
    std::string vendor;
    std::string brand;
    int family = 0;
    int model = 0;
    int stepping = 0;
    int cores_physical = 0;
    int cores_logical = 0;
    
    // Feature flags
    bool has_sse2 = false;
    bool has_sse4_2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_avx512bw = false;
    bool has_avx512vl = false;
};

// =============================================================================
// Backend Info Structure
// =============================================================================

struct BackendInfo {
    // CPU information
    CPUInfo cpu;
    
    // Detected SIMD level
    SIMDLevel simd_level = SIMDLevel::Scalar;
    
    // Available backends (compile-time)
    bool has_mkl = false;
    bool has_eigen = false;
    bool has_hnswlib = false;
    bool has_faiss = false;
    
    // Selected backends (best available)
    EigensolverBackend eigensolver = EigensolverBackend::Jacobi;
    KNNBackend knn = KNNBackend::BruteForce;
    
    // Build info
    std::string compiler;
    std::string build_type;
    
    // Generate summary string
    std::string summary() const {
        std::ostringstream ss;
        ss << "=== Hypercube Backend Configuration ===\n";
        ss << "CPU: " << cpu.brand << "\n";
        ss << "Vendor: " << cpu.vendor << "\n";
        ss << "Cores: " << cpu.cores_physical << "P / " << cpu.cores_logical << " threads\n";
        ss << "\n";
        ss << "SIMD Level: " << simd_level_name(simd_level) 
           << " (" << simd_width(simd_level) << " floats/op)\n";
        ss << "  SSE2:    " << (cpu.has_sse2 ? "Yes" : "No") << "\n";
        ss << "  SSE4.2:  " << (cpu.has_sse4_2 ? "Yes" : "No") << "\n";
        ss << "  AVX:     " << (cpu.has_avx ? "Yes" : "No") << "\n";
        ss << "  AVX2:    " << (cpu.has_avx2 ? "Yes" : "No") << "\n";
        ss << "  FMA:     " << (cpu.has_fma ? "Yes" : "No") << "\n";
        ss << "  AVX-512: " << (cpu.has_avx512f ? "Yes" : "No") << "\n";
        ss << "\n";
        ss << "Eigensolver: " << eigensolver_name(eigensolver);
        if (has_mkl) ss << " [MKL available]";
        if (has_eigen) ss << " [Eigen available]";
        ss << "\n";
        ss << "k-NN: " << knn_backend_name(knn);
        if (has_hnswlib) ss << " [HNSWLIB available]";
        if (has_faiss) ss << " [FAISS available]";
        ss << "\n";
        ss << "\n";
        ss << "Build: " << build_type << " (" << compiler << ")\n";
        return ss.str();
    }
};

// =============================================================================
// CPUID Helper Functions
// =============================================================================

namespace detail {

inline void cpuid(int info[4], int function_id) {
#if defined(_WIN32)
    __cpuid(info, function_id);
#else
    __cpuid(function_id, info[0], info[1], info[2], info[3]);
#endif
}

inline void cpuidex(int info[4], int function_id, int subfunction_id) {
#if defined(_WIN32)
    __cpuidex(info, function_id, subfunction_id);
#else
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
#endif
}

inline CPUInfo detect_cpu() {
    CPUInfo info;
    int cpuinfo[4] = {0};
    
    // Get vendor string
    cpuid(cpuinfo, 0);
    int max_function = cpuinfo[0];
    char vendor[13] = {0};
    *reinterpret_cast<int*>(vendor + 0) = cpuinfo[1];
    *reinterpret_cast<int*>(vendor + 4) = cpuinfo[3];
    *reinterpret_cast<int*>(vendor + 8) = cpuinfo[2];
    info.vendor = vendor;
    
    // Get CPU brand string
    cpuid(cpuinfo, 0x80000000);
    if (static_cast<unsigned>(cpuinfo[0]) >= 0x80000004) {
        char brand[49] = {0};
        cpuid(reinterpret_cast<int*>(brand + 0), 0x80000002);
        cpuid(reinterpret_cast<int*>(brand + 16), 0x80000003);
        cpuid(reinterpret_cast<int*>(brand + 32), 0x80000004);
        info.brand = brand;
        // Trim leading spaces
        size_t start = info.brand.find_first_not_of(' ');
        if (start != std::string::npos) {
            info.brand = info.brand.substr(start);
        }
    }
    
    // Get feature flags
    if (max_function >= 1) {
        cpuid(cpuinfo, 1);
        info.stepping = cpuinfo[0] & 0xF;
        info.model = ((cpuinfo[0] >> 4) & 0xF) | ((cpuinfo[0] >> 12) & 0xF0);
        info.family = ((cpuinfo[0] >> 8) & 0xF) + ((cpuinfo[0] >> 20) & 0xFF);
        
        info.has_sse2 = (cpuinfo[3] & (1 << 26)) != 0;
        info.has_sse4_2 = (cpuinfo[2] & (1 << 20)) != 0;
        info.has_avx = (cpuinfo[2] & (1 << 28)) != 0;
        info.has_fma = (cpuinfo[2] & (1 << 12)) != 0;
    }
    
    if (max_function >= 7) {
        cpuidex(cpuinfo, 7, 0);
        info.has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;
        info.has_avx512f = (cpuinfo[1] & (1 << 16)) != 0;
        info.has_avx512dq = (cpuinfo[1] & (1 << 17)) != 0;
        info.has_avx512bw = (cpuinfo[1] & (1 << 30)) != 0;
        info.has_avx512vl = (cpuinfo[1] & (1 << 31)) != 0;
    }
    
    // Get core count
#if defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    info.cores_logical = sysinfo.dwNumberOfProcessors;
    info.cores_physical = info.cores_logical / 2;  // Approximate
#else
    info.cores_logical = std::thread::hardware_concurrency();
    info.cores_physical = info.cores_logical / 2;
#endif
    
    return info;
}

}  // namespace detail

// =============================================================================
// Main Backend Detection Class
// =============================================================================

class Backend {
public:
    // Detect all available backends and select optimal configuration
    static BackendInfo detect() {
        BackendInfo info;
        
        // Detect CPU
        info.cpu = detail::detect_cpu();
        
        // Determine SIMD level
        if (info.cpu.has_avx512f && info.cpu.has_avx512dq && 
            info.cpu.has_avx512bw && info.cpu.has_avx512vl) {
            info.simd_level = SIMDLevel::AVX512;
        } else if (info.cpu.has_avx2 && info.cpu.has_fma) {
            info.simd_level = SIMDLevel::AVX2;
        } else if (info.cpu.has_avx) {
            info.simd_level = SIMDLevel::AVX;
        } else if (info.cpu.has_sse4_2) {
            info.simd_level = SIMDLevel::SSE4_2;
        } else if (info.cpu.has_sse2) {
            info.simd_level = SIMDLevel::SSE2;
        } else {
            info.simd_level = SIMDLevel::Scalar;
        }
        
        // Compile-time backend availability
#if defined(HAS_MKL)
        info.has_mkl = true;
#endif
#if defined(HAS_EIGEN)
        info.has_eigen = true;
#endif
#if defined(HAS_HNSWLIB)
        info.has_hnswlib = true;
#endif
#if defined(HAS_FAISS)
        info.has_faiss = true;
#endif
        
        // Select best eigensolver
        if (info.has_mkl) {
            info.eigensolver = EigensolverBackend::MKL;
        } else if (info.has_eigen) {
            info.eigensolver = EigensolverBackend::Eigen;
        } else {
            info.eigensolver = EigensolverBackend::Jacobi;
        }
        
        // Select best k-NN
        if (info.has_faiss) {
            info.knn = KNNBackend::FAISS;
        } else if (info.has_hnswlib) {
            info.knn = KNNBackend::HNSWLIB;
        } else {
            info.knn = KNNBackend::BruteForce;
        }
        
        // Build info
#if defined(_MSC_VER)
        info.compiler = "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
        info.compiler = "Clang " + std::to_string(__clang_major__) + "." + 
                        std::to_string(__clang_minor__);
#elif defined(__GNUC__)
        info.compiler = "GCC " + std::to_string(__GNUC__) + "." + 
                        std::to_string(__GNUC_MINOR__);
#else
        info.compiler = "Unknown";
#endif

#if defined(NDEBUG)
        info.build_type = "Release";
#else
        info.build_type = "Debug";
#endif
        
        return info;
    }
    
    // Get singleton instance (cached detection)
    static const BackendInfo& info() {
        static BackendInfo cached = detect();
        return cached;
    }
    
    // Convenience accessors
    static SIMDLevel simd_level() { return info().simd_level; }
    static EigensolverBackend eigensolver() { return info().eigensolver; }
    static KNNBackend knn() { return info().knn; }
    static bool has_avx512() { return info().cpu.has_avx512f; }
    static bool has_avx2() { return info().cpu.has_avx2; }
    static bool has_mkl() { return info().has_mkl; }
    static bool has_eigen() { return info().has_eigen; }
};

}  // namespace hypercube
