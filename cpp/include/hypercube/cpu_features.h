#pragma once

#include <string>
#include <cstdint>

namespace hypercube {

namespace cpu_features {

struct CpuFeatures {
    std::string vendor;

    // SSE
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;

    // AVX
    bool avx;
    bool avx2;

    // AVX-512
    bool avx512f;
    bool avx512dq;
    bool avx512ifma;
    bool avx512pf;
    bool avx512er;
    bool avx512cd;
    bool avx512bw;
    bool avx512vl;
    bool avx512_vbmi;
    bool avx512_vbmi2;
    bool avx512_vnni;
    bool avx512_bitalg;
    bool avx512_vpopcntdq;
    bool avx512_4vnniw;
    bool avx512_4fmaps;
    bool avx512_bf16;

    // VNNI
    bool avx_vnni;

    // AMX
    bool amx_bf16;
    bool amx_tile;
    bool amx_int8;

    // Misc
    bool fma;
    bool fma4;
    bool xop;
    bool f16c;
    bool bmi;
    bool bmi2;
    bool lzcnt;
    bool popcnt;
    bool aes;
    bool pclmulqdq;
    bool rdrand;
    bool rdseed;
    bool sha;
    bool adx;
};

CpuFeatures detect_cpu_features();

} // namespace cpu_features

} // namespace hypercube