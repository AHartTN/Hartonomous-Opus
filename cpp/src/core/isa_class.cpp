#include "hypercube/isa_class.h"

namespace hypercube {

IsaClass classify_isa(const cpu_features::CpuFeatures& features) {
    if (features.amx_tile) return IsaClass::AMX;
    if (features.avx512f && features.avx512_vnni) return IsaClass::AVX512_VNNI;
    if (features.avx512f) return IsaClass::AVX512;
    if (features.avx2 && features.avx_vnni) return IsaClass::AVX2_VNNI;
    if (features.avx2) return IsaClass::AVX2;
    if (features.sse4_2) return IsaClass::SSE42;
    return IsaClass::BASELINE;
}

std::string isa_class_name(IsaClass isa) {
    switch (isa) {
        case IsaClass::BASELINE: return "BASELINE";
        case IsaClass::SSE42: return "SSE42";
        case IsaClass::AVX2: return "AVX2";
        case IsaClass::AVX2_VNNI: return "AVX2_VNNI";
        case IsaClass::AVX512: return "AVX512";
        case IsaClass::AVX512_VNNI: return "AVX512_VNNI";
        case IsaClass::AMX: return "AMX";
        default: return "UNKNOWN";
    }
}

} // namespace hypercube
