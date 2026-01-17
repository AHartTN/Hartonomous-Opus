#pragma once

#include <string>
#include "cpu_features.h"

namespace hypercube {

enum class IsaClass {
    BASELINE,
    SSE42,
    AVX2,
    AVX2_VNNI,
    AVX512,
    AVX512_VNNI,
    AMX
};

IsaClass classify_isa(const cpu_features::CpuFeatures& features);

std::string isa_class_name(IsaClass isa);

} // namespace hypercube