#include "hypercube/runtime_dispatch.h"
#include <mutex>

namespace hypercube {

namespace {
    std::unique_ptr<RuntimeDispatchEngine> global_engine;
    std::once_flag init_flag;
}

RuntimeDispatchEngine& RuntimeDispatchEngine::instance() {
    std::call_once(init_flag, []() {
        global_engine = std::make_unique<RuntimeDispatchEngine>();
        global_engine->initialize(cpu_features::detect_runtime_cpu_features());
    });
    return *global_engine;
}

void RuntimeDispatchEngine::initialize(const cpu_features::RuntimeCpuFeatures& features) {
    cpu_features_ = features;
    initialized_ = true;
}

KernelPreference RuntimeDispatchEngine::get_kernel_preference(KernelType type) const {
    if (!initialized_) {
        return KernelPreference{KernelCapability::BASELINE, KernelCapability::BASELINE};
    }

    switch (type) {
        case KernelType::DISTANCE_L2:
            return select_distance_l2();
        case KernelType::DISTANCE_IP:
            return select_distance_ip();
        case KernelType::GEMM_F32:
            return select_gemm_f32();
        case KernelType::DOT_PRODUCT_D:
            return select_dot_product_d();
        case KernelType::DOT_PRODUCT_F:
            return select_dot_product_f();
        case KernelType::SCALE_INPLACE_D:
            return select_scale_inplace_d();
        case KernelType::SUBTRACT_SCALED_D:
            return select_subtract_scaled_d();
        case KernelType::NORM_D:
            return select_norm_d();
        default:
            return KernelPreference{KernelCapability::BASELINE, KernelCapability::BASELINE};
    }
}

bool RuntimeDispatchEngine::is_capability_safe(KernelCapability capability) const {
    if (!initialized_) return capability == KernelCapability::BASELINE;

    switch (capability) {
        case KernelCapability::BASELINE:
            return true; // Always safe

        case KernelCapability::SSE42:
            return cpu_features_.sse4_2;

        case KernelCapability::AVX2:
            return cpu_features_.has_avx2();

        case KernelCapability::AVX2_FMA3:
            return cpu_features_.has_avx2() && cpu_features_.fma3;

        case KernelCapability::AVX2_VNNI:
            return cpu_features_.has_avx2() && cpu_features_.avx_vnni;

        case KernelCapability::AVX512F:
            return cpu_features_.has_avx512f();

        case KernelCapability::AVX512_BW:
            return cpu_features_.has_avx512f() && cpu_features_.avx512bw;

        case KernelCapability::AVX512_VNNI:
            return cpu_features_.has_avx512f() && cpu_features_.avx512vnni;

        case KernelCapability::AVX512_BF16:
            return cpu_features_.has_avx512f() && cpu_features_.avx512bf16;

        case KernelCapability::AMX:
            return cpu_features_.has_amx();

        default:
            return false;
    }
}

std::string RuntimeDispatchEngine::capability_name(KernelCapability capability) {
    switch (capability) {
        case KernelCapability::BASELINE: return "BASELINE";
        case KernelCapability::SSE42: return "SSE42";
        case KernelCapability::AVX2: return "AVX2";
        case KernelCapability::AVX2_FMA3: return "AVX2_FMA3";
        case KernelCapability::AVX2_VNNI: return "AVX2_VNNI";
        case KernelCapability::AVX512F: return "AVX512F";
        case KernelCapability::AVX512_BW: return "AVX512_BW";
        case KernelCapability::AVX512_VNNI: return "AVX512_VNNI";
        case KernelCapability::AVX512_BF16: return "AVX512_BF16";
        case KernelCapability::AMX: return "AMX";
        default: return "UNKNOWN";
    }
}

KernelPreference RuntimeDispatchEngine::select_distance_l2() const {
    // Priority: AMX > AVX512_VNNI > AVX512F > AVX2_VNNI > AVX2 > SSE42 > BASELINE

    if (cpu_features_.has_amx()) {
        return {KernelCapability::AMX, KernelCapability::AVX512_VNNI};
    }

    if (cpu_features_.has_avx512f() && cpu_features_.avx512vnni) {
        return {KernelCapability::AVX512_VNNI, KernelCapability::AVX512F};
    }

    if (cpu_features_.has_avx512f()) {
        return {KernelCapability::AVX512F, KernelCapability::AVX2};
    }

    if (cpu_features_.has_avx2() && cpu_features_.avx_vnni) {
        return {KernelCapability::AVX2_VNNI, KernelCapability::AVX2};
    }

    if (cpu_features_.has_avx2()) {
        return {KernelCapability::AVX2, KernelCapability::SSE42};
    }

    if (cpu_features_.sse4_2) {
        return {KernelCapability::SSE42, KernelCapability::BASELINE};
    }

    return {KernelCapability::BASELINE, KernelCapability::BASELINE};
}

KernelPreference RuntimeDispatchEngine::select_distance_ip() const {
    // Same priority as L2 distance
    return select_distance_l2();
}

KernelPreference RuntimeDispatchEngine::select_gemm_f32() const {
    // GEMM benefits from FMA and higher vector widths
    // Priority: AVX512F > AVX2_FMA3 > AVX2 > SSE42 > BASELINE

    if (cpu_features_.has_avx512f()) {
        return {KernelCapability::AVX512F, KernelCapability::AVX2_FMA3};
    }

    if (cpu_features_.has_avx2() && cpu_features_.fma3) {
        return {KernelCapability::AVX2_FMA3, KernelCapability::AVX2};
    }

    if (cpu_features_.has_avx2()) {
        return {KernelCapability::AVX2, KernelCapability::SSE42};
    }

    if (cpu_features_.sse4_2) {
        return {KernelCapability::SSE42, KernelCapability::BASELINE};
    }

    return {KernelCapability::BASELINE, KernelCapability::BASELINE};
}

KernelPreference RuntimeDispatchEngine::select_dot_product_d() const {
    // Double precision dot product - prefer AVX512 for 512-bit vectors
    if (cpu_features_.has_avx512f()) {
        return {KernelCapability::AVX512F, KernelCapability::AVX2};
    }

    if (cpu_features_.has_avx2()) {
        return {KernelCapability::AVX2, KernelCapability::SSE42};
    }

    if (cpu_features_.sse4_2) {
        return {KernelCapability::SSE42, KernelCapability::BASELINE};
    }

    return {KernelCapability::BASELINE, KernelCapability::BASELINE};
}

KernelPreference RuntimeDispatchEngine::select_dot_product_f() const {
    // Single precision - can benefit from VNNI for quantized operations
    if (cpu_features_.has_avx512f() && cpu_features_.avx512vnni) {
        return {KernelCapability::AVX512_VNNI, KernelCapability::AVX512F};
    }

    if (cpu_features_.has_avx2() && cpu_features_.avx_vnni) {
        return {KernelCapability::AVX2_VNNI, KernelCapability::AVX2};
    }

    return select_dot_product_d(); // Same logic as double precision for base capabilities
}

KernelPreference RuntimeDispatchEngine::select_scale_inplace_d() const {
    // Scaling operations benefit from vectorization
    return select_dot_product_d();
}

KernelPreference RuntimeDispatchEngine::select_subtract_scaled_d() const {
    // axpy-like operations benefit from FMA
    if (cpu_features_.has_avx512f()) {
        return {KernelCapability::AVX512F, KernelCapability::AVX2_FMA3};
    }

    if (cpu_features_.has_avx2() && cpu_features_.fma3) {
        return {KernelCapability::AVX2_FMA3, KernelCapability::AVX2};
    }

    return select_dot_product_d();
}

KernelPreference RuntimeDispatchEngine::select_norm_d() const {
    // Norm operations can benefit from VNNI for squared accumulation
    if (cpu_features_.has_avx512f() && cpu_features_.avx512vnni) {
        return {KernelCapability::AVX512_VNNI, KernelCapability::AVX512F};
    }

    if (cpu_features_.has_avx2() && cpu_features_.avx_vnni) {
        return {KernelCapability::AVX2_VNNI, KernelCapability::AVX2};
    }

    return select_dot_product_d();
}

FallbackDispatcher& get_fallback_dispatcher() {
    static FallbackDispatcher dispatcher(RuntimeDispatchEngine::instance());
    return dispatcher;
}

} // namespace hypercube