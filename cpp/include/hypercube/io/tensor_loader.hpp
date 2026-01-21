#pragma once

#include "hypercube/ingest/safetensor.hpp"
#include <vector>
#include <string>

namespace hypercube {
namespace io {

class TensorLoader {
public:
    /**
     * @brief Loads a tensor from disk and converts it to F32.
     * 
     * Handles various input formats:
     * - F32: Direct copy
     * - BF16: Converts to F32 (handling NaN/Inf)
     * - F16: Converts to F32 (handling subnormals)
     * - F8_E4M3: Converts to F32
     * - I64: Converts to F32
     * 
     * @param meta Metadata describing the tensor location and format.
     * @return std::vector<float> The loaded data as F32. Returns empty vector on error.
     */
    static std::vector<float> load_tensor(const hypercube::safetensor::TensorMeta& meta);
};

} // namespace io
} // namespace hypercube
