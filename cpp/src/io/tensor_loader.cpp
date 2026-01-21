#include "hypercube/io/tensor_loader.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hypercube {
namespace io {

std::vector<float> TensorLoader::load_tensor(const hypercube::safetensor::TensorMeta& meta) {
    size_t n = meta.element_count();
    std::vector<float> data(n);

    std::cerr << "[LOAD_TENSOR] Loading tensor '" << meta.name << "' from " << meta.shard_file << "\n";
    std::cerr << "[LOAD_TENSOR] Shape: [";
    for (size_t i = 0; i < meta.shape.size(); ++i) {
        std::cerr << meta.shape[i];
        if (i < meta.shape.size() - 1) std::cerr << ",";
    }
    std::cerr << "] " << meta.dtype << " (" << n << " elements, " << (n * 4) << " bytes)\n";
    std::cerr << "[LOAD_TENSOR] Seeking to offset: " << meta.data_offset_start << "\n";

    std::ifstream f(meta.shard_file, std::ios::binary);
    if (!f) {
        std::cerr << "[LOAD_TENSOR] ERROR: Failed to open file!\n";
        return {};
    }

    f.seekg(static_cast<std::streamoff>(meta.data_offset_start));
    if (f.fail()) {
        std::cerr << "[LOAD_TENSOR] ERROR: Seek failed! Offset may be invalid.\n";
        return {};
    }

    if (meta.dtype == "F32") {
        f.read(reinterpret_cast<char*>(data.data()), n * 4);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F32 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 4) << ")\n";
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for BF16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
            std::memcpy(&data[i], &bits, 4);
        }

        // VALIDATE: Check for extreme values indicating corruption
        size_t corrupt_count = 0;
        size_t nan_count = 0;
        #pragma omp parallel for reduction(+:corrupt_count,nan_count)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            float val = data[i];
            if (std::isnan(val)) {
                nan_count++;
                data[i] = 0.0f;  // Replace NaN with 0
            } else if (std::abs(val) > 1e10f) {  // Extreme values
                corrupt_count++;
                data[i] = 0.0f;  // Replace corrupt values with 0
            }
        }
        if (corrupt_count > 0 || nan_count > 0) {
            std::cerr << "[LOAD_TENSOR] WARNING: Fixed " << corrupt_count << " extreme values and "
                      << nan_count << " NaNs in BF16 data\n";
        }
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint16_t h = raw[i];
            uint32_t s = (h & 0x8000) << 16;
            uint32_t e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;
            uint32_t fval = (e == 0) ? s : (e == 31) ? (s | 0x7F800000 | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
            std::memcpy(&data[i], &fval, 4);
        }
    } else if (meta.dtype == "F8_E4M3") {
        std::vector<uint8_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 1);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F8_E4M3 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 1) << ")\n";
        
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint8_t f8 = raw[i];
            uint32_t sign = (f8 >> 7) & 0x1;
            uint32_t exp = (f8 >> 3) & 0xF;
            uint32_t mant = f8 & 0x7;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    // Zero
                    f = sign << 31;
                } else {
                    // Denormalized number - shift mantissa until leading 1
                    int shift = 0;
                    uint32_t m = mant;
                    while ((m & 0x4) == 0 && shift < 3) {
                        m <<= 1;
                        shift++;
                    }
                    m &= 0x3;  // Remove leading 1
                    // FP32 exponent for subnormal: (127 - 7 - shift)
                    f = (sign << 31) | ((121 - shift) << 23) | (m << 20);
                }
            } else if (exp == 15 && mant == 7) {
                // NaN (the only NaN encoding in E4M3FN)
                f = (sign << 31) | 0x7FC00000;  // Quiet NaN
            } else {
                // Normalized number (including exp==15 with mant!=7, which are valid finite values)
                // FP32 exponent = exp - 7 (E4M3 bias) + 127 (FP32 bias) = exp + 120
                f = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
            }
            std::memcpy(&data[i], &f, 4);
        }
    } else if (meta.dtype == "I64") {
        std::vector<int64_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 8);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for I64 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 8) << ")\n";
        
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            data[i] = static_cast<float>(raw[i]);
        }
    } else {
        std::cerr << "[LOAD_TENSOR] ERROR: Unknown dtype '" << meta.dtype << "'\n";
        return {};
    }

    // Verify non-zero data
    int nonzero = 0;
    for (size_t i = 0; i < std::min(size_t(100), n); ++i) {
        if (std::abs(data[i]) > 1e-10f) ++nonzero;
    }
    std::cerr << "[LOAD_TENSOR] Non-zero values in first 100 elements: " << nonzero << "/100\n";

    return data;
}

} // namespace io
} // namespace hypercube
