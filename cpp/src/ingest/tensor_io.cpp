/**
 * @file tensor_io.cpp
 * @brief Tensor row reading utilities for DB operations
 * 
 * Provides read_tensor_row function that reads individual rows from
 * memory-mapped safetensor files with dtype conversion.
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/ingest/safetensor.hpp"

namespace hypercube {
namespace ingest {
namespace db {

std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row) {
    if (meta.shape.size() < 2 || row >= static_cast<size_t>(meta.shape[0])) {
        return {};
    }
    
    const safetensor::MappedFile* mf = safetensor::MappedFileCache::instance().get(meta.shard_file);
    if (!mf || !mf->data()) return {};
    
    // Read header size from file
    uint64_t header_size;
    std::memcpy(&header_size, mf->data(), 8);
    
    // Calculate base offset (header size + 8-byte size field)
    size_t base_offset = 8 + header_size;
    
    // Calculate bytes per element and row size
    size_t elem_size = 0;
    if (meta.dtype == "F8_E4M3") elem_size = 1;
    else if (meta.dtype == "F16" || meta.dtype == "BF16") elem_size = 2;
    else if (meta.dtype == "F32") elem_size = 4;
    else if (meta.dtype == "F64" || meta.dtype == "I64") elem_size = 8;
    else return {};  // Unsupported dtype
    
    size_t cols = static_cast<size_t>(meta.shape[1]);
    size_t row_bytes = cols * elem_size;
    size_t row_offset = base_offset + meta.data_offset_start + row * row_bytes;
    
    if (row_offset + row_bytes > mf->size()) return {};
    
    const uint8_t* data = mf->data() + row_offset;
    
    std::vector<float> result(cols);
    
    if (meta.dtype == "F32") {
        std::memcpy(result.data(), data, cols * sizeof(float));
    } else if (meta.dtype == "F16") {
        // FP16 to float conversion
        for (size_t i = 0; i < cols; ++i) {
            uint16_t h;
            std::memcpy(&h, data + i * 2, 2);
            
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    f = sign << 31;
                } else {
                    // Denormalized
                    exp = 1;
                    while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            std::memcpy(&result[i], &f, 4);
        }
    } else if (meta.dtype == "BF16") {
        // BF16 to float (just shift left by 16)
        for (size_t i = 0; i < cols; ++i) {
            uint16_t bf;
            std::memcpy(&bf, data + i * 2, 2);
            uint32_t f = static_cast<uint32_t>(bf) << 16;
            std::memcpy(&result[i], &f, 4);
        }
    } else if (meta.dtype == "F64") {
        // F64 to float conversion
        for (size_t i = 0; i < cols; ++i) {
            double d;
            std::memcpy(&d, data + i * 8, 8);
            result[i] = static_cast<float>(d);
        }
    } else if (meta.dtype == "F8_E4M3") {
        // F8_E4M3FN to float conversion (E4M3 Finite, no infinities)
        // Format: 1 sign bit, 4 exponent bits (biased by 7), 3 mantissa bits
        // This is the "finite" variant used by modern ML frameworks (NVIDIA, DeepSeek, etc.)
        // Range: [-448, 448], no infinities, NaN when exp=15 and mant=7
        for (size_t i = 0; i < cols; ++i) {
            uint8_t f8 = data[i];

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
            std::memcpy(&result[i], &f, 4);
        }
    } else if (meta.dtype == "I64") {
        // I64 to float conversion
        for (size_t i = 0; i < cols; ++i) {
            int64_t val;
            std::memcpy(&val, data + i * 8, 8);
            result[i] = static_cast<float>(val);
        }
    }

    return result;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
