// =============================================================================
// tensor_loader.hpp - Consolidated Tensor Loading Functionality
// =============================================================================
// This header provides a TensorLoader class that consolidates tensor loading
// from SafeTensor files with support for lazy loading, memory mapping, and
// metadata extraction. Integrates with the existing safetensor infrastructure.
// =============================================================================

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "hypercube/ingest/safetensor.hpp"

namespace hypercube {
namespace util {

// =============================================================================
// TensorLoader - Consolidated tensor loading with lazy evaluation
// =============================================================================

class TensorLoader {
public:
    // Constructor with optional header offset (default 8 for safetensor)
    explicit TensorLoader(size_t header_offset = 8);
    ~TensorLoader() = default;

    // Non-copyable, movable
    TensorLoader(const TensorLoader&) = delete;
    TensorLoader& operator=(const TensorLoader&) = delete;
    TensorLoader(TensorLoader&&) noexcept = default;
    TensorLoader& operator=(TensorLoader&&) noexcept = default;

    // =========================================================================
    // Core Loading Methods
    // =========================================================================

    // Load tensor metadata from a safetensor file
    bool load_metadata(const std::filesystem::path& path);

    // Load tensor metadata from multiple shard files
    bool load_metadata_sharded(const std::vector<std::filesystem::path>& shard_paths);

    // Get metadata for a specific tensor (returns nullptr if not found)
    const safetensor::TensorMeta* get_tensor_meta(const std::string& name) const;

    // Get all tensor names
    std::vector<std::string> get_tensor_names() const;

    // Check if tensor exists
    bool has_tensor(const std::string& name) const;

    // =========================================================================
    // Lazy Loading Interface
    // =========================================================================

    // Template method to load entire tensor (lazy - loads only when called)
    template<typename T>
    std::vector<T> load_tensor(const std::string& name);

    // Template method to load tensor slice (2D tensors: row range)
    template<typename T>
    std::vector<T> load_tensor_slice(const std::string& name,
                                   size_t start_row, size_t end_row);

    // Template method to load single element
    template<typename T>
    T load_element(const std::string& name, size_t index);

    // Template method to load tensor row (for 2D tensors)
    template<typename T>
    std::vector<T> load_tensor_row(const std::string& name, size_t row);

    // =========================================================================
    // Memory Mapping Interface
    // =========================================================================

    // Enable/disable memory mapping (default: enabled)
    void set_memory_mapping_enabled(bool enabled);

    // Get memory-mapped file for a tensor's shard (for direct access)
    const safetensor::MappedFile* get_mapped_file(const std::string& tensor_name) const;

    // =========================================================================
    // Metadata Extraction
    // =========================================================================

    // Get tensor shape
    std::vector<int64_t> get_shape(const std::string& name) const;

    // Get tensor data type string
    std::string get_dtype(const std::string& name) const;

    // Get tensor element count
    size_t get_element_count(const std::string& name) const;

    // Get tensor byte size
    size_t get_byte_size(const std::string& name) const;

    // Get shard file path for tensor
    std::string get_shard_file(const std::string& name) const;

    // =========================================================================
    // Utility Methods
    // =========================================================================

    // Clear all loaded metadata
    void clear();

    // Get memory usage statistics
    struct MemoryStats {
        size_t metadata_count = 0;
        size_t mapped_files_count = 0;
        size_t total_mapped_bytes = 0;
    };
    MemoryStats get_memory_stats() const;

private:
    // =========================================================================
    // Private Implementation
    // =========================================================================

    size_t header_offset_;  // Header size offset (usually 8 for safetensor)
    bool memory_mapping_enabled_;  // Whether to use memory mapping

    // Tensor metadata storage
    std::unordered_map<std::string, safetensor::TensorMeta> tensors_;

    // Validate tensor exists and type compatibility
    template<typename T>
    bool validate_tensor_access(const std::string& name, const safetensor::TensorMeta*& meta) const;

    // Get expected dtype string for type T
    template<typename T>
    std::string get_expected_dtype() const;

    // Load raw bytes for a tensor range
    std::vector<uint8_t> load_raw_bytes(const safetensor::TensorMeta& meta,
                                      size_t byte_start, size_t byte_count);

    // Convert raw bytes to typed vector
    template<typename T>
    std::vector<T> convert_bytes_to_typed(const std::vector<uint8_t>& bytes,
                                        const safetensor::TensorMeta& meta);
};

// =============================================================================
// Template Specializations for Common Types
// =============================================================================

// =============================================================================
// Implementation of Non-Template Methods
// =============================================================================

inline TensorLoader::TensorLoader(size_t header_offset)
    : header_offset_(header_offset), memory_mapping_enabled_(true) {
}

inline bool TensorLoader::load_metadata(const std::filesystem::path& path) {
    tensors_.clear();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read 8-byte header size (little-endian)
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Read JSON header
    std::vector<char> buf(header_size);
    file.read(buf.data(), header_size);
    std::string json(buf.begin(), buf.end());

    // Simple parser: find each tensor entry
    size_t pos = 0;
    while ((pos = json.find("\"dtype\"", pos)) != std::string::npos) {
        size_t entry_start = json.rfind("{", pos);
        size_t name_end = json.rfind("\":", entry_start);
        size_t name_start = json.rfind("\"", name_end - 1);

        if (name_start == std::string::npos || name_end == std::string::npos) {
            pos++;
            continue;
        }

        std::string name = json.substr(name_start + 1, name_end - name_start - 1);

        if (name == "__metadata__" || name == "format") {
            pos++;
            continue;
        }

        safetensor::TensorMeta meta;
        meta.name = name;
        meta.shard_file = path.string();

        // Extract dtype
        size_t dtype_pos = json.find(":", pos) + 1;
        size_t dtype_q1 = json.find("\"", dtype_pos);
        size_t dtype_q2 = json.find("\"", dtype_q1 + 1);
        meta.dtype = json.substr(dtype_q1 + 1, dtype_q2 - dtype_q1 - 1);

        // Extract shape
        size_t shape_pos = json.find("\"shape\"", pos);
        size_t shape_start = json.find("[", shape_pos);
        size_t shape_end = json.find("]", shape_start);
        std::string shape_str = json.substr(shape_start + 1, shape_end - shape_start - 1);
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            meta.shape.push_back(std::stoll(dim));
        }

        // Extract data offsets
        size_t off_pos = json.find("\"data_offsets\"", pos);
        size_t off_start = json.find("[", off_pos);
        size_t off_end = json.find("]", off_start);
        std::string off_str = json.substr(off_start + 1, off_end - off_start - 1);
        size_t comma = off_str.find(",");
        meta.data_offset_start = std::stoull(off_str.substr(0, comma));
        meta.data_offset_end = std::stoull(off_str.substr(comma + 1));

        tensors_[name] = meta;
        pos = off_end;
    }

    return !tensors_.empty();
}

inline bool TensorLoader::load_metadata_sharded(const std::vector<std::filesystem::path>& shard_paths) {
    tensors_.clear();

    for (const auto& path : shard_paths) {
        if (!load_metadata(path)) {
            // For sharded loading, we continue even if one shard fails
            // But we should at least validate the path exists
            if (!std::filesystem::exists(path)) {
                return false;
            }
        }
    }

    return !tensors_.empty();
}

inline const safetensor::TensorMeta* TensorLoader::get_tensor_meta(const std::string& name) const {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

inline std::vector<std::string> TensorLoader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

inline bool TensorLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

inline void TensorLoader::set_memory_mapping_enabled(bool enabled) {
    memory_mapping_enabled_ = enabled;
}

inline const safetensor::MappedFile* TensorLoader::get_mapped_file(const std::string& tensor_name) const {
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) {
        return nullptr;
    }

    if (!memory_mapping_enabled_) {
        return nullptr;
    }

    return safetensor::MappedFileCache::instance().get(it->second.shard_file);
}

inline std::vector<int64_t> TensorLoader::get_shape(const std::string& name) const {
    auto meta = get_tensor_meta(name);
    return meta ? meta->shape : std::vector<int64_t>{};
}

inline std::string TensorLoader::get_dtype(const std::string& name) const {
    auto meta = get_tensor_meta(name);
    return meta ? meta->dtype : "";
}

inline size_t TensorLoader::get_element_count(const std::string& name) const {
    auto meta = get_tensor_meta(name);
    return meta ? meta->element_count() : 0;
}

inline size_t TensorLoader::get_byte_size(const std::string& name) const {
    auto meta = get_tensor_meta(name);
    return meta ? meta->byte_size() : 0;
}

inline std::string TensorLoader::get_shard_file(const std::string& name) const {
    auto meta = get_tensor_meta(name);
    return meta ? meta->shard_file : "";
}

inline void TensorLoader::clear() {
    tensors_.clear();
}

inline TensorLoader::MemoryStats TensorLoader::get_memory_stats() const {
    MemoryStats stats;
    stats.metadata_count = tensors_.size();

    std::unordered_set<std::string> unique_shards;
    for (const auto& pair : tensors_) {
        unique_shards.insert(pair.second.shard_file);
    }

    stats.mapped_files_count = unique_shards.size();

    // Note: We can't easily track total mapped bytes without querying the cache
    // This would require changes to MappedFileCache
    stats.total_mapped_bytes = 0;

    return stats;
}

inline std::vector<uint8_t> TensorLoader::load_raw_bytes(const safetensor::TensorMeta& meta,
                                                       size_t byte_start, size_t byte_count) {
    if (memory_mapping_enabled_) {
        // Use memory mapping
        auto mf = safetensor::MappedFileCache::instance().get(meta.shard_file);
        if (!mf) {
            return {};
        }

        size_t data_start = header_offset_ + meta.data_offset_start + byte_start;
        if (data_start + byte_count > mf->size()) {
            return {};
        }

        std::vector<uint8_t> result(byte_count);
        std::memcpy(result.data(), mf->data() + data_start, byte_count);
        return result;
    } else {
        // Use regular file I/O
        std::ifstream file(meta.shard_file, std::ios::binary);
        if (!file) {
            return {};
        }

        size_t file_offset = header_offset_ + meta.data_offset_start + byte_start;
        file.seekg(file_offset);

        std::vector<uint8_t> result(byte_count);
        file.read(reinterpret_cast<char*>(result.data()), byte_count);

        if (file.gcount() != static_cast<std::streamsize>(byte_count)) {
            return {};
        }

        return result;
    }
}

// =============================================================================
// Template Method Implementations
// =============================================================================

template<typename T>
std::vector<T> TensorLoader::load_tensor(const std::string& name) {
    const safetensor::TensorMeta* meta;
    if (!validate_tensor_access<T>(name, meta)) {
        return {};
    }

    size_t byte_size = meta->byte_size();
    if (byte_size == 0) {
        return {};
    }

    auto raw_bytes = load_raw_bytes(*meta, 0, byte_size);
    return convert_bytes_to_typed<T>(raw_bytes, *meta);
}

template<typename T>
std::vector<T> TensorLoader::load_tensor_slice(const std::string& name,
                                              size_t start_row, size_t end_row) {
    const safetensor::TensorMeta* meta;
    if (!validate_tensor_access<T>(name, meta)) {
        return {};
    }

    if (meta->shape.size() < 2 || start_row >= static_cast<size_t>(meta->shape[0]) ||
        end_row <= start_row || end_row > static_cast<size_t>(meta->shape[0])) {
        return {};
    }

    size_t elem_size = meta->element_size();
    size_t row_size_bytes = static_cast<size_t>(meta->shape[1]) * elem_size;
    size_t start_byte = start_row * row_size_bytes;
    size_t byte_count = (end_row - start_row) * row_size_bytes;

    auto raw_bytes = load_raw_bytes(*meta, start_byte, byte_count);
    return convert_bytes_to_typed<T>(raw_bytes, *meta);
}

template<typename T>
T TensorLoader::load_element(const std::string& name, size_t index) {
    const safetensor::TensorMeta* meta;
    if (!validate_tensor_access<T>(name, meta)) {
        return T{};
    }

    size_t elem_count = meta->element_count();
    if (index >= elem_count) {
        return T{};
    }

    size_t elem_size = meta->element_size();
    size_t byte_start = index * elem_size;
    size_t byte_count = elem_size;

    auto raw_bytes = load_raw_bytes(*meta, byte_start, byte_count);
    if (raw_bytes.size() != elem_size) {
        return T{};
    }

    if constexpr (std::is_same_v<T, float>) {
        if (meta->dtype == "BF16") {
            uint16_t val;
            std::memcpy(&val, raw_bytes.data(), 2);
            return safetensor::bf16_to_float(val);
        } else if (meta->dtype == "F16") {
            uint16_t val;
            std::memcpy(&val, raw_bytes.data(), 2);
            return safetensor::fp16_to_float(val);
        } else if (meta->dtype == "F32") {
            float val;
            std::memcpy(&val, raw_bytes.data(), 4);
            return val;
        } else if (meta->dtype == "F64") {
            double val;
            std::memcpy(&val, raw_bytes.data(), 8);
            return static_cast<float>(val);
        }
    } else {
        // Direct copy for other types
        T val;
        std::memcpy(&val, raw_bytes.data(), sizeof(T));
        return val;
    }

    return T{};
}

template<typename T>
std::vector<T> TensorLoader::load_tensor_row(const std::string& name, size_t row) {
    const safetensor::TensorMeta* meta;
    if (!validate_tensor_access<T>(name, meta)) {
        return {};
    }

    if (meta->shape.size() < 2 || row >= static_cast<size_t>(meta->shape[0])) {
        return {};
    }

    size_t elem_size = meta->element_size();
    size_t row_size_bytes = static_cast<size_t>(meta->shape[1]) * elem_size;
    size_t start_byte = row * row_size_bytes;

    auto raw_bytes = load_raw_bytes(*meta, start_byte, row_size_bytes);
    return convert_bytes_to_typed<T>(raw_bytes, *meta);
}

template<typename T>
bool TensorLoader::validate_tensor_access(const std::string& name,
                                        const safetensor::TensorMeta*& meta) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return false;
    }

    meta = &it->second;

    // Allow float access to half-precision types
    if constexpr (std::is_same_v<T, float>) {
        if (meta->dtype == "BF16" || meta->dtype == "F16" || meta->dtype == "F32" || meta->dtype == "F64") {
            return true;
        }
    } else {
        std::string expected_dtype = get_expected_dtype<T>();
        if (meta->dtype == expected_dtype) {
            return true;
        }
        // Special case for uint16_t which could be F16 or BF16
        if constexpr (std::is_same_v<T, uint16_t>) {
            if (meta->dtype == "F16" || meta->dtype == "BF16") {
                return true;
            }
        }
    }

    return false;
}

template<typename T>
std::string TensorLoader::get_expected_dtype() const {
    // Default implementation - specializations below
    return "UNKNOWN";
}

// Template specializations for get_expected_dtype
template<>
inline std::string TensorLoader::get_expected_dtype<float>() const {
    return "F32";
}

template<>
inline std::string TensorLoader::get_expected_dtype<double>() const {
    return "F64";
}

template<>
inline std::string TensorLoader::get_expected_dtype<int64_t>() const {
    return "I64";
}

template<>
inline std::string TensorLoader::get_expected_dtype<int32_t>() const {
    return "I32";
}

template<>
inline std::string TensorLoader::get_expected_dtype<uint8_t>() const {
    return "U8";
}

template<typename T>
std::vector<T> TensorLoader::convert_bytes_to_typed(const std::vector<uint8_t>& bytes,
                                                  const safetensor::TensorMeta& meta) {
    size_t elem_size = meta.element_size();
    size_t elem_count = bytes.size() / elem_size;

    std::vector<T> result(elem_count);

    if constexpr (std::is_same_v<T, float>) {
        // Handle float conversions from various precisions
        for (size_t i = 0; i < elem_count; ++i) {
            result[i] = safetensor::read_element(bytes.data(), i, meta.dtype);
        }
    } else {
        // Direct copy for other types
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }

    return result;
}

} // namespace util
} // namespace hypercube