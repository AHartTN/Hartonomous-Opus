// =============================================================================
// safetensor.hpp - Unified Safetensor I/O Types
// =============================================================================
// This header consolidates types that were previously duplicated across:
// - ingest_safetensor.cpp
// - ingest_safetensor_4d.cpp
// - ingest_safetensor_universal.cpp
//
// ALL safetensor ingestion tools should use these shared types.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hypercube {
namespace safetensor {

// =============================================================================
// IngestConfig - Common configuration for all ingestion tools
// =============================================================================

struct IngestConfig {
    std::string conninfo;         // PostgreSQL connection string
    std::string model_name;       // Model identifier (e.g., "llama4-maverick")
    float weight_threshold = 0.5f; // Sparse edge threshold
    bool verbose = false;
    int batch_size = 10000;       // DB batch insert size
    
    // Per-type thresholds for embedding relations
    float token_threshold = 0.45f;
    float patch_threshold = 0.25f;
    float position_threshold = 0.02f;
    float projection_threshold = 0.15f;
};

// =============================================================================
// TensorMeta - Metadata for a tensor within a safetensor file
// =============================================================================

struct TensorMeta {
    std::string name;              // Full tensor name (e.g., "model.layers.0.attn.weight")
    std::string dtype;             // Data type: "BF16", "F16", "F32", "I64"
    std::vector<int64_t> shape;    // Tensor dimensions
    uint64_t data_offset_start;    // Byte offset in file (after header)
    uint64_t data_offset_end;      // End offset
    std::string shard_file;        // Path to containing .safetensors file
    int shard_index = 0;           // For multi-shard models
    
    // Computed properties
    size_t element_count() const {
        size_t count = 1;
        for (auto dim : shape) count *= static_cast<size_t>(dim);
        return count;
    }
    
    size_t byte_size() const {
        return data_offset_end - data_offset_start;
    }
    
    size_t element_size() const {
        if (dtype == "BF16" || dtype == "F16") return 2;
        if (dtype == "F32") return 4;
        if (dtype == "F64" || dtype == "I64") return 8;
        return 4; // Default to F32
    }
};

// =============================================================================
// MappedFile - Cross-platform memory-mapped file (zero-copy tensor access)
// =============================================================================

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { unmap(); }
    
    // Non-copyable, movable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            unmap();
            data_ = other.data_;
            size_ = other.size_;
            path_ = std::move(other.path_);
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            mapping_handle_ = other.mapping_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_ = nullptr;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    bool map(const std::string& path) {
        if (data_) unmap();
        path_ = path;
        
#ifdef _WIN32
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) return false;
        
        LARGE_INTEGER size_li;
        GetFileSizeEx(file_handle_, &size_li);
        size_ = static_cast<size_t>(size_li.QuadPart);
        
        mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        data_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        
        data_ = static_cast<const uint8_t*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            return false;
        }
#endif
        return true;
    }
    
    void unmap() {
        if (!data_) return;
#ifdef _WIN32
        UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        mapping_handle_ = nullptr;
#else
        munmap(const_cast<uint8_t*>(data_), size_);
#endif
        data_ = nullptr;
        size_ = 0;
    }
    
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }
    bool is_mapped() const { return data_ != nullptr; }
    
private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    std::string path_;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#endif
};

// =============================================================================
// MappedFileCache - Thread-safe cache of memory-mapped files
// =============================================================================

class MappedFileCache {
public:
    static MappedFileCache& instance() {
        static MappedFileCache cache;
        return cache;
    }

    // Set maximum cache size to prevent file handle exhaustion
    void set_max_size(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_cache_size_ = max_size;
        // If we're over the limit, clear oldest entries
        if (cache_.size() > max_cache_size_) {
            // Simple FIFO: remove oldest entries
            size_t to_remove = cache_.size() - max_cache_size_;
            auto it = cache_.begin();
            for (size_t i = 0; i < to_remove && it != cache_.end(); ++i, ++it) {
                // Evict oldest entries
            }
            cache_.erase(cache_.begin(), it);
        }
    }

    const MappedFile* get(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(path);
        if (it != cache_.end()) {
            return it->second.get();
        }

        // Check cache size limit before adding new file
        if (cache_.size() >= max_cache_size_ && max_cache_size_ > 0) {
            // Evict oldest entry (simple FIFO)
            auto oldest_it = cache_.begin();
            cache_.erase(oldest_it);
        }

        auto mf = std::make_unique<MappedFile>();
        if (!mf->map(path)) {
            return nullptr;
        }

        auto ptr = mf.get();
        cache_[path] = std::move(mf);
        return ptr;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }
    
private:
    MappedFileCache() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<MappedFile>> cache_;
    size_t max_cache_size_ = 50;  // Default limit to prevent file handle exhaustion
};

// =============================================================================
// Tensor Data Access Helpers
// =============================================================================

// Convert BF16 (bfloat16) to float
inline float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Convert F16 (IEEE 754 half precision) to float
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }
        // Subnormal: normalize
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Read a single float from tensor data at given element index
inline float read_element(const uint8_t* data, size_t index, const std::string& dtype) {
    if (dtype == "BF16") {
        uint16_t val;
        std::memcpy(&val, data + index * 2, 2);
        return bf16_to_float(val);
    } else if (dtype == "F16") {
        uint16_t val;
        std::memcpy(&val, data + index * 2, 2);
        return fp16_to_float(val);
    } else if (dtype == "F32") {
        float val;
        std::memcpy(&val, data + index * 4, 4);
        return val;
    } else if (dtype == "F64") {
        double val;
        std::memcpy(&val, data + index * 8, 8);
        return static_cast<float>(val);
    }
    return 0.0f;
}

// Read a row of floats from a 2D tensor
inline std::vector<float> read_tensor_row(
    const MappedFile* mf,
    const TensorMeta& meta,
    size_t row,
    size_t header_offset = 8  // Default safetensor header size field
) {
    if (meta.shape.size() < 2 || row >= static_cast<size_t>(meta.shape[0])) {
        return {};
    }
    
    size_t row_size = static_cast<size_t>(meta.shape[1]);
    std::vector<float> result(row_size);
    
    size_t data_start = header_offset + meta.data_offset_start;
    size_t elem_size = meta.element_size();
    const uint8_t* row_data = mf->data() + data_start + row * row_size * elem_size;
    
    for (size_t i = 0; i < row_size; ++i) {
        result[i] = read_element(row_data, i, meta.dtype);
    }
    
    return result;
}

} // namespace safetensor
} // namespace hypercube
