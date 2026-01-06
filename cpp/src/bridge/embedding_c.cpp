/**
 * Embedding Operations C API Implementation
 * 
 * C++ implementation of the C bridge interface.
 * Wraps SIMD-accelerated embedding operations for PostgreSQL extensions.
 * 
 * Uses runtime SIMD dispatch via backend.hpp:
 *   - AVX-512 on Intel 12th gen+ (16 floats/op)
 *   - AVX2+FMA on modern CPUs (8 floats/op)
 *   - SSE4.2 fallback (4 floats/op)
 *   - Scalar for compatibility
 */

#include "hypercube/embedding_c.h"
#include "hypercube/embedding_ops.hpp"
#include "hypercube/backend.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <memory>

using namespace hypercube;
using namespace hypercube::embedding;

// =============================================================================
// Embedding Cache (thread-safe singleton)
// =============================================================================

struct EmbeddingCacheEntry {
    std::vector<uint8_t> id;
    std::string label;
    std::vector<float> embedding;
};

class EmbeddingCache {
public:
    static EmbeddingCache& instance() {
        static EmbeddingCache cache;
        return cache;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        entries_.clear();
        flat_embeddings_.clear();
        dim_ = 0;
        flat_valid_ = false;
    }
    
    int64_t add(const uint8_t* id, size_t id_len, const char* label,
                const float* embedding, size_t dim) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // Set dimension on first add
        if (dim_ == 0) {
            dim_ = dim;
        } else if (dim_ != dim) {
            return -1;  // Dimension mismatch
        }
        
        EmbeddingCacheEntry entry;
        entry.id.assign(id, id + id_len);
        entry.label = label ? label : "";
        entry.embedding.assign(embedding, embedding + dim);
        
        int64_t idx = static_cast<int64_t>(entries_.size());
        entries_.push_back(std::move(entry));
        invalidate_flat();
        
        return idx;
    }
    
    size_t count() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return entries_.size();
    }
    
    size_t dim() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dim_;
    }
    
    int64_t find_label(const char* label) const {
        if (!label) return -1;
        std::lock_guard<std::mutex> lock(mtx_);
        
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].label == label) {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
    }
    
    const float* get_embedding(size_t index) const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (index >= entries_.size()) return nullptr;
        return entries_[index].embedding.data();
    }
    
    const char* get_label(size_t index) const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (index >= entries_.size()) return nullptr;
        return entries_[index].label.c_str();
    }
    
    size_t find_similar(size_t query_index, size_t k, EmbeddingSimilarityResult* results) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        if (query_index >= entries_.size() || !results) {
            return 0;
        }
        
        build_flat();
        if (flat_embeddings_.empty()) return 0;
        
        const float* query = entries_[query_index].embedding.data();
        size_t n = entries_.size();
        
        // Find top k+1 (to exclude self)
        auto cpp_results = find_top_k_cosine(
            query, flat_embeddings_.data(), n, dim_, k + 1);
        
        // Remove self from results
        cpp_results.erase(
            std::remove_if(cpp_results.begin(), cpp_results.end(),
                [query_index](const SimilarityResult& r) { 
                    return r.index == query_index; 
                }),
            cpp_results.end()
        );
        
        size_t n_results = std::min(cpp_results.size(), k);
        for (size_t i = 0; i < n_results; ++i) {
            results[i].index = cpp_results[i].index;
            results[i].similarity = cpp_results[i].similarity;
        }
        
        return n_results;
    }
    
    size_t find_analogy(size_t idx_a, size_t idx_b, size_t idx_c,
                        size_t k, EmbeddingSimilarityResult* results) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        size_t n = entries_.size();
        if (idx_a >= n || idx_b >= n || idx_c >= n || !results) {
            return 0;
        }
        
        // Compute target = c + (a - b)
        std::vector<float> target(dim_);
        analogy_target(
            entries_[idx_a].embedding.data(),
            entries_[idx_b].embedding.data(),
            entries_[idx_c].embedding.data(),
            target.data(),
            dim_
        );
        
        build_flat();
        
        // Find nearest to target
        auto cpp_results = find_top_k_cosine(
            target.data(), flat_embeddings_.data(), n, dim_, k + 3);
        
        // Remove input words
        cpp_results.erase(
            std::remove_if(cpp_results.begin(), cpp_results.end(),
                [idx_a, idx_b, idx_c](const SimilarityResult& r) {
                    return r.index == idx_a || r.index == idx_b || r.index == idx_c;
                }),
            cpp_results.end()
        );
        
        size_t n_results = std::min(cpp_results.size(), k);
        for (size_t i = 0; i < n_results; ++i) {
            results[i].index = cpp_results[i].index;
            results[i].similarity = cpp_results[i].similarity;
        }
        
        return n_results;
    }

private:
    EmbeddingCache() = default;
    ~EmbeddingCache() = default;
    
    EmbeddingCache(const EmbeddingCache&) = delete;
    EmbeddingCache& operator=(const EmbeddingCache&) = delete;
    
    void invalidate_flat() {
        flat_valid_ = false;
        flat_embeddings_.clear();
    }
    
    void build_flat() {
        if (flat_valid_) return;
        if (entries_.empty() || dim_ == 0) {
            flat_embeddings_.clear();
            flat_valid_ = true;
            return;
        }
        
        flat_embeddings_.resize(entries_.size() * dim_);
        for (size_t i = 0; i < entries_.size(); ++i) {
            std::memcpy(&flat_embeddings_[i * dim_], 
                       entries_[i].embedding.data(), 
                       dim_ * sizeof(float));
        }
        flat_valid_ = true;
    }
    
    std::vector<EmbeddingCacheEntry> entries_;
    std::vector<float> flat_embeddings_;
    size_t dim_ = 0;
    bool flat_valid_ = false;
    mutable std::mutex mtx_;
};

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

// --- Core Vector Operations ---

EMBEDDING_C_API double embedding_c_cosine_similarity(
    const float* a, const float* b, size_t n
) {
    return cosine_similarity(a, b, n);
}

EMBEDDING_C_API double embedding_c_l2_distance(
    const float* a, const float* b, size_t n
) {
    return l2_distance(a, b, n);
}

EMBEDDING_C_API void embedding_c_vector_add(
    const float* a, const float* b, float* result, size_t n
) {
    vector_add(a, b, result, n);
}

EMBEDDING_C_API void embedding_c_vector_sub(
    const float* a, const float* b, float* result, size_t n
) {
    vector_sub(a, b, result, n);
}

EMBEDDING_C_API void embedding_c_analogy_target(
    const float* a, const float* b, const float* c, float* result, size_t n
) {
    analogy_target(a, b, c, result, n);
}

EMBEDDING_C_API size_t embedding_c_find_top_k(
    const float* query, const float* embeddings,
    size_t n_embeddings, size_t dim, size_t k,
    EmbeddingSimilarityResult* results
) {
    auto cpp_results = find_top_k_cosine(query, embeddings, n_embeddings, dim, k);
    
    size_t n_results = std::min(cpp_results.size(), k);
    for (size_t i = 0; i < n_results; ++i) {
        results[i].index = cpp_results[i].index;
        results[i].similarity = cpp_results[i].similarity;
    }
    
    return n_results;
}

// --- Cache Management ---

EMBEDDING_C_API int embedding_c_cache_init(void) {
    // Singleton auto-initializes, this is just for API compatibility
    (void)EmbeddingCache::instance();
    return 0;
}

EMBEDDING_C_API void embedding_c_cache_clear(void) {
    EmbeddingCache::instance().clear();
}

EMBEDDING_C_API int64_t embedding_c_cache_add(
    const uint8_t* id, size_t id_len,
    const char* label, const float* embedding, size_t dim
) {
    return EmbeddingCache::instance().add(id, id_len, label, embedding, dim);
}

EMBEDDING_C_API size_t embedding_c_cache_count(void) {
    return EmbeddingCache::instance().count();
}

EMBEDDING_C_API size_t embedding_c_cache_dim(void) {
    return EmbeddingCache::instance().dim();
}

EMBEDDING_C_API int64_t embedding_c_cache_find_label(const char* label) {
    return EmbeddingCache::instance().find_label(label);
}

EMBEDDING_C_API const float* embedding_c_cache_get_embedding(size_t index) {
    return EmbeddingCache::instance().get_embedding(index);
}

EMBEDDING_C_API const char* embedding_c_cache_get_label(size_t index) {
    return EmbeddingCache::instance().get_label(index);
}

EMBEDDING_C_API size_t embedding_c_cache_similar(
    size_t query_index, size_t k, EmbeddingSimilarityResult* results
) {
    return EmbeddingCache::instance().find_similar(query_index, k, results);
}

EMBEDDING_C_API size_t embedding_c_cache_analogy(
    size_t idx_a, size_t idx_b, size_t idx_c,
    size_t k, EmbeddingSimilarityResult* results
) {
    return EmbeddingCache::instance().find_analogy(idx_a, idx_b, idx_c, k, results);
}

// --- Backend Info ---

EMBEDDING_C_API const char* embedding_c_simd_level(void) {
    return active_simd_implementation();
}

EMBEDDING_C_API int embedding_c_simd_width(void) {
    return simd_width(Backend::simd_level());
}

} // extern "C"
