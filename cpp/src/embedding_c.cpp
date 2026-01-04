/**
 * Embedding Operations C API Implementation
 * 
 * C++ implementation of the C bridge interface.
 * Wraps SIMD-accelerated embedding operations for PostgreSQL extensions.
 */

#include "hypercube/embedding_c.h"
#include "hypercube/embedding_ops.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <mutex>

using namespace hypercube::embedding;

// =============================================================================
// Embedding Cache (thread-safe singleton)
// =============================================================================

struct EmbeddingCacheEntry {
    std::vector<uint8_t> id;
    std::string label;
    std::vector<float> embedding;
};

struct EmbeddingCacheImpl {
    std::vector<EmbeddingCacheEntry> entries;
    std::vector<float> flat_embeddings;  // Contiguous for SIMD
    size_t dim = 0;
    bool flat_valid = false;
    mutable std::mutex mtx;
    
    void invalidate_flat() {
        flat_valid = false;
        flat_embeddings.clear();
    }
    
    void build_flat() {
        if (flat_valid) return;
        if (entries.empty() || dim == 0) {
            flat_embeddings.clear();
            flat_valid = true;
            return;
        }
        
        flat_embeddings.resize(entries.size() * dim);
        for (size_t i = 0; i < entries.size(); ++i) {
            std::memcpy(&flat_embeddings[i * dim], 
                       entries[i].embedding.data(), 
                       dim * sizeof(float));
        }
        flat_valid = true;
    }
};

static EmbeddingCacheImpl* g_cache = nullptr;

// =============================================================================
// Core Vector Operations
// =============================================================================

extern "C" {

EMBEDDING_C_API double embedding_c_cosine_similarity(
    const float* a,
    const float* b,
    size_t n
) {
    return cosine_similarity(a, b, n);
}

EMBEDDING_C_API double embedding_c_l2_distance(
    const float* a,
    const float* b,
    size_t n
) {
    return l2_distance(a, b, n);
}

EMBEDDING_C_API void embedding_c_vector_add(
    const float* a,
    const float* b,
    float* result,
    size_t n
) {
    vector_add(a, b, result, n);
}

EMBEDDING_C_API void embedding_c_vector_sub(
    const float* a,
    const float* b,
    float* result,
    size_t n
) {
    vector_sub(a, b, result, n);
}

EMBEDDING_C_API void embedding_c_analogy_target(
    const float* a,
    const float* b,
    const float* c,
    float* result,
    size_t n
) {
    analogy_target(a, b, c, result, n);
}

EMBEDDING_C_API size_t embedding_c_find_top_k(
    const float* query,
    const float* embeddings,
    size_t n_embeddings,
    size_t dim,
    size_t k,
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

// =============================================================================
// Cache Management
// =============================================================================

EMBEDDING_C_API int embedding_c_cache_init(void) {
    if (g_cache) {
        return 0;  // Already initialized
    }
    
    try {
        g_cache = new EmbeddingCacheImpl();
        return 0;
    } catch (...) {
        return -1;
    }
}

EMBEDDING_C_API void embedding_c_cache_clear(void) {
    if (!g_cache) return;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    g_cache->entries.clear();
    g_cache->flat_embeddings.clear();
    g_cache->dim = 0;
    g_cache->flat_valid = false;
}

EMBEDDING_C_API int64_t embedding_c_cache_add(
    const uint8_t* id,
    size_t id_len,
    const char* label,
    const float* embedding,
    size_t dim
) {
    if (!g_cache) {
        if (embedding_c_cache_init() != 0) {
            return -1;
        }
    }
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    // Set dimension on first add
    if (g_cache->dim == 0) {
        g_cache->dim = dim;
    } else if (g_cache->dim != dim) {
        return -1;  // Dimension mismatch
    }
    
    EmbeddingCacheEntry entry;
    entry.id.assign(id, id + id_len);
    entry.label = label;
    entry.embedding.assign(embedding, embedding + dim);
    
    int64_t idx = static_cast<int64_t>(g_cache->entries.size());
    g_cache->entries.push_back(std::move(entry));
    g_cache->invalidate_flat();
    
    return idx;
}

EMBEDDING_C_API size_t embedding_c_cache_count(void) {
    if (!g_cache) return 0;
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    return g_cache->entries.size();
}

EMBEDDING_C_API size_t embedding_c_cache_dim(void) {
    if (!g_cache) return 0;
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    return g_cache->dim;
}

EMBEDDING_C_API int64_t embedding_c_cache_find_label(const char* label) {
    if (!g_cache || !label) return -1;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    for (size_t i = 0; i < g_cache->entries.size(); ++i) {
        if (g_cache->entries[i].label == label) {
            return static_cast<int64_t>(i);
        }
    }
    
    return -1;
}

EMBEDDING_C_API const float* embedding_c_cache_get_embedding(size_t index) {
    if (!g_cache) return nullptr;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    if (index >= g_cache->entries.size()) {
        return nullptr;
    }
    
    return g_cache->entries[index].embedding.data();
}

EMBEDDING_C_API const char* embedding_c_cache_get_label(size_t index) {
    if (!g_cache) return nullptr;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    if (index >= g_cache->entries.size()) {
        return nullptr;
    }
    
    return g_cache->entries[index].label.c_str();
}

EMBEDDING_C_API size_t embedding_c_cache_similar(
    size_t query_index,
    size_t k,
    EmbeddingSimilarityResult* results
) {
    if (!g_cache || !results) return 0;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    if (query_index >= g_cache->entries.size()) {
        return 0;
    }
    
    // Build flat array if needed
    g_cache->build_flat();
    
    if (g_cache->flat_embeddings.empty()) {
        return 0;
    }
    
    const float* query = g_cache->entries[query_index].embedding.data();
    size_t n = g_cache->entries.size();
    size_t dim = g_cache->dim;
    
    // Find top k+1 (to exclude self)
    auto cpp_results = find_top_k_cosine(
        query, 
        g_cache->flat_embeddings.data(), 
        n, 
        dim, 
        k + 1
    );
    
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

EMBEDDING_C_API size_t embedding_c_cache_analogy(
    size_t idx_a,
    size_t idx_b,
    size_t idx_c,
    size_t k,
    EmbeddingSimilarityResult* results
) {
    if (!g_cache || !results) return 0;
    
    std::lock_guard<std::mutex> lock(g_cache->mtx);
    
    size_t n = g_cache->entries.size();
    if (idx_a >= n || idx_b >= n || idx_c >= n) {
        return 0;
    }
    
    size_t dim = g_cache->dim;
    
    // Compute target = c + (a - b)
    std::vector<float> target(dim);
    analogy_target(
        g_cache->entries[idx_a].embedding.data(),
        g_cache->entries[idx_b].embedding.data(),
        g_cache->entries[idx_c].embedding.data(),
        target.data(),
        dim
    );
    
    // Build flat array if needed
    g_cache->build_flat();
    
    // Find nearest to target
    auto cpp_results = find_top_k_cosine(
        target.data(),
        g_cache->flat_embeddings.data(),
        n,
        dim,
        k + 3  // Extra to remove inputs
    );
    
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

} // extern "C"
