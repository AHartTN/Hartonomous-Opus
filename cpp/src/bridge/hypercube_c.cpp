/**
 * hypercube_c.cpp - C API Implementation
 * 
 * Implements the C API declared in hypercube_c.h by wrapping the C++ core library.
 * This file is compiled as C++ but exports C-callable functions via extern "C".
 */

#include "hypercube_c.h"
#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

#include <cstring>
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace hypercube;

/* ============================================================================
 * Internal Conversion Helpers
 * ============================================================================ */

namespace {

inline Point4D to_cpp(hc_point4d_t p) {
    return Point4D(p.x, p.y, p.z, p.m);
}

inline hc_point4d_t to_c(const Point4D& p) {
    return hc_point4d_t{p.x, p.y, p.z, p.m};
}

inline HilbertIndex to_cpp(hc_hilbert_t h) {
    return HilbertIndex(h.lo, h.hi);
}

inline hc_hilbert_t to_c(const HilbertIndex& h) {
    return hc_hilbert_t{h.lo, h.hi};
}

inline Blake3Hash to_cpp(hc_hash_t h) {
    Blake3Hash result;
    std::memcpy(result.bytes.data(), h.bytes, 32);
    return result;
}

inline hc_hash_t to_c(const Blake3Hash& h) {
    hc_hash_t result;
    std::memcpy(result.bytes, h.bytes.data(), 32);
    return result;
}

inline AtomCategory to_cpp(hc_category_t c) {
    return static_cast<AtomCategory>(c);
}

inline hc_category_t to_c(AtomCategory c) {
    return static_cast<hc_category_t>(c);
}

} // anonymous namespace

/* ============================================================================
 * Hilbert Curve Functions
 * ============================================================================ */

extern "C" {

hc_hilbert_t hc_coords_to_hilbert(hc_point4d_t point) {
    return to_c(HilbertCurve::coords_to_index(to_cpp(point)));
}

hc_point4d_t hc_hilbert_to_coords(hc_hilbert_t index) {
    return to_c(HilbertCurve::index_to_coords(to_cpp(index)));
}

hc_hilbert_t hc_hilbert_distance(hc_hilbert_t a, hc_hilbert_t b) {
    return to_c(HilbertCurve::distance(to_cpp(a), to_cpp(b)));
}

int hc_hilbert_compare(hc_hilbert_t a, hc_hilbert_t b) {
    HilbertIndex ha = to_cpp(a);
    HilbertIndex hb = to_cpp(b);
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}

/* ============================================================================
 * Coordinate Mapping Functions
 * ============================================================================ */

hc_point4d_t hc_map_codepoint(uint32_t codepoint) {
    return to_c(CoordinateMapper::map_codepoint(codepoint));
}

hc_category_t hc_categorize(uint32_t codepoint) {
    return to_c(CoordinateMapper::categorize(codepoint));
}

const char* hc_category_name(hc_category_t cat) {
    return category_to_string(to_cpp(cat));
}

hc_point4d_t hc_centroid(const hc_point4d_t* points, size_t count) {
    if (count == 0 || points == nullptr) {
        return hc_point4d_t{0, 0, 0, 0};
    }
    
    std::vector<Point4D> cpp_points;
    cpp_points.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        cpp_points.push_back(to_cpp(points[i]));
    }
    
    return to_c(CoordinateMapper::centroid(cpp_points));
}

hc_point4d_t hc_weighted_centroid(const hc_point4d_t* points, 
                                   const double* weights, 
                                   size_t count) {
    if (count == 0 || points == nullptr || weights == nullptr) {
        return hc_point4d_t{0, 0, 0, 0};
    }
    
    std::vector<Point4D> cpp_points;
    std::vector<double> cpp_weights;
    cpp_points.reserve(count);
    cpp_weights.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        cpp_points.push_back(to_cpp(points[i]));
        cpp_weights.push_back(weights[i]);
    }
    
    return to_c(CoordinateMapper::weighted_centroid(cpp_points, cpp_weights));
}

bool hc_is_on_surface(hc_point4d_t point) {
    return CoordinateMapper::is_on_surface(to_cpp(point));
}

double hc_euclidean_distance(hc_point4d_t a, hc_point4d_t b) {
    return CoordinateMapper::euclidean_distance(to_cpp(a), to_cpp(b));
}

/* ============================================================================
 * BLAKE3 Hashing Functions
 * ============================================================================ */

hc_hash_t hc_blake3(const uint8_t* data, size_t len) {
    if (data == nullptr || len == 0) {
        hc_hash_t result;
        std::memset(result.bytes, 0, 32);
        return result;
    }
    return to_c(Blake3Hasher::hash(std::span<const uint8_t>(data, len)));
}

hc_hash_t hc_blake3_str(const char* str) {
    if (str == nullptr) {
        hc_hash_t result;
        std::memset(result.bytes, 0, 32);
        return result;
    }
    return to_c(Blake3Hasher::hash(std::string_view(str)));
}

hc_hash_t hc_blake3_codepoint(uint32_t codepoint) {
    return to_c(Blake3Hasher::hash_codepoint(codepoint));
}

hc_hash_t hc_blake3_children(const hc_hash_t* children, size_t count) {
    if (children == nullptr || count == 0) {
        hc_hash_t result;
        std::memset(result.bytes, 0, 32);
        return result;
    }
    
    std::vector<Blake3Hash> cpp_children;
    cpp_children.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        cpp_children.push_back(to_cpp(children[i]));
    }
    
    return to_c(Blake3Hasher::hash_children(cpp_children));
}

hc_hash_t hc_blake3_children_ordered(const hc_hash_t* children, size_t count) {
    if (children == nullptr || count == 0) {
        hc_hash_t result;
        std::memset(result.bytes, 0, 32);
        return result;
    }
    
    std::vector<Blake3Hash> cpp_children;
    cpp_children.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        cpp_children.push_back(to_cpp(children[i]));
    }
    
    return to_c(Blake3Hasher::hash_children_ordered(cpp_children));
}

void hc_hash_to_hex(hc_hash_t hash, char* out) {
    if (out == nullptr) return;
    
    static constexpr char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[i * 2] = hex_chars[hash.bytes[i] >> 4];
        out[i * 2 + 1] = hex_chars[hash.bytes[i] & 0x0F];
    }
    out[64] = '\0';
}

hc_hash_t hc_hash_from_hex(const char* hex) {
    hc_hash_t result;
    std::memset(result.bytes, 0, 32);
    
    if (hex == nullptr || std::strlen(hex) != 64) {
        return result;
    }
    
    auto hex_to_nibble = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + c - 'a';
        if (c >= 'A' && c <= 'F') return 10 + c - 'A';
        return 0;
    };
    
    for (int i = 0; i < 32; ++i) {
        result.bytes[i] = (hex_to_nibble(hex[i * 2]) << 4) | hex_to_nibble(hex[i * 2 + 1]);
    }
    
    return result;
}

int hc_hash_compare(hc_hash_t a, hc_hash_t b) {
    return std::memcmp(a.bytes, b.bytes, 32);
}

bool hc_hash_is_zero(hc_hash_t hash) {
    for (int i = 0; i < 32; ++i) {
        if (hash.bytes[i] != 0) return false;
    }
    return true;
}

/* ============================================================================
 * Batch Processing Functions
 * ============================================================================ */

void hc_map_codepoints_batch(const uint32_t* codepoints, 
                              size_t count, 
                              hc_point4d_t* out_points) {
    if (codepoints == nullptr || out_points == nullptr || count == 0) return;
    
    // Determine thread count based on hardware
    const size_t hw_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(hw_threads > 0 ? hw_threads : 4, count);
    
    if (count < 1000 || num_threads <= 1) {
        // Small batch - single threaded
        for (size_t i = 0; i < count; ++i) {
            out_points[i] = to_c(CoordinateMapper::map_codepoint(codepoints[i]));
        }
        return;
    }
    
    // Parallel processing
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    const size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        
        if (start >= count) break;
        
        threads.emplace_back([=]() {
            for (size_t i = start; i < end; ++i) {
                out_points[i] = to_c(CoordinateMapper::map_codepoint(codepoints[i]));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void hc_hash_codepoints_batch(const uint32_t* codepoints, 
                               size_t count, 
                               hc_hash_t* out_hashes) {
    if (codepoints == nullptr || out_hashes == nullptr || count == 0) return;
    
    const size_t hw_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(hw_threads > 0 ? hw_threads : 4, count);
    
    if (count < 1000 || num_threads <= 1) {
        for (size_t i = 0; i < count; ++i) {
            out_hashes[i] = to_c(Blake3Hasher::hash_codepoint(codepoints[i]));
        }
        return;
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    const size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        
        if (start >= count) break;
        
        threads.emplace_back([=]() {
            for (size_t i = start; i < end; ++i) {
                out_hashes[i] = to_c(Blake3Hasher::hash_codepoint(codepoints[i]));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void hc_coords_to_hilbert_batch(const hc_point4d_t* points,
                                 size_t count,
                                 hc_hilbert_t* out_indices) {
    if (points == nullptr || out_indices == nullptr || count == 0) return;
    
    const size_t hw_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(hw_threads > 0 ? hw_threads : 4, count);
    
    if (count < 1000 || num_threads <= 1) {
        for (size_t i = 0; i < count; ++i) {
            out_indices[i] = to_c(HilbertCurve::coords_to_index(to_cpp(points[i])));
        }
        return;
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    const size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        
        if (start >= count) break;
        
        threads.emplace_back([=]() {
            for (size_t i = start; i < end; ++i) {
                out_indices[i] = to_c(HilbertCurve::coords_to_index(to_cpp(points[i])));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

hc_atom_t hc_map_atom(uint32_t codepoint) {
    hc_atom_t atom;
    atom.codepoint = codepoint;
    
    Point4D coords = CoordinateMapper::map_codepoint(codepoint);
    atom.coords = to_c(coords);
    atom.hilbert = to_c(HilbertCurve::coords_to_index(coords));
    atom.hash = to_c(Blake3Hasher::hash_codepoint(codepoint));
    atom.category = to_c(CoordinateMapper::categorize(codepoint));
    
    return atom;
}

void hc_map_atoms_batch(const uint32_t* codepoints,
                         size_t count,
                         hc_atom_t* out_atoms) {
    if (codepoints == nullptr || out_atoms == nullptr || count == 0) return;
    
    const size_t hw_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(hw_threads > 0 ? hw_threads : 4, count);
    
    if (count < 1000 || num_threads <= 1) {
        for (size_t i = 0; i < count; ++i) {
            out_atoms[i] = hc_map_atom(codepoints[i]);
        }
        return;
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    const size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        
        if (start >= count) break;
        
        threads.emplace_back([=]() {
            for (size_t i = start; i < end; ++i) {
                out_atoms[i] = hc_map_atom(codepoints[i]);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

uint32_t hc_valid_codepoint_count(void) {
    return constants::MAX_CODEPOINT + 1 - (constants::SURROGATE_END - constants::SURROGATE_START + 1);
}

uint32_t hc_next_codepoint(uint32_t current) {
    if (current == UINT32_MAX) {
        // First call - return 0
        return 0;
    }
    
    uint32_t next = current + 1;
    
    // Skip surrogates
    if (next >= constants::SURROGATE_START && next <= constants::SURROGATE_END) {
        next = constants::SURROGATE_END + 1;
    }
    
    // Check if done
    if (next > constants::MAX_CODEPOINT) {
        return UINT32_MAX;
    }
    
    return next;
}

/* ============================================================================
 * Content Hash (CPE Cascade)
 * ============================================================================ */

hc_hash_t hc_content_hash_codepoints(const uint32_t* /* codepoints */, size_t count,
                                      const hc_hash_t* atom_hashes) {
    hc_hash_t result;
    std::memset(result.bytes, 0, 32);

    if (atom_hashes == nullptr || count == 0) {
        return result;
    }

    if (count == 1) {
        return atom_hashes[0];
    }

    // N-ary composition hash: BLAKE3(ord0 || hash0 || ord1 || hash1 || ... || ordN-1 || hashN-1)
    // This matches hash_children_ordered and cpe.cpp create_composition
    std::vector<uint8_t> input;
    input.reserve(count * 36);  // 4 bytes ordinal + 32 bytes hash per child

    for (size_t i = 0; i < count; ++i) {
        // Little-endian ordinal
        uint32_t ordinal = static_cast<uint32_t>(i);
        input.push_back(static_cast<uint8_t>(ordinal));
        input.push_back(static_cast<uint8_t>(ordinal >> 8));
        input.push_back(static_cast<uint8_t>(ordinal >> 16));
        input.push_back(static_cast<uint8_t>(ordinal >> 24));
        // Hash
        input.insert(input.end(), atom_hashes[i].bytes, atom_hashes[i].bytes + 32);
    }

    return hc_blake3(input.data(), input.size());
}

hc_hash_t hc_content_hash(const uint8_t* /* text */, size_t /* len */,
                           const hc_hash_t* atom_hashes, size_t atom_count) {
    // This function expects atom_hashes to already be in codepoint order
    // The caller must have decoded UTF-8 and looked up atom hashes
    return hc_content_hash_codepoints(nullptr, atom_count, atom_hashes);
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

void* hc_alloc(size_t size) {
    return std::malloc(size);
}

void hc_free(void* ptr) {
    std::free(ptr);
}

/* ============================================================================
 * Optimized Batch Operations (SIMD + Threading)
 * ============================================================================ */

void hc_batch_distances(const double* target, const double* points,
                        size_t count, double* distances_out) {
    if (target == nullptr || points == nullptr || distances_out == nullptr || count == 0) {
        return;
    }
    
#if defined(__AVX2__)
    // AVX2: Process 4 points at a time (each point has 4 coords)
    __m256d t = _mm256_loadu_pd(target);
    
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m256d p0 = _mm256_loadu_pd(&points[(i + 0) * 4]);
        __m256d p1 = _mm256_loadu_pd(&points[(i + 1) * 4]);
        __m256d p2 = _mm256_loadu_pd(&points[(i + 2) * 4]);
        __m256d p3 = _mm256_loadu_pd(&points[(i + 3) * 4]);
        
        __m256d d0 = _mm256_sub_pd(p0, t);
        __m256d d1 = _mm256_sub_pd(p1, t);
        __m256d d2 = _mm256_sub_pd(p2, t);
        __m256d d3 = _mm256_sub_pd(p3, t);
        
        d0 = _mm256_mul_pd(d0, d0);
        d1 = _mm256_mul_pd(d1, d1);
        d2 = _mm256_mul_pd(d2, d2);
        d3 = _mm256_mul_pd(d3, d3);
        
        __m256d s01 = _mm256_hadd_pd(d0, d1);
        __m256d s23 = _mm256_hadd_pd(d2, d3);
        
        __m256d lo = _mm256_permute2f128_pd(s01, s23, 0x20);
        __m256d hi = _mm256_permute2f128_pd(s01, s23, 0x31);
        __m256d sums = _mm256_add_pd(lo, hi);
        
        __m256d result = _mm256_sqrt_pd(sums);
        _mm256_storeu_pd(&distances_out[i], result);
    }
    
    // Handle remainder
    for (; i < count; ++i) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = points[i * 4 + d] - target[d];
            sum += diff * diff;
        }
        distances_out[i] = std::sqrt(sum);
    }
#else
    // Portable fallback
    for (size_t i = 0; i < count; ++i) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = points[i * 4 + d] - target[d];
            sum += diff * diff;
        }
        distances_out[i] = std::sqrt(sum);
    }
#endif
}

size_t hc_find_knn(const double* distances, size_t count, size_t k,
                   size_t* out_indices, double* out_distances) {
    if (distances == nullptr || out_indices == nullptr || out_distances == nullptr || count == 0) {
        return 0;
    }
    
    // Build index-distance pairs
    std::vector<std::pair<size_t, double>> pairs;
    pairs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pairs.push_back({i, distances[i]});
    }
    
    // Partial sort for k smallest
    size_t actual_k = std::min(k, count);
    std::partial_sort(pairs.begin(), pairs.begin() + actual_k, pairs.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Copy results
    for (size_t i = 0; i < actual_k; ++i) {
        out_indices[i] = pairs[i].first;
        out_distances[i] = pairs[i].second;
    }
    
    return actual_k;
}

double hc_frechet_distance(const double* traj1, size_t n1,
                           const double* traj2, size_t n2) {
    if (traj1 == nullptr || traj2 == nullptr || n1 == 0 || n2 == 0) {
        return 0.0;
    }
    
    // DP table
    std::vector<double> dp(n1 * n2, std::numeric_limits<double>::max());
    
    auto point_dist = [&](size_t i, size_t j) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = traj1[i * 4 + d] - traj2[j * 4 + d];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    };
    
    dp[0] = point_dist(0, 0);
    
    for (size_t j = 1; j < n2; ++j) {
        dp[j] = std::max(dp[j - 1], point_dist(0, j));
    }
    
    for (size_t i = 1; i < n1; ++i) {
        dp[i * n2] = std::max(dp[(i - 1) * n2], point_dist(i, 0));
    }
    
    for (size_t i = 1; i < n1; ++i) {
        for (size_t j = 1; j < n2; ++j) {
            double d = point_dist(i, j);
            double prev = std::min({
                dp[(i - 1) * n2 + j],
                dp[i * n2 + j - 1],
                dp[(i - 1) * n2 + j - 1]
            });
            dp[i * n2 + j] = std::max(d, prev);
        }
    }
    
    return dp[n1 * n2 - 1];
}

double hc_jaccard_similarity(const hc_hash_t* set1, size_t count1,
                             const hc_hash_t* set2, size_t count2) {
    if (set1 == nullptr || set2 == nullptr || count1 == 0 || count2 == 0) {
        return 0.0;
    }
    
    // Build hash set from first array
    std::unordered_set<std::string> s1;
    for (size_t i = 0; i < count1; ++i) {
        s1.insert(std::string(reinterpret_cast<const char*>(set1[i].bytes), 32));
    }
    
    // Count intersection
    size_t intersection = 0;
    for (size_t i = 0; i < count2; ++i) {
        if (s1.count(std::string(reinterpret_cast<const char*>(set2[i].bytes), 32))) {
            ++intersection;
        }
    }
    
    size_t union_size = count1 + count2 - intersection;
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

void hc_analogy_vector(const double* a, const double* b, 
                       const double* c, double* out_d) {
    if (a == nullptr || b == nullptr || c == nullptr || out_d == nullptr) {
        return;
    }
    
    // D = C + B - A
    for (int i = 0; i < 4; ++i) {
        out_d[i] = c[i] + b[i] - a[i];
    }
}

size_t hc_thread_count(void) {
    size_t count = std::thread::hardware_concurrency();
    return count > 0 ? count : 4;
}

} // extern "C"
