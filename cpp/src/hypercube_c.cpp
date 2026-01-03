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

hc_hash_t hc_content_hash_codepoints(const uint32_t* codepoints, size_t count,
                                      const hc_hash_t* atom_hashes) {
    hc_hash_t result;
    std::memset(result.bytes, 0, 32);
    
    if (atom_hashes == nullptr || count == 0) {
        return result;
    }
    
    if (count == 1) {
        return atom_hashes[0];
    }
    
    // CPE cascade: binary tree merging
    std::vector<hc_hash_t> hashes(atom_hashes, atom_hashes + count);
    
    // Little-endian ordinal constants
    uint8_t ord0[4] = {0, 0, 0, 0};  // ordinal 0
    uint8_t ord1[4] = {1, 0, 0, 0};  // ordinal 1
    
    while (hashes.size() > 1) {
        std::vector<hc_hash_t> merged;
        merged.reserve((hashes.size() + 1) / 2);
        
        for (size_t i = 0; i < hashes.size(); i += 2) {
            if (i + 1 < hashes.size()) {
                // Merge pair: BLAKE3(ord0 || left || ord1 || right)
                uint8_t input[72];  // 4 + 32 + 4 + 32
                std::memcpy(input, ord0, 4);
                std::memcpy(input + 4, hashes[i].bytes, 32);
                std::memcpy(input + 36, ord1, 4);
                std::memcpy(input + 40, hashes[i + 1].bytes, 32);
                
                merged.push_back(hc_blake3(input, 72));
            } else {
                // Odd element - carry forward
                merged.push_back(hashes[i]);
            }
        }
        hashes = std::move(merged);
    }
    
    return hashes[0];
}

hc_hash_t hc_content_hash(const uint8_t* text, size_t len,
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

} // extern "C"
