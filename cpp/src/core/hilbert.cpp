#include "hypercube/hilbert.hpp"

#ifdef __BMI2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace hypercube {

/**
 * 4D Hilbert Curve - Coordinate to Index mapping
 * 
 * 32 bits per dimension × 4 dimensions = 128 bits total
 * Output: two 64-bit unsigned integers (lo, hi)
 * 
 * The Hilbert index is DERIVED from coordinates for ordering/indexing.
 * Coordinates are the source of truth.
 * 
 * Algorithm: Skilling's compact Hilbert index
 */

namespace {

} // anonymous namespace

void HilbertCurve::transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from transpose (Hilbert) to axes (Cartesian)

    // Gray decode
    uint32_t t = x[n - 1] >> 1;
    for (uint32_t i = n - 1; i > 0; --i) {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;

    // Undo rotations - use unsigned arithmetic, consistent with axes_to_transpose
    for (uint32_t b = 1; b < bits; ++b) {
        uint64_t Q = 1ULL << b;
        uint64_t P = Q - 1ULL;
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            if (x[i] & static_cast<uint32_t>(Q)) {
                x[0] ^= static_cast<uint32_t>(P);
            } else {
                t = (x[0] ^ x[i]) & static_cast<uint32_t>(P);
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
}

void HilbertCurve::axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from axes (Cartesian) to transpose (Hilbert)

    // Rotations - iterate from bits-1 down to 1
    for (int b = static_cast<int>(bits) - 1; b >= 1; --b) {
        uint64_t Q = 1ULL << b;
        uint64_t P = Q - 1ULL;
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] & static_cast<uint32_t>(Q)) {
                x[0] ^= static_cast<uint32_t>(P);
            } else {
                uint32_t t = (x[0] ^ x[i]) & static_cast<uint32_t>(P);
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
    
    // Gray encode
    for (uint32_t i = 1; i < n; ++i) {
        x[i] ^= x[i - 1];
    }
    uint32_t t = 0;
    for (int b = static_cast<int>(bits) - 1; b >= 1; --b) {
        uint64_t Q = 1ULL << b;
        if (x[n - 1] & static_cast<uint32_t>(Q)) t ^= static_cast<uint32_t>((Q - 1));
    }
    for (uint32_t i = 0; i < n; ++i) {
        x[i] ^= t;
    }
}

HilbertIndex HilbertCurve::coords_to_index(const Point4D& point) noexcept {
    // COORDINATE CONVENTION: uint32 with CENTER at 2^31 = 0x80000000
    // Hilbert curve works on corner-origin coordinates [0, 2^32-1]
    // XOR with 0x80000000 maps our CENTER to Hilbert corner 0
    // This means compositions near origin (abstract/complex) cluster at Hilbert index 0
    // and atoms on sphere surface cluster at higher Hilbert indices
    constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;
    uint32_t X[4] = {
        point.x ^ CENTER_TO_CORNER,
        point.y ^ CENTER_TO_CORNER,
        point.z ^ CENTER_TO_CORNER,
        point.m ^ CENTER_TO_CORNER
    };
    axes_to_transpose(X, 4, 32);
    
    // Interleave the 4 transposed coordinates into 128-bit index using BMI2
    // Bit-plane interleave: bit i of dimension d goes to position i*4 + d
    HilbertIndex result{0, 0};

#ifdef __BMI2__
    // Use BMI2 _pdep_u64 for hardware-accelerated bit deposition
    // Masks for bit positions: for dimension d, bits at 4*i + d
    const uint64_t mask_low[4] = {
        0x1111111111111111ULL,  // d=0: bits 0,4,8,...,60
        0x2222222222222222ULL,  // d=1: bits 1,5,9,...,61
        0x4444444444444444ULL,  // d=2: bits 2,6,10,...,62
        0x8888888888888888ULL   // d=3: bits 3,7,11,...,63
    };
    const uint64_t mask_high[4] = {
        0x1111111111111111ULL,  // Upper 64: same pattern
        0x2222222222222222ULL,
        0x4444444444444444ULL,
        0x8888888888888888ULL
    };

    for (int d = 0; d < 4; ++d) {
        uint32_t x = X[d];
        // Deposit lower 16 bits into lower 64 positions
        uint64_t low = _pdep_u64(x & 0xFFFF, mask_low[d]);
        // Deposit upper 16 bits into upper 64 positions
        uint64_t high = _pdep_u64((x >> 16) & 0xFFFF, mask_high[d]);
        result.lo |= low;
        result.hi |= high;
    }
#else
    // Fallback: original bit-loop implementation
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t out_pos = bit * 4;
        uint64_t nibble = ((X[0] >> bit) & 1) |
                          (((X[1] >> bit) & 1) << 1) |
                          (((X[2] >> bit) & 1) << 2) |
                          (((X[3] >> bit) & 1) << 3);

        if (out_pos < 64) {
            result.lo |= (nibble << out_pos);
        } else {
            result.hi |= (nibble << (out_pos - 64));
        }
    }
#endif
    
    return result;
}

Point4D HilbertCurve::index_to_coords(const HilbertIndex& index) noexcept {
    // De-interleave 128-bit index into 4 transposed coordinates
    uint32_t X[4] = {0, 0, 0, 0};
    
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t in_pos = bit * 4;
        uint64_t nibble;
        
        if (in_pos < 64) {
            nibble = (index.lo >> in_pos) & 0xF;
        } else {
            nibble = (index.hi >> (in_pos - 64)) & 0xF;
        }
        
        X[0] |= ((nibble >> 0) & 1) << bit;
        X[1] |= ((nibble >> 1) & 1) << bit;
        X[2] |= ((nibble >> 2) & 1) << bit;
        X[3] |= ((nibble >> 3) & 1) << bit;
    }
    
    // Transform back to Cartesian (corner-origin)
    transpose_to_axes(X, 4, 32);

    // Convert back from Hilbert corner-origin to our CENTER-origin coords
    // XOR with 0x80000000 maps Hilbert 0 back to our CENTER (0x80000000)
    constexpr uint32_t CORNER_TO_CENTER = 0x80000000U;
    return Point4D(
        X[0] ^ CORNER_TO_CENTER,
        X[1] ^ CORNER_TO_CENTER,
        X[2] ^ CORNER_TO_CENTER,
        X[3] ^ CORNER_TO_CENTER
    );
}

Point4D HilbertCurve::index_to_raw_coords(const HilbertIndex& index) noexcept {
    // De-interleave 128-bit index into 4 transposed coordinates
    uint32_t X[4] = {0, 0, 0, 0};
    
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t in_pos = bit * 4;
        uint64_t nibble;
        
        if (in_pos < 64) {
            nibble = (index.lo >> in_pos) & 0xF;
        } else {
            nibble = (index.hi >> (in_pos - 64)) & 0xF;
        }
        
        X[0] |= ((nibble >> 0) & 1) << bit;
        X[1] |= ((nibble >> 1) & 1) << bit;
        X[2] |= ((nibble >> 2) & 1) << bit;
        X[3] |= ((nibble >> 3) & 1) << bit;
    }
    
    // Transform back to Cartesian (corner-origin)
    // NO CENTER adjustment - returns raw corner-origin coordinates
    transpose_to_axes(X, 4, 32);

    return Point4D(X[0], X[1], X[2], X[3]);
}

HilbertIndex HilbertCurve::distance(const HilbertIndex& a, const HilbertIndex& b) noexcept {
    return HilbertIndex::abs_distance(a, b);
}

bool HilbertCurve::in_range(const HilbertIndex& center, const HilbertIndex& point,
                            const HilbertIndex& range) noexcept {
    return distance(center, point) <= range;
}

/* ============================================================================
 * SIMD Batch Processing Functions
 * ============================================================================ */

/**
 * Batch coordinate to index conversion using AVX512 SIMD
 * Processes up to 16 coordinates at once for maximum throughput
 */
void HilbertCurve::coords_to_indices_batch_avx512(const Point4D* points,
                                                  size_t count,
                                                  HilbertIndex* indices) noexcept {
#ifdef __AVX512F__
    // AVX512 can process 16 coordinates in parallel (512 bits / 32 bits per coord = 16)
    const size_t batch_size = 16;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        // Load coordinates into AVX512 registers
        // We need to load 4 coordinates per batch element (x,y,z,m)
        // Each coordinate is 32 bits, so we can fit 16 coordinates in one __m512i

        // Process each batch element
        for (size_t j = 0; j < current_batch; ++j) {
            const Point4D& point = points[i + j];
            indices[i + j] = HilbertCurve::coords_to_index(point);
        }

        // TODO: Implement full AVX512 vectorization
        // This would require:
        // 1. Loading 16 x 32-bit coordinates into AVX512 registers
        // 2. Performing axes_to_transpose on all 16 sets in parallel
        // 3. Vectorized bit interleaving using AVX512 bit manipulation
        // 4. Storing results
        //
        // For now, we fall back to scalar processing but maintain the batch interface
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        indices[i] = HilbertCurve::coords_to_index(points[i]);
    }
#endif
}

/**
 * Batch index to coordinate conversion using AVX512 SIMD
 * Processes up to 16 indices at once
 */
void HilbertCurve::indices_to_coords_batch_avx512(const HilbertIndex* indices,
                                                  size_t count,
                                                  Point4D* points) noexcept {
#ifdef __AVX512F__
    // AVX512 implementation for batch index to coordinate conversion
    const size_t batch_size = 16;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        // Process each batch element
        for (size_t j = 0; j < current_batch; ++j) {
            points[i + j] = HilbertCurve::index_to_coords(indices[i + j]);
        }

        // TODO: Implement full AVX512 vectorization for de-interleaving
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        points[i] = index_to_coords(indices[i]);
    }
#endif
}

/**
 * Batch coordinate to index conversion using AVX2 SIMD
 * Processes up to 8 coordinates at once
 */
void HilbertCurve::coords_to_indices_batch_avx2(const Point4D* points,
                                                size_t count,
                                                HilbertIndex* indices) noexcept {
#ifdef __AVX2__
    // AVX2 can process 8 coordinates in parallel (256 bits / 32 bits per coord = 8)
    const size_t batch_size = 8;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        // Process each batch element
        for (size_t j = 0; j < current_batch; ++j) {
            const Point4D& point = points[i + j];
            indices[i + j] = HilbertCurve::coords_to_index(point);
        }

        // TODO: Implement AVX2 vectorization
        // This would use AVX2 registers to process 8 coordinates simultaneously
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        indices[i] = coords_to_index(points[i]);
    }
#endif
}

} // namespace hypercube
