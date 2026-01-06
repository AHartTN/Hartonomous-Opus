#include "hypercube/hilbert.hpp"

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

// Gray code: i XOR (i >> 1)
constexpr uint32_t gray_encode(uint32_t i) noexcept {
    return i ^ (i >> 1);
}

// Inverse Gray code
constexpr uint32_t gray_decode(uint32_t g) noexcept {
    uint32_t i = g;
    i ^= (i >> 1);
    i ^= (i >> 2);
    i ^= (i >> 4);
    i ^= (i >> 8);
    i ^= (i >> 16);
    return i;
}

} // anonymous namespace

void HilbertCurve::transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from transpose (Hilbert) to axes (Cartesian)
    
    // Gray decode
    uint32_t t = x[n - 1] >> 1;
    for (uint32_t i = n - 1; i > 0; --i) {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;
    
    // Undo rotations - use bit counter to avoid UB with 1<<32
    for (uint32_t b = 1; b < bits; ++b) {
        uint32_t Q = 1U << b;
        uint32_t P = Q - 1;
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            if (x[i] & Q) {
                x[0] ^= P;
            } else {
                t = (x[0] ^ x[i]) & P;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
}

void HilbertCurve::axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from axes (Cartesian) to transpose (Hilbert)
    
    // Rotations - iterate from bits-1 down to 1 to avoid UB
    for (uint32_t b = bits - 1; b >= 1; --b) {
        uint32_t Q = 1U << b;
        uint32_t P = Q - 1;
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] & Q) {
                x[0] ^= P;
            } else {
                uint32_t t = (x[0] ^ x[i]) & P;
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
    for (uint32_t b = bits - 1; b >= 1; --b) {
        uint32_t Q = 1U << b;
        if (x[n - 1] & Q) t ^= (Q - 1);
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
    
    // Interleave the 4 transposed coordinates into 128-bit index
    // Bit i of dimension d goes to position i*4 + d
    HilbertIndex result{0, 0};
    
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
    if (a > b) return a - b;
    return b - a;
}

bool HilbertCurve::in_range(const HilbertIndex& center, const HilbertIndex& point,
                            const HilbertIndex& range) noexcept {
    return distance(center, point) <= range;
}

} // namespace hypercube
