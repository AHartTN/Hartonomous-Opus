#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <string_view>
#include <cstring>
#include <cmath>

namespace hypercube {

// 32-bit unsigned coordinate per dimension
using Coord32 = uint32_t;

// 4D point with 32-bit coordinates
struct Point4D {
    Coord32 x, y, z, m;
    
    constexpr Point4D() noexcept : x(0), y(0), z(0), m(0) {}
    constexpr Point4D(Coord32 x_, Coord32 y_, Coord32 z_, Coord32 m_) noexcept
        : x(x_), y(y_), z(z_), m(m_) {}
    
    // Comparison operators defined below
    
    // DEPRECATED: Do not use for storage - raw uint32 values are stored directly
    // These were for old normalized [0,1] storage which loses precision
    // Kept for compatibility with validation code only
    constexpr double x_normalized() const noexcept { 
        return static_cast<double>(x) / static_cast<double>(UINT32_MAX); 
    }
    constexpr double y_normalized() const noexcept { 
        return static_cast<double>(y) / static_cast<double>(UINT32_MAX); 
    }
    constexpr double z_normalized() const noexcept { 
        return static_cast<double>(z) / static_cast<double>(UINT32_MAX); 
    }
    constexpr double m_normalized() const noexcept { 
        return static_cast<double>(m) / static_cast<double>(UINT32_MAX); 
    }
    
    // Convert to raw doubles for PostGIS POINTZM storage
    // PostGIS double has 53-bit mantissa, more than enough for 32-bit values
    constexpr double x_raw() const noexcept { return static_cast<double>(x); }
    constexpr double y_raw() const noexcept { return static_cast<double>(y); }
    constexpr double z_raw() const noexcept { return static_cast<double>(z); }
    constexpr double m_raw() const noexcept { return static_cast<double>(m); }
    
    // Check if on 3-sphere surface (r² ≈ 1 within quantization tolerance)
    // For atoms, all should be on surface; compositions are interior
    // COORDINATE CONVENTION: uint32 where [-1, 1] maps to [0, UINT32_MAX]
    // Formula: u = (v + 1.0) * 0.5 * UINT32_MAX, so v = 2 * u / UINT32_MAX - 1.0
    constexpr bool is_on_surface() const noexcept {
        // Convert uint32 coords to unit sphere [-1, 1] using same formula as Point4F constructor
        // This must match the quantization formula in coordinates.cpp
        constexpr double SCALE = 1.0 / static_cast<double>(UINT32_MAX);

        double ux = (static_cast<double>(x) * SCALE - 0.5) * 2.0;
        double uy = (static_cast<double>(y) * SCALE - 0.5) * 2.0;
        double uz = (static_cast<double>(z) * SCALE - 0.5) * 2.0;
        double um = (static_cast<double>(m) * SCALE - 0.5) * 2.0;
        double r_sq = ux*ux + uy*uy + uz*uz + um*um;

        // Strict tolerance for surface points (atoms should be exactly on S³)
        // Allow small tolerance for quantization errors: |r² - 1| < 0.01
        return r_sq >= 0.99 && r_sq <= 1.01;
    }
};

// Comparison operators for Point4D
constexpr bool operator==(const Point4D& lhs, const Point4D& rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.m == rhs.m;
}

constexpr bool operator!=(const Point4D& lhs, const Point4D& rhs) noexcept {
    return !(lhs == rhs);
}

constexpr bool operator<(const Point4D& lhs, const Point4D& rhs) noexcept {
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    if (lhs.z != rhs.z) return lhs.z < rhs.z;
    return lhs.m < rhs.m;
}

// Floating-point point on 4D unit sphere for optimization
struct Point4F {
    double x, y, z, m;

    constexpr Point4F() noexcept : x(0.0), y(0.0), z(0.0), m(0.0) {}
    constexpr Point4F(double x_, double y_, double z_, double m_) noexcept
        : x(x_), y(y_), z(z_), m(m_) {}

    // Convert from quantized Point4D to floating-point [-1, 1]
    // CRITICAL: Must normalize after dequantization to preserve S³ surface constraint
    // Research: "Optimal Quantization on Spheres" shows proper sphere quantization requires:
    //   1. Quantize: float → int
    //   2. Dequantize: int → float
    //   3. Normalize: project back to sphere surface
    // Without normalization, quantization rounding errors (~1-5%) accumulate and points
    // drift off the unit sphere, causing surface constraint violations.
    explicit Point4F(const Point4D& p) noexcept {
        constexpr double SCALE = 1.0 / static_cast<double>(UINT32_MAX);
        x = (static_cast<double>(p.x) * SCALE - 0.5) * 2.0;
        y = (static_cast<double>(p.y) * SCALE - 0.5) * 2.0;
        z = (static_cast<double>(p.z) * SCALE - 0.5) * 2.0;
        m = (static_cast<double>(p.m) * SCALE - 0.5) * 2.0;

        // Normalize back to unit sphere to correct quantization rounding errors
        double norm = std::sqrt(x*x + y*y + z*z + m*m);
        if (norm > 0.0) {
            double inv_norm = 1.0 / norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;
            m *= inv_norm;
        } else {
            // Degenerate case (should never happen for valid atoms)
            x = 1.0; y = 0.0; z = 0.0; m = 0.0;
        }
    }

    // Convert to quantized Point4D
    Point4D to_quantized() const noexcept {
        auto quantize = [](double v) -> Coord32 {
            if (v <= -1.0) return 0;
            if (v >= 1.0) return UINT32_MAX;
            double scaled = (v + 1.0) * 0.5 * static_cast<double>(UINT32_MAX);
            uint64_t rounded = static_cast<uint64_t>(std::floor(scaled + 0.5));
            return static_cast<Coord32>(rounded > static_cast<uint64_t>(UINT32_MAX) ? UINT32_MAX : rounded);
        };
        return Point4D(quantize(x), quantize(y), quantize(z), quantize(m));
    }

    // Normalize to unit sphere
    Point4F normalized() const noexcept {
        double norm = std::sqrt(x*x + y*y + z*z + m*m);
        if (norm > 0.0) {
            return Point4F(x/norm, y/norm, z/norm, m/norm);
        }
        return Point4F(1.0, 0.0, 0.0, 0.0); // Default to (1,0,0,0)
    }

    // Dot product
    double dot(const Point4F& other) const noexcept {
        return x*other.x + y*other.y + z*other.z + m*other.m;
    }

    // Euclidean distance
    double distance(const Point4F& other) const noexcept {
        double dx = x - other.x, dy = y - other.y, dz = z - other.z, dm = m - other.m;
        return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
    }

    // Geodesic distance on S^3 (arccos of dot product)
    double geodesic_distance(const Point4F& other) const noexcept {
        double cos_theta = dot(other);
        // Clamp to [-1, 1] for numerical stability
        if (cos_theta < -1.0) cos_theta = -1.0;
        else if (cos_theta > 1.0) cos_theta = 1.0;
        return std::acos(cos_theta);
    }

    // Addition, subtraction and scalar multiplication for updates
    Point4F operator+(const Point4F& other) const noexcept {
        return Point4F(x + other.x, y + other.y, z + other.z, m + other.m);
    }

    Point4F operator-(const Point4F& other) const noexcept {
        return Point4F(x - other.x, y - other.y, z - other.z, m - other.m);
    }

    Point4F operator*(double s) const noexcept {
        return Point4F(x * s, y * s, z * s, m * s);
    }

    // Euclidean norm
    double norm() const noexcept {
        return std::sqrt(x*x + y*y + z*z + m*m);
    }
};

// 128-bit Hilbert curve index (two 64-bit parts)
//
// SPATIAL INDEXING ONLY - NOT IDENTIFICATION!
//
// PURPOSE: Enables O(log N) nearest neighbor queries via B-tree/R-tree ordering
//
// PROPERTIES:
// - LOSSLESS: Deterministically computed from coordinates
// - DETERMINISTIC: Same coordinates → same Hilbert index (before collision resolution)
// - SPATIALLY ORDERED: Preserves locality for efficient range queries
// - NOT UNIQUE: Multiple coordinates can map to same index due to quantization
//
// COLLISION CAUSES:
// 1. Quantization: Float coordinates rounded to 32-bit integers
// 2. Jitter: Deterministic perturbations to resolve spatial conflicts
// 3. Finite precision: 32 bits per dimension limits uniqueness
//
// CORRECT USAGE:
// ✅ Spatial indexing: ORDER BY hilbert_hi, hilbert_lo
// ✅ Nearest neighbors: Range queries on Hilbert ordering
// ✅ B-tree/R-tree: Efficient spatial query acceleration
// ❌ Unique identifiers: Use Blake3Hash instead
// ❌ Primary keys: Use Blake3Hash instead
// ❌ Equality for identity: Use Blake3Hash instead
//
// Hilbert indices provide spatial locality for fast queries,
// but Blake3Hash provides the unique content-derived identifier.
//
struct HilbertIndex {
    uint64_t lo;  // Lower 64 bits
    uint64_t hi;  // Upper 64 bits

    constexpr HilbertIndex() noexcept : lo(0), hi(0) {}
    constexpr HilbertIndex(uint64_t lo_, uint64_t hi_) noexcept : lo(lo_), hi(hi_) {}

    // Compare: hi first, then lo (big-endian order)
    constexpr bool operator==(const HilbertIndex& other) const noexcept {
        return hi == other.hi && lo == other.lo;
    }
    constexpr bool operator!=(const HilbertIndex& other) const noexcept {
        return !(*this == other);
    }
    constexpr bool operator<(const HilbertIndex& other) const noexcept {
        if (hi != other.hi) return hi < other.hi;
        return lo < other.lo;
    }
    constexpr bool operator<=(const HilbertIndex& other) const noexcept {
        return *this == other || *this < other;
    }
    constexpr bool operator>(const HilbertIndex& other) const noexcept {
        return !(*this <= other);
    }
    constexpr bool operator>=(const HilbertIndex& other) const noexcept {
        return !(*this < other);
    }

    // Increment for iteration
    constexpr HilbertIndex& operator++() noexcept {
        if (++lo == 0) ++hi;
        return *this;
    }

    // Arithmetic for distance calculations
    constexpr HilbertIndex operator-(const HilbertIndex& other) const noexcept {
        HilbertIndex result;
        result.lo = lo - other.lo;
        result.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
        return result;
    }

    constexpr HilbertIndex operator+(const HilbertIndex& other) const noexcept {
        HilbertIndex result;
        result.lo = lo + other.lo;
        result.hi = hi + other.hi + (result.lo < lo ? 1 : 0);
        return result;
    }

    static HilbertIndex abs_distance(const HilbertIndex& a, const HilbertIndex& b) noexcept {
        return (a < b) ? (b - a) : (a - b);
    }
};

// BLAKE3 hash (32 bytes)
// NOTE: 256 bits is overkill for 4D coordinate space (~128 bits max needed).
// Full hash maintained for cryptographic content addressing (Merkle DAG).
// Spatial operations can use truncated_64() for performance.
struct Blake3Hash {
    std::array<uint8_t, 32> bytes;

    constexpr Blake3Hash() noexcept : bytes{} {}

    // Copy from raw pointer
    explicit Blake3Hash(const uint8_t* data) noexcept {
        std::memcpy(bytes.data(), data, 32);
    }

    explicit Blake3Hash(const uint8_t* data, size_t size) noexcept {
        if (size >= 32) {
            std::memcpy(bytes.data(), data, 32);
        }
    }

    bool operator==(const Blake3Hash& other) const noexcept {
        return bytes == other.bytes;
    }

    bool operator!=(const Blake3Hash& other) const noexcept {
        return !(*this == other);
    }

    bool operator<(const Blake3Hash& other) const noexcept {
        return bytes < other.bytes;
    }

    bool operator<=(const Blake3Hash& other) const noexcept {
        return bytes <= other.bytes;
    }

    bool operator>(const Blake3Hash& other) const noexcept {
        return bytes > other.bytes;
    }

    bool operator>=(const Blake3Hash& other) const noexcept {
        return bytes >= other.bytes;
    }

    // Hex string representation
    std::string to_hex() const {
        static constexpr char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (uint8_t b : bytes) {
            result.push_back(hex_chars[b >> 4]);
            result.push_back(hex_chars[b & 0x0F]);
        }
        return result;
    }

    static Blake3Hash from_hex(std::string_view hex) {
        Blake3Hash result;
        if (hex.size() != 64) return result;

        for (size_t i = 0; i < 32; ++i) {
            auto hex_to_nibble = [](char c) -> uint8_t {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + c - 'a';
                if (c >= 'A' && c <= 'F') return 10 + c - 'A';
                return 0;
            };
            result.bytes[i] = (hex_to_nibble(hex[i*2]) << 4) | hex_to_nibble(hex[i*2+1]);
        }
        return result;
    }

    // Raw bytes access
    constexpr const uint8_t* data() const noexcept { return bytes.data(); }
    constexpr uint8_t* data() noexcept { return bytes.data(); }
    static constexpr size_t size() noexcept { return 32; }

    // Check if zero (uninitialized)
    constexpr bool is_zero() const noexcept {
        for (uint8_t b : bytes) if (b != 0) return false;
        return true;
    }

    // Truncated hash for performance-critical operations (spatial indexing)
    // 64-bit hash sufficient for ~1.1M Unicode codepoints with low collision risk
    uint64_t truncated_64() const noexcept {
        uint64_t result;
        std::memcpy(&result, bytes.data(), 8);
        return result;
    }

    // 128-bit hash for higher collision resistance
    __int128 truncated_128() const noexcept {
        __int128 result;
        std::memcpy(&result, bytes.data(), 16);
        return result;
    }

    // CORRECTION: Hilbert indices CANNOT be used as unique identifiers
    // Multiple coordinates quantize to same Hilbert index due to finite precision
    // Blake3 hash provides the unique content-derived identifier
    // Hilbert index is for SPATIAL INDEXING only, not identification
};

// Hash functor for Blake3Hash (for use in std::unordered_map)
struct Blake3HashHasher {
    size_t operator()(const Blake3Hash& h) const noexcept {
        // Use truncated 64-bit hash for fast lookups
        return static_cast<size_t>(h.truncated_64());
    }
};

// Unicode codepoint category for semantic clustering
enum class AtomCategory : uint8_t {
    Control = 0,
    Format,
    PrivateUse,
    Surrogate,
    Noncharacter,
    Space,
    PunctuationOpen,
    PunctuationClose,
    PunctuationOther,
    Digit,
    NumberLetter,
    MathSymbol,
    Currency,
    Modifier,
    LetterUpper,
    LetterLower,
    LetterTitlecase,
    LetterModifier,
    LetterOther,
    MarkNonspacing,
    MarkSpacing,
    MarkEnclosing,
    SymbolOther,
    Separator,

    COUNT  // For iteration
};

// Convert category to SQL enum string
constexpr const char* category_to_string(AtomCategory cat) noexcept {
    switch (cat) {
        case AtomCategory::Control: return "control";
        case AtomCategory::Format: return "format";
        case AtomCategory::PrivateUse: return "private_use";
        case AtomCategory::Surrogate: return "surrogate";
        case AtomCategory::Noncharacter: return "noncharacter";
        case AtomCategory::Space: return "space";
        case AtomCategory::PunctuationOpen: return "punctuation_open";
        case AtomCategory::PunctuationClose: return "punctuation_close";
        case AtomCategory::PunctuationOther: return "punctuation_other";
        case AtomCategory::Digit: return "digit";
        case AtomCategory::NumberLetter: return "number_letter";
        case AtomCategory::MathSymbol: return "math_symbol";
        case AtomCategory::Currency: return "currency";
        case AtomCategory::Modifier: return "modifier";
        case AtomCategory::LetterUpper: return "letter_upper";
        case AtomCategory::LetterLower: return "letter_lower";
        case AtomCategory::LetterTitlecase: return "letter_titlecase";
        case AtomCategory::LetterModifier: return "letter_modifier";
        case AtomCategory::LetterOther: return "letter_other";
        case AtomCategory::MarkNonspacing: return "mark_nonspacing";
        case AtomCategory::MarkSpacing: return "mark_spacing";
        case AtomCategory::MarkEnclosing: return "mark_enclosing";
        case AtomCategory::SymbolOther: return "symbol_other";
        case AtomCategory::Separator: return "separator";
        default: return "symbol_other";
    }
}

// Unicode atom with full metadata
struct UnicodeAtom {
    uint32_t codepoint;
    AtomCategory category;
    Point4F coords_float;  // Canonical float coordinates on S³ surface
    Point4D coords;        // Quantized coordinates for indexing
    HilbertIndex hilbert;
    Blake3Hash hash;
};

// Composition node in the Merkle DAG
struct Composition {
    Blake3Hash hash;
    Point4F centroid_float;  // Canonical float centroid on S³ surface
    Point4D centroid;        // Quantized centroid for indexing
    HilbertIndex hilbert;
    uint32_t depth;
    uint32_t child_count;
    uint64_t atom_count;
};

// Semantic edge (weighted relationship between nodes)
// Used for: embedding similarities, co-occurrence, etc.
struct SemanticEdge {
    Blake3Hash source;
    Blake3Hash target;
    double weight;  // Relationship strength (cosine similarity, frequency, etc.)

    SemanticEdge() = default;
    SemanticEdge(const Blake3Hash& src, const Blake3Hash& tgt, double w)
        : source(src), target(tgt), weight(w) {}
};

// Constants
namespace constants {
    // Hypercube dimensions
    constexpr uint32_t DIMENSIONS = 4;
    constexpr uint32_t BITS_PER_DIM = 32;
    constexpr uint32_t TOTAL_BITS = DIMENSIONS * BITS_PER_DIM;  // 128 bits

    // Coordinate space origin (center of uint32 range)
    // This allows both positive and negative semantic directions
    constexpr Coord32 COORD_ORIGIN = UINT32_MAX / 2;  // 2^31 = 2147483648
    constexpr Coord32 COORD_RADIUS = COORD_ORIGIN - 1; // Max deviation from center

    // Unicode ranges
    constexpr uint32_t MAX_CODEPOINT = 0x10FFFF;
    constexpr uint32_t BMP_END = 0xFFFF;
    constexpr uint32_t SURROGATE_START = 0xD800;
    constexpr uint32_t SURROGATE_END = 0xDFFF;

    // Hypercube surface: points where at least one coordinate is 0 or MAX
    constexpr Coord32 SURFACE_MIN = 0;
    constexpr Coord32 SURFACE_MAX = UINT32_MAX;

    // Number of valid Unicode codepoints (excluding surrogates)
    constexpr uint32_t VALID_CODEPOINTS = MAX_CODEPOINT + 1 - (SURROGATE_END - SURROGATE_START + 1);

    // Semantic coordinate bounds
    constexpr double SEMANTIC_MIN = -1.0;  // Most negative semantic direction
    constexpr double SEMANTIC_MAX = 1.0;   // Most positive semantic direction
    constexpr double SEMANTIC_CENTER = 0.0; // Origin/neutral semantic position
}

} // namespace hypercube
