#pragma once

#include <cstdint>
#include <array>
#include <compare>
#include <span>
#include <string>
#include <string_view>
#include <cstring>

namespace hypercube {

// 32-bit unsigned coordinate per dimension
using Coord32 = uint32_t;

// 4D point with 32-bit coordinates
struct Point4D {
    Coord32 x, y, z, m;
    
    constexpr Point4D() noexcept : x(0), y(0), z(0), m(0) {}
    constexpr Point4D(Coord32 x_, Coord32 y_, Coord32 z_, Coord32 m_) noexcept
        : x(x_), y(y_), z(z_), m(m_) {}
    
    constexpr auto operator<=>(const Point4D&) const noexcept = default;
    
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
    // NOTE: Coordinates are stored as signed int32 bit-cast to uint32
    // The values represent [-1, 1] mapped to [INT32_MIN, INT32_MAX]
    constexpr bool is_on_surface() const noexcept {
        // Interpret uint32 bit patterns as signed int32 values
        auto as_signed = [](uint32_t v) -> double {
            int32_t signed_val;
            // Can't use memcpy in constexpr, use bit manipulation
            signed_val = static_cast<int32_t>(v);
            return static_cast<double>(signed_val) / static_cast<double>(INT32_MAX);
        };
        
        double ux = as_signed(x);
        double uy = as_signed(y);
        double uz = as_signed(z);
        double um = as_signed(m);
        double r_sq = ux*ux + uy*uy + uz*uz + um*um;
        // Allow tolerance for integer quantization
        return r_sq >= 0.9 && r_sq <= 1.1;
    }
};

// 128-bit Hilbert curve index (two 64-bit parts)
struct HilbertIndex {
    uint64_t lo;  // Lower 64 bits
    uint64_t hi;  // Upper 64 bits
    
    constexpr HilbertIndex() noexcept : lo(0), hi(0) {}
    constexpr HilbertIndex(uint64_t lo_, uint64_t hi_) noexcept : lo(lo_), hi(hi_) {}
    
    // Compare: hi first, then lo (big-endian order)
    constexpr auto operator<=>(const HilbertIndex& other) const noexcept {
        if (hi != other.hi) return hi <=> other.hi;
        return lo <=> other.lo;
    }
    constexpr bool operator==(const HilbertIndex& other) const noexcept {
        return hi == other.hi && lo == other.lo;
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
};

// BLAKE3 hash (32 bytes)
struct Blake3Hash {
    std::array<uint8_t, 32> bytes;
    
    constexpr Blake3Hash() noexcept : bytes{} {}
    
    // Copy from raw pointer
    explicit Blake3Hash(const uint8_t* data) noexcept {
        std::memcpy(bytes.data(), data, 32);
    }
    
    explicit Blake3Hash(std::span<const uint8_t> data) noexcept {
        if (data.size() >= 32) {
            std::memcpy(bytes.data(), data.data(), 32);
        }
    }
    
    constexpr auto operator<=>(const Blake3Hash&) const noexcept = default;
    
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
};

// Hash functor for Blake3Hash (for use in std::unordered_map)
struct Blake3HashHasher {
    size_t operator()(const Blake3Hash& h) const noexcept {
        // Use first 8 bytes as hash (already well-distributed)
        uint64_t result;
        std::memcpy(&result, h.bytes.data(), 8);
        return static_cast<size_t>(result);
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
    Point4D coords;
    HilbertIndex hilbert;
    Blake3Hash hash;
};

// Composition node in the Merkle DAG
struct Composition {
    Blake3Hash hash;
    Point4D centroid;
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
}

} // namespace hypercube
