#include "hypercube/semantic_ordering.hpp"
#include "hypercube/types.hpp"
#include "hypercube/error.hpp"
#include <algorithm>
#include <map>
#include <unordered_set>
#include <array>

namespace hypercube {

// Unicode decomposition and normalization helpers
namespace {

// Unicode decomposition mappings for common characters
// This is a simplified implementation - in production, use ICU or similar
uint32_t decompose_to_base(uint32_t cp) noexcept {
    // ASCII characters are already decomposed
    if (cp <= 0x7F) return cp;

    // Latin Extended-A decompositions
    if (cp >= 0x00C0 && cp <= 0x00C5) return 'A'; // À-Å → A
    if (cp >= 0x00C8 && cp <= 0x00CB) return 'E'; // È-Ë → E
    if (cp >= 0x00CC && cp <= 0x00CF) return 'I'; // Ì-Ï → I
    if (cp >= 0x00D2 && cp <= 0x00D6) return 'O'; // Ò-Ö → O
    if (cp >= 0x00D9 && cp <= 0x00DC) return 'U'; // Ù-Ü → U
    if (cp >= 0x00E0 && cp <= 0x00E5) return 'a'; // à-å → a
    if (cp >= 0x00E8 && cp <= 0x00EB) return 'e'; // è-ë → e
    if (cp >= 0x00EC && cp <= 0x00EF) return 'i'; // ì-ï → i
    if (cp >= 0x00F2 && cp <= 0x00F6) return 'o'; // ò-ö → o
    if (cp >= 0x00F9 && cp <= 0x00FC) return 'u'; // ù-ü → u

    // Common homoglyphs
    if (cp == 'O' || cp == 'o' || cp == '0') return '0'; // Visual similarity group
    if (cp == 'I' || cp == 'i' || cp == 'l' || cp == '1') return '1'; // Visual similarity group
    if (cp == '6' || cp == 'G' || cp == 'g') return '6'; // Visual similarity group
    if (cp == '8' || cp == 'B') return '8'; // Visual similarity group
    if (cp == '9' || cp == 'q') return '9'; // Visual similarity group

    // Greek decompositions
    if (cp >= 0x0391 && cp <= 0x03A9) return cp - 0x0391 + 0x03B1; // Upper to lower Greek
    if (cp >= 0x0410 && cp <= 0x042F) return cp - 0x0410 + 0x0430; // Upper to lower Cyrillic

    return cp; // No decomposition available
}

// Case folding with special handling
uint32_t case_fold(uint32_t cp) noexcept {
    // ASCII
    if (cp >= 'A' && cp <= 'Z') return cp + 32;

    // Greek
    if (cp >= 0x0391 && cp <= 0x03A9) return cp + 32;

    // Cyrillic (basic)
    if (cp >= 0x0410 && cp <= 0x042F) return cp + 32;

    return cp;
}

// Enhanced homoglyph clustering
uint32_t get_homoglyph_group(uint32_t cp) noexcept {
    static const std::unordered_map<uint32_t, uint32_t> homoglyph_groups = {
        // 0-group: 0, O, o, Ø, ø, etc.
        {'0', 0}, {'O', 0}, {'o', 0}, {0x00D8, 0}, {0x00F8, 0},
        // 1-group: 1, I, i, l, |, etc.
        {'1', 1}, {'I', 1}, {'i', 1}, {'l', 1}, {'|', 1},
        // 2-group: 2, Z, z, etc.
        {'2', 2}, {'Z', 2}, {'z', 2},
        // 3-group: 3, etc.
        {'3', 3},
        // 4-group: 4, etc.
        {'4', 4},
        // 5-group: 5, S, s, etc.
        {'5', 5}, {'S', 5}, {'s', 5},
        // 6-group: 6, G, g, etc.
        {'6', 6}, {'G', 6}, {'g', 6},
        // 7-group: 7, etc.
        {'7', 7},
        // 8-group: 8, B, etc.
        {'8', 8}, {'B', 8}, {'b', 8},
        // 9-group: 9, q, g, etc.
        {'9', 9}, {'q', 9}, {'g', 9}
    };

    auto it = homoglyph_groups.find(cp);
    return (it != homoglyph_groups.end()) ? it->second : cp;
}

// Get combining class for diacritic ordering
uint8_t get_combining_class(uint32_t cp) noexcept {
    // Simplified combining class mapping
    if (cp >= 0x0300 && cp <= 0x036F) {
        // Unicode combining diacritical marks
        return 1; // Above
    }
    return 0; // Base character
}

} // anonymous namespace

// Static member definitions
std::unordered_map<uint32_t, uint32_t> SemanticOrdering::codepoint_to_rank_;
std::vector<uint32_t> SemanticOrdering::rank_to_codepoint_;
bool SemanticOrdering::initialized_ = false;
std::mutex SemanticOrdering::init_mutex_;

void SemanticOrdering::initialize() {
    if (initialized_) return;

    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) return; // Double-checked locking

    // Collect all valid codepoints with their semantic keys
    std::vector<std::pair<uint64_t, uint32_t>> semantic_pairs;

    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        // Include all codepoints including surrogates as valid atoms
        uint64_t key = compute_semantic_key(cp);
        semantic_pairs.emplace_back(key, cp);
    }

    // Sort by semantic key for dense ranking
    std::sort(semantic_pairs.begin(), semantic_pairs.end());

    // Assign dense ranks
    rank_to_codepoint_.resize(semantic_pairs.size());
    for (size_t i = 0; i < semantic_pairs.size(); ++i) {
        uint32_t cp = semantic_pairs[i].second;
        uint32_t rank = static_cast<uint32_t>(i);
        codepoint_to_rank_[cp] = rank;
        rank_to_codepoint_[i] = cp;
    }

    initialized_ = true;
}

uint32_t SemanticOrdering::get_rank(uint32_t codepoint) {
    if (!initialized_) initialize();
    auto it = codepoint_to_rank_.find(codepoint);
    return (it != codepoint_to_rank_.end()) ? it->second : 0;
}

uint32_t SemanticOrdering::get_codepoint(uint32_t rank) {
    if (!initialized_) initialize();
    return (rank < rank_to_codepoint_.size()) ? rank_to_codepoint_[rank] : 0;
}

uint32_t SemanticOrdering::total_codepoints() {
    if (!initialized_) initialize();
    return static_cast<uint32_t>(rank_to_codepoint_.size());
}

bool SemanticOrdering::is_valid(uint32_t codepoint) {
    if (!initialized_) initialize();
    return codepoint_to_rank_.find(codepoint) != codepoint_to_rank_.end();
}

uint64_t SemanticOrdering::get_semantic_key(uint32_t codepoint) noexcept {
    return compute_semantic_key(codepoint);
}

uint64_t SemanticOrdering::compute_semantic_key(uint32_t codepoint) noexcept {
    // Enhanced semantic key computation with Unicode decomposition and homoglyph clustering

    // Step 1: Unicode decomposition and normalization
    uint32_t base_char = decompose_to_base(codepoint);
    uint32_t case_folded = case_fold(base_char);
    uint32_t homoglyph_group = get_homoglyph_group(case_folded);

    // Step 2: Script identification (highest level grouping)
    uint8_t script_id = 100; // other scripts
    if (codepoint <= 0x024F) script_id = 0; // Latin + Extended
    else if (codepoint >= 0x0370 && codepoint <= 0x03FF) script_id = 1; // Greek
    else if (codepoint >= 0x0400 && codepoint <= 0x04FF) script_id = 2; // Cyrillic
    else if (codepoint >= 0x0590 && codepoint <= 0x05FF) script_id = 3; // Hebrew
    else if (codepoint >= 0x0600 && codepoint <= 0x077F) script_id = 4; // Arabic
    else if (codepoint >= 0x0900 && codepoint <= 0x097F) script_id = 5; // Devanagari
    else if (codepoint >= 0x2E80 && codepoint <= 0x9FFF) script_id = 6; // CJK
    else if (codepoint >= 0x1F600 && codepoint <= 0x1F64F) script_id = 7; // Emoji
    else if (codepoint >= 0x1F300 && codepoint <= 0x1F5FF) script_id = 8; // Symbols

    // Step 3: Semantic classification with enhanced clustering
    uint8_t semantic_class = 15; // other
    uint32_t base_similarity = homoglyph_group; // Use homoglyph group as base
    uint8_t variant_order = 0;

    // Case variant ordering: uppercase first, then lowercase, then accented
    bool is_upper = (codepoint >= 'A' && codepoint <= 'Z') ||
                   (codepoint >= 0x0391 && codepoint <= 0x03A9) ||
                   (codepoint >= 0x0410 && codepoint <= 0x042F);
    bool is_lower = (codepoint >= 'a' && codepoint <= 'z') ||
                   (codepoint >= 0x03B1 && codepoint <= 0x03C9) ||
                   (codepoint >= 0x0430 && codepoint <= 0x044F);
    bool is_accented = (codepoint >= 0x00C0 && codepoint <= 0x00FF) ||
                      (codepoint >= 0x0100 && codepoint <= 0x024F);

    if (is_upper) variant_order = 0;        // Uppercase first
    else if (is_lower) variant_order = 1;   // Lowercase second
    else if (is_accented) variant_order = 2; // Accented third
    else variant_order = 3;                  // Other variants

    // Enhanced character classification
    if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z') ||
        (codepoint >= 0x00C0 && codepoint <= 0x024F)) {
        semantic_class = 1; // Latin letters with homoglyph clustering
        base_similarity = homoglyph_group * 1000 + (base_char % 1000);
    }
    else if (codepoint >= '0' && codepoint <= '9') {
        semantic_class = 3; // Digits with visual similarity clustering
        base_similarity = homoglyph_group * 10000;
    }
    else if ((codepoint >= 0x21 && codepoint <= 0x2F) || (codepoint >= 0x3A && codepoint <= 0x40) ||
             (codepoint >= 0x5B && codepoint <= 0x60) || (codepoint >= 0x7B && codepoint <= 0x7E)) {
        semantic_class = 10; // ASCII punctuation
        base_similarity = 100000 + codepoint;
    }
    else if (codepoint >= 0x0391 && codepoint <= 0x03C9) { // Greek letters
        semantic_class = 4; // Greek with case clustering
        base_similarity = 200000 + homoglyph_group * 1000 + (base_char % 1000);
    }
    else if (codepoint >= 0x0410 && codepoint <= 0x044F) { // Cyrillic letters
        semantic_class = 5; // Cyrillic with case clustering
        base_similarity = 300000 + homoglyph_group * 1000 + (base_char % 1000);
    }
    else if (codepoint >= 0x1F600 && codepoint <= 0x1F64F) {
        semantic_class = 20; // Emoji faces
        base_similarity = 400000 + (codepoint - 0x1F600);
    }
    else if (codepoint >= 0x1F300 && codepoint <= 0x1F5FF) {
        semantic_class = 21; // Emoji symbols
        base_similarity = 500000 + (codepoint - 0x1F300);
    }
    else if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
        semantic_class = 30; // CJK unified
        base_similarity = 600000 + (codepoint - 0x4E00);
    }
    else if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
        semantic_class = 50; // Surrogates - special handling
        base_similarity = 700000 + (codepoint - 0xD800);
    }
    else {
        semantic_class = 99; // Other characters
        base_similarity = 800000 + codepoint % 100000;
    }

    // Step 4: Diacritic ordering (combining class)
    uint8_t combining_class = get_combining_class(codepoint);
    variant_order = (variant_order << 4) | (combining_class & 0xF);

    // Step 5: Pack into 64-bit key with enhanced bit allocation
    // Script (8) > Semantic Class (8) > Base Similarity (20) > Variant Order (8) > Uniqueness (20)
    uint64_t key = 0;
    key |= (uint64_t(script_id) & 0xFF) << 56;           // Bits 63-56: script
    key |= (uint64_t(semantic_class) & 0xFF) << 48;      // Bits 55-48: semantic class
    key |= (uint64_t(base_similarity) & 0xFFFFF) << 28;  // Bits 47-28: base similarity (20 bits)
    key |= (uint64_t(variant_order) & 0xFF) << 20;       // Bits 27-20: variant order
    key |= (uint64_t(codepoint) & 0xFFFFF);              // Bits 19-0: codepoint for uniqueness (20 bits)

    return key;
}

// Legacy function for backward compatibility
uint64_t get_semantic_key(uint32_t cp) noexcept {
    return SemanticOrdering::get_semantic_key(cp);
}

} // namespace hypercube</xai:function_call">The file cpp/src/core/semantic_ordering.cpp was created successfully.