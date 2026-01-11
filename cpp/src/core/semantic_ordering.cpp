#include "hypercube/semantic_ordering.hpp"
#include "hypercube/types.hpp"
#include "hypercube/error.hpp"
#include <algorithm>
#include <map>

namespace hypercube {

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
        // Skip surrogates
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;

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

uint64_t SemanticOrdering::get_semantic_key(uint32_t codepoint) {
    return compute_semantic_key(codepoint);
}

uint64_t SemanticOrdering::compute_semantic_key(uint32_t codepoint) {
    // Default values
    uint8_t script_id = 100; // other scripts
    uint8_t semantic_class = 15; // other
    uint32_t base_similarity = codepoint; // fallback
    uint8_t variant_order = 0;

    // Script identification (highest level grouping)
    if (codepoint >= 0x0000 && codepoint <= 0x024F) script_id = 0; // Latin + Extended
    else if (codepoint >= 0x0370 && codepoint <= 0x03FF) script_id = 1; // Greek
    else if (codepoint >= 0x0400 && codepoint <= 0x04FF) script_id = 2; // Cyrillic
    else if (codepoint >= 0x0590 && codepoint <= 0x05FF) script_id = 3; // Hebrew
    else if (codepoint >= 0x0600 && codepoint <= 0x077F) script_id = 4; // Arabic
    else if (codepoint >= 0x0900 && codepoint <= 0x097F) script_id = 5; // Devanagari
    else if (codepoint >= 0x2E80 && codepoint <= 0x9FFF) script_id = 6; // CJK
    else if (codepoint >= 0x1F600 && codepoint <= 0x1F64F) script_id = 7; // Emoji
    else if (codepoint >= 0x1F300 && codepoint <= 0x1F5FF) script_id = 8; // Symbols

    // Semantic classification (within scripts)
    if (codepoint >= 'A' && codepoint <= 'Z') {
        semantic_class = 1; // ASCII uppercase
        base_similarity = codepoint - 'A'; // 0-25: A,B,C,...
        variant_order = 0; // uppercase first in base group
    } else if (codepoint >= 'a' && codepoint <= 'z') {
        semantic_class = 1; // ASCII lowercase (same class for grouping)
        base_similarity = codepoint - 'a'; // 0-25: a,b,c,... (same base as uppercase)
        variant_order = 1; // lowercase second in base group
    }
    // Digits: Group by visual similarity (0,O,o together)
    else if (codepoint >= '0' && codepoint <= '9') {
        semantic_class = 3; // digit
        // Group homoglyphs: 0→O, 1→I→l, 2, 3, 4, 5, 6→G, 7, 8→B, 9→g
        switch (codepoint) {
            case '0': base_similarity = 0; break; // 0,O,o group
            case '1': base_similarity = 10; break; // 1,I,l group
            case '2': base_similarity = 2; break;
            case '3': base_similarity = 3; break;
            case '4': base_similarity = 4; break;
            case '5': base_similarity = 5; break;
            case '6': base_similarity = 60; break; // 6,G group
            case '7': base_similarity = 7; break;
            case '8': base_similarity = 80; break; // 8,B group
            case '9': base_similarity = 90; break; // 9,g,q group
        }
        variant_order = 0;
    }
    // ASCII symbols: Group by function
    else if ((codepoint >= 0x21 && codepoint <= 0x2F) || (codepoint >= 0x3A && codepoint <= 0x40) ||
             (codepoint >= 0x5B && codepoint <= 0x60) || (codepoint >= 0x7B && codepoint <= 0x7E)) {
        semantic_class = 10; // punctuation
        base_similarity = codepoint; // keep original order for punctuation
        variant_order = 0;
    }
    // Greek: Group by base letter (case variants together)
    else if (codepoint >= 0x0391 && codepoint <= 0x03A9) { // uppercase greek
        semantic_class = 4; // greek uppercase
        base_similarity = 1000 + (codepoint - 0x0391); // base 1000-1024
        variant_order = 0; // uppercase first
    } else if (codepoint >= 0x03B1 && codepoint <= 0x03C9) { // lowercase greek
        semantic_class = 4; // greek lowercase (same class for grouping)
        base_similarity = 1000 + (codepoint - 0x03B1); // same base as uppercase
        variant_order = 1; // lowercase second
    }
    // Cyrillic: Group by base letter (case variants together)
    else if (codepoint >= 0x0410 && codepoint <= 0x042F) { // uppercase cyrillic
        semantic_class = 5; // cyrillic uppercase
        base_similarity = 2000 + (codepoint - 0x0410); // base 2000-2030
        variant_order = 0; // uppercase first
    } else if (codepoint >= 0x0430 && codepoint <= 0x044F) { // lowercase cyrillic
        semantic_class = 5; // cyrillic lowercase (same class for grouping)
        base_similarity = 2000 + (codepoint - 0x0430); // same base as uppercase
        variant_order = 1; // lowercase second
    }
    // Emoji: Group by category
    else if (codepoint >= 0x1F600 && codepoint <= 0x1F64F) {
        semantic_class = 20; // faces
        base_similarity = codepoint - 0x1F600;
        variant_order = 0;
    } else if (codepoint >= 0x1F300 && codepoint <= 0x1F5FF) {
        semantic_class = 21; // symbols
        base_similarity = codepoint - 0x1F300;
        variant_order = 0;
    }
    // CJK: Group by radical/stroke similarity (simplified)
    else if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
        semantic_class = 30; // CJK unified
        base_similarity = codepoint - 0x4E00;
        variant_order = 0;
    }
    // Fallback: Other characters
    else {
        semantic_class = 99; // other
        base_similarity = codepoint;
        variant_order = codepoint & 0xFF;
    }

    // Pack into 64-bit key: Script > Semantic Class > Base Similarity > Variant Order > Uniqueness
    uint64_t key = 0;
    key |= (uint64_t(script_id) & 0xFF) << 56;        // Bits 63-56: script (highest)
    key |= (uint64_t(semantic_class) & 0xFF) << 48;   // Bits 55-48: semantic class
    key |= (uint64_t(base_similarity) & 0xFFFF) << 24; // Bits 47-24: base similarity (16 bits)
    key |= (uint64_t(variant_order) & 0xFF) << 16;    // Bits 23-16: variant order
    key |= (uint64_t(codepoint & 0xFFFF));           // Bits 15-0: codepoint for uniqueness

    return key;
}

// Legacy function for backward compatibility
uint64_t get_semantic_key(uint32_t cp) {
    return SemanticOrdering::get_semantic_key(cp);
}

} // namespace hypercube</xai:function_call">The file cpp/src/core/semantic_ordering.cpp was created successfully.