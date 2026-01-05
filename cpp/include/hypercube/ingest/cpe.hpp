#pragma once

#include "hypercube/types.hpp"
#include "hypercube/db/atom_cache.hpp"
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <string>

namespace hypercube::ingest {

// ============================================================================
// PLUGGABLE TOKENIZATION - Language-agnostic configuration
// ============================================================================

// Tokenizer configuration for pluggable language support
// Defaults are Unicode-based (works for most scripts)
struct TokenizerConfig {
    // Detect word boundaries (whitespace/separator)
    // Default: Unicode Zs category + common whitespace
    std::function<bool(uint32_t)> is_whitespace;

    // Detect sentence-ending punctuation
    // Default: covers Latin, CJK, Arabic, Devanagari, etc.
    std::function<bool(uint32_t)> is_sentence_end;

    // Detect paragraph boundaries (returns true if prev+curr form a paragraph break)
    // Default: double newline or explicit paragraph separator
    std::function<bool(uint32_t prev, uint32_t curr)> is_paragraph_break;

    // Optional: custom word segmentation for languages without whitespace
    // If set, bypasses default whitespace tokenization
    // Takes codepoints, returns vector of word boundaries (indices)
    std::function<std::vector<size_t>(const std::vector<uint32_t>&)> segment_words;

    // Create with Unicode defaults (works for most languages)
    static TokenizerConfig unicode_default();

    // Create for CJK (no whitespace word boundaries - character-based)
    static TokenizerConfig cjk_character();

    // Create for whitespace-only (simple tokenization)
    static TokenizerConfig whitespace_only();
};

// Default Unicode-based predicates (used by TokenizerConfig::unicode_default)
namespace unicode {
    bool is_whitespace(uint32_t cp);
    bool is_sentence_end(uint32_t cp);
    bool is_paragraph_break(uint32_t prev, uint32_t curr);
}



// Node roles for semantic typing within the Merkle DAG
enum class NodeRole : int16_t {
    Generic = 0,            // default composition
    UnicodeAtom = 1,        // leaf (codepoint)
    Token = 2,              // word/subword
    Sentence = 3,           // sentence
    Paragraph = 4,          // paragraph
    DocumentContentRoot = 5,// root of document content tree
    FileInfoRoot = 6,       // file info node (name, size, etc.)
    FileMetadataRoot = 7,   // file metadata node (hash, timestamp, etc.)
    FileRoot = 8,           // parent ingestion node (combines info+meta+content)
    AstNode = 9,            // AST node (for code/structured content)
    KvPair = 10             // key-value metadata pair
};

// Child info for building LINESTRINGZM
struct ChildInfo {
    Blake3Hash hash;
    int32_t x, y, z, m;
    bool is_atom = true;  // true if child is an atom (depth=0), false if composition (depth>0)
};

// N-ary composition record - NOT limited to binary pairs
// "the" = one composition with 3 children [t, h, e]
// LINESTRINGZM geometry connects child centroids in order
struct CompositionRecord {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;  // Centroid (average of children, scaled by depth)
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;                               // Depth tier (1 = direct child of leaves)
    uint64_t atom_count;                          // Total leaves in subtree
    NodeRole node_role = NodeRole::Generic;       // Semantic role in DAG
    
    std::vector<ChildInfo> children;              // N children, ordered (NOT just 2)
};

// Create an N-ary composition from ordered children
// Hash = BLAKE3(child[0].hash || child[1].hash || ... || child[N-1].hash)
// Centroid = average of child centroids, radius scaled inward by depth
// Returns {record, is_new} where is_new=false if already in cache
std::pair<CompositionRecord, bool> create_composition(
    const std::vector<ChildInfo>& children,
    uint32_t max_child_depth,
    uint64_t total_atoms,
    std::unordered_map<std::string, CompositionRecord>& cache
);

// Token-aware ingestion: builds compositions at natural boundaries
// - Words: sequences of non-whitespace codepoints
// - Sentences: sequences of words ending in sentence punctuation
// - Paragraphs: sequences of sentences ending in newlines
// Returns root hash for the entire content
Blake3Hash ingest_text(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache
);

// Token-aware ingestion with custom tokenizer configuration
// Use this for language-specific tokenization rules
Blake3Hash ingest_text(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache,
    const TokenizerConfig& config
);

// Create composition for a single token (word/punctuation)
// "the" = composition of [t, h, e], NOT binary tree
Blake3Hash create_token_composition(
    const std::vector<uint32_t>& token_codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache
);

// ============================================================================
// DEPRECATED: Binary pair functions - DO NOT USE for new code
// ============================================================================

// DEPRECATED: Use create_composition with N children instead
struct CompositionRecordBinaryDeprecated {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;
    uint64_t atom_count;
    Blake3Hash left_hash, right_hash;
    int32_t left_x, left_y, left_z, left_m;
    int32_t right_x, right_y, right_z, right_m;
};

// DEPRECATED: Binary cascade creates wrong structure
[[deprecated("Use ingest_text() for proper N-ary compositions")]]
Blake3Hash cpe_cascade(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecordBinaryDeprecated>& new_compositions,
    std::unordered_map<std::string, CompositionRecordBinaryDeprecated>& comp_cache
);

} // namespace hypercube::ingest
