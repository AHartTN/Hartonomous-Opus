#include "hypercube/ingest/cpe.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/blake3.hpp"
#include <span>
#include <vector>
#include <algorithm>
#include <cctype>

namespace hypercube::ingest {

namespace {
    inline uint32_t int32_to_uint32(int32_t val) {
        return static_cast<uint32_t>(val);
    }

    inline int32_t uint32_to_int32(uint32_t val) {
        return static_cast<int32_t>(val);
    }
}

// ============================================================================
// UNICODE-BASED SEGMENTATION (Language-Agnostic per UAX #29)
// These are public in the unicode namespace for use in TokenizerConfig
// ============================================================================

namespace unicode {

bool is_whitespace(uint32_t cp) {
    // ASCII whitespace
    if (cp == 0x0009 || cp == 0x000A || cp == 0x000B || cp == 0x000C ||
        cp == 0x000D || cp == 0x0020) return true;
    // Unicode Zs (Space Separators)
    if (cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    // Line/paragraph separators
    if (cp == 0x2028 || cp == 0x2029) return true;
    return false;
}

bool is_sentence_end(uint32_t cp) {
    // ASCII sentence-enders
    if (cp == '.' || cp == '!' || cp == '?') return true;
    // CJK sentence-enders
    if (cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF0E) return true;
    // Arabic/Persian
    if (cp == 0x06D4 || cp == 0x061F) return true;
    // Devanagari
    if (cp == 0x0964 || cp == 0x0965) return true;
    // Armenian
    if (cp == 0x0589 || cp == 0x055C || cp == 0x055E) return true;
    // Greek
    if (cp == 0x037E) return true;
    // Ethiopic
    if (cp == 0x1362 || cp == 0x1367 || cp == 0x1368) return true;
    // Myanmar
    if (cp == 0x104A || cp == 0x104B) return true;
    // Tibetan
    if (cp == 0x0F0D || cp == 0x0F0E) return true;
    // Georgian
    if (cp == 0x10FB) return true;
    // Khmer
    if (cp == 0x17D4 || cp == 0x17D5) return true;
    // Lao (uses whitespace typically)
    // Thai has no sentence punctuation - uses space
    return false;
}

bool is_paragraph_break(uint32_t prev, uint32_t curr) {
    // Double newline
    if (prev == 0x000A && curr == 0x000A) return true;
    // Explicit paragraph separator
    if (prev == 0x2029 || curr == 0x2029) return true;
    // Form feed
    if (prev == 0x000C || curr == 0x000C) return true;
    return false;
}

} // namespace unicode

// ============================================================================
// TOKENIZER CONFIGURATION FACTORY FUNCTIONS
// ============================================================================

TokenizerConfig TokenizerConfig::unicode_default() {
    TokenizerConfig config;
    config.is_whitespace = unicode::is_whitespace;
    config.is_sentence_end = unicode::is_sentence_end;
    config.is_paragraph_break = unicode::is_paragraph_break;
    config.segment_words = nullptr;  // Use default whitespace tokenization
    return config;
}

TokenizerConfig TokenizerConfig::cjk_character() {
    TokenizerConfig config;
    // CJK scripts typically use punctuation as separators
    // Each character is treated as a potential word
    config.is_whitespace = unicode::is_whitespace;
    config.is_sentence_end = unicode::is_sentence_end;
    config.is_paragraph_break = unicode::is_paragraph_break;

    // Custom word segmentation: each CJK character is a "word"
    // Non-CJK sequences are kept together
    config.segment_words = [](const std::vector<uint32_t>& cps) -> std::vector<size_t> {
        std::vector<size_t> boundaries;
        boundaries.push_back(0);

        for (size_t i = 0; i < cps.size(); ++i) {
            uint32_t cp = cps[i];
            // CJK Unified Ideographs and extensions
            bool is_cjk = (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified
                          (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Ext A
                          (cp >= 0x20000 && cp <= 0x2A6DF) || // CJK Ext B
                          (cp >= 0x2A700 && cp <= 0x2CEAF) || // CJK Ext C-F
                          (cp >= 0x3040 && cp <= 0x30FF) ||   // Hiragana + Katakana
                          (cp >= 0xAC00 && cp <= 0xD7AF);     // Hangul

            if (is_cjk && i > 0) {
                boundaries.push_back(i);
            }
        }

        if (boundaries.back() != cps.size()) {
            boundaries.push_back(cps.size());
        }
        return boundaries;
    };

    return config;
}

TokenizerConfig TokenizerConfig::whitespace_only() {
    TokenizerConfig config;
    config.is_whitespace = unicode::is_whitespace;
    // No sentence detection - just whitespace tokenization
    config.is_sentence_end = [](uint32_t) { return false; };
    config.is_paragraph_break = [](uint32_t, uint32_t) { return false; };
    config.segment_words = nullptr;
    return config;
}

// ============================================================================
// N-ARY COMPOSITION: The correct way to build compositions
// ============================================================================

std::pair<CompositionRecord, bool> create_composition(
    const std::vector<ChildInfo>& children,
    uint32_t max_child_depth,
    uint64_t total_atoms,
    std::unordered_map<std::string, CompositionRecord>& cache
) {
    if (children.empty()) {
        return {CompositionRecord{}, false};
    }
    
    // Hash = BLAKE3(ord0 || hash0 || ord1 || hash1 || ... || ordN-1 || hashN-1)
    // Uses little-endian ordinals to match C API (hc_blake3_children_ordered)
    // This ensures consistency between C++ ingestion and SQL lookups
    std::vector<uint8_t> hash_input;
    hash_input.reserve(children.size() * 36);  // 4 bytes ordinal + 32 bytes hash per child

    for (size_t i = 0; i < children.size(); ++i) {
        // Little-endian ordinal
        uint32_t ordinal = static_cast<uint32_t>(i);
        hash_input.insert(hash_input.end(),
            reinterpret_cast<uint8_t*>(&ordinal),
            reinterpret_cast<uint8_t*>(&ordinal) + 4);
        // Child hash
        hash_input.insert(hash_input.end(),
            children[i].hash.bytes.begin(), children[i].hash.bytes.end());
    }

    Blake3Hash hash = Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
    
    // Check cache first
    std::string hash_key = hash.to_hex();
    auto it = cache.find(hash_key);
    if (it != cache.end()) {
        return {it->second, false};  // Already exists
    }
    
    // Compute centroid as average of all child centroids, scaled inward by depth
    // Atoms are on the 3-sphere surface; compositions move inward toward CENTER
    // CENTER = 2^31 = 2147483648 (origin of hypercube coordinate space)
    // This creates a layered structure where deeper compositions are closer to origin
    constexpr double CENTER = 2147483648.0;  // 2^31 - canonical center
    
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0, sum_m = 0.0;
    for (const auto& child : children) {
        // Children store coords as int32 (reinterpreted from uint32)
        // Convert back to uint32 for proper centroid calculation
        sum_x += static_cast<double>(static_cast<uint32_t>(child.x));
        sum_y += static_cast<double>(static_cast<uint32_t>(child.y));
        sum_z += static_cast<double>(static_cast<uint32_t>(child.z));
        sum_m += static_cast<double>(static_cast<uint32_t>(child.m));
    }

    size_t n = children.size();
    double avg_x = sum_x / static_cast<double>(n);
    double avg_y = sum_y / static_cast<double>(n);
    double avg_z = sum_z / static_cast<double>(n);
    double avg_m = sum_m / static_cast<double>(n);

    // Scale toward CENTER based on depth
    // Scale factor = 1 / (depth + 2), so depth 1 → 0.33, depth 2 → 0.25, etc.
    // This moves centroids inward while preserving relative angular positions
    uint32_t depth = max_child_depth + 1;
    double scale = 1.0 / static_cast<double>(depth + 2);

    // Interpolate: result = CENTER + (avg - CENTER) * (1 - scale)
    // As depth increases, centroids move closer to CENTER (origin)
    double factor = 1.0 - scale;
    
    auto scale_coord = [factor](double avg) -> uint32_t {
        double result = CENTER + (avg - CENTER) * factor;
        if (result < 0.0) result = 0.0;
        if (result > 4294967295.0) result = 4294967295.0;
        return static_cast<uint32_t>(std::round(result));
    };
    
    uint32_t cx = scale_coord(avg_x);
    uint32_t cy = scale_coord(avg_y);
    uint32_t cz = scale_coord(avg_z);
    uint32_t cm = scale_coord(avg_m);
    
    // Hilbert index from centroid
    Point4D coords(cx, cy, cz, cm);
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    
    CompositionRecord rec;
    rec.hash = hash;
    rec.coord_x = uint32_to_int32(cx);
    rec.coord_y = uint32_to_int32(cy);
    rec.coord_z = uint32_to_int32(cz);
    rec.coord_m = uint32_to_int32(cm);
    rec.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    rec.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    rec.depth = max_child_depth + 1;
    rec.atom_count = total_atoms;
    rec.children = children;  // Store all N children
    
    cache[hash_key] = rec;
    return {rec, true};
}

// Create composition for a single token (e.g., "the" = [t, h, e])
Blake3Hash create_token_composition(
    const std::vector<uint32_t>& token_codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache
) {
    if (token_codepoints.empty()) return Blake3Hash();
    
    // Single codepoint = return the atom directly (no composition needed)
    if (token_codepoints.size() == 1) {
        auto it = atom_cache.find(token_codepoints[0]);
        if (it != atom_cache.end()) {
            return it->second.hash;
        }
        return Blake3Hash();
    }
    
    // Build children list from codepoints
    std::vector<ChildInfo> children;
    children.reserve(token_codepoints.size());
    
    for (uint32_t cp : token_codepoints) {
        auto it = atom_cache.find(cp);
        if (it == atom_cache.end()) continue;
        
        ChildInfo child;
        child.hash = it->second.hash;
        child.x = it->second.coord_x;
        child.y = it->second.coord_y;
        child.z = it->second.coord_z;
        child.m = it->second.coord_m;
        children.push_back(child);
    }
    
    if (children.empty()) return Blake3Hash();
    if (children.size() == 1) return children[0].hash;
    
    // Create single N-ary composition for this token
    auto [rec, is_new] = create_composition(children, 0, children.size(), comp_cache);
    if (is_new) {
        new_compositions.push_back(rec);
    }
    
    return rec.hash;
}

// Main ingestion function with configurable tokenizer
Blake3Hash ingest_text(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache,
    const TokenizerConfig& config
) {
    if (codepoints.empty()) return Blake3Hash();

    // Use config functions or defaults
    auto is_ws = config.is_whitespace ? config.is_whitespace : unicode::is_whitespace;
    auto is_sent = config.is_sentence_end ? config.is_sentence_end : unicode::is_sentence_end;
    auto is_para = config.is_paragraph_break ? config.is_paragraph_break : unicode::is_paragraph_break;

    // ========================================================================
    // TIER 1: Tokenize into words (sequences between whitespace)
    // ========================================================================
    struct TokenInfo {
        std::vector<uint32_t> codepoints;
        bool ends_sentence;
        bool ends_paragraph;
    };

    std::vector<TokenInfo> tokens;
    std::vector<uint32_t> current_token;
    uint32_t prev_cp = 0;

    for (size_t i = 0; i < codepoints.size(); ++i) {
        uint32_t cp = codepoints[i];

        if (is_ws(cp)) {
            if (!current_token.empty()) {
                TokenInfo ti;
                ti.codepoints = std::move(current_token);
                ti.ends_sentence = false;
                ti.ends_paragraph = false;
                // Check if last codepoint of token is sentence-ender
                if (!ti.codepoints.empty() && is_sent(ti.codepoints.back())) {
                    ti.ends_sentence = true;
                }
                tokens.push_back(std::move(ti));
                current_token.clear();
            }
            // Check for paragraph break
            if (!tokens.empty() && is_para(prev_cp, cp)) {
                tokens.back().ends_paragraph = true;
            }
        } else {
            current_token.push_back(cp);
        }
        prev_cp = cp;
    }
    if (!current_token.empty()) {
        TokenInfo ti;
        ti.codepoints = std::move(current_token);
        ti.ends_sentence = !ti.codepoints.empty() && is_sent(ti.codepoints.back());
        ti.ends_paragraph = true;  // End of document = end of paragraph
        tokens.push_back(std::move(ti));
    }
    
    if (tokens.empty()) return Blake3Hash();
    
    // ========================================================================
    // Create compositions for each word token
    // ========================================================================
    struct WordInfo {
        ChildInfo child;
        bool ends_sentence;
        bool ends_paragraph;
    };
    
    std::vector<WordInfo> words;
    words.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        Blake3Hash token_hash = create_token_composition(
            token.codepoints, atom_cache, new_compositions, comp_cache);
        
        if (token_hash.is_zero()) continue;
        
        WordInfo wi;
        wi.ends_sentence = token.ends_sentence;
        wi.ends_paragraph = token.ends_paragraph;
        
        // Look up the composition we just created to get its centroid
        std::string hash_key = token_hash.to_hex();
        auto it = comp_cache.find(hash_key);
        if (it != comp_cache.end()) {
            wi.child.hash = it->second.hash;
            wi.child.x = it->second.coord_x;
            wi.child.y = it->second.coord_y;
            wi.child.z = it->second.coord_z;
            wi.child.m = it->second.coord_m;
            words.push_back(wi);
        } else if (token.codepoints.size() == 1) {
            // Single codepoint - get from atom cache
            auto atom_it = atom_cache.find(token.codepoints[0]);
            if (atom_it != atom_cache.end()) {
                wi.child.hash = atom_it->second.hash;
                wi.child.x = atom_it->second.coord_x;
                wi.child.y = atom_it->second.coord_y;
                wi.child.z = atom_it->second.coord_z;
                wi.child.m = atom_it->second.coord_m;
                words.push_back(wi);
            }
        }
    }
    
    if (words.empty()) return Blake3Hash();
    if (words.size() == 1) return words[0].child.hash;
    
    // ========================================================================
    // TIER 2: Group words into sentences
    // ========================================================================
    struct SentenceInfo {
        std::vector<ChildInfo> words;
        bool ends_paragraph;
    };
    
    std::vector<SentenceInfo> sentences;
    SentenceInfo current_sentence;
    current_sentence.ends_paragraph = false;
    
    for (const auto& word : words) {
        current_sentence.words.push_back(word.child);
        
        if (word.ends_sentence || word.ends_paragraph) {
            current_sentence.ends_paragraph = word.ends_paragraph;
            sentences.push_back(std::move(current_sentence));
            current_sentence.words.clear();
            current_sentence.ends_paragraph = false;
        }
    }
    if (!current_sentence.words.empty()) {
        current_sentence.ends_paragraph = true;
        sentences.push_back(std::move(current_sentence));
    }
    
    // Create sentence compositions (depth 2)
    struct ParagraphChild {
        ChildInfo child;
        bool ends_paragraph;
    };
    
    std::vector<ParagraphChild> sentence_children;
    sentence_children.reserve(sentences.size());
    
    for (const auto& sentence : sentences) {
        if (sentence.words.empty()) continue;
        
        ParagraphChild pc;
        pc.ends_paragraph = sentence.ends_paragraph;
        
        if (sentence.words.size() == 1) {
            pc.child = sentence.words[0];
        } else {
            uint64_t total_atoms = 0;
            total_atoms += sentence.words.size();  // Could sum actual atom counts
            
            auto [rec, is_new] = create_composition(sentence.words, 1, total_atoms, comp_cache);
            if (is_new) {
                new_compositions.push_back(rec);
            }
            
            pc.child.hash = rec.hash;
            pc.child.x = rec.coord_x;
            pc.child.y = rec.coord_y;
            pc.child.z = rec.coord_z;
            pc.child.m = rec.coord_m;
        }
        sentence_children.push_back(pc);
    }
    
    if (sentence_children.empty()) return Blake3Hash();
    if (sentence_children.size() == 1) return sentence_children[0].child.hash;
    
    // ========================================================================
    // TIER 3: Group sentences into paragraphs
    // ========================================================================
    std::vector<std::vector<ChildInfo>> paragraphs;
    std::vector<ChildInfo> current_paragraph;
    
    for (const auto& sc : sentence_children) {
        current_paragraph.push_back(sc.child);
        
        if (sc.ends_paragraph && !current_paragraph.empty()) {
            paragraphs.push_back(std::move(current_paragraph));
            current_paragraph.clear();
        }
    }
    if (!current_paragraph.empty()) {
        paragraphs.push_back(std::move(current_paragraph));
    }
    
    // Create paragraph compositions (depth 3)
    std::vector<ChildInfo> paragraph_children;
    paragraph_children.reserve(paragraphs.size());
    
    for (const auto& paragraph : paragraphs) {
        if (paragraph.empty()) continue;
        
        if (paragraph.size() == 1) {
            paragraph_children.push_back(paragraph[0]);
        } else {
            uint64_t total_atoms = paragraph.size();
            
            auto [rec, is_new] = create_composition(paragraph, 2, total_atoms, comp_cache);
            if (is_new) {
                new_compositions.push_back(rec);
            }
            
            ChildInfo child;
            child.hash = rec.hash;
            child.x = rec.coord_x;
            child.y = rec.coord_y;
            child.z = rec.coord_z;
            child.m = rec.coord_m;
            paragraph_children.push_back(child);
        }
    }
    
    if (paragraph_children.empty()) return Blake3Hash();
    if (paragraph_children.size() == 1) return paragraph_children[0].hash;
    
    // ========================================================================
    // TIER 4: Create document root composition (depth 4)
    // ========================================================================
    uint64_t total_atoms = paragraph_children.size();
    
    auto [root_rec, is_new] = create_composition(paragraph_children, 3, total_atoms, comp_cache);
    if (is_new) {
        new_compositions.push_back(root_rec);
    }
    
    return root_rec.hash;
}

// Default overload uses Unicode defaults
Blake3Hash ingest_text(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions,
    std::unordered_map<std::string, CompositionRecord>& comp_cache
) {
    return ingest_text(codepoints, atom_cache, new_compositions, comp_cache,
                       TokenizerConfig::unicode_default());
}

// ============================================================================
// DEPRECATED: Binary cascade (keeping for backwards compatibility only)
// ============================================================================

std::pair<CompositionRecordBinaryDeprecated, bool> create_pair_deprecated(
    const Blake3Hash& left_hash, int32_t left_x, int32_t left_y, int32_t left_z, int32_t left_m,
    uint32_t left_depth, uint64_t left_atoms,
    const Blake3Hash& right_hash, int32_t right_x, int32_t right_y, int32_t right_z, int32_t right_m,
    uint32_t right_depth, uint64_t right_atoms,
    std::unordered_map<std::string, CompositionRecordBinaryDeprecated>& cache
) {
    std::vector<uint8_t> hash_input;
    hash_input.reserve(72);
    
    uint32_t ord0 = 0;
    hash_input.insert(hash_input.end(), 
        reinterpret_cast<uint8_t*>(&ord0), 
        reinterpret_cast<uint8_t*>(&ord0) + 4);
    hash_input.insert(hash_input.end(), left_hash.bytes.begin(), left_hash.bytes.end());
    
    uint32_t ord1 = 1;
    hash_input.insert(hash_input.end(), 
        reinterpret_cast<uint8_t*>(&ord1), 
        reinterpret_cast<uint8_t*>(&ord1) + 4);
    hash_input.insert(hash_input.end(), right_hash.bytes.begin(), right_hash.bytes.end());
    
    Blake3Hash hash = Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
    
    std::string hash_key = hash.to_hex();
    auto it = cache.find(hash_key);
    if (it != cache.end()) {
        return {it->second, false};
    }
    
    uint64_t sum_x = static_cast<uint64_t>(int32_to_uint32(left_x)) + static_cast<uint64_t>(int32_to_uint32(right_x));
    uint64_t sum_y = static_cast<uint64_t>(int32_to_uint32(left_y)) + static_cast<uint64_t>(int32_to_uint32(right_y));
    uint64_t sum_z = static_cast<uint64_t>(int32_to_uint32(left_z)) + static_cast<uint64_t>(int32_to_uint32(right_z));
    uint64_t sum_m = static_cast<uint64_t>(int32_to_uint32(left_m)) + static_cast<uint64_t>(int32_to_uint32(right_m));
    
    uint32_t cx = static_cast<uint32_t>(sum_x / 2);
    uint32_t cy = static_cast<uint32_t>(sum_y / 2);
    uint32_t cz = static_cast<uint32_t>(sum_z / 2);
    uint32_t cm = static_cast<uint32_t>(sum_m / 2);
    
    Point4D coords(cx, cy, cz, cm);
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    
    CompositionRecordBinaryDeprecated rec;
    rec.hash = hash;
    rec.coord_x = uint32_to_int32(cx);
    rec.coord_y = uint32_to_int32(cy);
    rec.coord_z = uint32_to_int32(cz);
    rec.coord_m = uint32_to_int32(cm);
    rec.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    rec.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    rec.depth = std::max(left_depth, right_depth) + 1;
    rec.atom_count = left_atoms + right_atoms;
    rec.left_hash = left_hash;
    rec.right_hash = right_hash;
    rec.left_x = left_x; rec.left_y = left_y; rec.left_z = left_z; rec.left_m = left_m;
    rec.right_x = right_x; rec.right_y = right_y; rec.right_z = right_z; rec.right_m = right_m;
    
    cache[hash_key] = rec;
    return {rec, true};
}

Blake3Hash cpe_cascade(
    const std::vector<uint32_t>& codepoints,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecordBinaryDeprecated>& new_compositions,
    std::unordered_map<std::string, CompositionRecordBinaryDeprecated>& comp_cache
) {
    // DEPRECATED - use ingest_text instead
    // Keeping for backwards compatibility only
    
    if (codepoints.empty()) return Blake3Hash();
    
    struct CpeNode {
        Blake3Hash hash;
        int32_t x, y, z, m;
        uint32_t depth;
        uint64_t atom_count;
    };
    
    std::vector<CpeNode> nodes;
    nodes.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        auto it = atom_cache.find(cp);
        if (it == atom_cache.end()) continue;
        
        CpeNode node;
        node.hash = it->second.hash;
        node.x = it->second.coord_x;
        node.y = it->second.coord_y;
        node.z = it->second.coord_z;
        node.m = it->second.coord_m;
        node.depth = 0;
        node.atom_count = 1;
        nodes.push_back(node);
    }
    
    if (nodes.empty()) return Blake3Hash();
    if (nodes.size() == 1) return nodes[0].hash;
    
    while (nodes.size() > 1) {
        std::vector<CpeNode> next_tier;
        next_tier.reserve(nodes.size() - 1);
        
        for (size_t i = 0; i + 1 < nodes.size(); ++i) {
            const CpeNode& left = nodes[i];
            const CpeNode& right = nodes[i + 1];
            
            auto [rec, is_new] = create_pair_deprecated(
                left.hash, left.x, left.y, left.z, left.m, left.depth, left.atom_count,
                right.hash, right.x, right.y, right.z, right.m, right.depth, right.atom_count,
                comp_cache
            );
            
            if (is_new) {
                new_compositions.push_back(rec);
            }
            
            CpeNode node;
            node.hash = rec.hash;
            node.x = rec.coord_x;
            node.y = rec.coord_y;
            node.z = rec.coord_z;
            node.m = rec.coord_m;
            node.depth = rec.depth;
            node.atom_count = rec.atom_count;
            next_tier.push_back(node);
        }
        
        nodes = std::move(next_tier);
    }
    
    return nodes[0].hash;
}

} // namespace hypercube::ingest
