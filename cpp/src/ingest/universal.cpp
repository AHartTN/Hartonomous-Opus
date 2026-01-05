/**
 * Universal Substrate Ingester - Implementation
 * 
 * Pure sliding window pattern discovery. No linguistic rules.
 * Works on any sequence of integer tokens.
 */

#include "hypercube/ingest/universal.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/blake3.hpp"
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace hypercube::ingest {

namespace {
    inline uint32_t int32_to_uint32(int32_t val) {
        return static_cast<uint32_t>(val);
    }

    inline int32_t uint32_to_int32(uint32_t val) {
        return static_cast<int32_t>(val);
    }
}

// Extended child info with depth and atom count for proper tier tracking
struct ChildWithMeta {
    Blake3Hash hash;
    int32_t x, y, z, m;
    uint32_t depth;
    uint64_t atom_count;
    
    // Convert to ChildInfo for storage with proper is_atom flag
    ChildInfo to_child_info() const {
        ChildInfo ci;
        ci.hash = hash;
        ci.x = x;
        ci.y = y;
        ci.z = z;
        ci.m = m;
        ci.is_atom = (depth == 0);  // depth 0 = atom, depth > 0 = composition
        return ci;
    }
};

// ============================================================================
// COMPOSITION HASH
// ============================================================================

Blake3Hash compute_composition_hash(const std::vector<Blake3Hash>& children) {
    // hash = BLAKE3(ord_0 || hash_0 || ord_1 || hash_1 || ... || ord_N-1 || hash_N-1)
    // Each ordinal is 4 bytes (uint32_t), each hash is 32 bytes
    // Total: N * 36 bytes
    
    std::vector<uint8_t> buffer;
    buffer.reserve(children.size() * 36);
    
    for (size_t i = 0; i < children.size(); ++i) {
        // Ordinal (position in sequence) - 4 bytes little-endian
        uint32_t ordinal = static_cast<uint32_t>(i);
        buffer.push_back(static_cast<uint8_t>(ordinal & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 8) & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 16) & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 24) & 0xFF));
        
        // Hash - 32 bytes
        buffer.insert(buffer.end(), children[i].bytes.begin(), children[i].bytes.end());
    }
    
    return Blake3Hasher::hash(std::span<const uint8_t>(buffer.data(), buffer.size()));
}

// ============================================================================
// VOCABULARY TRIE IMPLEMENTATION
// ============================================================================

struct TrieNode {
    // Use Blake3Hash directly as key with proper hasher - no hex conversion
    std::unordered_map<Blake3Hash, std::unique_ptr<TrieNode>, Blake3HashHasher> children;
    std::optional<Blake3Hash> composition_hash;
};

struct VocabularyTrie::Impl {
    std::unique_ptr<TrieNode> root = std::make_unique<TrieNode>();
    size_t node_count = 0;
};

VocabularyTrie::VocabularyTrie() : impl_(std::make_unique<Impl>()) {}
VocabularyTrie::~VocabularyTrie() = default;

size_t VocabularyTrie::size() const { return impl_->node_count; }

void VocabularyTrie::insert(const std::vector<Blake3Hash>& children, const Blake3Hash& composition_hash) {
    if (children.empty()) return;
    
    TrieNode* node = impl_->root.get();
    for (const auto& child : children) {
        // Use Blake3Hash directly as key - no hex conversion
        auto it = node->children.find(child);
        if (it == node->children.end()) {
            node->children[child] = std::make_unique<TrieNode>();
            ++impl_->node_count;
        }
        node = node->children[child].get();
    }
    node->composition_hash = composition_hash;
}

std::pair<std::optional<Blake3Hash>, size_t> VocabularyTrie::longest_match(
    const std::vector<Blake3Hash>& sequence,
    size_t start_pos
) const {
    if (start_pos >= sequence.size()) return {std::nullopt, 0};
    
    const TrieNode* node = impl_->root.get();
    std::optional<Blake3Hash> best_match;
    size_t best_length = 0;
    
    for (size_t i = start_pos; i < sequence.size(); ++i) {
        // Use Blake3Hash directly - no hex conversion
        auto it = node->children.find(sequence[i]);
        if (it == node->children.end()) break;
        
        node = it->second.get();
        if (node->composition_hash) {
            best_match = node->composition_hash;
            best_length = i - start_pos + 1;
        }
    }
    
    return {best_match, best_length};
}

void VocabularyTrie::load_from_db(const std::vector<CompositionRecord>& compositions) {
    for (const auto& comp : compositions) {
        if (comp.children.empty()) continue;
        
        std::vector<Blake3Hash> child_hashes;
        child_hashes.reserve(comp.children.size());
        for (const auto& child : comp.children) {
            child_hashes.push_back(child.hash);
        }
        
        insert(child_hashes, comp.hash);
    }
}

// ============================================================================
// UNIVERSAL INGESTER IMPLEMENTATION
// ============================================================================

struct UniversalIngester::Impl {
    VocabularyTrie& vocab;
    // Use Blake3Hash directly as key with proper hasher - no hex conversion
    std::unordered_map<Blake3Hash, CompositionRecord, Blake3HashHasher> comp_cache;
    
    explicit Impl(VocabularyTrie& v) : vocab(v) {}
    
    // Create composition with proper depth and atom_count calculation
    std::pair<CompositionRecord, bool> create_composition(
        const std::vector<ChildWithMeta>& children
    );
    
    // Internal ingestion with metadata tracking
    Blake3Hash ingest_with_meta(
        const std::vector<ChildWithMeta>& children,
        std::vector<CompositionRecord>& new_compositions
    );
};

UniversalIngester::UniversalIngester(VocabularyTrie& vocab) 
    : impl_(std::make_unique<Impl>(vocab)) {}

UniversalIngester::~UniversalIngester() = default;

void UniversalIngester::set_min_ngram(size_t) { /* reserved for future use */ }
void UniversalIngester::set_max_ngram(size_t) { /* reserved for future use */ }

std::pair<CompositionRecord, bool> UniversalIngester::Impl::create_composition(
    const std::vector<ChildWithMeta>& children
) {
    if (children.empty()) {
        return {CompositionRecord{}, false};
    }
    
    // Compute hash from children (position-sensitive)
    std::vector<Blake3Hash> child_hashes;
    child_hashes.reserve(children.size());
    for (const auto& c : children) {
        child_hashes.push_back(c.hash);
    }
    Blake3Hash hash = compute_composition_hash(child_hashes);
    
    // Check cache - already exists? (use raw hash as key, no hex conversion)
    auto it = comp_cache.find(hash);
    if (it != comp_cache.end()) {
        return {it->second, false};
    }
    
    // Compute depth = max(child depths) + 1
    // Compute atom_count = sum(child atom_counts)
    uint32_t max_child_depth = 0;
    uint64_t total_atoms = 0;
    for (const auto& child : children) {
        max_child_depth = std::max(max_child_depth, child.depth);
        total_atoms += child.atom_count;
    }
    uint32_t depth = max_child_depth + 1;
    
    // Compute centroid as average of children, then scale toward CENTER by depth
    // CENTER = 2^31 = 2147483648 (origin of hypercube coordinate space)
    // Atoms are on 3-sphere SURFACE; compositions move INWARD as depth increases
    constexpr double CENTER = 2147483648.0;
    
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0, sum_m = 0.0;
    for (const auto& child : children) {
        // Children store coords as int32 (bit-cast from uint32)
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
    double scale = 1.0 / static_cast<double>(depth + 2);
    double factor = 1.0 - scale;
    
    auto scale_coord = [CENTER, factor](double avg) -> uint32_t {
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
    rec.coord_x = static_cast<int32_t>(cx);
    rec.coord_y = static_cast<int32_t>(cy);
    rec.coord_z = static_cast<int32_t>(cz);
    rec.coord_m = static_cast<int32_t>(cm);
    rec.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    rec.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    rec.depth = depth;
    rec.atom_count = total_atoms;
    
    // Convert ChildWithMeta to ChildInfo for storage
    rec.children.reserve(children.size());
    for (const auto& c : children) {
        rec.children.push_back(c.to_child_info());
    }
    
    // Add to cache and trie (use raw hash as key, no hex conversion)
    comp_cache[hash] = rec;
    vocab.insert(child_hashes, hash);
    
    return {rec, true};
}

Blake3Hash UniversalIngester::Impl::ingest_with_meta(
    const std::vector<ChildWithMeta>& children,
    std::vector<CompositionRecord>& new_compositions
) {
    if (children.empty()) return Blake3Hash();
    if (children.size() == 1) return children[0].hash;
    
    // Convert to hash sequence for trie lookup
    std::vector<Blake3Hash> hash_sequence;
    hash_sequence.reserve(children.size());
    for (const auto& c : children) {
        hash_sequence.push_back(c.hash);
    }
    
    // Check if entire sequence already exists
    auto [full_match, full_len] = vocab.longest_match(hash_sequence, 0);
    if (full_match && full_len == children.size()) {
        return *full_match;  // Already exists - return reference
    }
    
    // ==========================================================================
    // CPE-STYLE ITERATIVE PAIR DISCOVERY
    // ==========================================================================
    // Count adjacent pairs, find most frequent, create composition, replace, repeat
    // This discovers patterns like "th", "he", "the", "at", "cat", "sat" etc.
    
    std::vector<ChildWithMeta> current = children;
    
    while (current.size() > 1) {
        // Count all adjacent pairs
        std::unordered_map<std::string, size_t> pair_counts;
        for (size_t i = 0; i + 1 < current.size(); ++i) {
            // Create unique key for this pair (concatenated hashes)
            std::string key;
            key.reserve(64);
            for (auto b : current[i].hash.bytes) key += static_cast<char>(b);
            for (auto b : current[i+1].hash.bytes) key += static_cast<char>(b);
            pair_counts[key]++;
        }
        
        // Find most frequent pair
        std::string best_pair_key;
        size_t best_count = 0;
        for (const auto& [key, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best_pair_key = key;
            }
        }
        
        // If no pair occurs more than once, we're done discovering
        if (best_count <= 1) break;
        
        // Extract the two hashes from the key
        Blake3Hash hash1, hash2;
        std::memcpy(hash1.bytes.data(), best_pair_key.data(), 32);
        std::memcpy(hash2.bytes.data(), best_pair_key.data() + 32, 32);
        
        // Find the actual ChildWithMeta for these hashes (need coords)
        ChildWithMeta child1, child2;
        bool found1 = false, found2 = false;
        for (const auto& c : current) {
            if (!found1 && c.hash == hash1) { child1 = c; found1 = true; }
            if (!found2 && c.hash == hash2) { child2 = c; found2 = true; }
            if (found1 && found2) break;
        }
        
        // Create composition for this pair
        std::vector<ChildWithMeta> pair_children = {child1, child2};
        auto [rec, is_new] = create_composition(pair_children);
        if (is_new) {
            new_compositions.push_back(rec);
        }
        
        // Create ChildWithMeta for the new composition
        ChildWithMeta new_comp;
        new_comp.hash = rec.hash;
        new_comp.x = rec.coord_x;
        new_comp.y = rec.coord_y;
        new_comp.z = rec.coord_z;
        new_comp.m = rec.coord_m;
        new_comp.depth = rec.depth;
        new_comp.atom_count = rec.atom_count;
        
        // Replace all occurrences of this pair in current sequence
        std::vector<ChildWithMeta> next;
        next.reserve(current.size());
        size_t i = 0;
        while (i < current.size()) {
            if (i + 1 < current.size() && 
                current[i].hash == hash1 && current[i+1].hash == hash2) {
                next.push_back(new_comp);
                i += 2;  // Skip both elements of the pair
            } else {
                next.push_back(current[i]);
                i += 1;
            }
        }
        
        current = std::move(next);
    }
    
    // After pair discovery, do greedy tokenization with known vocabulary
    std::vector<ChildWithMeta> tokens;
    tokens.reserve(current.size());
    
    // Rebuild hash sequence for current state
    hash_sequence.clear();
    for (const auto& c : current) {
        hash_sequence.push_back(c.hash);
    }
    
    size_t i = 0;
    while (i < current.size()) {
        auto [match_hash, match_len] = vocab.longest_match(hash_sequence, i);
        
        if (match_hash && match_len > 1) {
            // Found existing composition - use it with its metadata
            auto cache_it = comp_cache.find(*match_hash);
            if (cache_it != comp_cache.end()) {
                ChildWithMeta info;
                info.hash = *match_hash;
                info.x = cache_it->second.coord_x;
                info.y = cache_it->second.coord_y;
                info.z = cache_it->second.coord_z;
                info.m = cache_it->second.coord_m;
                info.depth = cache_it->second.depth;
                info.atom_count = cache_it->second.atom_count;
                tokens.push_back(info);
                i += match_len;
                continue;
            }
        }
        
        // No multi-element match - take single element
        tokens.push_back(current[i]);
        ++i;
    }
    
    // If we reduced to fewer tokens, recurse
    if (tokens.size() < current.size() && tokens.size() > 1) {
        return ingest_with_meta(tokens, new_compositions);
    }
    
    // Create final N-ary composition for remaining sequence
    auto [rec, is_new] = create_composition(current);
    if (is_new) {
        new_compositions.push_back(rec);
    }
    
    return rec.hash;
}

Blake3Hash UniversalIngester::ingest(
    const std::vector<uint32_t>& tokens,
    const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
    std::vector<CompositionRecord>& new_compositions
) {
    if (tokens.empty()) return Blake3Hash();
    
    // Convert tokens to ChildWithMeta (atoms have depth=0, atom_count=1)
    std::vector<ChildWithMeta> atoms;
    atoms.reserve(tokens.size());
    
    for (uint32_t token : tokens) {
        auto it = atom_cache.find(token);
        if (it == atom_cache.end()) continue;  // Unknown token, skip
        
        ChildWithMeta info;
        info.hash = it->second.hash;
        info.x = it->second.coord_x;
        info.y = it->second.coord_y;
        info.z = it->second.coord_z;
        info.m = it->second.coord_m;
        info.depth = 0;        // Atoms are tier 0
        info.atom_count = 1;   // Atoms count as 1
        atoms.push_back(info);
    }
    
    if (atoms.empty()) return Blake3Hash();
    if (atoms.size() == 1) return atoms[0].hash;
    
    return impl_->ingest_with_meta(atoms, new_compositions);
}

Blake3Hash UniversalIngester::ingest_hashes(
    const std::vector<ChildInfo>& children,
    std::vector<CompositionRecord>& new_compositions
) {
    if (children.empty()) return Blake3Hash();
    if (children.size() == 1) return children[0].hash;
    
    // Convert ChildInfo to ChildWithMeta (assume depth=0, atom_count=1 if unknown)
    // For proper metadata, caller should use ingest() with atom_cache
    std::vector<ChildWithMeta> children_with_meta;
    children_with_meta.reserve(children.size());
    
    for (const auto& c : children) {
        // Check if this hash is in our composition cache for metadata (use raw hash, no hex)
        auto cache_it = impl_->comp_cache.find(c.hash);
        
        ChildWithMeta info;
        info.hash = c.hash;
        info.x = c.x;
        info.y = c.y;
        info.z = c.z;
        info.m = c.m;
        
        if (cache_it != impl_->comp_cache.end()) {
            info.depth = cache_it->second.depth;
            info.atom_count = cache_it->second.atom_count;
        } else {
            // Assume it's an atom if not in cache
            info.depth = 0;
            info.atom_count = 1;
        }
        children_with_meta.push_back(info);
    }
    
    return impl_->ingest_with_meta(children_with_meta, new_compositions);
}

} // namespace hypercube::ingest
