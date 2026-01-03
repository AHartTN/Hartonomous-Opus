/**
 * Vocabulary-Aware Ingester
 *
 * Implements proper grammar-based ingestion with persistent vocabulary:
 *
 * 1. VOCABULARY FIRST: Load all existing compositions from database
 * 2. N-GRAM DISCOVERY: Find unique patterns in content
 * 3. VOCABULARY LOOKUP: Reuse existing compositions when patterns match
 * 4. REFERENCE STORAGE: Content stored as references to vocabulary
 *
 * Key insight: "man" always produces the same hash because:
 * - First we find/create (m,a) composition
 * - Then we find/create ((m,a), n) composition
 * - Hash is deterministic: BLAKE3(ordinal||child_hash||ordinal||child_hash||...)
 *
 * This is like Sequitur but vocabulary-aware: we NEVER create a composition
 * that already exists - we reuse the existing one.
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Multi-threaded file processing
 * - Batch COPY inserts (not row-by-row)
 * - Connection pooling for parallel DB operations
 * - In-memory composition deduplication before DB insert
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <memory>
#include <optional>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;
namespace fs = std::filesystem;

// Thread-safe globals
static std::mutex g_vocab_mutex;
static std::mutex g_edge_mutex;
static std::atomic<size_t> g_total_new_compositions{0};
static std::atomic<size_t> g_total_edges{0};

// =============================================================================
// Composition Record
// =============================================================================

// Coordinates are signed (origin-centered sphere), but bit-compatible with uint32 for Hilbert
struct CompositionInfo {
    Blake3Hash hash;
    std::vector<Blake3Hash> children;
    std::vector<std::array<int32_t, 4>> child_centroids;  // For LINESTRINGZM
    int32_t centroid_x = 0, centroid_y = 0, centroid_z = 0, centroid_m = 0;
    int64_t hilbert_lo = 0, hilbert_hi = 0;
    uint32_t depth = 0;
    uint64_t atom_count = 0;
    bool from_db = false;
};

// Edge between two compositions (co-occurrence record)
struct EdgeRecord {
    Blake3Hash source;
    Blake3Hash target;
    int32_t source_x, source_y, source_z, source_m;
    int32_t target_x, target_y, target_z, target_m;
    float weight = 1.0f;
};

// =============================================================================
// Global Vocabulary Cache
// =============================================================================

// Blake3Hash hasher for use as map key (avoids .to_hex() string allocations)
struct Blake3HashHasher {
    size_t operator()(const Blake3Hash& h) const {
        size_t result;
        std::memcpy(&result, h.bytes.data(), sizeof(result));
        return result;
    }
};
struct Blake3HashEqual {
    bool operator()(const Blake3Hash& a, const Blake3Hash& b) const {
        return a.bytes == b.bytes;
    }
};

// Map from hash to composition info (NO string conversion)
static std::unordered_map<Blake3Hash, CompositionInfo, Blake3HashHasher, Blake3HashEqual> g_vocabulary;

// Map from children hash (concatenated child hashes) to composition hash
// This allows O(1) lookup: "do we have a composition for these children?"
static std::unordered_map<std::string, Blake3Hash> g_children_to_hash;

// Edge key: 64 bytes (source + target hashes concatenated)
struct EdgeKey {
    std::array<uint8_t, 64> bytes;
    bool operator==(const EdgeKey& other) const {
        return bytes == other.bytes;
    }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        // Use first 8 bytes as hash
        size_t h;
        memcpy(&h, k.bytes.data(), sizeof(h));
        return h;
    }
};
inline EdgeKey make_edge_key(const Blake3Hash& src, const Blake3Hash& tgt) {
    EdgeKey k;
    std::copy(src.bytes.begin(), src.bytes.end(), k.bytes.begin());
    std::copy(tgt.bytes.begin(), tgt.bytes.end(), k.bytes.begin() + 32);
    return k;
}

// Edge accumulator: (source_hash || target_hash) -> EdgeRecord
static std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> g_edges;

// Atom cache (leaf nodes)
struct AtomInfo {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
};
static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;
static std::unordered_map<Blake3Hash, uint32_t, Blake3HashHasher, Blake3HashEqual> g_hash_to_codepoint;  // Reverse lookup

// =============================================================================
// Helper Functions
// =============================================================================

// Bit-preserving conversions for Hilbert (which uses uint32 internally)
// These are just reinterpret casts - no value change, just type change
inline uint32_t as_uint32(int32_t val) { return static_cast<uint32_t>(val); }
inline int32_t as_int32(uint32_t val) { return static_cast<int32_t>(val); }
inline int64_t as_int64(uint64_t val) { return static_cast<int64_t>(val); }

// Build children key for lookup
// Children key using raw bytes (no string conversion)
struct ChildrenKey {
    std::vector<uint8_t> bytes;
    
    ChildrenKey() = default;
    explicit ChildrenKey(const std::vector<Blake3Hash>& children) {
        bytes.reserve(children.size() * 32);
        for (const auto& h : children) {
            bytes.insert(bytes.end(), h.bytes.begin(), h.bytes.end());
        }
    }
    
    bool operator==(const ChildrenKey& other) const {
        return bytes == other.bytes;
    }
};

struct ChildrenKeyHash {
    size_t operator()(const ChildrenKey& k) const {
        if (k.bytes.size() >= 8) {
            size_t h;
            std::memcpy(&h, k.bytes.data(), sizeof(h));
            return h;
        }
        return 0;
    }
};

// Legacy string version for compatibility during transition
std::string make_children_key(const std::vector<Blake3Hash>& children) {
    std::string key;
    key.reserve(children.size() * 64);
    for (const auto& h : children) {
        key += h.to_hex();
    }
    return key;
}

// Compute deterministic hash from children (matches C++ cpe_ingest and SQL atom_content_hash)
// Uses little-endian ordinals to match C++ native byte order
Blake3Hash compute_composition_hash(const std::vector<Blake3Hash>& children) {
    std::vector<uint8_t> hash_input;
    hash_input.reserve(children.size() * 36);  // 4 bytes ordinal + 32 bytes hash

    for (size_t i = 0; i < children.size(); ++i) {
        // Little-endian ordinal (matches C++ native on x86)
        uint32_t ordinal = static_cast<uint32_t>(i);
        hash_input.insert(hash_input.end(),
            reinterpret_cast<uint8_t*>(&ordinal),
            reinterpret_cast<uint8_t*>(&ordinal) + 4);
        hash_input.insert(hash_input.end(),
            children[i].bytes.begin(),
            children[i].bytes.end());
    }

    return Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
}

// =============================================================================
// Atom Computation (deterministic, no DB needed)
// =============================================================================

// Compute atom hash for a codepoint (deterministic)
// MUST match seed_atoms_parallel.cpp which uses UTF-8 encoding
Blake3Hash compute_atom_hash(uint32_t codepoint) {
    // Use the canonical hash_codepoint which hashes UTF-8 bytes
    return Blake3Hasher::hash_codepoint(codepoint);
}

// Thread-safe atom cache mutex
static std::mutex g_atom_mutex;

// Get or compute atom info for a codepoint (thread-safe)
const AtomInfo& get_atom(uint32_t codepoint) {
    // Fast path: check cache without lock (read-only, safe for concurrent access)
    {
        auto it = g_atom_cache.find(codepoint);
        if (it != g_atom_cache.end()) {
            return it->second;
        }
    }
    
    // Slow path: compute and cache with lock
    std::lock_guard<std::mutex> lock(g_atom_mutex);
    
    // Double-check after acquiring lock
    auto it = g_atom_cache.find(codepoint);
    if (it != g_atom_cache.end()) {
        return it->second;
    }
    
    // Compute on demand
    AtomInfo info;
    info.hash = compute_atom_hash(codepoint);
    
    Point4D coords = CoordinateMapper::map_codepoint(codepoint);
    info.coord_x = static_cast<int32_t>(coords.x);
    info.coord_y = static_cast<int32_t>(coords.y);
    info.coord_z = static_cast<int32_t>(coords.z);
    info.coord_m = static_cast<int32_t>(coords.m);
    
    g_atom_cache[codepoint] = info;
    g_hash_to_codepoint[info.hash] = codepoint;
    
    // Also add to vocabulary
    CompositionInfo comp;
    comp.hash = info.hash;
    comp.centroid_x = info.coord_x;
    comp.centroid_y = info.coord_y;
    comp.centroid_z = info.coord_z;
    comp.centroid_m = info.coord_m;
    comp.depth = 0;
    comp.atom_count = 1;
    comp.from_db = true;  // Treat as existing (deterministic)
    g_vocabulary[info.hash] = comp;
    
    return g_atom_cache[codepoint];
}

// =============================================================================
// Database Loading
// =============================================================================

// Load atoms that are referenced by compositions (for trie reconstruction)
// This is much faster than loading all 1.1M atoms
bool load_referenced_atoms(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Get unique atoms referenced by compositions
    PGresult* res = PQexec(conn,
        "WITH referenced AS ("
        "  SELECT DISTINCT unnest(children) as child_id FROM atom WHERE depth > 0"
        ") "
        "SELECT a.id, a.codepoint, "
        "       ST_X(a.geom), ST_Y(a.geom), ST_Z(a.geom), ST_M(a.geom) "
        "FROM atom a "
        "JOIN referenced r ON a.id = r.child_id "
        "WHERE a.depth = 0 AND a.codepoint IS NOT NULL");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load referenced atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    
    int rows = PQntuples(res);
    
    for (int i = 0; i < rows; ++i) {
        const char* id_hex = PQgetvalue(res, i, 0);
        Blake3Hash hash;
        if (id_hex[0] == '\\' && id_hex[1] == 'x') {
            hash = Blake3Hash::from_hex(std::string_view(id_hex + 2, 64));
        }
        
        uint32_t codepoint = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 1)));
        
        int32_t cx = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 2)));
        int32_t cy = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 3)));
        int32_t cz = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 4)));
        int32_t cm = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 5)));
        
        AtomInfo info;
        info.hash = hash;
        info.coord_x = cx;
        info.coord_y = cy;
        info.coord_z = cz;
        info.coord_m = cm;
        g_atom_cache[codepoint] = info;
        
        g_hash_to_codepoint[hash] = codepoint;
        
        CompositionInfo comp;
        comp.hash = hash;
        comp.centroid_x = cx;
        comp.centroid_y = cy;
        comp.centroid_z = cz;
        comp.centroid_m = cm;
        comp.depth = 0;
        comp.atom_count = 1;
        comp.from_db = true;
        g_vocabulary[hash] = comp;
    }
    
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[ATOMS] Loaded " << rows << " referenced atoms in " << ms << " ms\n";
    
    return true;
}

bool load_vocabulary(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();

    // Load all existing compositions (depth > 0)
    PGresult* res = PQexec(conn,
        "SELECT id, children, "
        "       ST_X(ST_Centroid(geom)), ST_Y(ST_Centroid(geom)), "
        "       ST_Z(ST_Centroid(geom)), ST_M(ST_Centroid(geom)), "
        "       hilbert_lo, hilbert_hi, depth, atom_count "
        "FROM atom WHERE depth > 0 AND children IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load vocabulary: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    std::cerr << "[VOCAB] Loading " << rows << " existing compositions...\n";

    for (int i = 0; i < rows; ++i) {
        CompositionInfo comp;
        comp.from_db = true;

        // Parse hash
        const char* id_hex = PQgetvalue(res, i, 0);
        if (id_hex[0] == '\\' && id_hex[1] == 'x') {
            comp.hash = Blake3Hash::from_hex(std::string_view(id_hex + 2, 64));
        }

        // Parse children array: {"\x...","\x..."}
        const char* children_str = PQgetvalue(res, i, 1);
        if (children_str && children_str[0] == '{') {
            std::string arr(children_str);
            size_t pos = 0;
            while ((pos = arr.find("\\\\x", pos)) != std::string::npos) {
                if (pos + 67 <= arr.size()) {
                    std::string hex = arr.substr(pos + 3, 64);
                    comp.children.push_back(Blake3Hash::from_hex(hex));
                }
                pos += 67;
            }
        }

        // Parse coordinates (with null check)
        const char* cx = PQgetvalue(res, i, 2);
        const char* cy = PQgetvalue(res, i, 3);
        const char* cz = PQgetvalue(res, i, 4);
        const char* cm = PQgetvalue(res, i, 5);
        
        if (cx && *cx && cy && *cy && cz && *cz && cm && *cm) {
            comp.centroid_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(cx)));
            comp.centroid_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(cy)));
            comp.centroid_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(cz)));
            comp.centroid_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(cm)));
        }
        comp.hilbert_lo = std::stoll(PQgetvalue(res, i, 6));
        comp.hilbert_hi = std::stoll(PQgetvalue(res, i, 7));
        comp.depth = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 8)));
        comp.atom_count = std::stoull(PQgetvalue(res, i, 9));

        g_vocabulary[comp.hash] = comp;

        // Index by children for fast lookup
        if (!comp.children.empty()) {
            std::string children_key = make_children_key(comp.children);
            g_children_to_hash[children_key] = comp.hash;
        }
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[VOCAB] Loaded " << g_vocabulary.size() << " entries in " << ms << " ms\n";

    return true;
}

// =============================================================================
// Vocabulary Trie for Greedy Longest-Match Tokenization
// =============================================================================

// Trie node for vocabulary lookup
struct TrieNode {
    std::unordered_map<uint32_t, std::unique_ptr<TrieNode>> children;
    std::optional<Blake3Hash> composition_hash;  // If this is end of a vocab entry
    size_t length = 0;  // Length in codepoints
};

class VocabTrie {
public:
    TrieNode root;
    
    // Insert a codepoint sequence with its composition hash
    void insert(const std::vector<uint32_t>& codepoints, const Blake3Hash& hash) {
        TrieNode* node = &root;
        for (uint32_t cp : codepoints) {
            if (!node->children.count(cp)) {
                node->children[cp] = std::make_unique<TrieNode>();
            }
            node = node->children[cp].get();
        }
        node->composition_hash = hash;
        node->length = codepoints.size();
    }
    
    // Find longest matching composition starting at position
    // Returns {hash, length} or {nullopt, 0} if no match
    std::pair<std::optional<Blake3Hash>, size_t> longest_match(
        const std::vector<uint32_t>& codepoints, size_t start
    ) const {
        std::optional<Blake3Hash> best_hash;
        size_t best_len = 0;
        
        const TrieNode* node = &root;
        for (size_t i = start; i < codepoints.size(); ++i) {
            auto it = node->children.find(codepoints[i]);
            if (it == node->children.end()) break;
            
            node = it->second.get();
            if (node->composition_hash) {
                best_hash = node->composition_hash;
                best_len = i - start + 1;
            }
        }
        
        return {best_hash, best_len};
    }
};

static VocabTrie g_vocab_trie;

// Reconstruct codepoints from a composition by traversing children
std::vector<uint32_t> reconstruct_codepoints(const CompositionInfo& comp) {
    std::vector<uint32_t> result;
    
    if (comp.depth == 0) {
        // O(1) reverse lookup instead of O(N) linear scan
        auto it = g_hash_to_codepoint.find(comp.hash);
        if (it != g_hash_to_codepoint.end()) {
            result.push_back(it->second);
        }
        return result;
    }
    
    // Recursively reconstruct from children
    for (const auto& child_hash : comp.children) {
        auto it = g_vocabulary.find(child_hash);
        if (it != g_vocabulary.end()) {
            auto child_cps = reconstruct_codepoints(it->second);
            result.insert(result.end(), child_cps.begin(), child_cps.end());
        }
    }
    
    return result;
}

// Build trie from loaded vocabulary
void build_vocab_trie() {
    auto start = std::chrono::high_resolution_clock::now();
    size_t indexed = 0;
    
    for (const auto& [hash, comp] : g_vocabulary) {
        if (comp.depth == 0) continue;  // Skip atoms
        
        std::vector<uint32_t> codepoints = reconstruct_codepoints(comp);
        if (!codepoints.empty()) {
            g_vocab_trie.insert(codepoints, hash);
            indexed++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[TRIE] Built vocabulary trie with " << indexed << " entries in " << ms << " ms\n";
}

// Add a new composition to the trie (for incremental updates)
void add_to_trie(const CompositionInfo& comp) {
    if (comp.depth == 0) return;  // Skip atoms
    
    std::vector<uint32_t> codepoints = reconstruct_codepoints(comp);
    if (!codepoints.empty()) {
        std::lock_guard<std::mutex> lock(g_vocab_mutex);
        g_vocab_trie.insert(codepoints, comp.hash);
    }
}

// =============================================================================
// Vocabulary-Aware Composition Creation
// =============================================================================

// Find or create a composition for the given children
// Returns the hash of the composition (existing or new)
const CompositionInfo& find_or_create_composition(
    const std::vector<Blake3Hash>& children,
    std::vector<CompositionInfo>& new_compositions
) {
    // Check if composition for these exact children already exists
    std::string children_key = make_children_key(children);
    auto it = g_children_to_hash.find(children_key);
    if (it != g_children_to_hash.end()) {
        // Found existing composition
        return g_vocabulary[it->second];
    }

    // Need to create new composition
    CompositionInfo comp;
    comp.hash = compute_composition_hash(children);
    comp.children = children;
    comp.from_db = false;

    // Check if hash already exists (shouldn't happen if children_key lookup works)
    auto hash_it = g_vocabulary.find(comp.hash);
    if (hash_it != g_vocabulary.end()) {
        return hash_it->second;
    }

    // Compute centroid from children
    uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    uint32_t max_depth = 0;
    uint64_t total_atoms = 0;

    for (const auto& child_hash : children) {
        auto child_it = g_vocabulary.find(child_hash);
        if (child_it != g_vocabulary.end()) {
            const CompositionInfo& child = child_it->second;
            sum_x += as_uint32(child.centroid_x);
            sum_y += as_uint32(child.centroid_y);
            sum_z += as_uint32(child.centroid_z);
            sum_m += as_uint32(child.centroid_m);
            max_depth = std::max(max_depth, child.depth);
            total_atoms += child.atom_count;
        }
    }

    size_t n = children.size();
    if (n > 0) {
        comp.centroid_x = as_int32(static_cast<uint32_t>(sum_x / n));
        comp.centroid_y = as_int32(static_cast<uint32_t>(sum_y / n));
        comp.centroid_z = as_int32(static_cast<uint32_t>(sum_z / n));
        comp.centroid_m = as_int32(static_cast<uint32_t>(sum_m / n));
    }

    // Compute Hilbert index
    Point4D coords(
        as_uint32(comp.centroid_x),
        as_uint32(comp.centroid_y),
        as_uint32(comp.centroid_z),
        as_uint32(comp.centroid_m)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    comp.hilbert_lo = as_int64(hilbert.lo);
    comp.hilbert_hi = as_int64(hilbert.hi);

    comp.depth = max_depth + 1;
    comp.atom_count = total_atoms;

    // Add to vocabulary
    g_vocabulary[comp.hash] = comp;
    g_children_to_hash[children_key] = comp.hash;
    
    // Also add to trie for future greedy matching
    add_to_trie(comp);

    // Track for database insertion
    new_compositions.push_back(comp);

    return g_vocabulary[comp.hash];
}

// Record an edge between two compositions (or accumulate weight if exists)
// Thread-safe version with mutex
void record_edge(const CompositionInfo& source, const CompositionInfo& target, float weight = 1.0f) {
    // Canonical key: smaller hash first for symmetric lookup
    bool src_first = std::memcmp(source.hash.bytes.data(), target.hash.bytes.data(), 32) < 0;
    EdgeKey edge_key = src_first 
        ? make_edge_key(source.hash, target.hash) 
        : make_edge_key(target.hash, source.hash);
    
    std::lock_guard<std::mutex> lock(g_edge_mutex);
    
    auto it = g_edges.find(edge_key);
    if (it != g_edges.end()) {
        // Accumulate weight
        it->second.weight += weight;
    } else {
        // Create new edge
        EdgeRecord edge;
        if (src_first) {
            edge.source = source.hash;
            edge.target = target.hash;
            edge.source_x = source.centroid_x;
            edge.source_y = source.centroid_y;
            edge.source_z = source.centroid_z;
            edge.source_m = source.centroid_m;
            edge.target_x = target.centroid_x;
            edge.target_y = target.centroid_y;
            edge.target_z = target.centroid_z;
            edge.target_m = target.centroid_m;
        } else {
            edge.source = target.hash;
            edge.target = source.hash;
            edge.source_x = target.centroid_x;
            edge.source_y = target.centroid_y;
            edge.source_z = target.centroid_z;
            edge.source_m = target.centroid_m;
            edge.target_x = source.centroid_x;
            edge.target_y = source.centroid_y;
            edge.target_z = source.centroid_z;
            edge.target_m = source.centroid_m;
        }
        edge.weight = weight;
        g_edges[edge_key] = edge;
    }
}

// =============================================================================
// Greedy Vocabulary-Aware Tokenization
// =============================================================================

// Node representing a token (either matched from vocabulary or single atom)
struct CascadeNode {
    Blake3Hash hash;
    int32_t cx, cy, cz, cm;
    uint32_t depth;
    uint64_t atom_count;
    bool from_vocab;  // True if this came from vocabulary match (no need to store)
};

// Cascade a sequence of atoms into a single composition
// Returns the root composition, adds new compositions to new_compositions
CascadeNode cascade_atoms(
    const std::vector<uint32_t>& codepoints,
    size_t start, size_t end,
    std::vector<CompositionInfo>& new_compositions
) {
    if (start >= end) return CascadeNode{};
    
    // Build initial atom nodes
    std::vector<CascadeNode> current;
    current.reserve(end - start);
    
    for (size_t i = start; i < end; ++i) {
        const AtomInfo& atom = get_atom(codepoints[i]);
        CascadeNode node;
        node.hash = atom.hash;
        node.cx = atom.coord_x;
        node.cy = atom.coord_y;
        node.cz = atom.coord_z;
        node.cm = atom.coord_m;
        node.depth = 0;
        node.atom_count = 1;
        node.from_vocab = true;  // Atoms always exist
        current.push_back(node);
    }
    
    if (current.size() == 1) {
        return current[0];
    }
    
    // Binary cascade until single root
    while (current.size() > 1) {
        std::vector<CascadeNode> next;
        next.reserve((current.size() + 1) / 2);
        
        for (size_t i = 0; i < current.size(); i += 2) {
            if (i + 1 >= current.size()) {
                // Odd element - carry forward
                next.push_back(current[i]);
                continue;
            }
            
            CascadeNode& left = current[i];
            CascadeNode& right = current[i + 1];
            
            // Compute pair hash
            std::vector<Blake3Hash> children = {left.hash, right.hash};
            Blake3Hash hash = compute_composition_hash(children);
            
            // Check if this composition already exists in vocabulary
            auto vocab_it = g_vocabulary.find(hash);
            
            CascadeNode node;
            node.hash = hash;
            node.cx = (left.cx + right.cx) / 2;
            node.cy = (left.cy + right.cy) / 2;
            node.cz = (left.cz + right.cz) / 2;
            node.cm = (left.cm + right.cm) / 2;
            node.depth = std::max(left.depth, right.depth) + 1;
            node.atom_count = left.atom_count + right.atom_count;
            
            if (vocab_it != g_vocabulary.end()) {
                // Already exists - reuse
                node.from_vocab = true;
            } else {
                // New composition - create and add to vocabulary
                node.from_vocab = false;
                
                CompositionInfo comp;
                comp.hash = hash;
                comp.children = children;
                comp.child_centroids = {
                    {left.cx, left.cy, left.cz, left.cm},
                    {right.cx, right.cy, right.cz, right.cm}
                };
                comp.centroid_x = node.cx;
                comp.centroid_y = node.cy;
                comp.centroid_z = node.cz;
                comp.centroid_m = node.cm;
                comp.depth = node.depth;
                comp.atom_count = node.atom_count;
                comp.from_db = false;
                
                // Compute Hilbert
                Point4D coords(
                    static_cast<uint32_t>(static_cast<int64_t>(node.cx) + INT32_MAX + 1),
                    static_cast<uint32_t>(static_cast<int64_t>(node.cy) + INT32_MAX + 1),
                    static_cast<uint32_t>(static_cast<int64_t>(node.cz) + INT32_MAX + 1),
                    static_cast<uint32_t>(static_cast<int64_t>(node.cm) + INT32_MAX + 1)
                );
                HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
                comp.hilbert_lo = static_cast<int64_t>(hilbert.lo);
                comp.hilbert_hi = static_cast<int64_t>(hilbert.hi);
                
                // Add to vocabulary for future reuse
                {
                    std::lock_guard<std::mutex> lock(g_vocab_mutex);
                    g_vocabulary[hash] = comp;
                    g_children_to_hash[make_children_key(children)] = hash;
                }
                
                // Also add to trie for future greedy matching
                add_to_trie(comp);
                
                new_compositions.push_back(comp);
            }
            
            next.push_back(node);
        }
        
        current = std::move(next);
    }
    
    return current[0];
}

// Greedy vocabulary-aware tokenization + cascade
// 1. Greedy longest-match against vocabulary trie
// 2. For unmatched gaps: cascade from atoms
// 3. Cascade all tokens into document root
// Returns the root hash of the document composition
Blake3Hash tokenize_and_ingest(
    const std::vector<uint32_t>& codepoints,
    std::vector<CompositionInfo>& new_compositions
) {
    if (codepoints.empty()) return Blake3Hash();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // ==========================================================================
    // GREEDY TOKENIZATION
    // ==========================================================================
    // Match longest compositions from vocabulary. Gaps become atom references.
    // Result: list of existing composition/atom hashes - NO new compositions yet.
    
    std::vector<Blake3Hash> token_hashes;
    token_hashes.reserve(codepoints.size() / 4);
    
    std::vector<std::array<int32_t, 4>> token_centroids;
    token_centroids.reserve(codepoints.size() / 4);
    
    size_t reused_count = 0;
    size_t atom_count = 0;
    
    size_t i = 0;
    while (i < codepoints.size()) {
        auto [match_hash, match_len] = g_vocab_trie.longest_match(codepoints, i);
        
        if (match_hash && match_len > 1) {
            // Found existing composition
            auto vocab_it = g_vocabulary.find(*match_hash);
            if (vocab_it != g_vocabulary.end()) {
                token_hashes.push_back(*match_hash);
                token_centroids.push_back({
                    vocab_it->second.centroid_x,
                    vocab_it->second.centroid_y,
                    vocab_it->second.centroid_z,
                    vocab_it->second.centroid_m
                });
                i += match_len;
                reused_count++;
                continue;
            }
        }
        
        // No match - use atom
        const AtomInfo& atom = get_atom(codepoints[i]);
        token_hashes.push_back(atom.hash);
        token_centroids.push_back({
            atom.coord_x, atom.coord_y, atom.coord_z, atom.coord_m
        });
        i++;
        atom_count++;
    }
    
    auto tokenize_time = std::chrono::high_resolution_clock::now();
    auto tokenize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tokenize_time - start_time).count();
    
    std::cerr << "[TOKENIZE] " << codepoints.size() << " codepoints → " 
              << token_hashes.size() << " tokens (" << reused_count << " vocab, "
              << atom_count << " atoms) in " << tokenize_ms << "ms\n";
    
    // ==========================================================================
    // DOCUMENT COMPOSITION
    // ==========================================================================
    // The document IS this list of token hashes. Create ONE composition that
    // references all of them. No binary cascade - just a flat children array.
    
    if (token_hashes.size() == 1) {
        return token_hashes[0];
    }
    
    // Compute document hash from all children
    Blake3Hash doc_hash = compute_composition_hash(token_hashes);
    
    // Check if document already exists
    auto doc_it = g_vocabulary.find(doc_hash);
    if (doc_it != g_vocabulary.end()) {
        std::cerr << "[DOC] Already exists\n";
        return doc_hash;
    }
    
    // Compute centroid as average of all tokens
    int64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    for (const auto& c : token_centroids) {
        sum_x += c[0];
        sum_y += c[1];
        sum_z += c[2];
        sum_m += c[3];
    }
    size_t n = token_centroids.size();
    
    CompositionInfo doc;
    doc.hash = doc_hash;
    doc.children = token_hashes;
    doc.child_centroids = token_centroids;
    doc.centroid_x = static_cast<int32_t>(sum_x / static_cast<int64_t>(n));
    doc.centroid_y = static_cast<int32_t>(sum_y / static_cast<int64_t>(n));
    doc.centroid_z = static_cast<int32_t>(sum_z / static_cast<int64_t>(n));
    doc.centroid_m = static_cast<int32_t>(sum_m / static_cast<int64_t>(n));
    doc.depth = 1;  // Flat structure - one level above tokens
    doc.atom_count = codepoints.size();
    doc.from_db = false;
    
    // Compute Hilbert
    Point4D coords(
        static_cast<uint32_t>(static_cast<int64_t>(doc.centroid_x) + INT32_MAX + 1),
        static_cast<uint32_t>(static_cast<int64_t>(doc.centroid_y) + INT32_MAX + 1),
        static_cast<uint32_t>(static_cast<int64_t>(doc.centroid_z) + INT32_MAX + 1),
        static_cast<uint32_t>(static_cast<int64_t>(doc.centroid_m) + INT32_MAX + 1)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    doc.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    doc.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    
    new_compositions.push_back(doc);
    
    // Add to vocabulary
    {
        std::lock_guard<std::mutex> lock(g_vocab_mutex);
        g_vocabulary[doc_hash] = doc;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "[DOC] Created document composition with " << token_hashes.size() 
              << " children in " << total_ms << "ms\n";
    
    return doc_hash;
}

// Legacy function name for compatibility
void discover_ngrams(
    const std::vector<uint32_t>& codepoints,
    std::vector<CompositionInfo>& new_compositions,
    size_t max_ngram_size = 32
) {
    tokenize_and_ingest(codepoints, new_compositions);
}

// =============================================================================
// Database Insertion
// =============================================================================

std::string build_linestringzm_ewkb(const std::vector<std::array<int32_t, 4>>& points) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string ewkb;

    ewkb += "01020000c0";  // Little-endian, LINESTRINGZM

    uint32_t npoints = static_cast<uint32_t>(points.size());
    for (int i = 0; i < 4; ++i) {
        uint8_t byte = (npoints >> (i * 8)) & 0xFF;
        ewkb += hex_chars[byte >> 4];
        ewkb += hex_chars[byte & 0x0F];
    }

    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };

    for (const auto& pt : points) {
        append_double(static_cast<double>(as_uint32(pt[0])));
        append_double(static_cast<double>(as_uint32(pt[1])));
        append_double(static_cast<double>(as_uint32(pt[2])));
        append_double(static_cast<double>(as_uint32(pt[3])));
    }

    return ewkb;
}

bool insert_compositions(PGconn* conn, const std::vector<CompositionInfo>& compositions) {
    if (compositions.empty()) return true;

    // Step 1: Deduplicate in memory
    std::unordered_map<std::string, const CompositionInfo*> unique;
    unique.reserve(compositions.size());
    
    for (const auto& comp : compositions) {
        if (comp.from_db || comp.children.empty()) continue;
        std::string key = comp.hash.to_hex();
        if (unique.find(key) == unique.end()) {
            unique[key] = &comp;
        }
    }
    
    if (unique.empty()) {
        std::cerr << "[DB] No new unique compositions to insert\n";
        return true;
    }

    std::cerr << "[DB] " << unique.size() << " unique compositions to check\n";
    
    // Step 2: Batch existence check using temp table (handles any size)
    auto check_start = std::chrono::high_resolution_clock::now();
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table for hashes to check
    res = PQexec(conn, "CREATE TEMP TABLE tmp_check (id BYTEA PRIMARY KEY) ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_check failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY hashes to temp table
    res = PQexec(conn, "COPY tmp_check (id) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY tmp_check failed\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 18);
    for (const auto& [hex, _] : unique) {
        batch += "\\\\x";
        batch += hex;
        batch += "\n";
        if (batch.size() > (1 << 17)) {
            PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    if (!batch.empty()) {
        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Find which ones already exist (single indexed join)
    res = PQexec(conn, "SELECT t.id FROM tmp_check t JOIN atom a ON t.id = a.id");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Existence check failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    // Remove existing from our set
    int existing_count = PQntuples(res);
    for (int i = 0; i < existing_count; ++i) {
        const char* id_hex = PQgetvalue(res, i, 0);
        if (id_hex && id_hex[0] == '\\' && id_hex[1] == 'x') {
            unique.erase(std::string(id_hex + 2, 64));
        }
    }
    PQclear(res);
    
    auto check_end = std::chrono::high_resolution_clock::now();
    auto check_ms = std::chrono::duration_cast<std::chrono::milliseconds>(check_end - check_start).count();
    
    if (unique.empty()) {
        std::cerr << "[DB] All " << existing_count << " compositions exist (" << check_ms << " ms)\n";
        PQexec(conn, "COMMIT");
        return true;
    }
    
    std::cerr << "[DB] " << unique.size() << " new, " << existing_count 
              << " exist (check: " << check_ms << " ms)\n";

    // Step 3: COPY only the truly new ones

    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_atom ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(GEOMETRYZM, 0) NOT NULL,"
        "  children BYTEA[],"
        "  hilbert_lo BIGINT NOT NULL,"
        "  hilbert_hi BIGINT NOT NULL,"
        "  depth INTEGER NOT NULL,"
        "  atom_count BIGINT NOT NULL"
        ") ON COMMIT DROP");

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    res = PQexec(conn,
        "COPY tmp_atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");

    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    std::string comp_batch;
    comp_batch.reserve(1 << 20);
    char num_buf[32];

    for (const auto& [hex, comp] : unique) {

        // Use pre-computed child centroids (no lookup needed)
        if (comp->child_centroids.empty()) continue;

        // id
        comp_batch += "\\\\x";
        comp_batch += comp->hash.to_hex();
        comp_batch += '\t';

        // geom
        comp_batch += build_linestringzm_ewkb(comp->child_centroids);
        comp_batch += '\t';

        // children
        comp_batch += "{";
        for (size_t i = 0; i < comp->children.size(); ++i) {
            if (i > 0) comp_batch += ",";
            comp_batch += "\"\\\\\\\\x";
            comp_batch += comp->children[i].to_hex();
            comp_batch += "\"";
        }
        comp_batch += "}";
        comp_batch += '\t';

        // hilbert_lo, hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(comp->hilbert_lo));
        comp_batch += num_buf;
        comp_batch += '\t';

        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(comp->hilbert_hi));
        comp_batch += num_buf;
        comp_batch += '\t';

        // depth
        snprintf(num_buf, sizeof(num_buf), "%u", comp->depth);
        comp_batch += num_buf;
        comp_batch += '\t';

        // atom_count
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(comp->atom_count));
        comp_batch += num_buf;
        comp_batch += '\n';

        if (comp_batch.size() > (1 << 19)) {
            if (PQputCopyData(conn, comp_batch.c_str(), static_cast<int>(comp_batch.size())) != 1) {
                std::cerr << "COPY data failed\n";
                PQputCopyEnd(conn, "error");
                return false;
            }
            comp_batch.clear();
        }
    }

    if (!comp_batch.empty()) {
        PQputCopyData(conn, comp_batch.c_str(), static_cast<int>(comp_batch.size()));
    }

    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);

    // Direct insert - no ON CONFLICT needed since we pre-filtered
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count FROM tmp_atom");

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Insert failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }

    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);

    res = PQexec(conn, "COMMIT");
    PQclear(res);

    std::cerr << "[DB] Inserted " << inserted << " compositions\n";
    return true;
}

// Insert accumulated edges into database
bool insert_edges(PGconn* conn, float threshold = 0.0f) {
    // Filter edges by threshold and compute IDs
    std::unordered_map<std::string, const EdgeRecord*> to_insert;
    for (const auto& [key, edge] : g_edges) {
        if (edge.weight >= threshold) {
            std::vector<uint8_t> id_input;
            id_input.insert(id_input.end(), edge.source.bytes.begin(), edge.source.bytes.end());
            id_input.insert(id_input.end(), edge.target.bytes.begin(), edge.target.bytes.end());
            Blake3Hash edge_id = Blake3Hasher::hash(std::span<const uint8_t>(id_input));
            to_insert[edge_id.to_hex()] = &edge;
        }
    }
    
    if (to_insert.empty()) {
        std::cerr << "[EDGES] No edges above threshold\n";
        return true;
    }
    
    std::cerr << "[EDGES] " << to_insert.size() << " edges to check\n";
    auto check_start = std::chrono::high_resolution_clock::now();
    
    // Batch existence check using temp table
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    res = PQexec(conn, "CREATE TEMP TABLE tmp_edge_check (id BYTEA PRIMARY KEY) ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COPY tmp_edge_check (id) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 18);
    for (const auto& [hex, _] : to_insert) {
        batch += "\\\\x";
        batch += hex;
        batch += "\n";
        if (batch.size() > (1 << 17)) {
            PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    if (!batch.empty()) {
        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    res = PQexec(conn, "SELECT encode(t.id, 'hex'), ST_M(ST_StartPoint(a.geom)) FROM tmp_edge_check t JOIN atom a ON t.id = a.id WHERE a.depth = 1");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    // Track existing edges and their current weights for accumulation
    std::unordered_map<std::string, float> existing_weights;
    int existing_count = PQntuples(res);
    for (int i = 0; i < existing_count; ++i) {
        const char* id_hex = PQgetvalue(res, i, 0);
        const char* weight_str = PQgetvalue(res, i, 1);
        if (id_hex && weight_str) {
            existing_weights[std::string(id_hex, 64)] = std::stof(weight_str);
        }
    }
    PQclear(res);
    
    // Separate new edges from edges needing weight update
    std::unordered_map<std::string, const EdgeRecord*> new_edges;
    std::unordered_map<std::string, float> edges_to_update;  // edge_hex -> new_weight
    
    for (const auto& [hex, edge] : to_insert) {
        auto existing_it = existing_weights.find(hex);
        if (existing_it != existing_weights.end()) {
            // Edge exists - accumulate weight
            edges_to_update[hex] = existing_it->second + edge->weight;
        } else {
            // New edge
            new_edges[hex] = edge;
        }
    }
    
    auto check_end = std::chrono::high_resolution_clock::now();
    auto check_ms = std::chrono::duration_cast<std::chrono::milliseconds>(check_end - check_start).count();
    
    std::cerr << "[EDGES] " << new_edges.size() << " new, " << edges_to_update.size() 
              << " to accumulate (check: " << check_ms << " ms)\n";
    
    // Update existing edges to accumulate weight in M coordinate
    if (!edges_to_update.empty()) {
        for (const auto& [hex, new_weight] : edges_to_update) {
            std::string sql = 
                "UPDATE atom SET geom = ST_SetSRID(ST_MakeLine("
                "  ST_SetSRID(ST_MakePoint(ST_X(ST_StartPoint(geom)), ST_Y(ST_StartPoint(geom)), "
                "    ST_Z(ST_StartPoint(geom)), " + std::to_string(new_weight) + "), 0),"
                "  ST_SetSRID(ST_MakePoint(ST_X(ST_EndPoint(geom)), ST_Y(ST_EndPoint(geom)), "
                "    ST_Z(ST_EndPoint(geom)), " + std::to_string(new_weight) + "), 0)"
                "), 0) WHERE id = '\\x" + hex + "'";
            res = PQexec(conn, sql.c_str());
            PQclear(res);
        }
        std::cerr << "[EDGES] Accumulated weight on " << edges_to_update.size() << " existing edges\n";
    }
    
    if (new_edges.empty()) {
        res = PQexec(conn, "COMMIT");
        PQclear(res);
        return true;
    }
    
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_edge ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(LINESTRINGZM, 0),"
        "  children BYTEA[],"
        "  hilbert_lo BIGINT,"
        "  hilbert_hi BIGINT,"
        "  depth INTEGER,"
        "  atom_count BIGINT"
        ") ON COMMIT DROP");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_edge failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COPY tmp_edge FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    static const char hex_chars[] = "0123456789abcdef";
    std::string edge_batch;
    edge_batch.reserve(1 << 20);
    
    for (const auto& [edge_hex, e] : new_edges) {
        // Recompute edge ID (we already have it as the key)
        Blake3Hash edge_id = Blake3Hash::from_hex(edge_hex);
        
        // Compute centroid
        uint64_t sum_x = as_uint32(e->source_x) + as_uint32(e->target_x);
        uint64_t sum_y = as_uint32(e->source_y) + as_uint32(e->target_y);
        uint64_t sum_z = as_uint32(e->source_z) + as_uint32(e->target_z);
        uint64_t sum_m = as_uint32(e->source_m) + as_uint32(e->target_m);
        
        uint32_t cx = static_cast<uint32_t>(sum_x / 2);
        uint32_t cy = static_cast<uint32_t>(sum_y / 2);
        uint32_t cz = static_cast<uint32_t>(sum_z / 2);
        uint32_t cm = static_cast<uint32_t>(sum_m / 2);
        
        Point4D coords(cx, cy, cz, cm);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        
        // ID
        edge_batch += "\\\\x";
        for (uint8_t b : edge_id.bytes) {
            edge_batch += hex_chars[b >> 4];
            edge_batch += hex_chars[b & 0x0F];
        }
        edge_batch += "\t";
        
        // GEOM - LINESTRINGZM with weight in M coordinate
        edge_batch += "01020000c002000000";  // LE + LINESTRINGZM + 2 points
        
        auto append_double = [&](double val) {
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            for (int i = 0; i < 8; ++i) {
                uint8_t byte = (bits >> (i * 8)) & 0xFF;
                edge_batch += hex_chars[byte >> 4];
                edge_batch += hex_chars[byte & 0x0F];
            }
        };
        
        // Point 1: source centroid, M = weight
        append_double(static_cast<double>(as_uint32(e->source_x)));
        append_double(static_cast<double>(as_uint32(e->source_y)));
        append_double(static_cast<double>(as_uint32(e->source_z)));
        append_double(static_cast<double>(e->weight));  // M = weight!
        
        // Point 2: target centroid, M = weight
        append_double(static_cast<double>(as_uint32(e->target_x)));
        append_double(static_cast<double>(as_uint32(e->target_y)));
        append_double(static_cast<double>(as_uint32(e->target_z)));
        append_double(static_cast<double>(e->weight));  // M = weight!
        
        edge_batch += "\t";
        
        // Children array [source, target]
        edge_batch += "{\"\\\\\\\\x";
        for (uint8_t b : e->source.bytes) {
            edge_batch += hex_chars[b >> 4];
            edge_batch += hex_chars[b & 0x0F];
        }
        edge_batch += "\",\"\\\\\\\\x";
        for (uint8_t b : e->target.bytes) {
            edge_batch += hex_chars[b >> 4];
            edge_batch += hex_chars[b & 0x0F];
        }
        edge_batch += "\"}\t";
        
        // Hilbert
        edge_batch += std::to_string(as_int64(hilbert.lo));
        edge_batch += "\t";
        edge_batch += std::to_string(as_int64(hilbert.hi));
        edge_batch += "\t";
        
        // Depth = 1 (edges are depth 1)
        edge_batch += "1\t";
        
        // Atom count = 2
        edge_batch += "2\n";
        
        if (edge_batch.size() > (1 << 19)) {
            PQputCopyData(conn, edge_batch.c_str(), static_cast<int>(edge_batch.size()));
            edge_batch.clear();
        }
    }
    
    if (!edge_batch.empty()) {
        PQputCopyData(conn, edge_batch.c_str(), static_cast<int>(edge_batch.size()));
    }
    
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Direct insert - no ON CONFLICT since we pre-filtered
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count FROM tmp_edge");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Edge insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[EDGES] Inserted " << inserted << " new edges\n";
    return true;
}

// =============================================================================
// UTF-8 Decoding
// =============================================================================

std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());

    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();

    while (p < end) {
        uint32_t cp;

        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            cp = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            cp = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            cp = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            cp = 0xFFFD;
            ++p;
        }

        codepoints.push_back(cp);
    }

    return codepoints;
}

// =============================================================================
// File Processing
// =============================================================================

struct IngestResult {
    size_t bytes = 0;
    size_t codepoints = 0;
    size_t new_compositions = 0;
    size_t reused_compositions = 0;
    double seconds = 0;
};

// Process a single file (CPU-bound, no DB operations)
// Returns compositions created - caller batches DB inserts
IngestResult process_file_content(const fs::path& path, std::vector<CompositionInfo>& new_compositions) {
    IngestResult result;
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot read: " << path << std::endl;
        return result;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    result.bytes = content.size();

    std::vector<uint32_t> codepoints = decode_utf8(content);
    result.codepoints = codepoints.size();

    if (codepoints.empty()) return result;

    size_t vocab_before = g_vocabulary.size();

    // Discover n-grams and create compositions (thread-safe with vocab mutex)
    discover_ngrams(codepoints, new_compositions);

    result.new_compositions = new_compositions.size();
    result.reused_compositions = (g_vocabulary.size() - vocab_before) - new_compositions.size();

    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();

    return result;
}

// =============================================================================
// Vocabulary File Ingestion (Optimized for vocab.txt format)
// =============================================================================
// Each line is a complete token - no overlapping n-grams needed
// Process in parallel, bulk insert

IngestResult ingest_vocab_file(PGconn* conn, const fs::path& path) {
    IngestResult result;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Read entire file
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot read: " << path << std::endl;
        return result;
    }
    
    // Parse lines into tokens
    std::vector<std::string> tokens;
    tokens.reserve(50000);  // Pre-allocate for typical vocab size
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            tokens.push_back(std::move(line));
        }
    }
    
    std::cerr << "[VOCAB] Read " << tokens.size() << " tokens from " << path.filename() << "\n";
    result.bytes = 0;
    result.codepoints = tokens.size();
    
    // Pre-compute all unique codepoints single-threaded (populates atom cache)
    std::unordered_set<uint32_t> unique_codepoints;
    for (const auto& token : tokens) {
        std::vector<uint32_t> cps = decode_utf8(token);
        for (uint32_t cp : cps) {
            unique_codepoints.insert(cp);
        }
    }
    std::cerr << "[VOCAB] Found " << unique_codepoints.size() << " unique codepoints\n";
    
    // Pre-populate atom cache (single-threaded, no race conditions)
    for (uint32_t cp : unique_codepoints) {
        get_atom(cp);  // This populates the cache
    }
    std::cerr << "[VOCAB] Pre-cached " << g_atom_cache.size() << " atoms\n";
    
    // Parallel processing - each thread has its own local storage, NO LOCKS during compute
    unsigned int num_threads = std::min(std::thread::hardware_concurrency(), 8u);
    if (num_threads == 0) num_threads = 4;
    
    // Thread-local results - each thread fills its own vector
    std::vector<std::vector<CompositionInfo>> thread_results(num_threads);
    std::vector<size_t> thread_processed(num_threads, 0);
    
    auto worker = [&](unsigned int thread_id, size_t start_idx, size_t end_idx) {
        // Thread-local vocabulary for deduplication within this thread's work
        std::unordered_map<Blake3Hash, bool, Blake3HashHasher, Blake3HashEqual> local_seen;
        std::vector<CompositionInfo>& local_new = thread_results[thread_id];
        local_new.reserve((end_idx - start_idx) * 5);  // ~5 compositions per token average
        
        for (size_t i = start_idx; i < end_idx; ++i) {
            const std::string& token = tokens[i];
            std::vector<uint32_t> codepoints = decode_utf8(token);
            
            if (codepoints.empty()) continue;
            
            // Build atom nodes - get_atom is deterministic, no lock needed
            std::vector<CascadeNode> atoms;
            atoms.reserve(codepoints.size());
            for (uint32_t cp : codepoints) {
                const AtomInfo& atom = get_atom(cp);
                CascadeNode node;
                node.hash = atom.hash;
                node.cx = atom.coord_x;
                node.cy = atom.coord_y;
                node.cz = atom.coord_z;
                node.cm = atom.coord_m;
                node.depth = 0;
                node.atom_count = 1;
                atoms.push_back(node);
            }
            
            // Binary cascade - create all intermediate compositions
            while (atoms.size() > 1) {
                std::vector<CascadeNode> next;
                next.reserve((atoms.size() + 1) / 2);
                
                for (size_t j = 0; j < atoms.size(); j += 2) {
                    if (j + 1 >= atoms.size()) {
                        next.push_back(atoms[j]);
                        continue;
                    }
                    
                    CascadeNode& left = atoms[j];
                    CascadeNode& right = atoms[j + 1];
                    
                    std::vector<Blake3Hash> children = {left.hash, right.hash};
                    Blake3Hash hash = compute_composition_hash(children);
                    
                    CascadeNode node;
                    node.hash = hash;
                    node.cx = (left.cx + right.cx) / 2;
                    node.cy = (left.cy + right.cy) / 2;
                    node.cz = (left.cz + right.cz) / 2;
                    node.cm = (left.cm + right.cm) / 2;
                    node.depth = std::max(left.depth, right.depth) + 1;
                    node.atom_count = left.atom_count + right.atom_count;
                    
                    // Check local cache first (no lock)
                    if (local_seen.find(hash) == local_seen.end()) {
                        // Check global vocabulary (read-only, thread-safe)
                        if (g_vocabulary.find(hash) == g_vocabulary.end()) {
                            // New composition
                            CompositionInfo comp;
                            comp.hash = hash;
                            comp.children = children;
                            comp.child_centroids = {
                                {left.cx, left.cy, left.cz, left.cm},
                                {right.cx, right.cy, right.cz, right.cm}
                            };
                            comp.centroid_x = node.cx;
                            comp.centroid_y = node.cy;
                            comp.centroid_z = node.cz;
                            comp.centroid_m = node.cm;
                            comp.depth = node.depth;
                            comp.atom_count = node.atom_count;
                            comp.from_db = false;
                            
                            Point4D coords(
                                static_cast<uint32_t>(static_cast<int64_t>(node.cx) + INT32_MAX + 1),
                                static_cast<uint32_t>(static_cast<int64_t>(node.cy) + INT32_MAX + 1),
                                static_cast<uint32_t>(static_cast<int64_t>(node.cz) + INT32_MAX + 1),
                                static_cast<uint32_t>(static_cast<int64_t>(node.cm) + INT32_MAX + 1)
                            );
                            HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
                            comp.hilbert_lo = static_cast<int64_t>(hilbert.lo);
                            comp.hilbert_hi = static_cast<int64_t>(hilbert.hi);
                            
                            local_new.push_back(std::move(comp));
                        }
                        local_seen[hash] = true;
                    }
                    
                    next.push_back(node);
                }
                atoms = std::move(next);
            }
            
            thread_processed[thread_id]++;
        }
    };
    
    // Launch threads
    std::vector<std::thread> threads;
    size_t chunk_size = (tokens.size() + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, tokens.size());
        if (start_idx < tokens.size()) {
            threads.emplace_back(worker, t, start_idx, end_idx);
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Merge results - dedupe across threads
    std::unordered_map<Blake3Hash, const CompositionInfo*, Blake3HashHasher, Blake3HashEqual> seen;
    std::vector<CompositionInfo> all_compositions;
    size_t total_processed = 0;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        total_processed += thread_processed[t];
        for (auto& comp : thread_results[t]) {
            if (seen.find(comp.hash) == seen.end()) {
                seen[comp.hash] = &comp;
                all_compositions.push_back(std::move(comp));
            }
        }
        thread_results[t].clear();  // Free memory
    }
    
    std::cerr << "[VOCAB] Processed " << total_processed << " tokens, " 
              << all_compositions.size() << " new compositions\n";
    
    // Single bulk insert - ON CONFLICT handles any remaining duplicates from DB
    if (!all_compositions.empty()) {
        std::cerr << "[DB] Inserting " << all_compositions.size() << " compositions...\n";
        insert_compositions(conn, all_compositions);
        
        // Update global vocabulary with new compositions
        for (const auto& comp : all_compositions) {
            g_vocabulary[comp.hash] = comp;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    result.new_compositions = all_compositions.size();
    
    std::cerr << "[VOCAB] Complete in " << std::fixed << std::setprecision(2) 
              << result.seconds << "s\n";
    
    return result;
}

// Detect if file is a vocab file (one token per line)
bool is_vocab_file(const fs::path& path) {
    std::string name = path.filename().string();
    return name == "vocab.txt" || name == "merges.txt" || 
           name.find("vocab") != std::string::npos;
}

// Legacy single-file ingest (includes DB operations for backward compat)
IngestResult ingest_file(PGconn* conn, const fs::path& path) {
    // Use optimized path for vocab files
    if (is_vocab_file(path)) {
        return ingest_vocab_file(conn, path);
    }
    
    std::vector<CompositionInfo> new_compositions;
    IngestResult result = process_file_content(path, new_compositions);
    
    if (!new_compositions.empty()) {
        insert_compositions(conn, new_compositions);
        insert_edges(conn, 0.0f);
    }
    
    return result;
}

void ingest_directory(PGconn* conn, const fs::path& dir) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Collect all text files
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".txt" || ext == ".md" || ext == ".json" ||
            ext == ".py" || ext == ".cpp" || ext == ".hpp" ||
            ext == ".c" || ext == ".h" || ext == ".js" ||
            ext == ".ts" || ext == ".yaml" || ext == ".yml" ||
            ext == ".xml" || ext == ".html" || ext == ".css" ||
            ext == ".sql" || ext == ".sh" || ext == ".rs" ||
            ext == ".go" || ext == ".java" || ext == ".rb") {
            files.push_back(entry.path());
        }
    }

    std::cerr << "[INGEST] Found " << files.size() << " text files\n";
    
    // Determine parallelism (use hardware concurrency, max 8 to avoid connection overload)
    unsigned int num_threads = std::min(std::thread::hardware_concurrency(), 8u);
    if (num_threads == 0) num_threads = 4;
    
    std::cerr << "[INGEST] Using " << num_threads << " parallel threads\n";
    
    // Atomic counters for thread-safe aggregation
    std::atomic<size_t> total_bytes{0};
    std::atomic<size_t> total_files{0};
    std::atomic<size_t> total_new{0};
    std::atomic<size_t> total_reused{0};
    std::atomic<size_t> file_idx{0};
    
    // Worker function - each thread processes files from shared queue
    auto worker = [&]() {
        // Each thread gets its own connection for parallel DB operations
        // But for now, we process files in memory and batch results
        while (true) {
            size_t idx = file_idx.fetch_add(1);
            if (idx >= files.size()) break;
            
            const fs::path& path = files[idx];
            
            // Read file into memory (fast, no DB access)
            std::ifstream file(path, std::ios::binary);
            if (!file) continue;
            
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            if (content.empty()) continue;
            
            // Decode UTF-8 to codepoints
            std::vector<uint32_t> codepoints = decode_utf8(content);
            if (codepoints.empty()) continue;
            
            // Thread-local composition storage
            std::vector<CompositionInfo> local_compositions;
            
            // Process content (g_edges has its own mutex, g_atom_cache is read-only)
            tokenize_and_ingest(codepoints, local_compositions);
            
            // Aggregate results
            total_bytes += content.size();
            total_files++;
            total_new += local_compositions.size();
        }
    };
    
    // Launch worker threads
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Now batch insert all new compositions (single-threaded for DB safety)
    std::vector<CompositionInfo> all_new;
    for (const auto& [key, comp] : g_vocabulary) {
        if (!comp.from_db) {
            all_new.push_back(comp);
        }
    }
    
    if (!all_new.empty()) {
        std::cerr << "[DB] Batch inserting " << all_new.size() << " compositions...\n";
        insert_compositions(conn, all_new);
    }
    
    // Insert edges
    insert_edges(conn, 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(end - start).count();

    std::cerr << "\n[COMPLETE]\n";
    std::cerr << "  Files: " << total_files.load() << "\n";
    std::cerr << "  Bytes: " << total_bytes.load() << " (" << (total_bytes.load() / 1024.0 / 1024.0) << " MB)\n";
    std::cerr << "  New compositions: " << all_new.size() << "\n";
    std::cerr << "  Edges: " << g_edges.size() << "\n";
    std::cerr << "  Vocabulary size: " << g_vocabulary.size() << "\n";
    std::cerr << "  Time: " << total_secs << " s\n";
    std::cerr << "  Throughput: " << (total_bytes.load() / 1024.0 / total_secs) << " KB/s\n";
}

// =============================================================================
// Validation: Test that content hash lookup works
// =============================================================================

bool validate_content_hash(PGconn* conn, const std::string& test_text) {
    std::cerr << "\n[VALIDATE] Testing content hash for: \"" << test_text << "\"\n";

    std::vector<uint32_t> codepoints = decode_utf8(test_text);
    if (codepoints.empty()) {
        std::cerr << "  ERROR: No valid codepoints\n";
        return false;
    }

    // Build composition hash using the same algorithm as ingestion
    std::vector<Blake3Hash> atom_hashes;
    for (uint32_t cp : codepoints) {
        const AtomInfo& atom = get_atom(cp);
        atom_hashes.push_back(atom.hash);
    }

    // Build cascade
    std::vector<Blake3Hash> current = atom_hashes;
    while (current.size() > 1) {
        std::vector<Blake3Hash> next;
        for (size_t i = 0; i + 1 < current.size(); i += 2) {
            std::vector<Blake3Hash> pair = {current[i], current[i + 1]};
            Blake3Hash hash = compute_composition_hash(pair);
            next.push_back(hash);
        }
        if (current.size() % 2 == 1) {
            next.push_back(current.back());
        }
        current = std::move(next);
    }

    if (current.empty()) {
        std::cerr << "  ERROR: No hash computed\n";
        return false;
    }

    Blake3Hash computed = current[0];
    std::string computed_hex = computed.to_hex();  // Only for display/DB query
    std::cerr << "  Computed hash: " << computed_hex << "\n";

    // Check if exists in vocabulary
    auto it = g_vocabulary.find(computed);
    if (it != g_vocabulary.end()) {
        std::cerr << "  FOUND in vocabulary! Depth: " << it->second.depth
                  << ", Atoms: " << it->second.atom_count << "\n";
        return true;
    }

    std::cerr << "  NOT FOUND in vocabulary\n";

    // Also check database directly
    std::string query = "SELECT id, depth, atom_count, atom_text(id) FROM atom WHERE id = '\\x" + computed_hex + "'";
    PGresult* res = PQexec(conn, query.c_str());

    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        std::cerr << "  FOUND in database! Reconstructed text: " << PQgetvalue(res, 0, 3) << "\n";
        PQclear(res);
        return true;
    }

    PQclear(res);
    return false;
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Vocabulary-Aware Ingester\n\n"
              << "Usage: " << prog << " [options] <path>\n\n"
              << "Arguments:\n"
              << "  <path>               File or directory to ingest\n\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  -P, --password PASS  Database password\n"
              << "  --validate TEXT      Validate content hash for TEXT after ingestion\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    std::string password = "";
    std::string target = "";
    std::string validate_text = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if ((arg == "-P" || arg == "--password") && i + 1 < argc) {
            password = argv[++i];
        } else if (arg == "--validate" && i + 1 < argc) {
            validate_text = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            target = arg;
        }
    }

    if (target.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::string conninfo = "dbname=" + dbname;
    if (!host.empty()) conninfo += " host=" + host;
    if (!port.empty()) conninfo += " port=" + port;
    if (!user.empty()) conninfo += " user=" + user;
    if (!password.empty()) conninfo += " password=" + password;

    std::cerr << "=== Vocabulary-Aware Ingester ===\n";
    std::cerr << "Database: " << dbname << "\n";
    std::cerr << "Target: " << target << "\n\n";

    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }

    // Load vocabulary first, then load only atoms referenced by compositions
    std::cerr << "[INIT] Loading vocabulary from database...\n";
    
    // Load existing compositions for greedy matching
    if (load_vocabulary(conn)) {
        // Load only atoms that are referenced by compositions (for trie reconstruction)
        load_referenced_atoms(conn);
        
        // Build trie for O(m) longest-match lookups
        build_vocab_trie();
        std::cerr << "[INIT] Ready: " << g_vocabulary.size() 
                  << " compositions, " << g_atom_cache.size() << " cached atoms\n";
    } else {
        std::cerr << "[INIT] No existing vocabulary, will build from scratch\n";
    }

    fs::path path(target);
    if (!fs::exists(path)) {
        std::cerr << "Not found: " << target << std::endl;
        PQfinish(conn);
        return 1;
    }

    if (fs::is_directory(path)) {
        ingest_directory(conn, path);
    } else {
        IngestResult res = ingest_file(conn, path);
        if (res.bytes > 0) {
            std::cerr << "[OK] " << res.bytes << " bytes → "
                      << res.new_compositions << " new compositions\n";
        }
    }

    // Validation
    if (!validate_text.empty()) {
        validate_content_hash(conn, validate_text);
    }

    // Also run standard validation tests
    std::cerr << "\n[VALIDATION TESTS]\n";
    validate_content_hash(conn, "ma");
    validate_content_hash(conn, "man");
    validate_content_hash(conn, "the");

    // Stats
    PGresult* res = PQexec(conn, "SELECT COUNT(*) FROM atom WHERE depth > 0");
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        std::cerr << "\nTotal compositions in DB: " << PQgetvalue(res, 0, 0) << "\n";
    }
    PQclear(res);

    PQfinish(conn);
    return 0;
}
