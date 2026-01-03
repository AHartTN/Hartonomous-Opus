/**
 * Grammar-Based Ingester (Re-Pair + Greedy Longest Match)
 *
 * PASS 1 - Grammar Inference (Re-Pair algorithm):
 *   1. RLE: Collapse consecutive identical atoms into run-length compositions
 *   2. Find most frequent digram (adjacent pair)
 *   3. Replace ALL occurrences with new composition
 *   4. Repeat until no digram appears more than once
 *   5. Store each unique composition to database (content-addressed)
 *
 * PASS 2 - Content Recording (Greedy Longest Match):
 *   1. Build trie from existing vocabulary (compositions)
 *   2. Tokenize input using greedy longest match
 *   3. Record edges between matched compositions (co-occurrence)
 *
 * This is Cascading Pair Encoding (CPE) done RIGHT:
 * - NOT binary cascade (which loses natural boundaries)
 * - NOT per-document trees (which explode storage)
 * - ONE global grammar that grows with ingestion
 * - Deduplication via content-addressing
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

#include <libpq-fe.h>

#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"

using namespace hypercube;

// =============================================================================
// Data Structures
// =============================================================================

struct AtomInfo {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
};

// Symbol in the sequence (can be atom or composition reference)
struct Symbol {
    Blake3Hash hash;
    int32_t cx, cy, cz, cm;
    uint32_t depth;
    uint64_t atom_count;
    
    bool operator==(const Symbol& other) const {
        return hash == other.hash;
    }
};

// Digram (pair of adjacent symbols)
struct Digram {
    Blake3Hash first;
    Blake3Hash second;
    
    bool operator==(const Digram& other) const {
        return first == other.first && second == other.second;
    }
};

struct DigramHash {
    size_t operator()(const Digram& d) const {
        size_t h1 = 0, h2 = 0;
        for (int i = 0; i < 8; i++) {
            h1 ^= static_cast<size_t>(d.first.bytes[i]) << (i * 8);
            h2 ^= static_cast<size_t>(d.second.bytes[i]) << (i * 8);
        }
        return h1 ^ (h2 << 1);
    }
};

// GrammarComposition to store in database
struct GrammarComposition {
    Blake3Hash hash;
    std::vector<Blake3Hash> children;
    std::vector<std::array<int32_t, 4>> child_coords;
    int32_t centroid_x, centroid_y, centroid_z, centroid_m;
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;
    uint64_t atom_count;
};

// =============================================================================
// Global State
// =============================================================================

static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;
static std::unordered_map<std::string, GrammarComposition> g_compositions;  // hash_hex -> comp

// =============================================================================
// Atom Cache
// =============================================================================

bool load_atom_cache(PGconn* conn) {
    // Load atoms using geom column (raw uint32 coords stored as double)
    PGresult* res = PQexec(conn,
        "SELECT encode(id, 'hex'), codepoint, "
        "ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom) "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    if (rows == 0) {
        std::cerr << "ERROR: No atoms found! Run setup-db.sh first.\n";
        PQclear(res);
        return false;
    }

    g_atom_cache.reserve(rows);

    for (int i = 0; i < rows; i++) {
        uint32_t cp = static_cast<uint32_t>(atoi(PQgetvalue(res, i, 1)));
        AtomInfo info;
        info.hash = Blake3Hash::from_hex(std::string_view(PQgetvalue(res, i, 0), 64));
        // Coordinates: stored as double (representing uint32), convert to int32 (same bit pattern)
        info.coord_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 2))));
        info.coord_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 3))));
        info.coord_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 4))));
        info.coord_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 5))));
        g_atom_cache[cp] = info;
    }

    PQclear(res);
    std::cerr << "[CACHE] Loaded " << g_atom_cache.size() << " atoms\n";
    return true;
}

// =============================================================================
// UTF-8 Decoding
// =============================================================================

std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());
    
    size_t i = 0;
    while (i < data.size()) {
        uint32_t cp;
        uint8_t c = static_cast<uint8_t>(data[i]);
        
        if ((c & 0x80) == 0) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6;
            if (i + 1 < data.size()) cp |= (static_cast<uint8_t>(data[i+1]) & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12;
            if (i + 1 < data.size()) cp |= (static_cast<uint8_t>(data[i+1]) & 0x3F) << 6;
            if (i + 2 < data.size()) cp |= (static_cast<uint8_t>(data[i+2]) & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = (c & 0x07) << 18;
            if (i + 1 < data.size()) cp |= (static_cast<uint8_t>(data[i+1]) & 0x3F) << 12;
            if (i + 2 < data.size()) cp |= (static_cast<uint8_t>(data[i+2]) & 0x3F) << 6;
            if (i + 3 < data.size()) cp |= (static_cast<uint8_t>(data[i+3]) & 0x3F);
            i += 4;
        } else {
            i += 1;
            continue;
        }
        codepoints.push_back(cp);
    }
    return codepoints;
}

// =============================================================================
// Hash Computation
// =============================================================================

Blake3Hash compute_composition_hash(const std::vector<Blake3Hash>& children) {
    std::vector<uint8_t> input;
    input.reserve(children.size() * 36);
    
    for (size_t i = 0; i < children.size(); i++) {
        uint32_t ordinal = static_cast<uint32_t>(i);
        input.insert(input.end(),
            reinterpret_cast<uint8_t*>(&ordinal),
            reinterpret_cast<uint8_t*>(&ordinal) + 4);
        input.insert(input.end(),
            children[i].bytes.begin(),
            children[i].bytes.end());
    }
    
    return Blake3Hasher::hash(std::span<const uint8_t>(input));
}

Blake3Hash compute_rle_hash(const Blake3Hash& child, uint32_t run_length) {
    std::vector<uint8_t> input;
    input.reserve(36);
    input.insert(input.end(), child.bytes.begin(), child.bytes.end());
    input.insert(input.end(),
        reinterpret_cast<uint8_t*>(&run_length),
        reinterpret_cast<uint8_t*>(&run_length) + 4);
    return Blake3Hasher::hash(std::span<const uint8_t>(input));
}

// =============================================================================
// Re-Pair Grammar Inference
// =============================================================================

// Reinterpret int32 as uint32 (same bit pattern)
inline uint32_t as_uint32(int32_t v) {
    return static_cast<uint32_t>(v);
}

// Reinterpret uint32 as int32 (same bit pattern)
inline int32_t as_int32(uint32_t v) {
    return static_cast<int32_t>(v);
}

// Create composition from two symbols
GrammarComposition create_composition(const Symbol& left, const Symbol& right) {
    GrammarComposition comp;
    comp.children = {left.hash, right.hash};
    comp.child_coords = {
        {left.cx, left.cy, left.cz, left.cm},
        {right.cx, right.cy, right.cz, right.cm}
    };
    comp.hash = compute_composition_hash(comp.children);

    // Centroid: average using uint64 to prevent overflow
    // Coordinates are stored as int32 but represent uint32 values
    uint64_t sum_x = static_cast<uint64_t>(as_uint32(left.cx)) + static_cast<uint64_t>(as_uint32(right.cx));
    uint64_t sum_y = static_cast<uint64_t>(as_uint32(left.cy)) + static_cast<uint64_t>(as_uint32(right.cy));
    uint64_t sum_z = static_cast<uint64_t>(as_uint32(left.cz)) + static_cast<uint64_t>(as_uint32(right.cz));
    uint64_t sum_m = static_cast<uint64_t>(as_uint32(left.cm)) + static_cast<uint64_t>(as_uint32(right.cm));

    comp.centroid_x = as_int32(static_cast<uint32_t>(sum_x / 2));
    comp.centroid_y = as_int32(static_cast<uint32_t>(sum_y / 2));
    comp.centroid_z = as_int32(static_cast<uint32_t>(sum_z / 2));
    comp.centroid_m = as_int32(static_cast<uint32_t>(sum_m / 2));
    
    comp.depth = std::max(left.depth, right.depth) + 1;
    comp.atom_count = left.atom_count + right.atom_count;

    // Coordinates are already corner-origin (stored as uint32 bit pattern in int32)
    // Hilbert uses these directly - no transformation needed
    Point4D coords(
        as_uint32(comp.centroid_x),
        as_uint32(comp.centroid_y),
        as_uint32(comp.centroid_z),
        as_uint32(comp.centroid_m)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    comp.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    comp.hilbert_hi = static_cast<int64_t>(hilbert.hi);

    return comp;
}

// =============================================================================
// Fast Binary Cascade - O(n log n)
// =============================================================================

// Binary cascade: reduce sequence to single root via pairwise merging
// This is fast (O(n log n)) and deterministic
Symbol binary_cascade(
    std::vector<Symbol>& sequence,
    std::vector<GrammarComposition>& new_compositions
) {
    if (sequence.empty()) return Symbol{};
    if (sequence.size() == 1) return sequence[0];

    while (sequence.size() > 1) {
        std::vector<Symbol> merged;
        merged.reserve((sequence.size() + 1) / 2);

        for (size_t i = 0; i < sequence.size(); i += 2) {
            if (i + 1 < sequence.size()) {
                // Merge pair
                const Symbol& left = sequence[i];
                const Symbol& right = sequence[i + 1];

                GrammarComposition comp = create_composition(left, right);
                std::string hash_hex = comp.hash.to_hex();

                if (g_compositions.find(hash_hex) == g_compositions.end()) {
                    g_compositions[hash_hex] = comp;
                    new_compositions.push_back(comp);
                }

                Symbol sym;
                sym.hash = comp.hash;
                sym.cx = comp.centroid_x;
                sym.cy = comp.centroid_y;
                sym.cz = comp.centroid_z;
                sym.cm = comp.centroid_m;
                sym.depth = comp.depth;
                sym.atom_count = comp.atom_count;
                merged.push_back(sym);
            } else {
                // Odd element - carry forward
                merged.push_back(sequence[i]);
            }
        }
        sequence = std::move(merged);
    }

    return sequence[0];
}

// =============================================================================
// Token-aware Cascade - respects natural boundaries
// =============================================================================

// Check if codepoint is a word boundary (whitespace or punctuation)
inline bool is_boundary(uint32_t cp) {
    // Whitespace
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return true;
    // Common punctuation
    if (cp >= 0x21 && cp <= 0x2F) return true;  // !"#$%&'()*+,-./
    if (cp >= 0x3A && cp <= 0x40) return true;  // :;<=>?@
    if (cp >= 0x5B && cp <= 0x60) return true;  // [\]^_`
    if (cp >= 0x7B && cp <= 0x7E) return true;  // {|}~
    return false;
}

// Tokenize into words and boundaries, cascade each, then cascade all
void tokenize_and_cascade(
    const std::vector<Symbol>& sequence,
    std::vector<GrammarComposition>& new_compositions,
    std::vector<Symbol>& token_symbols
) {
    if (sequence.empty()) return;

    std::vector<Symbol> current_token;
    size_t token_count = 0;
    size_t boundary_count = 0;

    auto flush_token = [&]() {
        if (current_token.empty()) return;

        if (current_token.size() == 1) {
            token_symbols.push_back(current_token[0]);
        } else {
            // Binary cascade the token
            Symbol root = binary_cascade(current_token, new_compositions);
            token_symbols.push_back(root);
        }
        token_count++;
        current_token.clear();
    };

    // Process symbols, grouping by word boundaries
    for (const auto& sym : sequence) {
        // Check if this is a boundary character
        // We need the codepoint, but we only have the hash
        // For efficiency, treat single-atom symbols specially
        bool is_bound = (sym.atom_count == 1);  // Single atoms might be boundaries

        if (is_bound && sym.depth == 0) {
            // This is a leaf atom - could be boundary
            // Flush current token, add boundary as separate symbol
            flush_token();
            token_symbols.push_back(sym);
            boundary_count++;
        } else {
            current_token.push_back(sym);
        }
    }
    flush_token();

    std::cerr << "[TOKENIZE] " << token_count << " word tokens, "
              << boundary_count << " boundary tokens, "
              << token_symbols.size() << " total\n";
}

// =============================================================================
// RLE Pass
// =============================================================================

std::vector<Symbol> apply_rle(
    const std::vector<Symbol>& input,
    std::vector<GrammarComposition>& new_compositions
) {
    std::vector<Symbol> output;
    output.reserve(input.size());
    
    size_t i = 0;
    while (i < input.size()) {
        size_t run_start = i;
        Blake3Hash run_hash = input[i].hash;
        
        while (i < input.size() && input[i].hash == run_hash) {
            i++;
        }
        size_t run_length = i - run_start;
        
        if (run_length == 1) {
            output.push_back(input[run_start]);
        } else {
            // Create RLE composition
            Blake3Hash rle_hash = compute_rle_hash(run_hash, static_cast<uint32_t>(run_length));
            std::string hash_hex = rle_hash.to_hex();
            
            if (g_compositions.find(hash_hex) == g_compositions.end()) {
                GrammarComposition comp;
                comp.hash = rle_hash;
                comp.children = {run_hash};  // Single child
                // Repeat child coords for RLE (geometry encodes run length)
                for (size_t r = 0; r < run_length; r++) {
                    comp.child_coords.push_back({
                        input[run_start].cx,
                        input[run_start].cy,
                        input[run_start].cz,
                        input[run_start].cm
                    });
                }
                comp.centroid_x = input[run_start].cx;
                comp.centroid_y = input[run_start].cy;
                comp.centroid_z = input[run_start].cz;
                comp.centroid_m = input[run_start].cm;
                comp.depth = input[run_start].depth + 1;
                comp.atom_count = run_length;
                
                // Hilbert
                auto uint32_from_int32 = [](int32_t v) -> uint32_t {
                    return static_cast<uint32_t>(static_cast<int64_t>(v) + INT32_MAX + 1);
                };
                Point4D coords(
                    uint32_from_int32(comp.centroid_x),
                    uint32_from_int32(comp.centroid_y),
                    uint32_from_int32(comp.centroid_z),
                    uint32_from_int32(comp.centroid_m)
                );
                HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
                comp.hilbert_lo = static_cast<int64_t>(hilbert.lo);
                comp.hilbert_hi = static_cast<int64_t>(hilbert.hi);
                
                g_compositions[hash_hex] = comp;
                new_compositions.push_back(comp);
            }
            
            const GrammarComposition& comp = g_compositions[hash_hex];
            Symbol rle_sym;
            rle_sym.hash = comp.hash;
            rle_sym.cx = comp.centroid_x;
            rle_sym.cy = comp.centroid_y;
            rle_sym.cz = comp.centroid_z;
            rle_sym.cm = comp.centroid_m;
            rle_sym.depth = comp.depth;
            rle_sym.atom_count = comp.atom_count;
            output.push_back(rle_sym);
        }
    }
    
    std::cerr << "[RLE] " << input.size() << " → " << output.size() << " symbols\n";
    return output;
}

// =============================================================================
// Database Operations
// =============================================================================

static inline uint32_t int32_to_uint32(int32_t v) {
    return static_cast<uint32_t>(v) + 2147483648U;
}

std::string build_linestringzm_ewkb(const std::vector<std::array<int32_t, 4>>& points) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string ewkb;
    
    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, 8);
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    auto append_uint32 = [&](uint32_t val) {
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (val >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    ewkb += "01";  // Little-endian
    ewkb += "020000c0";  // LINESTRINGZM type
    append_uint32(static_cast<uint32_t>(points.size()));
    
    for (const auto& p : points) {
        append_double(static_cast<double>(int32_to_uint32(p[0])));
        append_double(static_cast<double>(int32_to_uint32(p[1])));
        append_double(static_cast<double>(int32_to_uint32(p[2])));
        append_double(static_cast<double>(int32_to_uint32(p[3])));
    }
    
    return ewkb;
}

bool insert_compositions(PGconn* conn, const std::vector<GrammarComposition>& comps) {
    if (comps.empty()) return true;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_comp ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(GEOMETRYZM, 0) NOT NULL,"
        "  children BYTEA[],"
        "  hilbert_lo BIGINT NOT NULL,"
        "  hilbert_hi BIGINT NOT NULL,"
        "  depth INTEGER NOT NULL,"
        "  atom_count BIGINT NOT NULL"
        ") ON COMMIT DROP");
    PQclear(res);
    
    res = PQexec(conn,
        "COPY tmp_comp (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    char num_buf[32];
    
    for (const auto& c : comps) {
        if (c.child_coords.empty()) continue;
        
        batch += "\\\\x";
        batch += c.hash.to_hex();
        batch += '\t';
        
        batch += build_linestringzm_ewkb(c.child_coords);
        batch += '\t';
        
        batch += "{";
        for (size_t i = 0; i < c.children.size(); i++) {
            if (i > 0) batch += ",";
            batch += "\"\\\\\\\\x";
            batch += c.children[i].to_hex();
            batch += "\"";
        }
        batch += "}";
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(c.hilbert_lo));
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(c.hilbert_hi));
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%u", c.depth);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(c.atom_count));
        batch += num_buf;
        batch += '\n';
        
        if (batch.size() > (1 << 19)) {
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
    
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count "
        "FROM tmp_comp "
        "ON CONFLICT (id) DO NOTHING");
    
    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[DB] Inserted " << inserted << " / " << comps.size() << " compositions in " << ms << " ms\n";
    
    return true;
}

// =============================================================================
// Main Ingestion
// =============================================================================

Blake3Hash ingest_content(
    const std::vector<uint32_t>& codepoints,
    std::vector<GrammarComposition>& new_compositions
) {
    if (codepoints.empty()) return Blake3Hash();
    
    // Convert to symbols
    std::vector<Symbol> sequence;
    sequence.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        auto it = g_atom_cache.find(cp);
        if (it == g_atom_cache.end()) continue;
        
        Symbol s;
        s.hash = it->second.hash;
        s.cx = it->second.coord_x;
        s.cy = it->second.coord_y;
        s.cz = it->second.coord_z;
        s.cm = it->second.coord_m;
        s.depth = 0;
        s.atom_count = 1;
        sequence.push_back(s);
    }
    
    if (sequence.empty()) return Blake3Hash();
    if (sequence.size() == 1) return sequence[0].hash;
    
    // Pass 1: RLE
    sequence = apply_rle(sequence, new_compositions);
    
    // Pass 2: Re-Pair grammar inference
    repair_grammar(sequence, new_compositions);
    
    // Return root (final sequence should be small, possibly 1 symbol)
    if (sequence.size() == 1) {
        return sequence[0].hash;
    }
    
    // If still multiple symbols, create final composition
    GrammarComposition root;
    for (const auto& s : sequence) {
        root.children.push_back(s.hash);
        root.child_coords.push_back({s.cx, s.cy, s.cz, s.cm});
    }
    root.hash = compute_composition_hash(root.children);
    
    int64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    uint64_t total_atoms = 0;
    uint32_t max_depth = 0;
    for (const auto& s : sequence) {
        sum_x += s.cx;
        sum_y += s.cy;
        sum_z += s.cz;
        sum_m += s.cm;
        total_atoms += s.atom_count;
        max_depth = std::max(max_depth, s.depth);
    }
    root.centroid_x = static_cast<int32_t>(sum_x / static_cast<int64_t>(sequence.size()));
    root.centroid_y = static_cast<int32_t>(sum_y / static_cast<int64_t>(sequence.size()));
    root.centroid_z = static_cast<int32_t>(sum_z / static_cast<int64_t>(sequence.size()));
    root.centroid_m = static_cast<int32_t>(sum_m / static_cast<int64_t>(sequence.size()));
    root.depth = max_depth + 1;
    root.atom_count = total_atoms;
    
    auto uint32_from_int32 = [](int32_t v) -> uint32_t {
        return static_cast<uint32_t>(static_cast<int64_t>(v) + INT32_MAX + 1);
    };
    Point4D coords(
        uint32_from_int32(root.centroid_x),
        uint32_from_int32(root.centroid_y),
        uint32_from_int32(root.centroid_z),
        uint32_from_int32(root.centroid_m)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    root.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    root.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    
    std::string hash_hex = root.hash.to_hex();
    if (g_compositions.find(hash_hex) == g_compositions.end()) {
        g_compositions[hash_hex] = root;
        new_compositions.push_back(root);
    }
    
    return root.hash;
}

// =============================================================================
// File Processing
// =============================================================================

void ingest_file(PGconn* conn, const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return;
    }
    
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();
    
    auto codepoints = decode_utf8(content);
    std::cerr << "[FILE] " << path.filename().string() << ": " 
              << content.size() << " bytes, " << codepoints.size() << " codepoints\n";
    
    std::vector<GrammarComposition> new_compositions;
    Blake3Hash root = ingest_content(codepoints, new_compositions);
    
    if (!new_compositions.empty()) {
        insert_compositions(conn, new_compositions);
    }
    
    std::cout << root.to_hex() << std::endl;
    std::cerr << "[OK] " << content.size() << " bytes → " << new_compositions.size() << " new compositions\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::string db_name = "hypercube";
    std::string db_user = "hartonomous";
    std::string db_host = "localhost";
    std::string db_port = "5432";
    std::vector<std::string> targets;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) db_name = argv[++i];
        else if (arg == "-U" && i + 1 < argc) db_user = argv[++i];
        else if (arg == "-h" && i + 1 < argc) db_host = argv[++i];
        else if (arg == "-p" && i + 1 < argc) db_port = argv[++i];
        else targets.push_back(arg);
    }
    
    if (targets.empty()) {
        std::cerr << "Usage: grammar_ingest [options] <file_or_directory>...\n";
        std::cerr << "Options:\n";
        std::cerr << "  -d <database>  Database name (default: hypercube)\n";
        std::cerr << "  -U <user>      Username (default: hartonomous)\n";
        std::cerr << "  -h <host>      Host (default: localhost)\n";
        std::cerr << "  -p <port>      Port (default: 5432)\n";
        return 1;
    }
    
    std::cerr << "=== Grammar-Based Ingester (Re-Pair + RLE) ===\n";
    std::cerr << "Database: " << db_name << "\n";
    
    std::string conninfo = "dbname=" + db_name + " user=" + db_user +
                          " host=" + db_host + " port=" + db_port;
    PGconn* conn = PQconnectdb(conninfo.c_str());
    
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }
    
    if (!load_atom_cache(conn)) {
        PQfinish(conn);
        return 1;
    }
    
    for (const auto& target : targets) {
        std::filesystem::path path(target);
        if (std::filesystem::is_directory(path)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    ingest_file(conn, entry.path());
                }
            }
        } else {
            ingest_file(conn, path);
        }
    }
    
    // Print final stats
    PGresult* res = PQexec(conn, "SELECT COUNT(*) FROM atom WHERE depth > 0");
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        std::cerr << "\nTotal compositions in DB: " << PQgetvalue(res, 0, 0) << "\n";
    }
    PQclear(res);
    
    PQfinish(conn);
    return 0;
}
