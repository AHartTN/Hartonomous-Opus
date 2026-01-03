/**
 * Semantic Ingester - The Real Content Pipeline
 * 
 * This is the correct ingestion flow for the Hartonomous substrate:
 * 
 * 1. VOCABULARY DISCOVERY (Sequitur)
 *    - Discover repeated patterns → compositions
 *    - These are "nouns" - meaningless on their own
 * 
 * 2. GREEDY TOKENIZATION
 *    - Match longest known composition at each position
 *    - Result: sequence of composition references
 * 
 * 3. RELATIONSHIP RECORDING (THE ACTUAL CONTENT)
 *    - For each pair within context window: create/update edge
 *    - Edge weight = accumulated co-occurrence strength
 *    - This is where MEANING is created
 * 
 * The source is discarded after ingestion. Only relationships remain.
 * 
 * For AI models: tokens = text, cosines = edge weights, below threshold = zero
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;
namespace fs = std::filesystem;

// =============================================================================
// Configuration
// =============================================================================

struct IngestConfig {
    std::string conninfo;
    int context_window = 5;           // How far to look for co-occurrence
    float edge_threshold = 0.01f;     // Minimum weight to record an edge
    int top_k_neighbors = 64;         // Max neighbors per composition
    bool verbose = false;
};

// =============================================================================
// Composition Cache (Vocabulary Trie)
// =============================================================================

struct CompositionInfo {
    Blake3Hash hash;
    int32_t cx, cy, cz, cm;  // Centroid coordinates
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;
    std::vector<uint32_t> codepoints;  // The actual content for matching
};

// Trie node for longest-match tokenization
struct TrieNode {
    std::unordered_map<uint32_t, std::unique_ptr<TrieNode>> children;
    CompositionInfo* composition = nullptr;  // Non-null if this is end of a vocab entry
};

class VocabularyTrie {
public:
    TrieNode root;
    std::vector<std::unique_ptr<CompositionInfo>> compositions;
    
    void insert(const std::vector<uint32_t>& codepoints, CompositionInfo info) {
        auto comp = std::make_unique<CompositionInfo>(std::move(info));
        comp->codepoints = codepoints;
        
        TrieNode* node = &root;
        for (uint32_t cp : codepoints) {
            if (!node->children[cp]) {
                node->children[cp] = std::make_unique<TrieNode>();
            }
            node = node->children[cp].get();
        }
        node->composition = comp.get();
        compositions.push_back(std::move(comp));
    }
    
    // Returns longest matching composition starting at position, or nullptr
    CompositionInfo* longest_match(const std::vector<uint32_t>& codepoints, size_t start) {
        TrieNode* node = &root;
        CompositionInfo* best_match = nullptr;
        
        for (size_t i = start; i < codepoints.size() && node; ++i) {
            auto it = node->children.find(codepoints[i]);
            if (it == node->children.end()) break;
            
            node = it->second.get();
            if (node->composition) {
                best_match = node->composition;
            }
        }
        
        return best_match;
    }
    
    size_t size() const { return compositions.size(); }
};

// Global vocabulary
static VocabularyTrie g_vocab;

// Atom cache for leaf lookups
struct AtomInfo {
    Blake3Hash hash;
    int32_t cx, cy, cz, cm;
    int64_t hilbert_lo, hilbert_hi;
};
static std::unordered_map<uint32_t, AtomInfo> g_atoms;

// =============================================================================
// Database Loading
// =============================================================================

bool load_atoms(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();
    
    PGresult* res = PQexec(conn,
        "SELECT codepoint, id, ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom), "
        "       hilbert_lo, hilbert_hi "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    
    int rows = PQntuples(res);
    g_atoms.reserve(rows);
    
    for (int i = 0; i < rows; ++i) {
        uint32_t cp = std::stoul(PQgetvalue(res, i, 0));
        
        AtomInfo info;
        const char* hex = PQgetvalue(res, i, 1);
        if (hex[0] == '\\' && hex[1] == 'x') {
            info.hash = Blake3Hash::from_hex(std::string_view(hex + 2, 64));
        }
        info.cx = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 2)));
        info.cy = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 3)));
        info.cz = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 4)));
        info.cm = static_cast<int32_t>(std::stod(PQgetvalue(res, i, 5)));
        info.hilbert_lo = std::stoll(PQgetvalue(res, i, 6));
        info.hilbert_hi = std::stoll(PQgetvalue(res, i, 7));
        
        g_atoms[cp] = info;
    }
    
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[ATOMS] Loaded " << g_atoms.size() << " atoms in " << ms << " ms\n";
    
    return true;
}

bool load_vocabulary(PGconn* conn) {
    // Skip vocabulary for now - will tokenize as atoms only
    // Full vocabulary loading requires reconstructing each composition's text,
    // which is too slow for 400K+ compositions
    std::cerr << "[VOCAB] Skipping vocabulary (tokenizing as atoms)\n";
    return false;
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
// Tokenization Result
// =============================================================================

struct TokenRef {
    Blake3Hash hash;
    int32_t cx, cy, cz, cm;
    size_t length;  // Number of codepoints consumed
    bool is_atom;   // True if this is a single codepoint atom
};

// =============================================================================
// Phase 2: Greedy Tokenization
// =============================================================================

std::vector<TokenRef> tokenize(const std::vector<uint32_t>& codepoints) {
    std::vector<TokenRef> tokens;
    tokens.reserve(codepoints.size() / 4);  // Rough estimate
    
    size_t i = 0;
    while (i < codepoints.size()) {
        // Try longest match first
        CompositionInfo* match = g_vocab.longest_match(codepoints, i);
        
        if (match) {
            TokenRef ref;
            ref.hash = match->hash;
            ref.cx = match->cx;
            ref.cy = match->cy;
            ref.cz = match->cz;
            ref.cm = match->cm;
            ref.length = match->codepoints.size();
            ref.is_atom = false;
            tokens.push_back(ref);
            i += match->codepoints.size();
        } else {
            // Fall back to single atom
            auto it = g_atoms.find(codepoints[i]);
            if (it != g_atoms.end()) {
                TokenRef ref;
                ref.hash = it->second.hash;
                ref.cx = it->second.cx;
                ref.cy = it->second.cy;
                ref.cz = it->second.cz;
                ref.cm = it->second.cm;
                ref.length = 1;
                ref.is_atom = true;
                tokens.push_back(ref);
            }
            i++;
        }
    }
    
    return tokens;
}

// =============================================================================
// Phase 3: Relationship Recording
// =============================================================================

struct EdgeKey {
    Blake3Hash a;
    Blake3Hash b;
    
    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        size_t h = 0;
        for (int i = 0; i < 8; ++i) {
            h ^= std::hash<uint32_t>()(
                (static_cast<uint32_t>(k.a.bytes[i*4]) << 24) |
                (static_cast<uint32_t>(k.a.bytes[i*4+1]) << 16) |
                (static_cast<uint32_t>(k.a.bytes[i*4+2]) << 8) |
                static_cast<uint32_t>(k.a.bytes[i*4+3])
            );
            h ^= std::hash<uint32_t>()(
                (static_cast<uint32_t>(k.b.bytes[i*4]) << 24) |
                (static_cast<uint32_t>(k.b.bytes[i*4+1]) << 16) |
                (static_cast<uint32_t>(k.b.bytes[i*4+2]) << 8) |
                static_cast<uint32_t>(k.b.bytes[i*4+3])
            ) << 1;
        }
        return h;
    }
};

struct EdgeData {
    Blake3Hash a_hash;
    Blake3Hash b_hash;
    int32_t a_cx, a_cy, a_cz, a_cm;
    int32_t b_cx, b_cy, b_cz, b_cm;
    float weight;
};

// Accumulate co-occurrence edges from token sequence
std::unordered_map<EdgeKey, EdgeData, EdgeKeyHash> record_relationships(
    const std::vector<TokenRef>& tokens,
    int context_window
) {
    std::unordered_map<EdgeKey, EdgeData, EdgeKeyHash> edges;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        const TokenRef& a = tokens[i];
        
        // Look at tokens within context window
        for (size_t j = i + 1; j < tokens.size() && j <= i + context_window; ++j) {
            const TokenRef& b = tokens[j];
            
            // Skip self-edges
            if (a.hash == b.hash) continue;
            
            // Canonical ordering: smaller hash first (symmetric edges)
            EdgeKey key;
            bool a_first = std::lexicographical_compare(
                a.hash.bytes.begin(), a.hash.bytes.end(),
                b.hash.bytes.begin(), b.hash.bytes.end()
            );
            
            if (a_first) {
                key.a = a.hash;
                key.b = b.hash;
            } else {
                key.a = b.hash;
                key.b = a.hash;
            }
            
            // Weight based on distance (closer = stronger)
            float distance = static_cast<float>(j - i);
            float weight = 1.0f / distance;
            
            auto it = edges.find(key);
            if (it != edges.end()) {
                // Accumulate weight
                it->second.weight += weight;
            } else {
                // Create new edge
                EdgeData data;
                if (a_first) {
                    data.a_hash = a.hash;
                    data.b_hash = b.hash;
                    data.a_cx = a.cx; data.a_cy = a.cy; data.a_cz = a.cz; data.a_cm = a.cm;
                    data.b_cx = b.cx; data.b_cy = b.cy; data.b_cz = b.cz; data.b_cm = b.cm;
                } else {
                    data.a_hash = b.hash;
                    data.b_hash = a.hash;
                    data.a_cx = b.cx; data.a_cy = b.cy; data.a_cz = b.cz; data.a_cm = b.cm;
                    data.b_cx = a.cx; data.b_cy = a.cy; data.b_cz = a.cz; data.b_cm = a.cm;
                }
                data.weight = weight;
                edges[key] = data;
            }
        }
    }
    
    return edges;
}

// =============================================================================
// Database Insertion - Edges as Compositions
// =============================================================================

inline uint32_t int32_to_uint32(int32_t v) { return static_cast<uint32_t>(v); }
inline int32_t uint32_to_int32(uint32_t v) { return static_cast<int32_t>(v); }
inline int64_t uint64_to_int64(uint64_t v) { return static_cast<int64_t>(v); }

bool insert_edges(
    PGconn* conn,
    const std::unordered_map<EdgeKey, EdgeData, EdgeKeyHash>& edges,
    float threshold
) {
    // Filter edges by threshold
    std::vector<const EdgeData*> to_insert;
    for (const auto& [key, data] : edges) {
        if (data.weight >= threshold) {
            to_insert.push_back(&data);
        }
    }
    
    if (to_insert.empty()) return true;
    
    std::cerr << "[EDGES] Inserting " << to_insert.size() << " edges (threshold=" << threshold << ")\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_edge ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(LINESTRINGZM, 0),"
        "  children BYTEA[],"
        "  hilbert_lo BIGINT,"
        "  hilbert_hi BIGINT,"
        "  depth INTEGER,"
        "  atom_count BIGINT,"
        "  weight REAL"
        ") ON COMMIT DROP");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create temp failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY to temp
    res = PQexec(conn, "COPY tmp_edge FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    static const char hex[] = "0123456789abcdef";
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const EdgeData* e : to_insert) {
        // Edge ID = BLAKE3(A.id || B.id)
        std::vector<uint8_t> id_input;
        id_input.insert(id_input.end(), e->a_hash.bytes.begin(), e->a_hash.bytes.end());
        id_input.insert(id_input.end(), e->b_hash.bytes.begin(), e->b_hash.bytes.end());
        Blake3Hash edge_id = Blake3Hasher::hash(std::span<const uint8_t>(id_input));
        
        // Compute centroid
        uint64_t sum_x = int32_to_uint32(e->a_cx) + int32_to_uint32(e->b_cx);
        uint64_t sum_y = int32_to_uint32(e->a_cy) + int32_to_uint32(e->b_cy);
        uint64_t sum_z = int32_to_uint32(e->a_cz) + int32_to_uint32(e->b_cz);
        uint64_t sum_m = int32_to_uint32(e->a_cm) + int32_to_uint32(e->b_cm);
        
        uint32_t cx = static_cast<uint32_t>(sum_x / 2);
        uint32_t cy = static_cast<uint32_t>(sum_y / 2);
        uint32_t cz = static_cast<uint32_t>(sum_z / 2);
        uint32_t cm = static_cast<uint32_t>(sum_m / 2);
        
        // Hilbert index
        Point4D coords(cx, cy, cz, cm);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        
        // ID
        batch += "\\\\x";
        for (uint8_t b : edge_id.bytes) {
            batch += hex[b >> 4];
            batch += hex[b & 0x0F];
        }
        batch += "\t";
        
        // GEOM - LINESTRINGZM with weight in M
        // EWKB: 01 02 00 00 c0 (LE + LINESTRINGZM) + npoints + points
        batch += "01020000c002000000";  // LE + LINESTRINGZM + 2 points
        
        auto append_double = [&](double val) {
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            for (int i = 0; i < 8; ++i) {
                uint8_t byte = (bits >> (i * 8)) & 0xFF;
                batch += hex[byte >> 4];
                batch += hex[byte & 0x0F];
            }
        };
        
        // Point A with weight in M
        append_double(static_cast<double>(int32_to_uint32(e->a_cx)));
        append_double(static_cast<double>(int32_to_uint32(e->a_cy)));
        append_double(static_cast<double>(int32_to_uint32(e->a_cz)));
        append_double(static_cast<double>(e->weight));  // M = weight
        
        // Point B with weight in M
        append_double(static_cast<double>(int32_to_uint32(e->b_cx)));
        append_double(static_cast<double>(int32_to_uint32(e->b_cy)));
        append_double(static_cast<double>(int32_to_uint32(e->b_cz)));
        append_double(static_cast<double>(e->weight));  // M = weight
        
        batch += "\t";
        
        // Children array
        batch += "{\"\\\\\\\\x";
        for (uint8_t b : e->a_hash.bytes) {
            batch += hex[b >> 4];
            batch += hex[b & 0x0F];
        }
        batch += "\",\"\\\\\\\\x";
        for (uint8_t b : e->b_hash.bytes) {
            batch += hex[b >> 4];
            batch += hex[b & 0x0F];
        }
        batch += "\"}\t";
        
        // Hilbert
        batch += std::to_string(uint64_to_int64(hilbert.lo));
        batch += "\t";
        batch += std::to_string(uint64_to_int64(hilbert.hi));
        batch += "\t";
        
        // Depth (edges are depth 1 above their constituents, but we mark as 1 for simplicity)
        batch += "1\t";
        
        // Atom count
        batch += "2\t";
        
        // Weight (for upsert logic)
        batch += std::to_string(e->weight);
        batch += "\n";
        
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
    
    // Upsert: if edge exists, ADD to weight; else insert
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count "
        "FROM tmp_edge "
        "ON CONFLICT (id) DO UPDATE SET "
        "  geom = ST_SetSRID(ST_MakeLine("
        "    ST_SetSRID(ST_MakePoint("
        "      ST_X(ST_StartPoint(atom.geom)),"
        "      ST_Y(ST_StartPoint(atom.geom)),"
        "      ST_Z(ST_StartPoint(atom.geom)),"
        "      ST_M(ST_StartPoint(atom.geom)) + ST_M(ST_StartPoint(EXCLUDED.geom))"
        "    ), 0),"
        "    ST_SetSRID(ST_MakePoint("
        "      ST_X(ST_EndPoint(atom.geom)),"
        "      ST_Y(ST_EndPoint(atom.geom)),"
        "      ST_Z(ST_EndPoint(atom.geom)),"
        "      ST_M(ST_EndPoint(atom.geom)) + ST_M(ST_EndPoint(EXCLUDED.geom))"
        "    ), 0)"
        "  ), 0)");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Upsert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    int rows = atoi(PQcmdTuples(res));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[EDGES] Upserted " << rows << " edges\n";
    return true;
}

// =============================================================================
// Main Ingestion Flow
// =============================================================================

void ingest_file(PGconn* conn, const fs::path& path, const IngestConfig& config) {
    std::cerr << "\n[INGEST] " << path.filename().string() << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    // Read file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "  Cannot read file\n";
        return;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    std::cerr << "  Size: " << content.size() << " bytes\n";
    
    // Decode UTF-8
    std::vector<uint32_t> codepoints = decode_utf8(content);
    std::cerr << "  Codepoints: " << codepoints.size() << "\n";
    
    // Phase 2: Greedy tokenization
    std::vector<TokenRef> tokens = tokenize(codepoints);
    std::cerr << "  Tokens: " << tokens.size() << "\n";
    
    // Phase 3: Record relationships
    auto edges = record_relationships(tokens, config.context_window);
    std::cerr << "  Raw edges: " << edges.size() << "\n";
    
    // Insert to database
    insert_edges(conn, edges, config.edge_threshold);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "  Time: " << ms << " ms\n";
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Semantic Ingester - Record relationships, not just structure\n\n"
              << "Usage: " << prog << " [options] <path>\n\n"
              << "Options:\n"
              << "  -d, --dbname NAME     Database name (default: hypercube)\n"
              << "  -h, --host HOST       Database host\n"
              << "  -p, --port PORT       Database port\n"
              << "  -U, --user USER       Database user\n"
              << "  -w, --window N        Context window size (default: 5)\n"
              << "  -t, --threshold F     Edge weight threshold (default: 0.01)\n"
              << "  -v, --verbose         Verbose output\n"
              << "  --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    IngestConfig config;
    std::string dbname = "hypercube";
    std::string host, port, user;
    std::string target;
    
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
        } else if ((arg == "-w" || arg == "--window") && i + 1 < argc) {
            config.context_window = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            config.edge_threshold = std::stof(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
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
    
    config.conninfo = "dbname=" + dbname;
    if (!host.empty()) config.conninfo += " host=" + host;
    if (!port.empty()) config.conninfo += " port=" + port;
    if (!user.empty()) config.conninfo += " user=" + user;
    
    std::cerr << "=== Semantic Ingester ===\n";
    std::cerr << "Target: " << target << "\n";
    std::cerr << "Context window: " << config.context_window << "\n";
    std::cerr << "Edge threshold: " << config.edge_threshold << "\n\n";
    
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    // Load atoms and vocabulary
    if (!load_atoms(conn)) {
        PQfinish(conn);
        return 1;
    }
    
    if (!load_vocabulary(conn)) {
        std::cerr << "[WARN] No vocabulary loaded - will tokenize as atoms only\n";
    }
    
    // Ingest
    fs::path path(target);
    if (!fs::exists(path)) {
        std::cerr << "Not found: " << target << "\n";
        PQfinish(conn);
        return 1;
    }
    
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".md" || ext == ".json" || ext == ".py" ||
                ext == ".cpp" || ext == ".hpp" || ext == ".c" || ext == ".h") {
                ingest_file(conn, entry.path(), config);
            }
        }
    } else {
        ingest_file(conn, path, config);
    }
    
    // Final stats
    PGresult* res = PQexec(conn,
        "SELECT "
        "  COUNT(*) FILTER (WHERE depth = 0) as atoms, "
        "  COUNT(*) FILTER (WHERE depth > 0) as compositions, "
        "  MAX(ST_M(ST_StartPoint(geom))) FILTER (WHERE depth > 0) as max_weight "
        "FROM atom");
    
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        std::cerr << "\n=== Database Stats ===\n";
        std::cerr << "  Atoms: " << PQgetvalue(res, 0, 0) << "\n";
        std::cerr << "  Compositions: " << PQgetvalue(res, 0, 1) << "\n";
        std::cerr << "  Max edge weight: " << PQgetvalue(res, 0, 2) << "\n";
    }
    PQclear(res);
    
    PQfinish(conn);
    return 0;
}
