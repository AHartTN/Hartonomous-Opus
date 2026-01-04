/**
 * Vocabulary Token Ingester
 *
 * Creates word-level compositions for vocabulary tokens (e.g., from MiniLM).
 * This bridges the gap between CPE (which creates subword motifs) and
 * whole-word embeddings from transformer models.
 *
 * Each vocabulary token becomes a composition:
 * - Hash: BLAKE3 of the token bytes (content-addressed)
 * - Geometry: LINESTRINGZM through child atom coordinates
 * - Children: Codepoint atoms in sequence order
 * - Centroid: 4D average of child coordinates
 *
 * This enables analogy() to find "man", "woman", "king", "queen" etc.
 *
 * Usage:
 *   vocab_ingest --vocab vocab.txt [--db hypercube] [--host localhost]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <chrono>
#include <algorithm>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;

// =============================================================================
// Atom Cache (codepoint -> hash, coords)
// =============================================================================

struct AtomInfo {
    Blake3Hash hash;
    double x, y, z, m;  // 4D coordinates
};

static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;

bool load_atom_cache(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();

    PGresult* res = PQexec(conn,
        "SELECT codepoint, id, ST_X(centroid), ST_Y(centroid), "
        "       ST_Z(centroid), ST_M(centroid) "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    g_atom_cache.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        uint32_t cp = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 0)));

        AtomInfo info;
        const char* hex = PQgetvalue(res, i, 1);
        if (hex[0] == '\\' && hex[1] == 'x') {
            info.hash = Blake3Hash::from_hex(std::string_view(hex + 2, 64));
        }
        info.x = std::stod(PQgetvalue(res, i, 2));
        info.y = std::stod(PQgetvalue(res, i, 3));
        info.z = std::stod(PQgetvalue(res, i, 4));
        info.m = std::stod(PQgetvalue(res, i, 5));

        g_atom_cache[cp] = info;
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "Loaded " << g_atom_cache.size() << " atom entries in " << ms << " ms\n";

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
            cp = 0xFFFD;  // Replacement character
            ++p;
        }

        codepoints.push_back(cp);
    }

    return codepoints;
}

// =============================================================================
// Token Composition Record
// =============================================================================

struct TokenComposition {
    Blake3Hash hash;
    std::string token;  // Original token string for debugging
    std::vector<Blake3Hash> children;  // Child hashes in order
    double cx, cy, cz, cm;  // 4D centroid
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;
    uint64_t atom_count;
};

// Build a token composition using BINARY CASCADE (same as vocabulary_ingest)
// This ensures "captain" from vocab and "captain" from content produce the SAME hash
bool build_token_composition(const std::string& token, TokenComposition& out) {
    auto codepoints = decode_utf8(token);
    if (codepoints.empty()) return false;
    
    // Convert codepoints to atom hashes and coords
    struct Node {
        Blake3Hash hash;
        double x, y, z, m;
        uint32_t depth;
        uint64_t atoms;
    };
    
    std::vector<Node> nodes;
    nodes.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        auto it = g_atom_cache.find(cp);
        if (it == g_atom_cache.end()) {
            return false;  // Unknown codepoint
        }
        Node n;
        n.hash = it->second.hash;
        n.x = it->second.x;
        n.y = it->second.y;
        n.z = it->second.z;
        n.m = it->second.m;
        n.depth = 0;
        n.atoms = 1;
        nodes.push_back(n);
    }
    
    if (nodes.size() == 1) {
        // Single character - just return the atom
        out.hash = nodes[0].hash;
        out.children.push_back(nodes[0].hash);
        out.cx = nodes[0].x;
        out.cy = nodes[0].y;
        out.cz = nodes[0].z;
        out.cm = nodes[0].m;
        out.token = token;
        out.depth = 0;
        out.atom_count = 1;
        
        Point4D coords(static_cast<uint32_t>(out.cx), static_cast<uint32_t>(out.cy),
                       static_cast<uint32_t>(out.cz), static_cast<uint32_t>(out.cm));
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        out.hilbert_lo = static_cast<int64_t>(hilbert.lo);
        out.hilbert_hi = static_cast<int64_t>(hilbert.hi);
        return true;
    }
    
    // Binary cascade: pair adjacent nodes until single root
    while (nodes.size() > 1) {
        std::vector<Node> next;
        next.reserve((nodes.size() + 1) / 2);
        
        for (size_t i = 0; i + 1 < nodes.size(); i += 2) {
            const Node& left = nodes[i];
            const Node& right = nodes[i + 1];
            
            // Hash: ordinal(0) || left_hash || ordinal(1) || right_hash
            std::vector<uint8_t> hash_input;
            hash_input.reserve(72);
            
            uint32_t ord0 = 0;
            hash_input.insert(hash_input.end(),
                reinterpret_cast<uint8_t*>(&ord0),
                reinterpret_cast<uint8_t*>(&ord0) + 4);
            hash_input.insert(hash_input.end(),
                left.hash.bytes.begin(), left.hash.bytes.end());
            
            uint32_t ord1 = 1;
            hash_input.insert(hash_input.end(),
                reinterpret_cast<uint8_t*>(&ord1),
                reinterpret_cast<uint8_t*>(&ord1) + 4);
            hash_input.insert(hash_input.end(),
                right.hash.bytes.begin(), right.hash.bytes.end());
            
            Node merged;
            merged.hash = Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
            merged.x = (left.x + right.x) / 2.0;
            merged.y = (left.y + right.y) / 2.0;
            merged.z = (left.z + right.z) / 2.0;
            merged.m = (left.m + right.m) / 2.0;
            merged.depth = std::max(left.depth, right.depth) + 1;
            merged.atoms = left.atoms + right.atoms;
            
            next.push_back(merged);
        }
        
        // Handle odd element
        if (nodes.size() % 2 == 1) {
            next.push_back(nodes.back());
        }
        
        nodes = std::move(next);
    }
    
    // Root node is the final composition
    out.hash = nodes[0].hash;
    out.cx = nodes[0].x;
    out.cy = nodes[0].y;
    out.cz = nodes[0].z;
    out.cm = nodes[0].m;
    out.depth = nodes[0].depth;
    out.atom_count = nodes[0].atoms;
    out.token = token;
    
    // Store original atoms as children (for LINESTRINGZM trajectory)
    for (uint32_t cp : codepoints) {
        auto it = g_atom_cache.find(cp);
        if (it != g_atom_cache.end()) {
            out.children.push_back(it->second.hash);
        }
    }
    
    Point4D coords(static_cast<uint32_t>(out.cx), static_cast<uint32_t>(out.cy),
                   static_cast<uint32_t>(out.cz), static_cast<uint32_t>(out.cm));
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    out.hilbert_lo = static_cast<int64_t>(hilbert.lo);
    out.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    
    return true;
}

// =============================================================================
// Batch Insert
// =============================================================================

bool batch_insert_tokens(PGconn* conn, const std::vector<TokenComposition>& tokens) {
    if (tokens.empty()) return true;

    std::cerr << "Inserting " << tokens.size() << " token compositions...\n";

    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);

    // Create temp table
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_vocab_token ("
        "  id BYTEA, "
        "  cx DOUBLE PRECISION, cy DOUBLE PRECISION, "
        "  cz DOUBLE PRECISION, cm DOUBLE PRECISION, "
        "  hilbert_lo BIGINT, hilbert_hi BIGINT, "
        "  depth INTEGER, atom_count BIGINT, "
        "  children BYTEA[]"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create temp table: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Insert each token (could use COPY for higher performance)
    for (const auto& tok : tokens) {
        std::string children_arr = "ARRAY[";
        for (size_t i = 0; i < tok.children.size(); ++i) {
            if (i > 0) children_arr += ",";
            children_arr += "'\\x" + tok.children[i].to_hex() + "'::bytea";
        }
        children_arr += "]";

        char query[4096];
        snprintf(query, sizeof(query),
            "INSERT INTO tmp_vocab_token VALUES ("
            "'\\x%s'::bytea, %.1f, %.1f, %.1f, %.1f, "
            "%lld, %lld, %u, %llu, %s)",
            tok.hash.to_hex().c_str(),
            tok.cx, tok.cy, tok.cz, tok.cm,
            (long long)tok.hilbert_lo, (long long)tok.hilbert_hi,
            tok.depth, (unsigned long long)tok.atom_count,
            children_arr.c_str());

        res = PQexec(conn, query);
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            // Silently skip duplicates
            PQclear(res);
            continue;
        }
        PQclear(res);
    }

    // Insert into atom table with LINESTRINGZM geometry
    // Build geometry from child centroids
    res = PQexec(conn, R"(
        INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count)
        SELECT
            t.id,
            -- Build LINESTRINGZM from child centroids
            ST_SetSRID(ST_MakeLine(ARRAY(
                SELECT ST_MakePoint(
                    ST_X(a.centroid), ST_Y(a.centroid),
                    ST_Z(a.centroid), ST_M(a.centroid))
                FROM unnest(t.children) WITH ORDINALITY AS c(child_id, ord)
                JOIN atom a ON a.id = c.child_id
                ORDER BY c.ord
            )), 0),
            t.children,
            t.hilbert_lo,
            t.hilbert_hi,
            t.depth,
            t.atom_count
        FROM tmp_vocab_token t
        WHERE NOT EXISTS (SELECT 1 FROM atom WHERE id = t.id)
    )");

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }

    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);

    res = PQexec(conn, "COMMIT");
    PQclear(res);

    std::cerr << "  Inserted " << inserted << " new token compositions\n";
    return true;
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --vocab PATH      Path to vocab.txt (required)\n"
              << "  -d, --dbname NAME Database name (default: hypercube)\n"
              << "  -h, --host HOST   Database host (default: localhost)\n"
              << "  -p, --port PORT   Database port (default: 5432)\n"
              << "  -U, --user USER   Database user\n"
              << "  --batch SIZE      Batch size for inserts (default: 1000)\n"
              << "  --help            Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string vocab_path;
    std::string dbname = "hypercube";
    std::string host = "localhost";
    std::string port = "5432";
    std::string user;
    size_t batch_size = 1000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoul(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (vocab_path.empty()) {
        std::cerr << "Error: --vocab is required\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cerr << "=== Vocabulary Token Ingester ===\n";
    std::cerr << "Vocab: " << vocab_path << "\n";
    std::cerr << "Database: " << dbname << " @ " << host << ":" << port << "\n\n";

    // Connect to database
    std::string conninfo = "dbname=" + dbname + " host=" + host + " port=" + port;
    if (!user.empty()) conninfo += " user=" + user;

    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }

    // Load atom cache
    if (!load_atom_cache(conn)) {
        PQfinish(conn);
        return 1;
    }

    // Load vocabulary
    std::vector<std::string> vocab;
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Error: Cannot open vocab file: " << vocab_path << "\n";
        PQfinish(conn);
        return 1;
    }

    std::string line;
    while (std::getline(vocab_file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            vocab.push_back(line.substr(start, end - start + 1));
        }
    }
    vocab_file.close();

    std::cerr << "Loaded " << vocab.size() << " vocabulary tokens\n";

    // Check which tokens already exist
    std::cerr << "Checking existing tokens...\n";
    std::unordered_set<std::string> existing_hashes;

    PGresult* res = PQexec(conn, "SELECT encode(id, 'hex') FROM atom WHERE depth > 0");
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        int rows = PQntuples(res);
        for (int i = 0; i < rows; ++i) {
            existing_hashes.insert(PQgetvalue(res, i, 0));
        }
    }
    PQclear(res);
    std::cerr << "  Found " << existing_hashes.size() << " existing compositions\n";

    // Build token compositions
    std::vector<TokenComposition> pending;
    pending.reserve(batch_size);

    size_t skipped = 0, failed = 0, total_inserted = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < vocab.size(); ++i) {
        TokenComposition tok;
        if (!build_token_composition(vocab[i], tok)) {
            failed++;
            continue;
        }

        // Check if already exists
        if (existing_hashes.count(tok.hash.to_hex())) {
            skipped++;
            continue;
        }

        pending.push_back(std::move(tok));

        // Batch insert when full
        if (pending.size() >= batch_size) {
            if (batch_insert_tokens(conn, pending)) {
                total_inserted += pending.size();
            }
            pending.clear();

            std::cerr << "\r  Progress: " << (i + 1) << "/" << vocab.size()
                      << " (" << total_inserted << " inserted, "
                      << skipped << " skipped)...";
        }
    }

    // Insert remaining
    if (!pending.empty()) {
        if (batch_insert_tokens(conn, pending)) {
            total_inserted += pending.size();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cerr << "\n\n=== Complete ===\n";
    std::cerr << "  Total tokens: " << vocab.size() << "\n";
    std::cerr << "  Inserted: " << total_inserted << "\n";
    std::cerr << "  Skipped (existing): " << skipped << "\n";
    std::cerr << "  Failed (unknown codepoints): " << failed << "\n";
    std::cerr << "  Time: " << ms << " ms\n";

    // Show final stats
    res = PQexec(conn,
        "SELECT COUNT(*) as total_compositions, "
        "       MAX(depth) as max_depth "
        "FROM atom WHERE depth > 0");
    if (PQntuples(res) > 0) {
        std::cerr << "\nDatabase now has:\n";
        std::cerr << "  Total compositions: " << PQgetvalue(res, 0, 0) << "\n";
        std::cerr << "  Max depth: " << PQgetvalue(res, 0, 1) << "\n";
    }
    PQclear(res);

    PQfinish(conn);
    return 0;
}
