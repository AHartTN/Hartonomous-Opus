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
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <algorithm>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;
namespace fs = std::filesystem;

// =============================================================================
// Composition Record
// =============================================================================

struct CompositionInfo {
    Blake3Hash hash;
    std::vector<Blake3Hash> children;  // Variable-length children
    int32_t centroid_x = 0, centroid_y = 0, centroid_z = 0, centroid_m = 0;
    int64_t hilbert_lo = 0, hilbert_hi = 0;
    uint32_t depth = 0;
    uint64_t atom_count = 0;
    bool from_db = false;  // True if loaded from database
};

// =============================================================================
// Global Vocabulary Cache
// =============================================================================

// Map from hash hex string to composition info
static std::unordered_map<std::string, CompositionInfo> g_vocabulary;

// Map from children hash (concatenated child hashes) to composition hash
// This allows O(1) lookup: "do we have a composition for these children?"
static std::unordered_map<std::string, std::string> g_children_to_hash;

// Atom cache (leaf nodes)
struct AtomInfo {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
};
static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;

// =============================================================================
// Helper Functions
// =============================================================================

inline uint32_t int32_to_uint32(int32_t val) {
    return static_cast<uint32_t>(val);
}

inline int32_t uint32_to_int32(uint32_t val) {
    return static_cast<int32_t>(val);
}

inline int64_t uint64_to_int64(uint64_t val) {
    return static_cast<int64_t>(val);
}

// Build children key for lookup
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
// Database Loading
// =============================================================================

bool load_atom_cache(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();

    PGresult* res = PQexec(conn,
        "SELECT codepoint, id, ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom) "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    if (rows == 0) {
        std::cerr << "ERROR: No atoms found. Run setup first.\n";
        PQclear(res);
        return false;
    }

    g_atom_cache.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        uint32_t cp = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 0)));

        AtomInfo info;
        const char* hex = PQgetvalue(res, i, 1);
        if (hex[0] == '\\' && hex[1] == 'x') {
            info.hash = Blake3Hash::from_hex(std::string_view(hex + 2, 64));
        }
        info.coord_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 2))));
        info.coord_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 3))));
        info.coord_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 4))));
        info.coord_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 5))));

        g_atom_cache[cp] = info;

        // Also add atoms to vocabulary for unified lookup
        CompositionInfo comp;
        comp.hash = info.hash;
        comp.centroid_x = info.coord_x;
        comp.centroid_y = info.coord_y;
        comp.centroid_z = info.coord_z;
        comp.centroid_m = info.coord_m;
        comp.depth = 0;
        comp.atom_count = 1;
        comp.from_db = true;
        g_vocabulary[info.hash.to_hex()] = comp;
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[ATOMS] Loaded " << g_atom_cache.size() << " atoms in " << ms << " ms\n";

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

        // Parse coordinates
        comp.centroid_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 2))));
        comp.centroid_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 3))));
        comp.centroid_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 4))));
        comp.centroid_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 5))));
        comp.hilbert_lo = std::stoll(PQgetvalue(res, i, 6));
        comp.hilbert_hi = std::stoll(PQgetvalue(res, i, 7));
        comp.depth = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 8)));
        comp.atom_count = std::stoull(PQgetvalue(res, i, 9));

        std::string hash_key = comp.hash.to_hex();
        g_vocabulary[hash_key] = comp;

        // Index by children for fast lookup
        if (!comp.children.empty()) {
            std::string children_key = make_children_key(comp.children);
            g_children_to_hash[children_key] = hash_key;
        }
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[VOCAB] Loaded " << g_vocabulary.size() << " entries in " << ms << " ms\n";

    return true;
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
    std::string hash_key = comp.hash.to_hex();
    auto hash_it = g_vocabulary.find(hash_key);
    if (hash_it != g_vocabulary.end()) {
        return hash_it->second;
    }

    // Compute centroid from children
    uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    uint32_t max_depth = 0;
    uint64_t total_atoms = 0;

    for (const auto& child_hash : children) {
        std::string child_key = child_hash.to_hex();
        auto child_it = g_vocabulary.find(child_key);
        if (child_it != g_vocabulary.end()) {
            const CompositionInfo& child = child_it->second;
            sum_x += int32_to_uint32(child.centroid_x);
            sum_y += int32_to_uint32(child.centroid_y);
            sum_z += int32_to_uint32(child.centroid_z);
            sum_m += int32_to_uint32(child.centroid_m);
            max_depth = std::max(max_depth, child.depth);
            total_atoms += child.atom_count;
        }
    }

    size_t n = children.size();
    if (n > 0) {
        comp.centroid_x = uint32_to_int32(static_cast<uint32_t>(sum_x / n));
        comp.centroid_y = uint32_to_int32(static_cast<uint32_t>(sum_y / n));
        comp.centroid_z = uint32_to_int32(static_cast<uint32_t>(sum_z / n));
        comp.centroid_m = uint32_to_int32(static_cast<uint32_t>(sum_m / n));
    }

    // Compute Hilbert index
    Point4D coords(
        int32_to_uint32(comp.centroid_x),
        int32_to_uint32(comp.centroid_y),
        int32_to_uint32(comp.centroid_z),
        int32_to_uint32(comp.centroid_m)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    comp.hilbert_lo = uint64_to_int64(hilbert.lo);
    comp.hilbert_hi = uint64_to_int64(hilbert.hi);

    comp.depth = max_depth + 1;
    comp.atom_count = total_atoms;

    // Add to vocabulary
    g_vocabulary[hash_key] = comp;
    g_children_to_hash[children_key] = hash_key;

    // Track for database insertion
    new_compositions.push_back(comp);

    return g_vocabulary[hash_key];
}

// =============================================================================
// N-Gram Discovery and Ingestion
// =============================================================================

// Discover all unique n-grams in content and add to vocabulary
// Uses bottom-up approach: bigrams first, then trigrams from bigrams, etc.
void discover_ngrams(
    const std::vector<uint32_t>& codepoints,
    std::vector<CompositionInfo>& new_compositions,
    size_t max_ngram_size = 32  // Maximum n-gram size
) {
    if (codepoints.empty()) return;

    // Convert codepoints to atom hashes
    std::vector<Blake3Hash> atom_hashes;
    atom_hashes.reserve(codepoints.size());

    for (uint32_t cp : codepoints) {
        auto it = g_atom_cache.find(cp);
        if (it != g_atom_cache.end()) {
            atom_hashes.push_back(it->second.hash);
        }
    }

    if (atom_hashes.size() < 2) return;

    // Current level hashes (start with atoms)
    std::vector<Blake3Hash> current_level = atom_hashes;

    // Build compositions bottom-up
    // Level 1: pairs of atoms → bigrams
    // Level 2: pairs of bigrams or (bigram, atom) → 4-grams or trigrams
    // etc.

    size_t level = 1;
    while (current_level.size() > 1 && level <= max_ngram_size) {
        std::vector<Blake3Hash> next_level;
        next_level.reserve((current_level.size() + 1) / 2);

        // Create pairs
        for (size_t i = 0; i + 1 < current_level.size(); i += 2) {
            std::vector<Blake3Hash> pair = {current_level[i], current_level[i + 1]};
            const CompositionInfo& comp = find_or_create_composition(pair, new_compositions);
            next_level.push_back(comp.hash);
        }

        // Handle odd element
        if (current_level.size() % 2 == 1) {
            next_level.push_back(current_level.back());
        }

        current_level = std::move(next_level);
        level++;
    }
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
        append_double(static_cast<double>(int32_to_uint32(pt[0])));
        append_double(static_cast<double>(int32_to_uint32(pt[1])));
        append_double(static_cast<double>(int32_to_uint32(pt[2])));
        append_double(static_cast<double>(int32_to_uint32(pt[3])));
    }

    return ewkb;
}

bool insert_compositions(PGconn* conn, const std::vector<CompositionInfo>& compositions) {
    if (compositions.empty()) return true;

    std::cerr << "[DB] Inserting " << compositions.size() << " new compositions...\n";

    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);

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

    std::string batch;
    batch.reserve(1 << 20);
    char num_buf[32];

    for (const auto& comp : compositions) {
        if (comp.from_db || comp.children.empty()) continue;

        // Build geometry points from children
        std::vector<std::array<int32_t, 4>> child_points;
        for (const auto& child_hash : comp.children) {
            auto it = g_vocabulary.find(child_hash.to_hex());
            if (it != g_vocabulary.end()) {
                child_points.push_back({
                    it->second.centroid_x,
                    it->second.centroid_y,
                    it->second.centroid_z,
                    it->second.centroid_m
                });
            }
        }

        if (child_points.empty()) continue;

        // id
        batch += "\\\\x";
        batch += comp.hash.to_hex();
        batch += '\t';

        // geom
        batch += build_linestringzm_ewkb(child_points);
        batch += '\t';

        // children
        batch += "{";
        for (size_t i = 0; i < comp.children.size(); ++i) {
            if (i > 0) batch += ",";
            batch += "\"\\\\\\\\x";
            batch += comp.children[i].to_hex();
            batch += "\"";
        }
        batch += "}";
        batch += '\t';

        // hilbert_lo, hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(comp.hilbert_lo));
        batch += num_buf;
        batch += '\t';

        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(comp.hilbert_hi));
        batch += num_buf;
        batch += '\t';

        // depth
        snprintf(num_buf, sizeof(num_buf), "%u", comp.depth);
        batch += num_buf;
        batch += '\t';

        // atom_count
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(comp.atom_count));
        batch += num_buf;
        batch += '\n';

        if (batch.size() > (1 << 19)) {
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "COPY data failed\n";
                PQputCopyEnd(conn, "error");
                return false;
            }
            batch.clear();
        }
    }

    if (!batch.empty()) {
        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
    }

    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);

    // Upsert
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count FROM tmp_atom "
        "ON CONFLICT (id) DO NOTHING");

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Upsert failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }

    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);

    res = PQexec(conn, "COMMIT");
    PQclear(res);

    std::cerr << "[DB] Inserted " << inserted << " new compositions\n";
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

IngestResult ingest_file(PGconn* conn, const fs::path& path) {
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

    // Discover n-grams and create compositions
    std::vector<CompositionInfo> new_compositions;
    discover_ngrams(codepoints, new_compositions);

    result.new_compositions = new_compositions.size();
    result.reused_compositions = (g_vocabulary.size() - vocab_before) - new_compositions.size();

    // Insert to database
    insert_compositions(conn, new_compositions);

    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();

    return result;
}

void ingest_directory(PGconn* conn, const fs::path& dir) {
    size_t total_bytes = 0;
    size_t total_files = 0;
    size_t total_new = 0;
    size_t total_reused = 0;
    auto start = std::chrono::high_resolution_clock::now();

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

    for (const auto& path : files) {
        IngestResult res = ingest_file(conn, path);
        if (res.bytes > 0) {
            total_bytes += res.bytes;
            total_files++;
            total_new += res.new_compositions;
            total_reused += res.reused_compositions;

            double rate_kbps = (res.bytes / 1024.0) / std::max(res.seconds, 0.001);
            std::cerr << "  " << path.filename().string()
                      << " (" << res.bytes << " B, "
                      << res.new_compositions << " new, "
                      << static_cast<int>(rate_kbps) << " KB/s)\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(end - start).count();

    std::cerr << "\n[COMPLETE]\n";
    std::cerr << "  Files: " << total_files << "\n";
    std::cerr << "  Bytes: " << total_bytes << " (" << (total_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cerr << "  New compositions: " << total_new << "\n";
    std::cerr << "  Vocabulary size: " << g_vocabulary.size() << "\n";
    std::cerr << "  Time: " << total_secs << " s\n";
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
        auto it = g_atom_cache.find(cp);
        if (it != g_atom_cache.end()) {
            atom_hashes.push_back(it->second.hash);
        } else {
            std::cerr << "  ERROR: Unknown codepoint " << cp << "\n";
            return false;
        }
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

    std::string computed_hash = current[0].to_hex();
    std::cerr << "  Computed hash: " << computed_hash << "\n";

    // Check if exists in vocabulary
    auto it = g_vocabulary.find(computed_hash);
    if (it != g_vocabulary.end()) {
        std::cerr << "  FOUND in vocabulary! Depth: " << it->second.depth
                  << ", Atoms: " << it->second.atom_count << "\n";
        return true;
    }

    std::cerr << "  NOT FOUND in vocabulary\n";

    // Also check database directly
    std::string query = "SELECT id, depth, atom_count, atom_text(id) FROM atom WHERE id = '\\x" + computed_hash + "'";
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

    // Load atoms first
    if (!load_atom_cache(conn)) {
        PQfinish(conn);
        return 1;
    }

    // Load existing vocabulary
    if (!load_vocabulary(conn)) {
        PQfinish(conn);
        return 1;
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
