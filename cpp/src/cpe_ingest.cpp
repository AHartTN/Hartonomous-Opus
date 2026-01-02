/**
 * Cascading Pair Encoding (CPE) Ingester - High Performance
 * 
 * Ingests files/directories using CPE entirely in C++:
 * 1. Load atom lookup table once (codepoint → hash,coords)
 * 2. Read file as UTF-8 codepoints
 * 3. Perform CPE cascade in memory (binary tree merging)
 * 4. Batch-insert all compositions via COPY
 * 
 * Performance target: 1MB/s+ ingestion rate
 * 
 * Key insight: CPE reduces N characters to ~2N compositions (binary tree)
 * Each merge pass halves the count: N → N/2 → N/4 → ... → 1
 * Total compositions = N + N/2 + N/4 + ... ≈ 2N
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <queue>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;
namespace fs = std::filesystem;

// =============================================================================
// Atom Lookup Cache
// =============================================================================

struct AtomInfo {
    Blake3Hash hash;
    // Raw 32-bit coordinates as signed int (same bit pattern as uint32)
    int32_t coord_x, coord_y, coord_z, coord_m;
};

// Global atom cache - loaded once at startup
static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;

bool load_atom_cache(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();

    // Load atoms from unified schema using codepoint column
    // Extract coordinates from POINTZM geometry (normalized to [0,1], scale back to uint32)
    PGresult* res = PQexec(conn,
        "SELECT "
        "  codepoint, "
        "  id, "
        "  ST_X(geom) * 4294967295, "
        "  ST_Y(geom) * 4294967295, "
        "  ST_Z(geom) * 4294967295, "
        "  ST_M(geom) * 4294967295 "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    if (rows == 0) {
        std::cerr << "ERROR: No atoms found in database!\n";
        std::cerr << "       The atom table may need seeding with: ./setup.sh init\n";
        PQclear(res);
        return false;
    }

    g_atom_cache.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        uint32_t cp = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 0)));

        AtomInfo info;
        // Parse bytea hex: \x followed by hex digits
        const char* hex = PQgetvalue(res, i, 1);
        if (hex[0] == '\\' && hex[1] == 'x') {
            info.hash = Blake3Hash::from_hex(std::string_view(hex + 2, 64));
        }
        // Convert normalized double back to uint32 as int32
        info.coord_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 2))));
        info.coord_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 3))));
        info.coord_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 4))));
        info.coord_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 5))));

        g_atom_cache[cp] = info;
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[CACHE] Loaded " << g_atom_cache.size() << " atoms in " << ms << " ms\n";

    return true;
}

// =============================================================================
// UTF-8 Decoding
// =============================================================================

// Decode UTF-8 bytes to codepoints
std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());  // Worst case: all ASCII
    
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();
    
    while (p < end) {
        uint32_t cp;
        
        if (*p < 0x80) {
            // ASCII
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            // 2-byte
            cp = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            // 3-byte
            cp = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            // 4-byte
            cp = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            // Invalid - use replacement character
            cp = 0xFFFD;
            ++p;
        }
        
        codepoints.push_back(cp);
    }
    
    return codepoints;
}

// =============================================================================
// CPE Composition
// =============================================================================

struct CompositionRecord {
    Blake3Hash hash;
    // Raw 32-bit centroid coordinates as signed int (same bit pattern as uint32)
    int32_t coord_x, coord_y, coord_z, coord_m;
    int64_t hilbert_lo, hilbert_hi;
    uint32_t depth;
    uint64_t atom_count;

    // Children (unified schema uses BYTEA[] array)
    Blake3Hash left_hash;
    Blake3Hash right_hash;
    // Child coordinates (for building LINESTRINGZM)
    int32_t left_x, left_y, left_z, left_m;
    int32_t right_x, right_y, right_z, right_m;
};

// Convert int32 to uint32 (reinterpret bits, no data loss)
inline uint32_t int32_to_uint32(int32_t val) {
    return static_cast<uint32_t>(val);
}

// Convert uint32 to int32 (reinterpret bits, no data loss)
inline int32_t uint32_to_int32(uint32_t val) {
    return static_cast<int32_t>(val);
}

// Convert uint64 to int64 (reinterpret bits, no data loss)
inline int64_t uint64_to_int64(uint64_t val) {
    return static_cast<int64_t>(val);
}

// Cache of compositions we've created (deduplication)
static std::unordered_map<std::string, CompositionRecord> g_comp_cache;

// Create a pair composition from two children using lossless integer arithmetic
// Returns {record, is_new} - is_new is false if already cached
std::pair<CompositionRecord, bool> create_pair(
    const Blake3Hash& left_hash, int32_t left_x, int32_t left_y, int32_t left_z, int32_t left_m,
    uint32_t left_depth, uint64_t left_atoms, bool left_is_atom,
    const Blake3Hash& right_hash, int32_t right_x, int32_t right_y, int32_t right_z, int32_t right_m,
    uint32_t right_depth, uint64_t right_atoms, bool right_is_atom
) {
    // Build hash input: ordinal(4) + hash(32) for each child
    std::vector<uint8_t> hash_input;
    hash_input.reserve(72);  // 4 + 32 + 4 + 32
    
    // Left child: ordinal 0
    uint32_t ord0 = 0;
    hash_input.insert(hash_input.end(), 
        reinterpret_cast<uint8_t*>(&ord0), 
        reinterpret_cast<uint8_t*>(&ord0) + 4);
    hash_input.insert(hash_input.end(), left_hash.bytes.begin(), left_hash.bytes.end());
    
    // Right child: ordinal 1
    uint32_t ord1 = 1;
    hash_input.insert(hash_input.end(), 
        reinterpret_cast<uint8_t*>(&ord1), 
        reinterpret_cast<uint8_t*>(&ord1) + 4);
    hash_input.insert(hash_input.end(), right_hash.bytes.begin(), right_hash.bytes.end());
    
    Blake3Hash hash = Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
    
    // Check cache
    std::string hash_key = hash.to_hex();
    auto it = g_comp_cache.find(hash_key);
    if (it != g_comp_cache.end()) {
        return {it->second, false};  // Already exists
    }
    
    // Compute centroid using unsigned integer arithmetic (lossless)
    // Convert signed int32 to unsigned uint32, average, then back to signed
    uint64_t sum_x = static_cast<uint64_t>(int32_to_uint32(left_x)) + static_cast<uint64_t>(int32_to_uint32(right_x));
    uint64_t sum_y = static_cast<uint64_t>(int32_to_uint32(left_y)) + static_cast<uint64_t>(int32_to_uint32(right_y));
    uint64_t sum_z = static_cast<uint64_t>(int32_to_uint32(left_z)) + static_cast<uint64_t>(int32_to_uint32(right_z));
    uint64_t sum_m = static_cast<uint64_t>(int32_to_uint32(left_m)) + static_cast<uint64_t>(int32_to_uint32(right_m));
    
    // Average (integer division)
    uint32_t cx = static_cast<uint32_t>(sum_x / 2);
    uint32_t cy = static_cast<uint32_t>(sum_y / 2);
    uint32_t cz = static_cast<uint32_t>(sum_z / 2);
    uint32_t cm = static_cast<uint32_t>(sum_m / 2);
    
    // Compute Hilbert index from raw coords
    Point4D coords(cx, cy, cz, cm);
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    
    CompositionRecord rec;
    rec.hash = hash;
    rec.coord_x = uint32_to_int32(cx);
    rec.coord_y = uint32_to_int32(cy);
    rec.coord_z = uint32_to_int32(cz);
    rec.coord_m = uint32_to_int32(cm);
    rec.hilbert_lo = uint64_to_int64(hilbert.lo);
    rec.hilbert_hi = uint64_to_int64(hilbert.hi);
    rec.depth = std::max(left_depth, right_depth) + 1;
    rec.atom_count = left_atoms + right_atoms;
    rec.left_hash = left_hash;
    rec.right_hash = right_hash;
    // Store child coordinates for LINESTRINGZM
    rec.left_x = left_x;
    rec.left_y = left_y;
    rec.left_z = left_z;
    rec.left_m = left_m;
    rec.right_x = right_x;
    rec.right_y = right_y;
    rec.right_z = right_z;
    rec.right_m = right_m;
    
    g_comp_cache[hash_key] = rec;
    return {rec, true};  // New composition
}

// CPE cascade: reduce sequence to single root
struct CpeNode {
    Blake3Hash hash;
    // Raw 32-bit coordinates as signed int (same bit pattern as uint32)
    int32_t x, y, z, m;
    uint32_t depth;
    uint64_t atoms;
    bool is_atom;
};

Blake3Hash cpe_cascade(const std::vector<uint32_t>& codepoints, 
                       std::vector<CompositionRecord>& new_compositions) {
    if (codepoints.empty()) {
        return Blake3Hash();
    }
    
    // Convert codepoints to nodes
    std::vector<CpeNode> nodes;
    nodes.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        auto it = g_atom_cache.find(cp);
        if (it == g_atom_cache.end()) {
            // Unknown character - skip (or use replacement)
            continue;
        }
        
        CpeNode node;
        node.hash = it->second.hash;
        node.x = it->second.coord_x;
        node.y = it->second.coord_y;
        node.z = it->second.coord_z;
        node.m = it->second.coord_m;
        node.depth = 0;
        node.atoms = 1;
        node.is_atom = true;
        nodes.push_back(node);
    }
    
    if (nodes.empty()) {
        return Blake3Hash();
    }
    
    if (nodes.size() == 1) {
        return nodes[0].hash;
    }
    
    // Cascade until single root
    while (nodes.size() > 1) {
        std::vector<CpeNode> merged;
        merged.reserve((nodes.size() + 1) / 2);
        
        for (size_t i = 0; i < nodes.size(); i += 2) {
            if (i + 1 < nodes.size()) {
                // Merge pair
                const CpeNode& left = nodes[i];
                const CpeNode& right = nodes[i + 1];
                
                auto [rec, is_new] = create_pair(
                    left.hash, left.x, left.y, left.z, left.m,
                    left.depth, left.atoms, left.is_atom,
                    right.hash, right.x, right.y, right.z, right.m,
                    right.depth, right.atoms, right.is_atom
                );
                
                if (is_new) {
                    new_compositions.push_back(rec);
                }
                
                CpeNode merged_node;
                merged_node.hash = rec.hash;
                merged_node.x = rec.coord_x;
                merged_node.y = rec.coord_y;
                merged_node.z = rec.coord_z;
                merged_node.m = rec.coord_m;
                merged_node.depth = rec.depth;
                merged_node.atoms = rec.atom_count;
                merged_node.is_atom = false;
                merged.push_back(merged_node);
            } else {
                // Odd element - carry forward
                merged.push_back(nodes[i]);
            }
        }
        
        nodes = std::move(merged);
    }
    
    return nodes[0].hash;
}

// =============================================================================
// Database Insertion
// =============================================================================

// Build EWKB hex for LINESTRINGZM from two 4D points
// Format: type(4) + npoints(4) + point1(32) + point2(32) = 72 bytes = 144 hex chars
std::string build_linestringzm_ewkb(
    int32_t x1, int32_t y1, int32_t z1, int32_t m1,
    int32_t x2, int32_t y2, int32_t z2, int32_t m2
) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string ewkb;
    ewkb.reserve(146);  // "01" + type(8) + npoints(8) + 2 * coords(64)

    // Byte order: little-endian
    ewkb += "01";

    // Type: LINESTRINGZM = 0xC0000002 (little-endian: 020000c0)
    ewkb += "020000c0";

    // Number of points: 2 (little-endian: 02000000)
    ewkb += "02000000";

    // Helper to append double as little-endian hex
    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };

    // Convert int32 coords to normalized doubles [0, 1] for PostGIS
    constexpr double COORD_SCALE = 1.0 / 4294967295.0;

    // Point 1
    append_double(static_cast<double>(int32_to_uint32(x1)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(y1)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(z1)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(m1)) * COORD_SCALE);

    // Point 2
    append_double(static_cast<double>(int32_to_uint32(x2)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(y2)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(z2)) * COORD_SCALE);
    append_double(static_cast<double>(int32_to_uint32(m2)) * COORD_SCALE);

    return ewkb;
}

bool insert_compositions(PGconn* conn, const std::vector<CompositionRecord>& comps) {
    if (comps.empty()) return true;

    // Begin transaction
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);

    // Create temp table matching unified atom schema
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_atom ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(GEOMETRYZM, 0) NOT NULL,"
        "  children BYTEA[],"
        "  value BYTEA,"
        "  hilbert_lo BIGINT NOT NULL,"
        "  hilbert_hi BIGINT NOT NULL,"
        "  depth INTEGER NOT NULL,"
        "  atom_count BIGINT NOT NULL"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_atom failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // COPY to temp table
    res = PQexec(conn,
        "COPY tmp_atom (id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");

    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    std::string batch;
    batch.reserve(1 << 20);
    char num_buf[32];

    for (const auto& c : comps) {
        // id (bytea hex)
        batch += "\\\\x";
        batch += c.hash.to_hex();
        batch += '\t';

        // geom: LINESTRINGZM from child centroids
        batch += build_linestringzm_ewkb(
            c.left_x, c.left_y, c.left_z, c.left_m,
            c.right_x, c.right_y, c.right_z, c.right_m
        );
        batch += '\t';

        // children: BYTEA array format {"\\x<hex>","\\x<hex>"}
        // For COPY format, each element needs to be quoted, with \\x prefix
        batch += "{\"\\\\\\\\x";
        batch += c.left_hash.to_hex();
        batch += "\",\"\\\\\\\\x";
        batch += c.right_hash.to_hex();
        batch += "\"}";
        batch += '\t';

        // value: NULL for compositions
        batch += "\\N";
        batch += '\t';

        // hilbert_lo, hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%ld", c.hilbert_lo);
        batch += num_buf;
        batch += '\t';

        snprintf(num_buf, sizeof(num_buf), "%ld", c.hilbert_hi);
        batch += num_buf;
        batch += '\t';

        // depth
        snprintf(num_buf, sizeof(num_buf), "%u", c.depth);
        batch += num_buf;
        batch += '\t';

        // atom_count
        snprintf(num_buf, sizeof(num_buf), "%lu", c.atom_count);
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
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "COPY final failed\n";
            PQputCopyEnd(conn, "error");
            return false;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "COPY end failed: " << PQerrorMessage(conn) << std::endl;
        return false;
    }

    res = PQgetResult(conn);
    bool success = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!success) {
        std::cerr << "COPY result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Upsert from temp to atom table
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count FROM tmp_atom "
        "ON CONFLICT (id) DO NOTHING");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Upsert failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Commit transaction
    res = PQexec(conn, "COMMIT");
    PQclear(res);

    return true;
}

// =============================================================================
// File Processing
// =============================================================================

struct IngestResult {
    Blake3Hash root;
    size_t bytes;
    size_t codepoints;
    size_t compositions;
    double seconds;
};

// Process a file without database insert - just cascade in memory
IngestResult process_file(const fs::path& path, std::vector<CompositionRecord>& all_comps) {
    IngestResult result{};
    auto start = std::chrono::high_resolution_clock::now();
    
    // Read file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot read: " << path << std::endl;
        return result;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    result.bytes = content.size();
    
    // Decode UTF-8
    std::vector<uint32_t> codepoints = decode_utf8(content);
    result.codepoints = codepoints.size();
    
    if (codepoints.empty()) {
        return result;
    }
    
    // CPE cascade - appends to all_comps
    size_t before = all_comps.size();
    result.root = cpe_cascade(codepoints, all_comps);
    result.compositions = all_comps.size() - before;
    
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    
    return result;
}

IngestResult ingest_file(PGconn* conn, const fs::path& path) {
    std::vector<CompositionRecord> new_comps;
    new_comps.reserve(100000);
    
    IngestResult result = process_file(path, new_comps);
    
    // Insert to database
    if (!new_comps.empty()) {
        insert_compositions(conn, new_comps);
    }
    
    return result;
}

void ingest_directory(PGconn* conn, const fs::path& dir) {
    size_t total_bytes = 0;
    size_t total_files = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Collect text files
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
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
    
    // Process all files, accumulate compositions
    std::vector<CompositionRecord> all_comps;
    all_comps.reserve(1000000);  // Pre-allocate for speed
    
    for (const auto& path : files) {
        auto file_start = std::chrono::high_resolution_clock::now();
        size_t before = all_comps.size();
        
        IngestResult res = process_file(path, all_comps);
        
        auto file_end = std::chrono::high_resolution_clock::now();
        double file_secs = std::chrono::duration<double>(file_end - file_start).count();
        
        if (res.bytes > 0) {
            total_bytes += res.bytes;
            total_files++;
            
            double rate_kbps = (res.bytes / 1024.0) / file_secs;
            std::cerr << "  " << path.filename().string() 
                      << " (" << res.bytes << " B, " 
                      << res.compositions << " comps, "
                      << static_cast<int>(rate_kbps) << " KB/s) → "
                      << res.root.to_hex().substr(0, 16) << "...\n";
        }
    }
    
    // Batch insert all compositions at once
    std::cerr << "\n[DB] Inserting " << all_comps.size() << " compositions...\n";
    auto db_start = std::chrono::high_resolution_clock::now();
    
    if (!all_comps.empty()) {
        insert_compositions(conn, all_comps);
    }
    
    auto db_end = std::chrono::high_resolution_clock::now();
    double db_secs = std::chrono::duration<double>(db_end - db_start).count();
    std::cerr << "[DB] Insert completed in " << db_secs << " s\n";
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(end - start).count();
    double rate_kbps = (total_bytes / 1024.0) / total_secs;
    
    std::cerr << "\n[COMPLETE]\n";
    std::cerr << "  Files: " << total_files << "\n";
    std::cerr << "  Bytes: " << total_bytes << " (" << (total_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cerr << "  Compositions: " << all_comps.size() << "\n";
    std::cerr << "  Time: " << total_secs << " s\n";
    std::cerr << "  Rate: " << rate_kbps << " KB/s\n";
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Cascading Pair Encoding Ingester\n\n"
              << "Usage: " << prog << " [options] <path>\n\n"
              << "Arguments:\n"
              << "  <path>               File or directory to ingest\n\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  --help               Show this help\n\n"
              << "Examples:\n"
              << "  " << prog << " -d hypercube myfile.txt\n"
              << "  " << prog << " -d hypercube ~/Documents/\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    std::string target = "";
    
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
    
    std::cerr << "=== CPE Ingester ===\n";
    std::cerr << "Database: " << dbname << "\n";
    std::cerr << "Target: " << target << "\n\n";
    
    // Connect
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
    
    // Process target
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
            std::cout << res.root.to_hex() << std::endl;
            std::cerr << "[OK] " << res.bytes << " bytes → " << res.compositions << " compositions\n";
        }
    }
    
    // Stats
    PGresult* res = PQexec(conn, "SELECT COUNT(*) FROM atom WHERE depth > 0");
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        std::cerr << "\nTotal compositions in DB: " << PQgetvalue(res, 0, 0) << "\n";
    }
    PQclear(res);
    
    PQfinish(conn);
    return 0;
}
