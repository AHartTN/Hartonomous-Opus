/**
 * Direct PostgreSQL Atom Seeder - Maximum Performance
 * 
 * Strategy:
 * 1. Generate atoms in parallel (C++ threads)
 * 2. COPY to UNLOGGED staging table (no WAL overhead, text format for simplicity)
 * 3. Single INSERT ... SELECT with ST_MakePoint (database handles geometry)
 * 4. Rebuild indexes
 * 
 * Target: Seed all 1.1M atoms in <5 seconds (limited by PostGIS geometry creation)
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;

// Connection string
std::string g_conninfo = "dbname=hypercube";

// Number of parallel generators
int g_num_generators = 8;

struct AtomRecord {
    Blake3Hash hash;
    int32_t codepoint;
    AtomCategory category;
    // Raw 32-bit coordinates stored as signed int32 (same bit pattern as uint32)
    int32_t coord_x, coord_y, coord_z, coord_m;
    // Hilbert indices as signed int64 (same bit pattern as uint64)
    int64_t hilbert_lo, hilbert_hi;
};

// Convert uint32 to int32 (reinterpret bits, no data loss)
inline int32_t uint32_to_int32(uint32_t val) {
    return static_cast<int32_t>(val);
}

// Convert uint64 to int64 (reinterpret bits, no data loss)
inline int64_t uint64_to_int64(uint64_t val) {
    return static_cast<int64_t>(val);
}

// Generate atoms for a codepoint range
void generate_range(uint32_t start, uint32_t end, std::vector<AtomRecord>& out) {
    out.reserve(end - start);
    
    for (uint32_t cp = start; cp < end; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) {
            continue;
        }
        
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        
        AtomRecord rec;
        rec.hash = Blake3Hasher::hash_codepoint(cp);
        rec.codepoint = static_cast<int32_t>(cp);
        rec.category = CoordinateMapper::categorize(cp);
        // Store raw uint32 coordinates as int32 (same bit pattern)
        rec.coord_x = uint32_to_int32(coords.x);
        rec.coord_y = uint32_to_int32(coords.y);
        rec.coord_z = uint32_to_int32(coords.z);
        rec.coord_m = uint32_to_int32(coords.m);
        // Store raw uint64 Hilbert indices as int64 (same bit pattern)
        rec.hilbert_lo = uint64_to_int64(hilbert.lo);
        rec.hilbert_hi = uint64_to_int64(hilbert.hi);
        
        out.push_back(rec);
    }
}

// Insert atoms using COPY directly into atom table with raw integer coordinates
bool copy_atoms_lossless(PGconn* conn, const std::vector<AtomRecord>& atoms) {
    // Updated COPY to include both raw integer coords and PostGIS geometry
    PGresult* res = PQexec(conn, 
        "COPY atom (id, codepoint, category, coords, coord_x, coord_y, coord_z, coord_m, hilbert_lo, hilbert_hi) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // Pre-allocate buffer for batch sending
    std::string batch;
    batch.reserve(1 << 20);  // 1MB buffer
    
    // EWKB header for POINTZM: "01010000c0" + 64 bytes of coords = 74 hex chars
    static const char hex_chars[] = "0123456789abcdef";
    char ewkb[75];
    std::memcpy(ewkb, "01010000c0", 10);
    ewkb[74] = '\0';
    
    // Convert int32 to normalized double for PostGIS (preserves as much precision as possible)
    auto int32_to_normalized_double = [](int32_t val) -> double {
        uint32_t uval = static_cast<uint32_t>(val);
        return static_cast<double>(uval) / 4294967295.0;
    };
    
    auto double_to_hex = [&](double val, char* out) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            out[i * 2] = hex_chars[byte >> 4];
            out[i * 2 + 1] = hex_chars[byte & 0x0F];
        }
    };
    
    char num_buf[32];
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        const auto& a = atoms[i];
        
        // Build EWKB from raw integer coords (converted to normalized double for PostGIS)
        double_to_hex(int32_to_normalized_double(a.coord_x), ewkb + 10);
        double_to_hex(int32_to_normalized_double(a.coord_y), ewkb + 26);
        double_to_hex(int32_to_normalized_double(a.coord_z), ewkb + 42);
        double_to_hex(int32_to_normalized_double(a.coord_m), ewkb + 58);
        
        // Build row: id, codepoint, category, coords(EWKB), coord_x, coord_y, coord_z, coord_m, hilbert_lo, hilbert_hi
        batch += "\\\\x";
        batch += a.hash.to_hex();
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%d", a.codepoint);
        batch += num_buf;
        batch += '\t';
        
        batch += category_to_string(a.category);
        batch += '\t';
        
        batch += ewkb;
        batch += '\t';
        
        // Raw integer coordinates (lossless)
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_x);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_y);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_z);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_m);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%ld", a.hilbert_lo);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%ld", a.hilbert_hi);
        batch += num_buf;
        batch += '\n';
        
        // Send batch when full
        if (batch.size() > (1 << 19)) {  // 512KB
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "COPY data failed: " << PQerrorMessage(conn) << std::endl;
                PQputCopyEnd(conn, "error");
                return false;
            }
            batch.clear();
        }
    }
    
    // Send remaining
    if (!batch.empty()) {
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "COPY data failed: " << PQerrorMessage(conn) << std::endl;
            PQputCopyEnd(conn, "error");
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "COPY end failed: " << PQerrorMessage(conn) << std::endl;
        return false;
    }
    
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY result failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    return true;
}

// Insert atoms using COPY to staging table (legacy, updated for lossless)
bool copy_to_staging(PGconn* conn, const std::vector<AtomRecord>& atoms) {
    PGresult* res = PQexec(conn, 
        "COPY atom_staging (hash, codepoint, category, coord_x, coord_y, coord_z, coord_m, hilbert_lo, hilbert_hi) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // Pre-allocate buffer
    std::string row;
    row.reserve(256);
    char num_buf[64];
    
    for (const auto& a : atoms) {
        row.clear();
        
        // Hash as bytea hex
        row += "\\\\x";
        row += a.hash.to_hex();
        row += '\t';
        
        // Codepoint
        row += std::to_string(a.codepoint);
        row += '\t';
        
        // Category
        row += category_to_string(a.category);
        row += '\t';
        
        // Raw integer coordinates (lossless)
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_x);
        row += num_buf;
        row += '\t';
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_y);
        row += num_buf;
        row += '\t';
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_z);
        row += num_buf;
        row += '\t';
        snprintf(num_buf, sizeof(num_buf), "%d", a.coord_m);
        row += num_buf;
        row += '\t';
        
        // Hilbert indices
        row += std::to_string(a.hilbert_lo);
        row += '\t';
        row += std::to_string(a.hilbert_hi);
        row += '\n';
        
        if (PQputCopyData(conn, row.c_str(), static_cast<int>(row.size())) != 1) {
            std::cerr << "COPY data failed: " << PQerrorMessage(conn) << std::endl;
            PQputCopyEnd(conn, "error");
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "COPY end failed: " << PQerrorMessage(conn) << std::endl;
        return false;
    }
    
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY result failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    return true;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  -j, --jobs N         Number of parallel generators (default: 8)\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    
    // Parse args
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
        } else if ((arg == "-j" || arg == "--jobs") && i + 1 < argc) {
            g_num_generators = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Build connection string
    g_conninfo = "dbname=" + dbname;
    if (!host.empty()) g_conninfo += " host=" + host;
    if (!port.empty()) g_conninfo += " port=" + port;
    if (!user.empty()) g_conninfo += " user=" + user;
    
    std::cerr << "=== Direct Atom Seeder ===\n";
    std::cerr << "Connection: " << g_conninfo << "\n";
    std::cerr << "Generators: " << g_num_generators << "\n\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 1: Generate atoms in parallel
    std::cerr << "Generating atoms...\n";
    std::vector<std::vector<AtomRecord>> thread_results(g_num_generators);
    std::vector<std::thread> threads;
    
    uint32_t total_codepoints = constants::MAX_CODEPOINT + 1;
    uint32_t chunk_size = (total_codepoints + g_num_generators - 1) / g_num_generators;
    
    for (int t = 0; t < g_num_generators; ++t) {
        uint32_t t_start = t * chunk_size;
        uint32_t t_end = std::min(t_start + chunk_size, total_codepoints);
        threads.emplace_back([t, t_start, t_end, &thread_results]() {
            generate_range(t_start, t_end, thread_results[t]);
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    // Merge results
    std::vector<AtomRecord> all_atoms;
    size_t total = 0;
    for (const auto& v : thread_results) total += v.size();
    all_atoms.reserve(total);
    for (auto& v : thread_results) {
        all_atoms.insert(all_atoms.end(),
                         std::make_move_iterator(v.begin()),
                         std::make_move_iterator(v.end()));
    }
    
    auto gen_time = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_time - start_time).count();
    std::cerr << "Generated " << all_atoms.size() << " atoms in " << gen_ms << " ms\n";
    
    // Step 2: Connect and truncate atom table
    PGconn* conn = PQconnectdb(g_conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }
    
    // Drop indexes on atom table for faster insert
    std::cerr << "Dropping indexes...\n";
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coords");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_hilbert");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_category");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_letters");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_digits");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coord_x");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coord_y");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coord_z");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coord_m");
    PQexec(conn, "DROP INDEX IF EXISTS idx_atom_coords_int");
    PQexec(conn, "TRUNCATE atom CASCADE");
    
    auto setup_time = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - gen_time).count();
    std::cerr << "Setup complete: " << setup_ms << " ms\n";
    
    // Step 3: COPY directly to atom table with lossless integer coordinates
    std::cerr << "COPY to atom table (lossless integers)...\n";
    if (!copy_atoms_lossless(conn, all_atoms)) {
        PQfinish(conn);
        return 1;
    }
    
    auto copy_time = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(copy_time - setup_time).count();
    std::cerr << "COPY complete: " << copy_ms << " ms\n";
    
    // Step 4: Rebuild indexes
    std::cerr << "Rebuilding indexes...\n";
    PQexec(conn, "CREATE INDEX idx_atom_coords ON atom USING GIST(coords)");
    PQexec(conn, "CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo)");
    PQexec(conn, "CREATE INDEX idx_atom_category ON atom(category)");
    PQexec(conn, "CREATE INDEX idx_atom_letters ON atom(codepoint) WHERE category IN ('letter_upper', 'letter_lower', 'letter_titlecase', 'letter_other')");
    PQexec(conn, "CREATE INDEX idx_atom_digits ON atom(codepoint) WHERE category = 'digit'");
    // New indexes on raw integer coordinates
    PQexec(conn, "CREATE INDEX idx_atom_coord_x ON atom(coord_x)");
    PQexec(conn, "CREATE INDEX idx_atom_coord_y ON atom(coord_y)");
    PQexec(conn, "CREATE INDEX idx_atom_coord_z ON atom(coord_z)");
    PQexec(conn, "CREATE INDEX idx_atom_coord_m ON atom(coord_m)");
    PQexec(conn, "ANALYZE atom");
    
    auto index_time = std::chrono::high_resolution_clock::now();
    auto index_ms = std::chrono::duration_cast<std::chrono::milliseconds>(index_time - copy_time).count();
    std::cerr << "Index rebuild: " << index_ms << " ms\n";
    
    PQfinish(conn);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total atoms: " << all_atoms.size() << "\n";
    std::cerr << "Total time: " << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n";
    std::cerr << "Rate: " << (all_atoms.size() * 1000 / std::max(total_ms, 1L)) << " atoms/sec\n";
    
    return 0;
}
