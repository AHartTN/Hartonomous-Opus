/**
 * Parallel Partitioned Atom Seeder - Maximum Performance
 * 
 * Strategy:
 * 1. Generate all atoms in parallel (C++ threads)
 * 2. Partition by blake3 hash prefix (12 partitions)
 * 3. COPY to 12 unlogged staging tables in parallel (12 connections)
 * 4. Single INSERT ... SELECT to merge into atom table
 * 5. Rebuild indexes in parallel where possible
 * 
 * Target: Seed all 1.1M atoms in <2 seconds total
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <future>
#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;

// Configuration
static constexpr int NUM_PARTITIONS = 12;
static constexpr int NUM_GENERATORS = 8;

struct AtomRecord {
    Blake3Hash hash;
    int32_t codepoint;
    AtomCategory category;
    double x, y, z, m;
    int64_t hilbert_lo, hilbert_hi;
    
    // Partition based on first nibble of hash (0-15, mapped to 0-11)
    int partition() const {
        return (hash.bytes[0] >> 4) % NUM_PARTITIONS;
    }
};

// Generate atoms for a codepoint range
void generate_range(uint32_t start, uint32_t end, std::vector<AtomRecord>& out) {
    out.reserve(end - start);
    constexpr double COORD_SCALE = 1.0 / 4294967295.0;
    
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
        rec.x = static_cast<double>(coords.x) * COORD_SCALE;
        rec.y = static_cast<double>(coords.y) * COORD_SCALE;
        rec.z = static_cast<double>(coords.z) * COORD_SCALE;
        rec.m = static_cast<double>(coords.m) * COORD_SCALE;
        rec.hilbert_lo = static_cast<int64_t>(hilbert.lo);
        rec.hilbert_hi = static_cast<int64_t>(hilbert.hi);
        
        out.push_back(rec);
    }
}

// Partition atoms by hash prefix
void partition_atoms(const std::vector<AtomRecord>& all, 
                     std::array<std::vector<AtomRecord>, NUM_PARTITIONS>& partitions) {
    // Count per partition for pre-allocation
    std::array<size_t, NUM_PARTITIONS> counts{};
    for (const auto& a : all) {
        counts[a.partition()]++;
    }
    
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        partitions[i].reserve(counts[i]);
    }
    
    for (const auto& a : all) {
        partitions[a.partition()].push_back(a);
    }
}

// COPY a partition directly to atom table
bool copy_partition(const std::string& conninfo, int partition_id, 
                    const std::vector<AtomRecord>& atoms) {
    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Partition " << partition_id << " connection failed: " 
                  << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return false;
    }
    
    // COPY directly to atom table - no staging
    std::string copy_cmd = 
        "COPY atom (id, codepoint, category, coords, hilbert_lo, hilbert_hi) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";
    
    PGresult* res = PQexec(conn, copy_cmd.c_str());
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "Partition " << partition_id << " COPY start failed: " 
                  << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQfinish(conn);
        return false;
    }
    PQclear(res);
    
    // Build batch buffer
    std::string batch;
    batch.reserve(1 << 20);  // 1MB
    
    static const char hex_chars[] = "0123456789abcdef";
    char ewkb[75];
    std::memcpy(ewkb, "01010000c0", 10);  // POINTZM little-endian, SRID=0
    ewkb[74] = '\0';
    
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
    
    for (const auto& a : atoms) {
        // Build EWKB geometry
        double_to_hex(a.x, ewkb + 10);
        double_to_hex(a.y, ewkb + 26);
        double_to_hex(a.z, ewkb + 42);
        double_to_hex(a.m, ewkb + 58);
        
        // Build row
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
        
        snprintf(num_buf, sizeof(num_buf), "%ld", a.hilbert_lo);
        batch += num_buf;
        batch += '\t';
        
        snprintf(num_buf, sizeof(num_buf), "%ld", a.hilbert_hi);
        batch += num_buf;
        batch += '\n';
        
        // Send when buffer full
        if (batch.size() > (1 << 19)) {
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "Partition " << partition_id << " COPY data failed\n";
                PQputCopyEnd(conn, "error");
                PQfinish(conn);
                return false;
            }
            batch.clear();
        }
    }
    
    // Send remaining
    if (!batch.empty()) {
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "Partition " << partition_id << " COPY final failed\n";
            PQputCopyEnd(conn, "error");
            PQfinish(conn);
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "Partition " << partition_id << " COPY end failed\n";
        PQfinish(conn);
        return false;
    }
    
    res = PQgetResult(conn);
    bool success = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!success) {
        std::cerr << "Partition " << partition_id << " result: " 
                  << PQerrorMessage(conn) << std::endl;
    }
    PQclear(res);
    PQfinish(conn);
    
    return success;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    
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
        }
    }
    
    std::string conninfo = "dbname=" + dbname;
    if (!host.empty()) conninfo += " host=" + host;
    if (!port.empty()) conninfo += " port=" + port;
    if (!user.empty()) conninfo += " user=" + user;
    
    std::cerr << "=== Parallel Partitioned Atom Seeder ===\n";
    std::cerr << "Connection: " << conninfo << "\n";
    std::cerr << "Partitions: " << NUM_PARTITIONS << "\n";
    std::cerr << "Generators: " << NUM_GENERATORS << "\n\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === STEP 1: Generate atoms in parallel ===
    std::cerr << "[1/5] Generating atoms...\n";
    std::vector<std::vector<AtomRecord>> thread_results(NUM_GENERATORS);
    std::vector<std::thread> threads;
    
    uint32_t total_codepoints = constants::MAX_CODEPOINT + 1;
    uint32_t chunk_size = (total_codepoints + NUM_GENERATORS - 1) / NUM_GENERATORS;
    
    for (int t = 0; t < NUM_GENERATORS; ++t) {
        uint32_t t_start = t * chunk_size;
        uint32_t t_end = std::min(t_start + chunk_size, total_codepoints);
        threads.emplace_back([t, t_start, t_end, &thread_results]() {
            generate_range(t_start, t_end, thread_results[t]);
        });
    }
    
    for (auto& th : threads) th.join();
    
    // Merge and partition
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
    std::cerr << "      Generated " << all_atoms.size() << " atoms in " << gen_ms << " ms\n";
    
    // === STEP 2: Partition atoms ===
    std::cerr << "[2/5] Partitioning by hash...\n";
    std::array<std::vector<AtomRecord>, NUM_PARTITIONS> partitions;
    partition_atoms(all_atoms, partitions);
    
    // Free original vector
    all_atoms.clear();
    all_atoms.shrink_to_fit();
    
    auto part_time = std::chrono::high_resolution_clock::now();
    auto part_ms = std::chrono::duration_cast<std::chrono::milliseconds>(part_time - gen_time).count();
    std::cerr << "      Partitioned in " << part_ms << " ms\n";
    
    // === STEP 3: Setup database - create staging tables ===
    std::cerr << "[3/5] Creating staging tables...\n";
    PGconn* main_conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(main_conn) != CONNECTION_OK) {
        std::cerr << "Main connection failed: " << PQerrorMessage(main_conn) << std::endl;
        PQfinish(main_conn);
        return 1;
    }
    
    // Drop all indexes for fast bulk insert
    PQexec(main_conn, "DROP INDEX IF EXISTS idx_atom_coords");
    PQexec(main_conn, "DROP INDEX IF EXISTS idx_atom_hilbert");
    PQexec(main_conn, "DROP INDEX IF EXISTS idx_atom_category");
    PQexec(main_conn, "DROP INDEX IF EXISTS idx_atom_letters");
    PQexec(main_conn, "DROP INDEX IF EXISTS idx_atom_digits");
    
    // Truncate for clean seed
    PQexec(main_conn, "TRUNCATE atom CASCADE");
    
    auto setup_time = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - part_time).count();
    std::cerr << "      Setup in " << setup_ms << " ms\n";
    
    // === STEP 4: Parallel COPY to staging tables ===
    std::cerr << "[4/5] Parallel COPY to atom table (" << NUM_PARTITIONS << " connections)...\n";
    
    std::vector<std::future<bool>> futures;
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        futures.push_back(std::async(std::launch::async, 
            [&conninfo, i, &partitions]() {
                return copy_partition(conninfo, i, partitions[i]);
            }));
    }
    
    bool all_success = true;
    for (auto& f : futures) {
        if (!f.get()) all_success = false;
    }
    
    if (!all_success) {
        std::cerr << "Some partitions failed!\n";
        PQfinish(main_conn);
        return 1;
    }
    
    auto copy_time = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(copy_time - setup_time).count();
    std::cerr << "      Parallel COPY in " << copy_ms << " ms\n";
    
    // === STEP 5: Rebuild indexes ===
    std::cerr << "[5/5] Building indexes...\n";
    
    PQexec(main_conn, "CREATE INDEX idx_atom_coords ON atom USING GIST(coords)");
    PQexec(main_conn, "CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo)");
    PQexec(main_conn, "CREATE INDEX idx_atom_category ON atom(category)");
    PQexec(main_conn, "CREATE INDEX idx_atom_letters ON atom(codepoint) WHERE category IN ('letter_upper', 'letter_lower', 'letter_titlecase', 'letter_other')");
    PQexec(main_conn, "CREATE INDEX idx_atom_digits ON atom(codepoint) WHERE category = 'digit'");
    PQexec(main_conn, "ANALYZE atom");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto index_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - copy_time).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "      Index build in " << index_ms << " ms\n";
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total atoms: " << total << "\n";
    std::cerr << "Total time: " << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n";
    std::cerr << "Rate: " << (total * 1000 / std::max(total_ms, 1L)) << " atoms/sec\n";
    
    PQfinish(main_conn);
    return 0;
}
