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
#include "hypercube/blake3_simd.hpp"
#include "hypercube/cpu_features.hpp"
#include "hypercube/db/operations.hpp"

using namespace hypercube;
using namespace hypercube::db;

// Configuration
static constexpr int NUM_PARTITIONS = 1;
static constexpr int NUM_GENERATORS = 8;

struct AtomRecord {
    Blake3Hash hash;
    int32_t codepoint;
    AtomCategory category;
    // Raw 32-bit coordinates as signed int (same bit pattern as uint32)
    // This is LOSSLESS storage - int32 and uint32 have same bits
    int32_t coord_x, coord_y, coord_z, coord_m;
    // PostGIS normalized double (for spatial queries only)
    double x, y, z, m;
    int64_t hilbert_lo, hilbert_hi;
    
    // Partition based on first nibble of hash (0-15, mapped to 0-11)
    int partition() const {
        return (hash.bytes[0] >> 4) % NUM_PARTITIONS;
    }
};

// Batch coordinate mapping for multiple codepoints
void map_codepoints_batch(const std::vector<uint32_t>& codepoints,
                         std::vector<Point4D>& coords_out,
                         std::vector<HilbertIndex>& hilberts_out,
                         std::vector<Blake3Hash>& hashes_out) {
    size_t batch_size = codepoints.size();
    coords_out.resize(batch_size);
    hilberts_out.resize(batch_size);
    hashes_out.resize(batch_size);

    // Check CPU features for SIMD optimization
    bool has_avx2 = cpu_features::has_feature(cpu_features::Feature::AVX2);
    bool has_avx512 = cpu_features::has_feature(cpu_features::Feature::AVX512F);

    // Process in batches for SIMD efficiency
    const size_t SIMD_BATCH_SIZE = has_avx512 ? 16 : (has_avx2 ? 8 : 1);

    for (size_t i = 0; i < batch_size; i += SIMD_BATCH_SIZE) {
        size_t current_batch = std::min(SIMD_BATCH_SIZE, batch_size - i);

        // Coordinate mapping (currently scalar, could be vectorized)
        for (size_t j = 0; j < current_batch; ++j) {
            coords_out[i + j] = CoordinateMapper::map_codepoint(codepoints[i + j]);
        }

        // Batch Hilbert computation if SIMD available
        if (has_avx512 && current_batch >= 16) {
            HilbertCurve::coords_to_indices_batch_avx512(&coords_out[i], current_batch, &hilberts_out[i]);
        } else if (has_avx2 && current_batch >= 8) {
            HilbertCurve::coords_to_indices_batch_avx2(&coords_out[i], current_batch, &hilberts_out[i]);
        } else {
            // Scalar fallback
            for (size_t j = 0; j < current_batch; ++j) {
                hilberts_out[i + j] = HilbertCurve::coords_to_index(coords_out[i + j]);
            }
        }

        // Batch BLAKE3 hashing
        // Use SIMD-optimized BLAKE3 when available, fallback to scalar
        for (size_t j = 0; j < current_batch; ++j) {
            hashes_out[i + j] = Blake3Hasher::hash_codepoint(codepoints[i + j]);
        }
    }
}

// Generate atoms for a codepoint range
void generate_range(uint32_t start, uint32_t end, std::vector<AtomRecord>& out) {
    // Collect valid codepoints first (skip surrogates)
    std::vector<uint32_t> codepoints;
    codepoints.reserve(end - start);

    for (uint32_t cp = start; cp < end; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) {
            continue;
        }
        codepoints.push_back(cp);
    }

    if (codepoints.empty()) return;

    // Batch process coordinates, Hilbert indices, and hashes
    std::vector<Point4D> coords;
    std::vector<HilbertIndex> hilberts;
    std::vector<Blake3Hash> hashes;

    map_codepoints_batch(codepoints, coords, hilberts, hashes);

    // Build AtomRecord structures
    out.reserve(out.size() + codepoints.size());

    for (size_t i = 0; i < codepoints.size(); ++i) {
        uint32_t cp = codepoints[i];
        const Point4D& coord = coords[i];
        const HilbertIndex& hilbert = hilberts[i];
        const Blake3Hash& hash = hashes[i];

        AtomRecord rec;
        rec.hash = hash;
        rec.codepoint = static_cast<int32_t>(cp);
        rec.category = CoordinateMapper::categorize(cp);

        // Store uint32 coordinates as int32 (bit-preserving cast)
        // This preserves the full 32-bit value - no information loss
        rec.coord_x = static_cast<int32_t>(coord.x);
        rec.coord_y = static_cast<int32_t>(coord.y);
        rec.coord_z = static_cast<int32_t>(coord.z);
        rec.coord_m = static_cast<int32_t>(coord.m);

        // Store as double for PostGIS - DIRECT from uint32, no int32 reinterpretation
        // This ensures CENTER (2^31) is stored as 2147483648.0, not as negative
        rec.x = static_cast<double>(coord.x);
        rec.y = static_cast<double>(coord.y);
        rec.z = static_cast<double>(coord.z);
        rec.m = static_cast<double>(coord.m);

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

// Encode UTF-8 for a codepoint (returns escaped BYTEA format for COPY)
std::string encode_utf8_value(uint32_t codepoint) {
    std::string result = "\\\\x";
    static const char hex_chars[] = "0123456789abcdef";

    auto append_byte = [&](uint8_t b) {
        result += hex_chars[b >> 4];
        result += hex_chars[b & 0x0F];
    };

    if (codepoint < 0x80) {
        append_byte(static_cast<uint8_t>(codepoint));
    } else if (codepoint < 0x800) {
        append_byte(0xC0 | (codepoint >> 6));
        append_byte(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        append_byte(0xE0 | (codepoint >> 12));
        append_byte(0x80 | ((codepoint >> 6) & 0x3F));
        append_byte(0x80 | (codepoint & 0x3F));
    } else {
        append_byte(0xF0 | (codepoint >> 18));
        append_byte(0x80 | ((codepoint >> 12) & 0x3F));
        append_byte(0x80 | ((codepoint >> 6) & 0x3F));
        append_byte(0x80 | (codepoint & 0x3F));
    }

    return result;
}

// COPY a partition directly to unified atom table
bool copy_partition(const std::string& conninfo, int partition_id,
                    const std::vector<AtomRecord>& atoms) {
    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Partition " << partition_id << " connection failed: "
                  << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return false;
    }

    // COPY to atom table using CopyStream
    // New schema: id, codepoint, value, geom, hilbert_lo, hilbert_hi
    std::string copy_cmd =
        "COPY atom (id, codepoint, value, geom, hilbert_lo, hilbert_hi) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";

    CopyStream copy(conn, copy_cmd.c_str());
    if (!copy.ok()) {
        std::cerr << "Partition " << partition_id << " COPY start failed: "
                  << copy.error() << std::endl;
        PQfinish(conn);
        return false;
    }

    // Build batch buffer - 8MB for better throughput
    std::string batch;
    batch.reserve(8 << 20);  // 8MB

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
        // Build EWKB geometry (POINTZM)
        double_to_hex(a.x, ewkb + 10);
        double_to_hex(a.y, ewkb + 26);
        double_to_hex(a.z, ewkb + 42);
        double_to_hex(a.m, ewkb + 58);

        // id (BYTEA)
        copy_bytea(batch, a.hash);
        batch += '\t';

        // codepoint
        snprintf(num_buf, sizeof(num_buf), "%d", a.codepoint);
        batch += num_buf;
        batch += '\t';

        // value (UTF-8 bytes)
        batch += encode_utf8_value(static_cast<uint32_t>(a.codepoint));
        batch += '\t';

        // geom (POINTZM)
        batch += ewkb;
        batch += '\t';

        // hilbert_lo
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(a.hilbert_lo));
        batch += num_buf;
        batch += '\t';

        // hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(a.hilbert_hi));
        batch += num_buf;
        batch += '\n';

        // Send when buffer full (4MB threshold)
        if (batch.size() > (4 << 20)) {
            if (!copy.put(batch)) {
                std::cerr << "Partition " << partition_id << " COPY data failed: " << copy.error() << "\n";
                PQfinish(conn);
                return false;
            }
            batch.clear();
        }
    }

    // Send remaining
    if (!batch.empty()) {
        if (!copy.put(batch)) {
            std::cerr << "Partition " << partition_id << " COPY final failed: " << copy.error() << "\n";
            PQfinish(conn);
            return false;
        }
    }

    bool success = copy.end();
    if (!success) {
        std::cerr << "Partition " << partition_id << " COPY end failed: " << copy.error() << std::endl;
    }
    
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
    std::cerr << "Generators: " << NUM_GENERATORS << "\n";

    // CPU feature detection and SIMD optimization info
    std::cerr << "CPU Features: " << cpu_features::get_cpu_info();
    bool has_avx2 = cpu_features::has_feature(cpu_features::Feature::AVX2);
    bool has_avx512 = cpu_features::has_feature(cpu_features::Feature::AVX512F);
    bool has_avx_vnni = cpu_features::has_feature(cpu_features::Feature::AVX_VNNI);

    if (has_avx512) {
        std::cerr << "SIMD: AVX-512 enabled (16-way batch processing)\n";
    } else if (has_avx2) {
        std::cerr << "SIMD: AVX2 enabled (8-way batch processing)\n";
    } else {
        std::cerr << "SIMD: Scalar processing (no AVX support)\n";
    }

    if (has_avx_vnni) {
        std::cerr << "VNNI: AVX-VNNI available for enhanced integer operations\n";
    }
    std::cerr << "\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === STEP 1: Generate atoms in parallel ===
    std::cerr << "[1/5] Generating atoms (" << NUM_GENERATORS << " threads)...\n";
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
    thread_results.clear();
    thread_results.shrink_to_fit();
    
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
    
    // === STEP 3: Setup database ===
    std::cerr << "[3/5] Preparing database...\n";
    PGconn* main_conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(main_conn) != CONNECTION_OK) {
        std::cerr << "Main connection failed: " << PQerrorMessage(main_conn) << std::endl;
        PQfinish(main_conn);
        return 1;
    }
    
    // Use stored procedure for database setup
    PQexec(main_conn, "CALL seed_atoms_setup()");
    
    auto setup_time = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - part_time).count();
    std::cerr << "      Setup in " << setup_ms << " ms\n";
    
    // === STEP 4: Parallel COPY ===
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

    // Use stored procedure for finalization
    PQexec(main_conn, "CALL seed_atoms_finalize()");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto index_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - copy_time).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "      Index build in " << index_ms << " ms\n";
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total atoms: " << total << "\n";
    std::cerr << "Total time: " << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n";
    std::cerr << "Rate: " << (total * 1000 / std::max(static_cast<long long>(total_ms), 1LL)) << " atoms/sec\n";
    
    PQfinish(main_conn);
    return 0;
}
