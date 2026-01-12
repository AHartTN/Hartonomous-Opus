/**
 * Parallel Partitioned Atom Seeder - Maximum Performance
 *
 * Strategy:
 * 1. Generate all atoms in parallel (C++ threads)
 * 2. Partition by blake3 hash prefix (12 partitions)
 * 3. COPY directly to atom table in parallel (12 connections)
 * 4. Rebuild indexes in parallel where possible
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

#include "hypercube_c.h"
#include "hypercube/types.hpp"
#include "hypercube/db/connection.hpp"
#include "hypercube/cpu_features.hpp"

using namespace hypercube;

// Configuration
static constexpr int NUM_PARTITIONS = 8;
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
    uint64_t hilbert_lo, hilbert_hi;
    
    // Partition based on first byte of hash (0-255, mapped to 0-11)
    int partition() const {
        return hash.bytes[0] % NUM_PARTITIONS;
    }
};

// Generate atoms for a codepoint range
void generate_range(uint32_t start, uint32_t end, std::vector<AtomRecord>& out) {
    uint32_t range_size = end - start;
    out.reserve(out.size() + range_size);

    // Check CPU features for SIMD optimization info
    bool has_avx2 = cpu_features::has_feature(cpu_features::Feature::AVX2);
    bool has_avx512 = cpu_features::has_feature(cpu_features::Feature::AVX512F);

    // Use batch processing for better SIMD utilization
    // Process in chunks that fit SIMD registers efficiently
    const size_t BATCH_SIZE = has_avx512 ? 512 : (has_avx2 ? 256 : 64);

    std::vector<uint32_t> codepoints;
    codepoints.reserve(BATCH_SIZE);

    for (uint32_t cp = start; cp < end; ) {
        // Collect batch of codepoints
        codepoints.clear();
        uint32_t batch_end = std::min(end, static_cast<uint32_t>(cp + BATCH_SIZE));
        for (uint32_t c = cp; c < batch_end; ++c) {
            codepoints.push_back(c);
        }

        size_t batch_count = codepoints.size();
        if (batch_count == 0) break;

        // Use batch API functions for SIMD acceleration
        std::vector<hc_point4d_t> coords(batch_count);
        std::vector<hc_hash_t> hashes(batch_count);
        std::vector<hc_hilbert_t> hilberts(batch_count);

        // Batch coordinate mapping
        hc_map_codepoints_batch(codepoints.data(), batch_count, coords.data());

        // Batch hashing
        hc_hash_codepoints_batch(codepoints.data(), batch_count, hashes.data());

        // Batch Hilbert computation
        hc_coords_to_hilbert_batch(coords.data(), batch_count, hilberts.data());

        // Convert to AtomRecord structures
        for (size_t i = 0; i < batch_count; ++i) {
            uint32_t current_cp = codepoints[i];
            hc_category_t category = hc_categorize(current_cp);

            AtomRecord rec;
            std::memcpy(rec.hash.bytes.data(), hashes[i].bytes, 32);
            rec.codepoint = static_cast<int32_t>(current_cp);
            rec.category = static_cast<AtomCategory>(category);

            // Store uint32 coordinates as int32 (bit-preserving cast)
            // This preserves the full 32-bit value - no information loss
            rec.coord_x = coords[i].x;
            rec.coord_y = coords[i].y;
            rec.coord_z = coords[i].z;
            rec.coord_m = coords[i].m;

            // Store as double for PostGIS - DIRECT from uint32, no int32 reinterpretation
            // This ensures CENTER (2^31) is stored as 2147483648.0, not as negative
            rec.x = static_cast<double>(coords[i].x);
            rec.y = static_cast<double>(coords[i].y);
            rec.z = static_cast<double>(coords[i].z);
            rec.m = static_cast<double>(coords[i].m);

            rec.hilbert_lo = hilberts[i].lo;
            rec.hilbert_hi = hilberts[i].hi;

            out.push_back(rec);
        }

        cp = batch_end;
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

    std::cerr << "Partition sizes: ";
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        std::cerr << partitions[i].size() << " ";
    }
    std::cerr << "\n";
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

// COPY a partition directly to atom table
bool copy_partition(hypercube::db::ConnectionPool& pool, int partition_id,
                    const std::vector<AtomRecord>& atoms) {
    auto copy_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<hypercube::db::Connection> conn_handle;
    try {
        conn_handle = pool.acquire();
    } catch (const std::runtime_error& e) {
        std::cerr << "Partition " << partition_id << " connection failed: " << e.what() << std::endl;
        return false;
    }
    PGconn* conn = conn_handle->get();
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Partition " << partition_id << " connection failed: "
                  << PQerrorMessage(conn) << std::endl;
        return false;
    }

    // COPY directly to atom table
    // Schema: id, codepoint, value, geom, hilbert_lo, hilbert_hi
    std::string copy_cmd = "COPY atom (id, codepoint, value, geom, hilbert_lo, hilbert_hi) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";

    PGresult* res = PQexec(conn, copy_cmd.c_str());
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "Partition " << partition_id << " COPY start failed: "
                  << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Build batch buffer - 8MB for better throughput
    std::string batch;
    batch.reserve(8 << 20);  // 8MB

    static const char hex_chars[] = "0123456789abcdef";
    uint8_t ewkb[37];  // 5 header + 4*8 doubles = 37 bytes
    // WKB header for POINTZM little-endian: byte order 0x01, type 0x00000BB9 (3001)
    uint8_t header[5] = {0x01, 0xB9, 0x0B, 0x00, 0x00};
    std::memcpy(ewkb, header, 5);

    char num_buf[32];

    auto build_start = std::chrono::high_resolution_clock::now();
    for (const auto& a : atoms) {
        // Build EWKB geometry (POINTZM)
        std::memcpy(ewkb + 5, &a.x, sizeof(double));
        std::memcpy(ewkb + 13, &a.y, sizeof(double));
        std::memcpy(ewkb + 21, &a.z, sizeof(double));
        std::memcpy(ewkb + 29, &a.m, sizeof(double));

        // id (BYTEA)
        batch += "\\\\x";
        batch += a.hash.to_hex();
        batch += '\t';

        // codepoint
        snprintf(num_buf, sizeof(num_buf), "%d", a.codepoint);
        batch += num_buf;
        batch += '\t';

        // value (UTF-8 bytes)
        batch += encode_utf8_value(static_cast<uint32_t>(a.codepoint));
        batch += '\t';

        // geom (POINTZM as hex-encoded WKB)
        for (int i = 0; i < 37; ++i) {
            uint8_t b = ewkb[i];
            batch += hex_chars[b >> 4];
            batch += hex_chars[b & 0x0F];
        }
        batch += '\t';

        // hilbert_lo
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(a.hilbert_lo));
        batch += num_buf;
        batch += '\t';

        // hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(a.hilbert_hi));
        batch += num_buf;
        batch += '\n';

        // Send when buffer full (4MB threshold)
        if (batch.size() > (4 << 20)) {
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "Partition " << partition_id << " COPY data failed\n";
                PQputCopyEnd(conn, "error");
                return false;
            }
            batch.clear();
        }
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cerr << "Partition " << partition_id << " build time: " << build_ms << " ms\n";

    // Send remaining
    if (!batch.empty()) {
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "Partition " << partition_id << " COPY final failed\n";
            PQputCopyEnd(conn, "error");
            return false;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "Partition " << partition_id << " COPY end failed\n";
        return false;
    }

    PGresult* final_res;
    bool success = true;
    while ((final_res = PQgetResult(conn)) != nullptr) {
        if (PQresultStatus(final_res) != PGRES_COMMAND_OK) {
            std::cerr << "Partition " << partition_id << " result: "
                      << PQerrorMessage(conn) << std::endl;
            success = false;
        }
        PQclear(final_res);
    }
    // PooledConnection automatically returns connection to pool

    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start).count();
    auto io_ms = copy_ms - build_ms;
    std::cerr << "Partition " << partition_id << " total time: " << copy_ms << " ms, I/O: " << io_ms << " ms, atoms: " << atoms.size() << ", rate: " << (atoms.size() * 1000LL / (copy_ms > 0 ? copy_ms : 1)) << " atoms/sec\n";

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
    
    hypercube::db::ConnectionConfig config;
    config.dbname = dbname;
    config.host = host;
    config.port = port;
    config.user = user;

    // Create connection pool with NUM_PARTITIONS connections
    hypercube::db::ConnectionPool pool(config, NUM_PARTITIONS + 1);  // +1 for main connection

    std::string conninfo = config.to_conninfo();
    std::cerr << "=== Parallel Partitioned Atom Seeder ===\n";
    std::cerr << "Connection: " << conninfo << "\n";
    std::cerr << "Partitions: " << NUM_PARTITIONS << "\n";
    std::cerr << "Generators: " << NUM_GENERATORS << "\n";
    std::cerr << "Connection Pool: " << NUM_PARTITIONS << " connections\n";

    // CPU feature detection and SIMD optimization info
    std::cerr << "CPU Features: " << cpu_features::get_cpu_info();
    bool has_avx2 = cpu_features::has_feature(cpu_features::Feature::AVX2);
    bool has_avx512 = cpu_features::has_feature(cpu_features::Feature::AVX512F);
    bool has_avx_vnni = cpu_features::has_feature(cpu_features::Feature::AVX_VNNI);

    if (has_avx512) {
        std::cerr << "SIMD: AVX-512 enabled (512-codepoint batch processing)\n";
    } else if (has_avx2) {
        std::cerr << "SIMD: AVX2 enabled (256-codepoint batch processing)\n";
    } else {
        std::cerr << "SIMD: Scalar processing (64-codepoint batch processing)\n";
    }

    if (has_avx_vnni) {
        std::cerr << "VNNI: AVX-VNNI available for enhanced integer operations\n";
    }
    std::cerr << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === STEP 1: Initialize coordinate mapping (single-threaded) ===
    std::cerr << "[1/5] Initializing coordinate mapping...\n";
    // Force initialization of static data structures before parallel execution
    // This prevents race conditions in DenseRegistry and other static initialization
    hc_map_codepoint(0);  // Initialize coordinate mapping system

    // === STEP 1.5: Generate atoms in parallel ===
    std::cerr << "[1.5/5] Generating atoms (" << NUM_GENERATORS << " threads)...\n";
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
    std::unique_ptr<hypercube::db::Connection> main_conn_handle;
    try {
        main_conn_handle = pool.acquire();
    } catch (const std::runtime_error& e) {
        std::cerr << "Main connection failed: " << e.what() << std::endl;
        return 1;
    }
    PGconn* main_conn = main_conn_handle->get();
    if (PQstatus(main_conn) != CONNECTION_OK) {
        std::cerr << "Main connection failed: " << PQerrorMessage(main_conn) << std::endl;
        return 1;
    }

    // Use stored procedure for database setup
    PQexec(main_conn, "CALL seed_atoms_setup()");

    auto setup_time = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - part_time).count();
    std::cerr << "      Setup in " << setup_ms << " ms\n";
    
    // === STEP 4: Parallel COPY ===
    std::cerr << "[4/5] Parallel COPY to atom table (" << NUM_PARTITIONS << " connections)...\n";

    std::vector<std::thread> copy_threads;
    std::vector<bool> results(NUM_PARTITIONS, false);
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        copy_threads.emplace_back([i, &pool, &partitions, &results]() {
            results[i] = copy_partition(pool, i, partitions[i]);
        });
    }

    for (auto& t : copy_threads) {
        t.join();
    }

    bool all_success = true;
    for (bool r : results) {
        if (!r) all_success = false;
    }

    if (!all_success) {
        std::cerr << "Some partitions failed!\n";
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

    // PooledConnection automatically returns connection to pool
    return 0;
}
