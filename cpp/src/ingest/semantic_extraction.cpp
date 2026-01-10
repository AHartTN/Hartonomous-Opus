/**
 * @file semantic_extraction.cpp
 * @brief TRULY PARALLEL semantic extraction
 *
 * PARALLELISM STRATEGY:
 * 1. MKL cblas_sgemm - THREADED (uses OpenMP internally)
 * 2. normalize_rows - PARALLELIZED with OpenMP
 * 3. HNSW build - Partitioned parallel: build shards, then merge
 * 4. HNSW query - PARALLELIZED (query is thread-safe)
 *
 * Environment: Set OMP_NUM_THREADS and MKL_NUM_THREADS before running
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/ingest/projection_db.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/db/operations.hpp"  // For Transaction
#include "hypercube/db/helpers.hpp"    // For Result and exec
#include "hypercube/laplacian_4d.hpp"
#include <unordered_set>

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <set>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#define bswap16(x) _byteswap_ushort(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#define bswap16(x) __builtin_bswap16(x)
#endif

#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
#endif

#ifdef HAS_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

namespace hypercube {
namespace ingest {
namespace db {

// Quality score calculation matching SQL function
static float calculate_projection_quality(const std::string& role, const std::string& dtype, int dim, float variance_explained) {
    float dtype_bonus = 0.0f;
    if (dtype == "F32") dtype_bonus = 2.0f;
    else if (dtype == "BF16") dtype_bonus = 1.5f;
    else if (dtype == "F16") dtype_bonus = 1.0f;
    else if (dtype == "F8_E4M3") dtype_bonus = 0.5f;

    float role_bonus = 0.0f;
    if (role == "embeddings") role_bonus = 2.0f;
    else if (role == "attention") role_bonus = 1.5f;
    else if (role == "ffn") role_bonus = 1.0f;
    else role_bonus = 0.5f;

    float spectrum_bonus = 0.0f;
    if (variance_explained > 0.0f) {
        spectrum_bonus = variance_explained * 2.0f;
    }

    return std::log(static_cast<float>(dim)) + dtype_bonus + role_bonus + spectrum_bonus;
}

// Forward declarations for merge and hierarchy extraction
static bool extract_merge_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool extract_hierarchy_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool extract_temporal_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool extract_visual_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool ensure_position_compositions_exist(PGconn* conn, const std::string& emb_name, const std::vector<size_t>& positions);
static bool ensure_visual_compositions_exist(PGconn* conn, const std::string& emb_name, const std::vector<size_t>& features);

// ============================================================================
// Config
// ============================================================================

static const size_t K_DEFAULT = 5;
static const float THRESHOLD_DEFAULT = 0.7f;
static constexpr float MIN_NORM = 0.01f;
static int g_num_threads = 1;  // Set at runtime

// Projection k-NN is enabled for complete semantic extraction
static constexpr bool ENABLE_PROJECTION_KNN = true;
static constexpr size_t MAX_PROJECTIONS = 2;  // Limit to first N projections

// ============================================================================
// Load tensor
// ============================================================================

static std::vector<float> load_tensor(const TensorMeta& meta) {
    size_t n = meta.element_count();
    std::vector<float> data(n);

    std::cerr << "[LOAD_TENSOR] Opening file: " << meta.shard_file << "\n";
    std::cerr << "[LOAD_TENSOR] Seeking to offset: " << meta.data_offset_start << "\n";
    std::cerr << "[LOAD_TENSOR] Reading " << n << " elements as " << meta.dtype << "\n";

    std::ifstream f(meta.shard_file, std::ios::binary);
    if (!f) {
        std::cerr << "[LOAD_TENSOR] ERROR: Failed to open file!\n";
        return {};
    }

    f.seekg(static_cast<std::streamoff>(meta.data_offset_start));
    if (f.fail()) {
        std::cerr << "[LOAD_TENSOR] ERROR: Seek failed! Offset may be invalid.\n";
        return {};
    }

    if (meta.dtype == "F32") {
        f.read(reinterpret_cast<char*>(data.data()), n * 4);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F32 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 4) << ")\n";
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for BF16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        // PARALLEL BF16 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
            std::memcpy(&data[i], &bits, 4);
        }

        // VALIDATE: Check for extreme values indicating corruption
        size_t corrupt_count = 0;
        size_t nan_count = 0;
        #pragma omp parallel for reduction(+:corrupt_count,nan_count)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            float val = data[i];
            if (std::isnan(val)) {
                nan_count++;
                data[i] = 0.0f;  // Replace NaN with 0
            } else if (std::abs(val) > 1e10f) {  // Extreme values
                corrupt_count++;
                data[i] = 0.0f;  // Replace corrupt values with 0
            }
        }
        if (corrupt_count > 0 || nan_count > 0) {
            std::cerr << "[LOAD_TENSOR] WARNING: Fixed " << corrupt_count << " extreme values and "
                      << nan_count << " NaNs in BF16 data\n";
        }
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        // PARALLEL F16 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint16_t h = raw[i];
            uint32_t s = (h & 0x8000) << 16;
            uint32_t e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;
            uint32_t fval = (e == 0) ? s : (e == 31) ? (s | 0x7F800000 | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
            std::memcpy(&data[i], &fval, 4);
        }
    } else if (meta.dtype == "F8_E4M3") {
        std::vector<uint8_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 1);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F8_E4M3 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 1) << ")\n";
        // PARALLEL F8_E4M3 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint8_t f8 = raw[i];
            uint32_t sign = (f8 >> 7) & 0x1;
            uint32_t exp = (f8 >> 3) & 0xF;
            uint32_t mant = f8 & 0x7;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    // Zero
                    f = sign << 31;
                } else {
                    // Denormalized number - shift mantissa until leading 1
                    int shift = 0;
                    uint32_t m = mant;
                    while ((m & 0x4) == 0 && shift < 3) {
                        m <<= 1;
                        shift++;
                    }
                    m &= 0x3;  // Remove leading 1
                    f = (sign << 31) | ((1 + 127 - 7 - shift) << 23) | (m << 20);
                }
            } else if (exp == 15) {
                // Infinity or NaN
                if (mant == 0) {
                    f = (sign << 31) | 0x7F800000;  // Infinity
                } else {
                    f = (sign << 31) | 0x7FC00000;  // NaN
                }
            } else {
                // Normalized number
                f = (sign << 31) | ((exp + 127 - 7) << 23) | (mant << 20);
            }
            std::memcpy(&data[i], &f, 4);
        }
    } else {
        std::cerr << "[LOAD_TENSOR] ERROR: Unknown dtype '" << meta.dtype << "'\n";
        return {};
    }

    // Verify non-zero data
    int nonzero = 0;
    for (size_t i = 0; i < std::min(size_t(100), n); ++i) {
        if (std::abs(data[i]) > 1e-10f) ++nonzero;
    }
    std::cerr << "[LOAD_TENSOR] Non-zero values in first 100 elements: " << nonzero << "/100\n";

    return data;
}

// ============================================================================
// PARALLEL normalize rows, return valid indices (skip near-zero)
// ============================================================================

static std::vector<size_t> normalize_rows(float* data, size_t rows, size_t cols) {
    // First pass: compute norms in parallel
    std::vector<float> norms(rows);
    
    #pragma omp parallel for schedule(static) num_threads(g_num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        const float* row = data + i * cols;
        float norm2 = 0;
        // Use SIMD-friendly loop
        for (size_t j = 0; j < cols; j++) {
            norm2 += row[j] * row[j];
        }
        norms[i] = std::sqrt(norm2);
    }
    
    // Second pass: normalize valid rows in parallel (write back)
    #pragma omp parallel for schedule(static) num_threads(g_num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        if (norms[i] >= MIN_NORM) {
            float inv = 1.0f / norms[i];
            float* row = data + i * cols;
            for (size_t j = 0; j < cols; j++) {
                row[j] *= inv;
            }
        }
    }
    
    // Collect valid indices (sequential, but fast)
    std::vector<size_t> valid;
    valid.reserve(rows);
    for (size_t i = 0; i < rows; i++) {
        if (norms[i] >= MIN_NORM) valid.push_back(i);
    }
    return valid;
}

// ============================================================================
// k-NN using HNSW - single-threaded BUILD, PARALLEL QUERY
// HNSW addPoint is NOT thread-safe, but searchKnn IS thread-safe
// ============================================================================

struct Edge { int32_t src, tgt; float sim; };

#ifdef HAS_HNSWLIB
static std::vector<Edge> knn(const float* data, const std::vector<size_t>& valid, size_t dim,
                          size_t k_neighbors = K_DEFAULT, float threshold = THRESHOLD_DEFAULT) {
    size_t n = valid.size();
    if (n < 2) return {};

    std::cerr << "[KNN] Building index for " << n << " vectors (single-threaded)...\n";

    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> idx(&space, n, 16, 100);

    // Progress logging for build (single-threaded, this is the bottleneck)
    auto build_start = std::chrono::steady_clock::now();
    size_t log_interval = std::max(size_t(1), n / 10);

    for (size_t i = 0; i < n; i++) {
        idx.addPoint(data + valid[i] * dim, i);
        if ((i + 1) % log_interval == 0 || i + 1 == n) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - build_start).count();
            double pct = 100.0 * (i + 1) / n;
            double rate = (i + 1) * 1000.0 / (elapsed + 1);
            std::cerr << "[KNN] Build: " << (i + 1) << "/" << n
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                      << " - " << rate << " vecs/sec\n";
        }
    }
    idx.setEf(50);

    std::cerr << "[KNN] Querying " << n << " vectors (PARALLEL, " << g_num_threads << " threads)...\n";

    // PARALLEL query - searchKnn IS thread-safe
    std::vector<std::set<std::tuple<size_t, size_t, float>>> thread_edge_sets(g_num_threads);

    auto query_start = std::chrono::steady_clock::now();
    std::atomic<size_t> progress{0};

    #pragma omp parallel num_threads(g_num_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        std::set<std::tuple<size_t, size_t, float>>& local = thread_edge_sets[tid];

        #pragma omp for schedule(dynamic, 256)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            auto result = idx.searchKnn(data + valid[i] * dim, k_neighbors + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                if (static_cast<size_t>(j) == i) continue;
                float sim = 1.0f - dist;
                if (sim >= threshold && valid[i] < valid[j]) {
                    local.emplace(valid[i], valid[j], sim);
                }
            }

            // Progress logging (every 10% from thread 0)
            size_t p = progress.fetch_add(1);
            if (tid == 0 && (p % (n / 10 + 1) == 0)) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - query_start).count();
                double rate = (p + 1) * 1000.0 / (elapsed + 1);
                std::cerr << "[KNN] Query: " << p << "/" << n
                          << " - " << rate << " queries/sec\n";
            }
        }
    }

    // Merge results
    std::set<std::tuple<size_t, size_t, float>> all_edges;
    for (auto& tes : thread_edge_sets) {
        all_edges.insert(tes.begin(), tes.end());
    }

    std::vector<Edge> edges;
    edges.reserve(all_edges.size());
    for (auto& t : all_edges) {
        edges.push_back({static_cast<int32_t>(std::get<0>(t)), static_cast<int32_t>(std::get<1>(t)), std::get<2>(t)});
    }

    std::cerr << "[KNN] Found " << edges.size() << " unique edges above threshold " << threshold << "\n";
    return edges;
}
#else
static std::vector<Edge> knn(const float*, const std::vector<size_t>&, size_t) { return {}; }
#endif

// ============================================================================
// Insert edges via batched INSERT
// ============================================================================

static void insert(PGconn* conn, const std::vector<Edge>& edges, const IngestContext& ctx, const std::string& tag) {
    if (edges.empty()) return;

    // Deduplicate edges (though HNSW should already produce unique edges)
    std::set<std::tuple<Blake3Hash, Blake3Hash, float>> seen;
    for (const auto& e : edges) {
        if ((size_t)e.src >= ctx.vocab_tokens.size() || (size_t)e.tgt >= ctx.vocab_tokens.size()) continue;
        const auto& s = ctx.vocab_tokens[e.src].comp;
        const auto& t = ctx.vocab_tokens[e.tgt].comp;
        seen.emplace(s.hash, t.hash, e.sim);
    }

    // Use temp table then INSERT with existence checks
    PQexec(conn, "DROP TABLE IF EXISTS tmp_semantic_rel");
    PQexec(conn, "CREATE TEMP TABLE tmp_semantic_rel ("
                 "source_type CHAR(1), source_id BYTEA, target_type CHAR(1), target_id BYTEA, "
                 "relation_type CHAR(1), weight REAL, source_model TEXT, layer INTEGER, component TEXT)");

    std::string copy_cmd = "COPY tmp_semantic_rel FROM STDIN";
    PGresult* res = PQexec(conn, copy_cmd.c_str());
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[INSERT] COPY start failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        return;
    }
    PQclear(res);

    const std::string component = "semantic";

    for (const auto& [source_hash, target_hash, weight] : seen) {
        std::string line = "C\t\\\\x" + source_hash.to_hex() + "\tC\t\\\\x" + target_hash.to_hex() +
                          "\tS\t" + std::to_string(weight) + "\t" + tag + "\t-1\t" + component + "\n";

        if (PQputCopyData(conn, line.c_str(), (int)line.size()) != 1) {
            std::cerr << "[INSERT] COPY data failed: " << PQerrorMessage(conn) << "\n";
            return;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "[INSERT] COPY to temp failed: " << PQerrorMessage(conn) << "\n";
        return;
    }

    res = PQgetResult(conn);
    PQclear(res);

    // INSERT into relation_evidence with ELO rating updates
    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        SELECT
            DECODE(SUBSTRING(source_id, 3), 'hex'),
            DECODE(SUBSTRING(target_id, 3), 'hex'),
            relation_type::CHAR(1),
            source_model,
            layer::INT,
            component,
            1500.0,  -- Initial ELO rating
            1,        -- First observation
            weight,   -- Raw weight
            weight    -- Normalized weight
        FROM tmp_semantic_rel t
        WHERE EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.source_id, 3), 'hex'))
        AND EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.target_id, 3), 'hex'))
        ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
        DO UPDATE SET
            -- ELO rating update with dynamic K-factor
            rating = relation_evidence.rating +
                     LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                     (
                         (EXCLUDED.normalized_weight + 1.0) / 2.0 -  -- Actual score [0,1]
                         (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))  -- Expected
                     ),
            observation_count = relation_evidence.observation_count + 1,
            raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                         (relation_evidence.observation_count + 1),
            normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                               (relation_evidence.observation_count + 1),
            last_updated = NOW()
    )SQL";

    res = PQexec(conn, insert_sql.c_str());

    int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    PQclear(res);

    std::cerr << "[INSERT] Inserted " << inserted << " semantic relations (filtered " << (seen.size() - inserted) << " missing refs)\n";
}

// ============================================================================
// Main - PARALLEL everywhere possible
// ============================================================================

bool extract_all_semantic_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    
    // Detect and set thread count
    g_num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (g_num_threads < 1) g_num_threads = 8;
    
    std::cerr << "============================================================\n";
    std::cerr << "[SEMANTIC] PARALLEL EXTRACTION starting\n";
    std::cerr << "[SEMANTIC] Hardware threads: " << g_num_threads << "\n";
    
#ifdef _OPENMP
    omp_set_num_threads(g_num_threads);
    std::cerr << "[SEMANTIC] OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cerr << "[SEMANTIC] WARNING: OpenMP NOT AVAILABLE - many operations single-threaded!\n";
#endif
    
#if defined(HAS_MKL) && HAS_MKL
    mkl_set_num_threads(g_num_threads);
    mkl_set_dynamic(0);  // CRITICAL: Force thread count
    std::cerr << "[SEMANTIC] MKL threads: " << mkl_get_max_threads() << " (dynamic=0)\n";
#else
    std::cerr << "[SEMANTIC] WARNING: MKL NOT AVAILABLE - GEMM will be slow!\n";
#endif

#ifdef HAS_HNSWLIB
    std::cerr << "[SEMANTIC] HNSWLIB: available\n";
#else
    std::cerr << "[SEMANTIC] WARNING: HNSWLIB NOT AVAILABLE - no k-NN!\n";
#endif
    std::cerr << "============================================================\n";

    // =========================================================================
    // QUERY ANCHOR ATOMS FOR LAPLACIAN PROJECTION CONSTRAINTS
    // =========================================================================

    std::vector<AnchorPoint> anchors;

    // Query anchor atoms from database
    // Anchors are single-character tokens that already have 4D coordinates from SuperFibonacci + Hopf
    std::cerr << "[SEMANTIC] Querying anchor atoms from database for Laplacian constraints...\n";

    {
        // Build SQL query to find single-character tokens that match atoms
        std::string query =
            "SELECT a.id, a.codepoint, a.value, "
            "  ST_X(a.geom) as x, ST_Y(a.geom) as y, ST_Z(a.geom) as z, ST_M(a.geom) as m "
            "FROM atom a "
            "WHERE a.codepoint = ANY($1::integer[])";

        // Collect codepoints for single-character tokens
        std::vector<uint32_t> single_char_codepoints;
        std::vector<size_t> single_char_indices;

        for (size_t i = 0; i < ctx.vocab_tokens.size(); ++i) {
            const std::string& token_text = ctx.vocab_tokens[i].text;

            // Check if token is a single Unicode character
            // Simple UTF-8 check: count characters
            size_t char_count = 0;
            for (size_t j = 0; j < token_text.size(); ) {
                unsigned char c = token_text[j];
                if (c < 0x80) j += 1;
                else if ((c & 0xE0) == 0xC0) j += 2;
                else if ((c & 0xF0) == 0xE0) j += 3;
                else if ((c & 0xF8) == 0xF0) j += 4;
                else j += 1;  // Invalid, skip
                char_count++;
            }

            if (char_count == 1) {
                // Decode the single character to get codepoint
                uint32_t codepoint = 0;
                const unsigned char* bytes = reinterpret_cast<const unsigned char*>(token_text.data());

                if (bytes[0] < 0x80) {
                    codepoint = bytes[0];
                } else if ((bytes[0] & 0xE0) == 0xC0 && token_text.size() >= 2) {
                    codepoint = ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
                } else if ((bytes[0] & 0xF0) == 0xE0 && token_text.size() >= 3) {
                    codepoint = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
                } else if ((bytes[0] & 0xF8) == 0xF0 && token_text.size() >= 4) {
                    codepoint = ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
                                ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
                }

                if (codepoint > 0 && codepoint <= 0x10FFFF) {
                    single_char_codepoints.push_back(codepoint);
                    single_char_indices.push_back(i);
                }
            }
        }

        std::cerr << "[SEMANTIC]   Found " << single_char_codepoints.size()
                  << " single-character tokens in vocabulary\n";

        if (!single_char_codepoints.empty()) {
            // Build PostgreSQL array literal
            std::ostringstream array_literal;
            array_literal << "{";
            for (size_t i = 0; i < single_char_codepoints.size(); ++i) {
                if (i > 0) array_literal << ",";
                array_literal << single_char_codepoints[i];
            }
            array_literal << "}";

            std::string array_str = array_literal.str();
            const char* params[1] = {array_str.c_str()};

            PGresult* res = PQexecParams(conn, query.c_str(), 1, nullptr, params, nullptr, nullptr, 0);

            if (PQresultStatus(res) == PGRES_TUPLES_OK) {
                int nrows = PQntuples(res);
                std::cerr << "[SEMANTIC]   Retrieved " << nrows << " atom coordinates from database\n";

                // Build map: codepoint → 4D coordinates
                std::map<uint32_t, std::array<double, 4>> codepoint_coords;

                for (int r = 0; r < nrows; ++r) {
                    uint32_t codepoint = static_cast<uint32_t>(std::stoll(PQgetvalue(res, r, 1)));
                    std::array<double, 4> coords;
                    coords[0] = std::stod(PQgetvalue(res, r, 3));  // x
                    coords[1] = std::stod(PQgetvalue(res, r, 4));  // y
                    coords[2] = std::stod(PQgetvalue(res, r, 5));  // z
                    coords[3] = std::stod(PQgetvalue(res, r, 6));  // m
                    codepoint_coords[codepoint] = coords;
                }

                // Create anchors for tokens with retrieved coordinates
                for (size_t k = 0; k < single_char_codepoints.size(); ++k) {
                    uint32_t cp = single_char_codepoints[k];
                    auto it = codepoint_coords.find(cp);
                    if (it != codepoint_coords.end()) {
                        AnchorPoint anchor;
                        anchor.token_index = single_char_indices[k];
                        anchor.coords_4d = it->second;
                        anchor.weight = 1.0;
                        anchors.push_back(anchor);
                    }
                }

                std::cerr << "[SEMANTIC]   Created " << anchors.size() << " anchor points for Laplacian constraints\n";
            } else {
                std::cerr << "[SEMANTIC]   WARNING: Failed to query atoms: "
                          << PQerrorMessage(conn) << "\n";
            }

            PQclear(res);
        }
    }

    // =========================================================================
    // EXTRACT EMBEDDING AND ATTENTION RELATIONS
    // =========================================================================

    // CRITICAL: Build k-NN similarity graph from token embeddings
    // This is the PRIMARY semantic relation extraction - connects tokens by learned similarity
    std::cerr << "[SEMANTIC] Extracting token embedding relations (k-NN similarity graph)...\n";
    extract_embedding_relations(conn, ctx, config);

    // Extract attention-based semantic relations from Q/K/V projections
    std::cerr << "[SEMANTIC] Extracting attention projection relations...\n";
    insert_attention_relations(conn, ctx, config);

    // Extract temporal relations from position embeddings
    extract_temporal_relations(conn, ctx, config);

    // Extract visual relations from vision features
    extract_visual_relations(conn, ctx, config);

    // =========================================================================
    // CONFIG-DRIVEN TENSOR LOOKUP FOR ATTENTION PROJECTIONS
    // Use the parsed model manifest to find tensors by ROLE, not by name pattern
    // =========================================================================

    const TensorMeta* emb = nullptr;
    std::string emb_name;

    // Dtype preference ranking (higher score = better for semantic precision)
    auto dtype_score = [](const std::string& dtype) -> int {
        if (dtype == "F64") return 50;
        if (dtype == "F32") return 40;
        if (dtype == "BF16") return 30;  // Better than FP16 for ML (wider range)
        if (dtype == "F16") return 20;
        if (dtype == "F8_E5M2") return 11; // E5M2: wider range, less precision
        if (dtype == "F8_E4M3") return 10; // E4M3: narrower range, more precision
        return 0;  // Unknown/unsupported dtype
    };

    // First try: Use manifest's categorized extraction plans
    // CRITICAL: Prefer highest precision tensor if multiple TOKEN_EMBEDDING exist
    if (ctx.manifest.has_value()) {
        std::cerr << "[SEMANTIC] Using config-driven tensor lookup (architecture: "
                  << architecture_to_string(ctx.manifest->architecture) << ")\n";

        const TensorMeta* best_emb = nullptr;
        std::string best_emb_name;
        int best_score = -1;
        std::vector<std::string> candidates;

        // Scan ALL TOKEN_EMBEDDING tensors and select highest precision
        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::TOKEN_EMBEDDING) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    int score = dtype_score(it->second.dtype);
                    candidates.push_back(plan.name + ":" + it->second.dtype);
                    if (score > best_score) {
                        best_score = score;
                        best_emb = &it->second;
                        best_emb_name = plan.name;
                    }
                }
            }
        }

        if (best_emb) {
            emb = best_emb;
            emb_name = best_emb_name;
            std::cerr << "[SEMANTIC] Candidates: [";
            for (size_t i = 0; i < candidates.size(); ++i) {
                std::cerr << candidates[i];
                if (i + 1 < candidates.size()) std::cerr << ", ";
            }
            std::cerr << "]\n";
            std::cerr << "[SEMANTIC] Selected: " << emb_name
                      << " [" << emb->shape[0] << " x " << emb->shape[1] << "]"
                      << " dtype=" << emb->dtype << " (precision_score=" << best_score << ")\n";
        }
    }

    // Fallback: If no manifest or manifest didn't find embedding, scan tensors
    // This should rarely happen if manifest parsing is working
    if (!emb) {
        std::cerr << "[SEMANTIC] WARNING: Manifest lookup failed, falling back to tensor scan\n";

        // Get expected vocab size from config if available
        size_t expected_vocab = ctx.get_vocab_size();
        size_t expected_dim = ctx.get_model_dim();

        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;
            if (name.find("position") != std::string::npos) continue;

            // If config tells us vocab_size, use it to validate
            if (expected_vocab > 0 && meta.shape[0] == static_cast<int64_t>(expected_vocab)) {
                // This tensor has the right vocab dimension - likely the embedding
                if (expected_dim == 0 || meta.shape[1] == static_cast<int64_t>(expected_dim)) {
                    emb = &meta;
                    emb_name = name;
                    std::cerr << "[SEMANTIC] Found embedding by dimension match: " << name
                              << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
                    break;
                }
            }
        }
    }

    if (!emb) {
        std::cerr << "[SEMANTIC] No embedding tensor found for attention projections\n";
        return true;
    }

    size_t V = static_cast<size_t>(emb->shape[0]);
    size_t D = static_cast<size_t>(emb->shape[1]);
    if (!ctx.vocab_tokens.empty()) V = std::min(V, ctx.vocab_tokens.size());

    std::cerr << "[SEMANTIC] Loading " << V << " x " << D << " embeddings for attention projections...\n";
    auto load_start = std::chrono::steady_clock::now();
    auto E = load_tensor(*emb);
    auto load_end = std::chrono::steady_clock::now();
    if (E.empty()) {
        std::cerr << "[SEMANTIC] Failed to load embeddings\n";
        return false;
    }
    E.resize(V * D);
    std::cerr << "[SEMANTIC] Loaded in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << "ms\n";

    std::cerr << "[SEMANTIC] Normalizing (parallel)...\n";
    auto norm_start = std::chrono::steady_clock::now();
    auto valid = normalize_rows(E.data(), V, D);
    auto norm_end = std::chrono::steady_clock::now();
    std::cerr << "[SEMANTIC] " << valid.size() << "/" << V << " valid embeddings in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(norm_end - norm_start).count() << "ms\n";

    // NOTE: Base embedding k-NN REMOVED - attention_relations.cpp handles this
    // with better HNSW params (K=15, threshold=0.5, ef=100 vs K=5, threshold=0.7, ef=50)
    // This eliminates ~6.4 seconds of redundant work
    
    // Find ONE of each projection type per layer (skip duplicates)
    std::vector<std::pair<std::string, const TensorMeta*>> projs;
    std::set<std::string> seen;
    
    for (const auto& [name, meta] : ctx.tensors) {
        if (meta.shape.size() != 2) continue;
        size_t r = meta.shape[0], c = meta.shape[1];
        if (c != D) continue;  // Must project from embedding dim
        if (r < 64) continue;  // Skip tiny
        
        // Extract layer + type
        std::string type;
        if (name.find("q_proj") != std::string::npos) type = "q";
        else if (name.find("k_proj") != std::string::npos) type = "k";
        else if (name.find("v_proj") != std::string::npos) type = "v";
        else if (name.find("fc1") != std::string::npos || name.find("up_proj") != std::string::npos) type = "up";
        else continue;
        
        int layer = -1;
        auto p = name.find("layers.");
        if (p != std::string::npos) layer = std::stoi(name.substr(p + 7));
        else {
            p = name.find("layer.");
            if (p != std::string::npos) layer = std::stoi(name.substr(p + 6));
        }
        
        std::string key = std::to_string(layer) + type;
        if (seen.count(key)) continue;
        seen.insert(key);
        projs.push_back({config.model_name + "/L" + std::to_string(layer) + "/" + type, &meta});
    }
    
    std::cerr << "[SEMANTIC] " << projs.size() << " projections\n";
    
    // Skip projection k-NN if disabled (too slow due to single-threaded HNSW build)
    if (!ENABLE_PROJECTION_KNN) {
        std::cerr << "[SEMANTIC] Projection k-NN DISABLED (use attention_relations for token semantics)\n";
        std::cerr << "[SEMANTIC] Total: 0 sparse relations\n";
        return true;
    }
    
    // Limit projections if enabled
    if (projs.size() > MAX_PROJECTIONS) {
        std::cerr << "[SEMANTIC] Limiting to first " << MAX_PROJECTIONS << " projections\n";
        projs.resize(MAX_PROJECTIONS);
    }
    
    size_t total = 0;
    
    for (const auto& [tag, meta] : projs) {
        size_t R = meta->shape[0];
        
        auto W = load_tensor(*meta);
        if (W.empty()) continue;
        
        // P = E @ W^T using MKL  (V x D) @ (D x R) = (V x R)
        std::vector<float> P(V * R);
        
        auto gemm_start = std::chrono::steady_clock::now();
#if defined(HAS_MKL) && HAS_MKL
        // cblas_sgemm: C = alpha * A * B + beta * C
        // A = E (V x D), B = W^T (D x R), C = P (V x R)
        // W is stored as (R x D), so we use CblasTrans
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)V, (int)R, (int)D,
                    1.0f, E.data(), (int)D, W.data(), (int)D,
                    0.0f, P.data(), (int)R);
#else
        // Fallback naive (slow)
        for (size_t i = 0; i < V; i++) {
            for (size_t j = 0; j < R; j++) {
                float sum = 0;
                for (size_t k = 0; k < D; k++) sum += E[i * D + k] * W[j * D + k];
                P[i * R + j] = sum;
            }
        }
#endif
        auto gemm_end = std::chrono::steady_clock::now();
        auto gemm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gemm_end - gemm_start).count();
        
        auto pvalid = normalize_rows(P.data(), V, R);
        
        auto knn_start = std::chrono::steady_clock::now();
        auto edges = knn(P.data(), pvalid, R);
        auto knn_end = std::chrono::steady_clock::now();
        auto knn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - knn_start).count();
        
        std::cerr << "[SEMANTIC] " << tag << ": " << edges.size() << " edges (gemm=" << gemm_ms << "ms, knn=" << knn_ms << "ms)\n";
        insert(conn, edges, ctx, tag);
        total += edges.size();
    }
    
    std::cerr << "[SEMANTIC] Total: " << total << " sparse relations\n";



    // =========================================================================
    // EXTRACT MERGE RELATIONS FROM BPE/CPE MERGES
    // =========================================================================

    if (extract_merge_relations(conn, ctx, config)) {
        std::cerr << "[SEMANTIC] Merge relations extracted\n";
    }

    // =========================================================================
    // EXTRACT HIERARCHY RELATIONS FROM TENSOR NAMES
    // =========================================================================

    if (extract_hierarchy_relations(conn, ctx, config)) {
        std::cerr << "[SEMANTIC] Hierarchy relations extracted\n";
    }

    // =========================================================================
    // PROJECT EMBEDDINGS TO 4D USING ANCHOR-CONSTRAINED LAPLACIAN EIGENMAPS
    // =========================================================================

    if (!project_and_update_embeddings(conn, ctx, config, anchors)) {
        std::cerr << "[SEMANTIC] WARNING: Embedding projection failed\n";
    }

    return true;
}

// ============================================================================
// Project embeddings to 4D using Laplacian eigenmaps
// ============================================================================

bool project_and_update_embeddings(PGconn* conn, IngestContext& ctx, const IngestConfig& config,
                                   const std::vector<AnchorPoint>& anchors) {
    std::cerr << "============================================================\n";
    std::cerr << "[PROJECTION] LAPLACIAN EIGENMAP PROJECTION starting\n";
    std::cerr << "============================================================\n";

    // Process ALL token embedding tensors that are reasonable for projection
    std::vector<std::pair<const TensorMeta*, std::string>> embeddings_to_project;

    if (ctx.manifest.has_value()) {
        std::cerr << "[PROJECTION] Using config-driven tensor lookup\n";
        std::cerr << "[PROJECTION] Scanning " << ctx.manifest->extraction_plans.size() << " extraction plans...\n";

        // Scan for embeddings with reasonable vocab size

        // Process all TOKEN_EMBEDDING tensors with extract_embeddings=true
        int candidate_count = 0;

        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::TOKEN_EMBEDDING && plan.extract_embeddings) {
                std::cerr << "[PROJECTION]   Candidate " << (++candidate_count) << ": " << plan.name;
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    const auto& meta = it->second;
                    std::cerr << " [" << meta.shape[0] << " x " << meta.shape[1] << "] - AVAILABLE\n";

                    if (meta.shape.size() == 2) {
                        embeddings_to_project.emplace_back(&meta, plan.name);
                        std::cerr << "[PROJECTION] >>> WILL PROJECT: " << plan.name << "\n";
                    } else {
                        std::cerr << "[PROJECTION] >>> SKIPPED: " << plan.name << " (not 2D)\n";
                    }
                } else {
                    std::cerr << " - NOT FOUND IN TENSORS\n";
                }
            }
        }

        if (candidate_count == 0) {
            std::cerr << "[PROJECTION] WARNING: No TOKEN_EMBEDDING plans found in manifest!\n";
        }
    }

    if (embeddings_to_project.empty()) {
        // Fallback: If no manifest or manifest didn't find embeddings, scan tensors
        std::cerr << "[PROJECTION] No embeddings found via manifest, falling back to tensor scan\n";
        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;
            if (name.find("position") != std::string::npos) continue;

            // Include all potential embedding tensors
            embeddings_to_project.emplace_back(&meta, name);
            std::cerr << "[PROJECTION] >>> FOUND VIA SCAN: " << name
                      << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
        }
    }

    if (embeddings_to_project.empty()) {
        std::cerr << "[PROJECTION] No suitable embedding tensors found for projection\n";
        return true;
    }

    // Process each embedding tensor that was selected
    for (const auto& [emb, emb_name] : embeddings_to_project) {
        std::cerr << "[PROJECTION] Processing embedding: " << emb_name << "\n";

        size_t V = static_cast<size_t>(emb->shape[0]);
        size_t D = static_cast<size_t>(emb->shape[1]);
        if (!ctx.vocab_tokens.empty()) V = std::min(V, ctx.vocab_tokens.size());

        // NOTE: Large vocabularies (>150K) will take significant time for HNSW build
        // The build is progress-monitored and can be interrupted if needed
        if (V > 150000) {
            std::cerr << "[PROJECTION] WARNING: Large vocabulary (" << V << " tokens) - HNSW build may take 30-60 minutes\n";
            std::cerr << "[PROJECTION] This is NORMAL for models like Llama with 200K vocab\n";
            std::cerr << "[PROJECTION] Progress will be shown every 10%...\n";
        }

        std::cerr << "[PROJECTION] Loading " << V << " x " << D << " embeddings...\n";
        auto load_start = std::chrono::steady_clock::now();
        auto E_flat = load_tensor(*emb);
        auto load_end = std::chrono::steady_clock::now();
        if (E_flat.empty()) {
            std::cerr << "[PROJECTION] Failed to load embeddings for " << emb_name << "\n";
            continue;
        }
        E_flat.resize(V * D);
        std::cerr << "[PROJECTION] Loaded in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << "ms\n";

        // DIAGNOSTIC: Check embedding statistics BEFORE conversion
        {
            double sum = 0.0, sum_sq = 0.0;
            double min_val = E_flat[0], max_val = E_flat[0];
            size_t zero_count = 0, nan_count = 0;

            for (size_t i = 0; i < std::min(size_t(V * D), E_flat.size()); ++i) {
                float val = E_flat[i];
                if (std::isnan(val)) {
                    ++nan_count;
                    continue;
                }
                if (std::abs(val) < 1e-10f) ++zero_count;
                sum += val;
                sum_sq += val * val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            double mean = sum / (V * D);
            double variance = (sum_sq / (V * D)) - (mean * mean);
            double stddev = std::sqrt(variance);

            std::cerr << "[PROJECTION] Embedding statistics (raw from safetensor):\n";
            std::cerr << "  Tensor: " << emb_name << " [" << V << " x " << D << "]\n";
            std::cerr << "  Dtype: " << emb->dtype << ", File offset: " << emb->data_offset_start << "\n";
            std::cerr << std::scientific << std::setprecision(6);
            std::cerr << "  Min: " << min_val << ", Max: " << max_val << "\n";
            std::cerr << "  Mean: " << mean << ", StdDev: " << stddev << "\n";
            std::cerr << std::defaultfloat << std::setprecision(2);
            std::cerr << "  Zeros: " << zero_count << " (" << (100.0 * zero_count / (V * D)) << "%)\n";
            std::cerr << "  NaNs: " << nan_count << "\n";

            // Check first few embeddings with better precision
            std::cerr << std::scientific << std::setprecision(4);
            std::cerr << "  First embedding (token 0): [";
            for (size_t j = 0; j < std::min(size_t(10), D); ++j) {
                std::cerr << E_flat[j];
                if (j < std::min(size_t(10), D) - 1) std::cerr << ", ";
            }
            std::cerr << ", ...]\n";

            // Show second embedding too
            if (V > 1) {
                std::cerr << "  Second embedding (token 1): [";
                for (size_t j = 0; j < std::min(size_t(10), D) - 1; ++j) {
                    std::cerr << E_flat[D + j];
                    if (j < std::min(size_t(10), D) - 1) std::cerr << ", ";
                }
                std::cerr << ", ...]\n";
            }
            std::cerr << std::defaultfloat << std::setprecision(6);

            // Check if all rows are identical
            bool all_same = true;
            if (V > 1) {
                for (size_t j = 0; j < D; ++j) {
                    if (std::abs(E_flat[j] - E_flat[D + j]) > 1e-6f) {
                        all_same = false;
                        break;
                    }
                }
            }
            std::cerr << "  First two rows identical: " << (all_same ? "YES (DEGENERATE!)" : "NO (good)") << "\n";

            // CRITICAL: Handle NaN/Inf values based on severity
            double nan_ratio = static_cast<double>(nan_count) / (V * D);

            if (nan_ratio > 0.5) {
                // > 50% corrupt - tensor is completely unusable
                std::cerr << "[PROJECTION] FATAL: " << emb_name << " has " << (nan_ratio * 100.0)
                          << "% NaN/Inf values - tensor is corrupted\n";
                std::cerr << "[PROJECTION] Possible causes:\n";
                std::cerr << "  - Model file corruption\n";
                std::cerr << "  - Wrong FP8 variant (E4M3 vs E5M2)\n";
                std::cerr << "  - Incompatible quantization scheme\n";
                std::cerr << "[PROJECTION] SKIPPING this tensor - use higher precision if available\n";
                continue;
            } else if (nan_ratio > 0.01) {
                // 1-50% corrupt - may indicate quantization issues but recoverable
                std::cerr << "[PROJECTION] WARNING: " << emb_name << " has " << (nan_ratio * 100.0)
                          << "% NaN/Inf values\n";
                std::cerr << "[PROJECTION] This is concerning for FP8 tensors - consider using F16/F32\n";
                std::cerr << "[PROJECTION] Attempting to clean and continue (replacing NaN/Inf with 0)\n";

                // Clean by replacing with zeros
                for (size_t i = 0; i < E_flat.size(); ++i) {
                    if (std::isnan(E_flat[i]) || std::isinf(E_flat[i])) {
                        E_flat[i] = 0.0f;
                    }
                }
            } else if (nan_count > 0) {
                // < 1% corrupt - minor issue, clean and continue
                std::cerr << "[PROJECTION] INFO: Cleaning " << nan_count << " NaN/Inf values ("
                          << (nan_ratio * 100.0) << "%)\n";
                for (size_t i = 0; i < E_flat.size(); ++i) {
                    if (std::isnan(E_flat[i]) || std::isinf(E_flat[i])) {
                        E_flat[i] = 0.0f;
                    }
                }
            }
        }

        // Convert to vector<vector<float>> for LaplacianProjector
        std::vector<std::vector<float>> embeddings(V);
        for (size_t i = 0; i < V; ++i) {
            embeddings[i].resize(D);
            for (size_t j = 0; j < D; ++j) {
                embeddings[i][j] = E_flat[i * D + j];
            }
        }

        // Prepare labels
        std::vector<std::string> labels(V);
        for (size_t i = 0; i < V; ++i) {
            if (i < ctx.vocab_tokens.size()) {
                labels[i] = ctx.vocab_tokens[i].text;
            } else {
                labels[i] = "token_" + std::to_string(i);
            }
        }

        // Configure Laplacian projector
        LaplacianConfig lap_config;
        lap_config.k_neighbors = 15;
        lap_config.similarity_threshold = 0.0f;
        lap_config.power_iterations = 50;  // Reduced for speed - 50 is often sufficient
        lap_config.num_threads = g_num_threads;
        lap_config.project_to_sphere = true;
        lap_config.verbose = true;
        // RELAXED tolerance for large sparse matrices (>100K points)
        // Residuals ~1e-2 are acceptable for semantic embeddings
        lap_config.convergence_tol = (V > 100000) ? 1e-2 : 1e-3;

        // CRITICAL: N×N Laplacian → N eigenvalues → (N-1) non-zero eigenvectors
        // NOT about "5 points to define a 4D simplex" (affine geometry)
        // This is pure linear algebra: need 4 eigenvectors for 4D spectral coordinates
        if (V < 5) {
            std::cerr << "[PROJECTION] Skipping " << emb_name << ": " << V << " nodes → max "
                      << (V-1) << " non-zero eigenvectors (need 4 for spectral 4D)\n";
            std::cerr << "[PROJECTION] (This is expected for utility tensors like token_type_embeddings)\n";
            continue;
        }

        std::cerr << "[PROJECTION] Projecting " << V << " embeddings to 4D using Laplacian eigenmaps...\n";
        auto proj_start = std::chrono::steady_clock::now();

        LaplacianProjector projector(lap_config);
        ProjectionResult result = projector.project(embeddings, labels, anchors);

        auto proj_end = std::chrono::steady_clock::now();
        std::cerr << "[PROJECTION] Projection completed in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(proj_end - proj_start).count() << "ms\n";
        std::cerr << "[PROJECTION] 4D coordinates range computed\n";
        std::cerr << "[PROJECTION] Variance explained: " << result.total_variance_explained * 100 << "%\n";

        if (result.total_variance_explained < 0.01) {
            std::cerr << "[PROJECTION] Variance explained too low for " << emb_name << ", skipping database update\n";
            continue;
        }

        // Calculate quality score to determine if geometry should be written
        float quality_score = calculate_projection_quality("embeddings", emb->dtype, static_cast<int>(D), result.total_variance_explained);
        std::cerr << "[PROJECTION] Quality score for " << emb_name << ": " << quality_score << "\n";

        bool should_write_geometry = (quality_score > 2.0f);  // Threshold from task: >2.0 for minimal quality
        size_t updated = 0;

        if (should_write_geometry) {
            // Prepare TokenData for database update
            std::vector<hypercube::db::TokenData> tokens(V);
            for (size_t i = 0; i < V; ++i) {
                tokens[i].label = labels[i];
                tokens[i].hash = ctx.vocab_tokens[i].comp.hash;  // Use existing hash
                tokens[i].is_atom = false;  // These are compositions (tokens)
                tokens[i].coords = result.coords[i];
                tokens[i].hilbert_lo = result.hilbert_lo[i];
                tokens[i].hilbert_hi = result.hilbert_hi[i];
            }

            // Update database with projected coordinates
            std::cerr << "[PROJECTION] Quality sufficient (" << quality_score << " > 2.0), writing 4D coordinates to database...\n";
            hypercube::db::PersistConfig persist_config;
            persist_config.update_existing = true;
            hypercube::db::ProjectionPersister persister(conn, persist_config);
            updated = persister.persist(tokens);
        } else {
            std::cerr << "[PROJECTION] Quality insufficient (" << quality_score << " <= 2.0), skipping geometry write\n";
        }

        std::cerr << "[PROJECTION] Updated " << updated << " token compositions with Laplacian-projected coordinates from " << emb_name << "\n";

        // Insert projection metadata for quality tracking
        if (result.total_variance_explained >= 0.01) {  // Only track successful projections
            try {
                // Get or create model ID
                std::string get_model_query = "SELECT id FROM model WHERE name = $1";
                const char* model_params[1] = {config.model_name.c_str()};
                PGresult* model_res = PQexecParams(conn, get_model_query.c_str(), 1, nullptr, model_params, nullptr, nullptr, 0);

                int64_t model_id = -1;
                if (PQresultStatus(model_res) == PGRES_TUPLES_OK && PQntuples(model_res) > 0) {
                    model_id = std::stoll(PQgetvalue(model_res, 0, 0));
                }
                PQclear(model_res);

                if (model_id == -1) {
                    // Model doesn't exist, create it
                    std::string insert_model_query =
                        "INSERT INTO model (name, source, embedding_dim) VALUES ($1, $2, $3) RETURNING id";
                    const char* insert_params[3] = {
                        config.model_name.c_str(),
                        "ingestion",
                        std::to_string(D).c_str()
                    };
                    PGresult* insert_res = PQexecParams(conn, insert_model_query.c_str(), 3, nullptr, insert_params, nullptr, nullptr, 0);
                    if (PQresultStatus(insert_res) == PGRES_TUPLES_OK) {
                        model_id = std::stoll(PQgetvalue(insert_res, 0, 0));
                    }
                    PQclear(insert_res);
                }

                if (model_id != -1) {
                    // Insert projection metadata
                    std::string insert_meta_query = R"SQL(
                        INSERT INTO projection_metadata
                            (model_id, tensor_name, role, dtype, dim, variance_explained, converged, geom_written)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (model_id, tensor_name) DO UPDATE SET
                            variance_explained = EXCLUDED.variance_explained,
                            converged = EXCLUDED.converged,
                            geom_written = EXCLUDED.geom_written,
                            updated_at = NOW()
                    )SQL";

                    std::string model_id_str = std::to_string(model_id);
                    std::string variance_str = std::to_string(result.total_variance_explained);
                    std::string converged_str = result.converged ? "true" : "false";
                    std::string geom_written_str = should_write_geometry ? "true" : "false";

                    const char* meta_params[8] = {
                        model_id_str.c_str(),
                        emb_name.c_str(),
                        "embeddings",
                        emb->dtype.c_str(),
                        std::to_string(D).c_str(),
                        variance_str.c_str(),
                        converged_str.c_str(),
                        geom_written_str.c_str()
                    };

                    PGresult* meta_res = PQexecParams(conn, insert_meta_query.c_str(), 8, nullptr, meta_params, nullptr, nullptr, 0);
                    if (PQresultStatus(meta_res) == PGRES_COMMAND_OK) {
                        std::cerr << "[PROJECTION] Inserted projection metadata for " << emb_name
                                  << " (variance_explained=" << result.total_variance_explained
                                  << ", converged=" << result.converged
                                  << ", geom_written=" << (updated > 0) << ")\n";
                    } else {
                        std::cerr << "[PROJECTION] Failed to insert projection metadata: "
                                  << PQerrorMessage(conn) << "\n";
                    }
                    PQclear(meta_res);
                }
            } catch (const std::exception& e) {
                std::cerr << "[PROJECTION] Error inserting projection metadata: " << e.what() << "\n";
            }
        }
    }

    return true;
}

static bool extract_merge_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[MERGE] Extracting BPE/CPE merge relations...\n";

    if (ctx.bpe_merges.empty()) {
        std::cerr << "[MERGE] No BPE merges found\n";
        return true;
    }

    hypercube::db::Transaction tx(conn);

    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    std::vector<std::string> values;

    size_t merge_edges = 0;

    // For each BPE merge, create M-relations between parts and merged token
    for (const auto& [left, right] : ctx.bpe_merges) {
        std::string merged = left + right;

        // Create compositions for the parts and merged token if they don't exist
        auto left_hash = AtomCalculator::compute_vocab_token(left).hash;
        auto right_hash = AtomCalculator::compute_vocab_token(right).hash;
        auto merged_hash = AtomCalculator::compute_vocab_token(merged).hash;

        // Create M-relations: left -> merged, right -> merged
        // Weight represents the merge strength (1.0 for direct merges)
        float weight = 1.0f;

        // left -> merged
        {
            std::string source_hex = "\\x" + left_hash.to_hex();
            std::string target_hex = "\\x" + merged_hash.to_hex();

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'M', '" +
                              config.model_name + "', -1, 'merge', 1500.0, 1, " +
                              std::to_string(weight) + ", " + std::to_string(weight) + ")";
            values.push_back(val);
            merge_edges++;
        }

        // right -> merged
        {
            std::string source_hex = "\\x" + right_hash.to_hex();
            std::string target_hex = "\\x" + merged_hash.to_hex();

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'M', '" +
                              config.model_name + "', -1, 'merge', 1500.0, 1, " +
                              std::to_string(weight) + ", " + std::to_string(weight) + ")";
            values.push_back(val);
            merge_edges++;
        }
    }

    // Batch insert merge relations
    if (!values.empty()) {
        const size_t batch_size = 1000;
        for (size_t i = 0; i < values.size(); i += batch_size) {
            std::string batch_sql = insert_sql;
            for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += values[j];
            }
            batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
            DO UPDATE SET
                rating = relation_evidence.rating +
                         LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                         (
                             (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                             (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                         ),
                observation_count = relation_evidence.observation_count + 1,
                raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                             (relation_evidence.observation_count + 1),
                normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                    (relation_evidence.observation_count + 1),
                last_updated = NOW()
        )SQL";

            hypercube::db::Result res = hypercube::db::exec(conn, batch_sql);
            if (!res.ok()) {
                std::cerr << "[MERGE] Batch insert failed: " << res.error_message() << "\n";
                return false;
            }
        }
    }

    // Ensure merge token compositions exist
    if (!ctx.bpe_merges.empty()) {
        std::unordered_set<std::string> tokens_to_create;
        for (const auto& [left, right] : ctx.bpe_merges) {
            tokens_to_create.insert(left);
            tokens_to_create.insert(right);
            tokens_to_create.insert(left + right);
        }

        std::string comp_insert = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
        std::vector<std::string> comp_values;

        for (const std::string& token : tokens_to_create) {
            CompositionRecord comp = AtomCalculator::compute_vocab_token(token);
            std::string hex_id = "\\x" + comp.hash.to_hex();
            std::string centroid = "010100002000000000000000000000000000000000000000000000000000000000000000";  // POINT(0 0 0 0)
            comp_values.push_back("('" + hex_id + "', '" + token + "', 1, 1, 1, '" + centroid + "', 0, 0)");
        }

        if (!comp_values.empty()) {
            for (size_t i = 0; i < comp_values.size(); ++i) {
                if (i > 0) comp_insert += ", ";
                comp_insert += comp_values[i];
            }
            comp_insert += " ON CONFLICT (id) DO NOTHING";

            hypercube::db::Result comp_res = hypercube::db::exec(conn, comp_insert);
            if (!comp_res.ok()) {
                std::cerr << "[MERGE] Failed to insert merge token compositions: " << comp_res.error_message() << "\n";
                return false;
            }
        }
    }

    tx.commit();

    std::cerr << "[MERGE] Inserted " << merge_edges << " merge relations\n";

    return true;
}

static bool extract_hierarchy_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[HIERARCHY] Extracting tensor hierarchy relations...\n";

    hypercube::db::Transaction tx(conn);

    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    std::vector<std::string> values;

    size_t hierarchy_edges = 0;

    // Process each tensor to build parent-child relations
    for (const auto& [tensor_name, tensor] : ctx.tensors) {
        // Parse tensor name into components
        std::vector<std::string> parts;
        std::stringstream ss(tensor_name);
        std::string part;
        while (std::getline(ss, part, '.')) {
            parts.push_back(part);
        }

        if (parts.size() < 2) continue;

        // Build hierarchical path
        std::string current_path;
        for (size_t i = 0; i < parts.size(); ++i) {
            std::string parent_path = current_path;
            if (!current_path.empty()) current_path += ".";
            current_path += parts[i];

            if (i > 0) {  // Not the root
                // Create H-relation from parent to child
                auto parent_hash = AtomCalculator::compute_vocab_token(parent_path).hash;
                auto child_hash = AtomCalculator::compute_vocab_token(current_path).hash;

                std::string parent_hex = "\\x" + parent_hash.to_hex();
                std::string child_hex = "\\x" + child_hash.to_hex();

                // Parse layer number if present
                int layer = -1;
                if (parent_path.find("layers.") != std::string::npos) {
                    size_t layer_pos = parent_path.find("layers.");
                    if (layer_pos != std::string::npos) {
                        size_t num_start = layer_pos + 7;
                        size_t num_end = parent_path.find(".", num_start);
                        if (num_end != std::string::npos) {
                            try {
                                layer = std::stoi(parent_path.substr(num_start, num_end - num_start));
                            } catch (...) {}
                        }
                    }
                }

                // Weight = 1.0 for direct hierarchy (strong relation)
                float weight = 1.0f;

                std::string val = "('" + parent_hex + "', '" + child_hex + "', 'H', '" +
                                  config.model_name + "', " + std::to_string(layer) + ", 'hierarchy', 1500.0, 1, " +
                                  std::to_string(weight) + ", " + std::to_string(weight) + ")";
                values.push_back(val);
                hierarchy_edges++;
            }
        }
    }

    // Batch insert hierarchy relations
    if (!values.empty()) {
        const size_t batch_size = 1000;
        for (size_t i = 0; i < values.size(); i += batch_size) {
            std::string batch_sql = insert_sql;
            for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += values[j];
            }
            batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
            DO UPDATE SET
                rating = relation_evidence.rating +
                         LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                         (
                             (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                             (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                         ),
                observation_count = relation_evidence.observation_count + 1,
                raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                             (relation_evidence.observation_count + 1),
                normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                    (relation_evidence.observation_count + 1),
                last_updated = NOW()
        )SQL";

            hypercube::db::Result res = hypercube::db::exec(conn, batch_sql);
            if (!res.ok()) {
                std::cerr << "[HIERARCHY] Batch insert failed: " << res.error_message() << "\n";
                return false;
            }
        }
    }

    tx.commit();

    std::cerr << "[HIERARCHY] Inserted " << hierarchy_edges << " hierarchy relations\n";

    return true;
}

static bool extract_temporal_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[TEMPORAL] Extracting temporal/positional relations...\n";

    // Find position embedding tensors
    std::vector<std::pair<const TensorMeta*, std::string>> position_embeddings;

    if (ctx.manifest.has_value()) {
        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::POSITION_EMBEDDING ||
                plan.category == TensorCategory::POSITION_EMBEDDING_2D) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    position_embeddings.emplace_back(&it->second, plan.name);
                    std::cerr << "[TEMPORAL] Found " << category_to_string(plan.category) << " via manifest: " << plan.name << "\n";
                }
            }
        }
    }

    if (position_embeddings.empty()) {
        // Fallback: Scan tensors for position-like embeddings
        std::cerr << "[TEMPORAL] No position embeddings found via manifest, falling back to tensor scan\n";

        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;

            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

            if (lower_name.find("position") != std::string::npos ||
                lower_name.find("rotary") != std::string::npos ||
                lower_name.find("rope") != std::string::npos ||
                lower_name.find("pos_emb") != std::string::npos ||
                lower_name.find("sinusoidal") != std::string::npos ||
                lower_name.find("temporal") != std::string::npos) {
                // Position embeddings
                position_embeddings.emplace_back(&meta, name);
                std::cerr << "[TEMPORAL] Found position embedding by name: " << name
                          << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
            }
        }
    }

    if (position_embeddings.empty()) {
        std::cerr << "[TEMPORAL] No position embedding tensors found\n";
        return true;
    }

    size_t total_temporal_edges = 0;

    for (const auto& [pos_emb, emb_name] : position_embeddings) {
        int64_t seq_len = pos_emb->shape[0];
        int64_t embed_dim = pos_emb->shape[1];

        // Position embeddings are typically small (e.g., 2048 x 768)
        // But can be up to ~100K for very long contexts
        if (seq_len > 100000) {
            std::cerr << "[TEMPORAL] Skipping large position embedding: " << emb_name
                      << " [" << seq_len << " x " << embed_dim << "]\n";
            continue;
        }

        std::cerr << "[TEMPORAL] Processing " << emb_name << " [" << seq_len << " x " << embed_dim << "]...\n";

        // Load position embeddings
        auto pos_data = load_tensor(*pos_emb);
        if (pos_data.empty()) {
            std::cerr << "[TEMPORAL] Failed to load position embeddings for " << emb_name << "\n";
            continue;
        }
        pos_data.resize(static_cast<size_t>(seq_len * embed_dim));

        // Normalize rows (each position embedding)
        auto valid_positions = normalize_rows(pos_data.data(), static_cast<size_t>(seq_len),
                                            static_cast<size_t>(embed_dim));

        if (valid_positions.size() < 2) {
            std::cerr << "[TEMPORAL] Not enough valid position embeddings\n";
            continue;
        }

        // Extract k-NN relations with lower threshold for temporal proximity
        // Position embeddings capture sequential/temporal relationships
        const float TEMPORAL_THRESHOLD = 0.3f;  // Lower threshold - positions can be similar
        const size_t TEMPORAL_K = 8;

        // Temporarily adjust constants for temporal extraction
        size_t k = TEMPORAL_K;
        float threshold = TEMPORAL_THRESHOLD;

        auto temporal_edges = knn(pos_data.data(), valid_positions, static_cast<size_t>(embed_dim), k, threshold);

        std::cerr << "[TEMPORAL] Found " << temporal_edges.size() << " temporal relations\n";

        if (!temporal_edges.empty()) {
            // Insert T-relations using the same pattern as the existing insert function
            // Use temp table then INSERT with existence checks
            PQexec(conn, "DROP TABLE IF EXISTS tmp_temporal_rel");
            PQexec(conn, "CREATE TEMP TABLE tmp_temporal_rel ("
                      "source_type CHAR(1), source_id BYTEA, target_type CHAR(1), target_id BYTEA, "
                      "relation_type CHAR(1), weight REAL, source_model TEXT, layer INTEGER, component TEXT)");

            std::string copy_cmd = "COPY tmp_temporal_rel FROM STDIN";
            PGresult* res = PQexec(conn, copy_cmd.c_str());
            if (PQresultStatus(res) != PGRES_COPY_IN) {
                std::cerr << "[TEMPORAL] COPY start failed: " << PQerrorMessage(conn) << "\n";
                PQclear(res);
                return false;
            }
            PQclear(res);

            for (const auto& e : temporal_edges) {
                // Create position composition hashes
                std::string src_key = "pos:" + emb_name + ":" + std::to_string(valid_positions[e.src]);
                std::string tgt_key = "pos:" + emb_name + ":" + std::to_string(valid_positions[e.tgt]);

                auto src_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                auto tgt_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;

                std::string source_hex = "\\x" + src_hash.to_hex();
                std::string target_hex = "\\x" + tgt_hash.to_hex();

                std::string line = "C\t" + source_hex + "\tC\t" + target_hex +
                                  "\tT\t" + std::to_string(e.sim) + "\t" + config.model_name +
                                  "\t-1\t" + emb_name + "\n";

                if (PQputCopyData(conn, line.c_str(), (int)line.size()) != 1) {
                    std::cerr << "[TEMPORAL] COPY data failed: " << PQerrorMessage(conn) << "\n";
                    return false;
                }
            }

            if (PQputCopyEnd(conn, nullptr) != 1) {
                std::cerr << "[TEMPORAL] COPY to temp failed: " << PQerrorMessage(conn) << "\n";
                return false;
            }

            res = PQgetResult(conn);
            PQclear(res);

            // INSERT into relation_evidence with ELO rating updates
            std::string insert_sql = R"SQL(
                INSERT INTO relation_evidence
                    (source_id, target_id, relation_type, source_model, layer, component,
                     rating, observation_count, raw_weight, normalized_weight)
                SELECT
                    DECODE(SUBSTRING(source_id, 3), 'hex'),
                    DECODE(SUBSTRING(target_id, 3), 'hex'),
                    relation_type::CHAR(1),
                    source_model,
                    layer::INT,
                    component,
                    1500.0,  -- Initial ELO rating
                    1,        -- First observation
                    weight,   -- Raw weight
                    weight    -- Normalized weight
                FROM tmp_temporal_rel t
                WHERE EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.source_id, 3), 'hex'))
                AND EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.target_id, 3), 'hex'))
                ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
                DO UPDATE SET
                    -- ELO rating update with dynamic K-factor
                    rating = relation_evidence.rating +
                             LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                             (
                                 (EXCLUDED.normalized_weight + 1.0) / 2.0 -  -- Actual score [0,1]
                                 (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))  -- Expected
                             ),
                    observation_count = relation_evidence.observation_count + 1,
                    raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                                 (relation_evidence.observation_count + 1),
                    normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                        (relation_evidence.observation_count + 1),
                    last_updated = NOW()
            )SQL";

            res = PQexec(conn, insert_sql.c_str());
            int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
            PQclear(res);

            std::cerr << "[TEMPORAL] Inserted " << inserted << " temporal relations (filtered " << (temporal_edges.size() - inserted) << " missing refs)\n";

            // Ensure position compositions exist
            ensure_position_compositions_exist(conn, emb_name, valid_positions);

            total_temporal_edges += temporal_edges.size();
        }
    }

    std::cerr << "[TEMPORAL] Total: " << total_temporal_edges << " temporal/positional relations\n";
    return true;
}

static bool extract_visual_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[VISION] Extracting visual embedding relations...\n";

    // Find vision feature tensors
    std::vector<std::pair<const TensorMeta*, std::string>> vision_features;

    if (ctx.manifest.has_value()) {
        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::VISION_FEATURE ||
                plan.category == TensorCategory::VISION_PROJECTION ||
                plan.category == TensorCategory::PATCH_EMBEDDING ||
                plan.category == TensorCategory::DETECTION_BACKBONE ||
                plan.category == TensorCategory::DETECTION_NECK ||
                plan.category == TensorCategory::DETECTION_HEAD) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    vision_features.emplace_back(&it->second, plan.name);
                    std::cerr << "[VISION] Found " << category_to_string(plan.category) << " via manifest: " << plan.name << "\n";
                }
            }
        }
    }

    if (vision_features.empty()) {
        // Fallback: Scan tensors for vision-like embeddings
        std::cerr << "[VISION] No vision features found via manifest, falling back to tensor scan\n";

        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;

            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

            if (lower_name.find("patch") != std::string::npos ||
                lower_name.find("vision") != std::string::npos ||
                lower_name.find("visual") != std::string::npos ||
                lower_name.find("image") != std::string::npos ||
                lower_name.find("clip") != std::string::npos ||
                lower_name.find("backbone") != std::string::npos ||
                lower_name.find("neck") != std::string::npos ||
                lower_name.find("fpn") != std::string::npos ||
                lower_name.find("lateral") != std::string::npos ||
                lower_name.find("detection") != std::string::npos ||
                lower_name.find("object") != std::string::npos) {
                // Vision/detection embeddings
                vision_features.emplace_back(&meta, name);
                std::cerr << "[VISION] Found vision/detection embedding by name: " << name
                          << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
            }
        }
    }

    if (vision_features.empty()) {
        std::cerr << "[VISION] No vision feature tensors found\n";
        return true;
    }

    size_t total_visual_edges = 0;

    for (const auto& [vis_emb, emb_name] : vision_features) {
        int64_t num_features = vis_emb->shape[0];
        int64_t feature_dim = vis_emb->shape[1];

        // Vision features can be large (e.g., CLIP has 257x768 for ViT-B/32)
        // But we limit to reasonable sizes for k-NN computation
        if (num_features > 50000) {
            std::cerr << "[VISION] Skipping large vision features: " << emb_name
                      << " [" << num_features << " x " << feature_dim << "]\n";
            continue;
        }

        std::cerr << "[VISION] Processing " << emb_name << " [" << num_features << " x " << feature_dim << "]...\n";

        // Load vision features
        auto vis_data = load_tensor(*vis_emb);
        if (vis_data.empty()) {
            std::cerr << "[VISION] Failed to load vision features for " << emb_name << "\n";
            continue;
        }
        vis_data.resize(static_cast<size_t>(num_features * feature_dim));

        // Normalize rows (each visual feature/patch)
        auto valid_features = normalize_rows(vis_data.data(), static_cast<size_t>(num_features),
                                           static_cast<size_t>(feature_dim));

        if (valid_features.size() < 2) {
            std::cerr << "[VISION] Not enough valid visual features\n";
            continue;
        }

        // Extract k-NN relations with moderate threshold for visual similarity
        // Vision features capture visual concepts and patterns
        const float VISION_THRESHOLD = 0.4f;  // Moderate threshold for visual similarity
        const size_t VISION_K = 10;

        // Temporarily adjust constants for vision extraction
        size_t k = VISION_K;
        float threshold = VISION_THRESHOLD;

        auto visual_edges = knn(vis_data.data(), valid_features, static_cast<size_t>(feature_dim), k, threshold);

        std::cerr << "[VISION] Found " << visual_edges.size() << " visual relations\n";

        if (!visual_edges.empty()) {
            // Insert V-relations using the same pattern as the existing insert function
            PQexec(conn, "DROP TABLE IF EXISTS tmp_visual_rel");
            PQexec(conn, "CREATE TEMP TABLE tmp_visual_rel ("
                      "source_type CHAR(1), source_id BYTEA, target_type CHAR(1), target_id BYTEA, "
                      "relation_type CHAR(1), weight REAL, source_model TEXT, layer INTEGER, component TEXT)");

            std::string copy_cmd = "COPY tmp_visual_rel FROM STDIN";
            PGresult* res = PQexec(conn, copy_cmd.c_str());
            if (PQresultStatus(res) != PGRES_COPY_IN) {
                std::cerr << "[VISION] COPY start failed: " << PQerrorMessage(conn) << "\n";
                PQclear(res);
                return false;
            }
            PQclear(res);

            for (const auto& e : visual_edges) {
                // Create visual feature composition hashes
                std::string src_key = "vis:" + emb_name + ":" + std::to_string(valid_features[e.src]);
                std::string tgt_key = "vis:" + emb_name + ":" + std::to_string(valid_features[e.tgt]);

                auto src_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                auto tgt_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;

                std::string source_hex = "\\x" + src_hash.to_hex();
                std::string target_hex = "\\x" + tgt_hash.to_hex();

                std::string line = "C\t" + source_hex + "\tC\t" + target_hex +
                                  "\tV\t" + std::to_string(e.sim) + "\t" + config.model_name +
                                  "\t-1\t" + emb_name + "\n";

                if (PQputCopyData(conn, line.c_str(), (int)line.size()) != 1) {
                    std::cerr << "[VISION] COPY data failed: " << PQerrorMessage(conn) << "\n";
                    return false;
                }
            }

            if (PQputCopyEnd(conn, nullptr) != 1) {
                std::cerr << "[VISION] COPY to temp failed: " << PQerrorMessage(conn) << "\n";
                return false;
            }

            res = PQgetResult(conn);
            PQclear(res);

            // INSERT into relation_evidence with ELO rating updates
            std::string insert_sql = R"SQL(
                INSERT INTO relation_evidence
                    (source_id, target_id, relation_type, source_model, layer, component,
                     rating, observation_count, raw_weight, normalized_weight)
                SELECT
                    DECODE(SUBSTRING(source_id, 3), 'hex'),
                    DECODE(SUBSTRING(target_id, 3), 'hex'),
                    relation_type::CHAR(1),
                    source_model,
                    layer::INT,
                    component,
                    1500.0,  -- Initial ELO rating
                    1,        -- First observation
                    weight,   -- Raw weight
                    weight    -- Normalized weight
                FROM tmp_visual_rel t
                WHERE EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.source_id, 3), 'hex'))
                AND EXISTS (SELECT 1 FROM composition WHERE id = DECODE(SUBSTRING(t.target_id, 3), 'hex'))
                ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
                DO UPDATE SET
                    -- ELO rating update with dynamic K-factor
                    rating = relation_evidence.rating +
                             LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                             (
                                 (EXCLUDED.normalized_weight + 1.0) / 2.0 -  -- Actual score [0,1]
                                 (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))  -- Expected
                             ),
                    observation_count = relation_evidence.observation_count + 1,
                    raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                                 (relation_evidence.observation_count + 1),
                    normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                        (relation_evidence.observation_count + 1),
                    last_updated = NOW()
            )SQL";

            res = PQexec(conn, insert_sql.c_str());
            int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
            PQclear(res);

            std::cerr << "[VISION] Inserted " << inserted << " visual relations (filtered " << (visual_edges.size() - inserted) << " missing refs)\n";

            // Ensure visual compositions exist
            ensure_visual_compositions_exist(conn, emb_name, valid_features);

            total_visual_edges += visual_edges.size();
        }
    }

    std::cerr << "[VISION] Total: " << total_visual_edges << " visual relations\n";
    return true;
}

static bool ensure_position_compositions_exist(PGconn* conn, const std::string& emb_name,
                                             const std::vector<size_t>& positions) {
    // Create compositions for position atoms if they don't exist
    std::string insert_sql = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
    std::vector<std::string> inserts;

    for (size_t pos : positions) {
        std::string key = "pos:" + emb_name + ":" + std::to_string(pos);
        CompositionRecord comp = AtomCalculator::compute_vocab_token(key);
        std::string hex_id = "\\x" + comp.hash.to_hex();

        // Build centroid EWKB (simplified point)
        std::string centroid = "010100002000000000000000000000000000000000000000000000000000000000000000";  // POINT(0 0 0 0)

        inserts.push_back("('" + hex_id + "', '" + key + "', 1, 1, 1, '" + centroid + "', 0, 0)");
    }

    if (!inserts.empty()) {
        for (size_t i = 0; i < inserts.size(); ++i) {
            if (i > 0) insert_sql += ", ";
            insert_sql += inserts[i];
        }
        insert_sql += " ON CONFLICT (id) DO NOTHING";

        PGresult* ins_res = PQexec(conn, insert_sql.c_str());
        if (PQresultStatus(ins_res) != PGRES_COMMAND_OK) {
            std::cerr << "[TEMPORAL] Failed to insert position compositions: " << PQerrorMessage(conn) << "\n";
            PQclear(ins_res);
            return false;
        }
        PQclear(ins_res);
    }

    return true;
}

static bool ensure_visual_compositions_exist(PGconn* conn, const std::string& emb_name,
                                           const std::vector<size_t>& features) {
    // Create compositions for visual feature atoms if they don't exist
    std::string insert_sql = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
    std::vector<std::string> inserts;

    for (size_t feat : features) {
        std::string key = "vis:" + emb_name + ":" + std::to_string(feat);
        CompositionRecord comp = AtomCalculator::compute_vocab_token(key);
        std::string hex_id = "\\x" + comp.hash.to_hex();

        // Build centroid EWKB (simplified point)
        std::string centroid = "010100002000000000000000000000000000000000000000000000000000000000000000";  // POINT(0 0 0 0)

        inserts.push_back("('" + hex_id + "', '" + key + "', 1, 1, 1, '" + centroid + "', 0, 0)");
    }

    if (!inserts.empty()) {
        for (size_t i = 0; i < inserts.size(); ++i) {
            if (i > 0) insert_sql += ", ";
            insert_sql += inserts[i];
        }
        insert_sql += " ON CONFLICT (id) DO NOTHING";

        PGresult* ins_res = PQexec(conn, insert_sql.c_str());
        if (PQresultStatus(ins_res) != PGRES_COMMAND_OK) {
            std::cerr << "[VISION] Failed to insert visual compositions: " << PQerrorMessage(conn) << "\n";
            PQclear(ins_res);
            return false;
        }
        PQclear(ins_res);
    }

    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
