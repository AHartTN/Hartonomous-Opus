/**
 * Safetensor Embedding Extractor
 *
 * Complete model ingestion pipeline:
 * 1. Parse safetensor header to discover tensors
 * 2. Load word embeddings matrix [vocab_size, hidden_dim]
 * 3. Compute pairwise cosine similarities
 * 4. Record sparse edges above threshold
 * 5. Link edges to vocab compositions in the hypercube
 *
 * The embedding matrix is just an "address book" - we extract the RELATIONSHIPS
 * and discard the raw vectors. Only edges >= threshold are stored (sparse).
 *
 * Usage:
 *   extract_embeddings --model model.safetensors --vocab vocab.txt --threshold 0.5
 *   extract_embeddings --model model.safetensors --config config.json -t 0.6
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/embedding_ops.hpp"  // Centralized SIMD operations
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/ingest/projection_db.hpp"

using namespace hypercube;
using namespace hypercube::db;

// Safetensor header structures (JSON parsed manually)
struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset_start;
    size_t offset_end;
};

// Parse safetensor header (JSON)
std::vector<TensorInfo> parse_safetensor_header(const std::string& filepath) {
    std::vector<TensorInfo> tensors;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return tensors;
    }

    // Read header size (8 bytes, little-endian uint64)
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Read header JSON
    std::string header(header_size, '\0');
    file.read(&header[0], header_size);

    // Simple JSON parsing (no external library)
    // Format: {"tensor_name": {"dtype": "F32", "shape": [30522, 384], "data_offsets": [0, 46881408]}, ...}

    size_t pos = 0;
    while ((pos = header.find("\"dtype\"", pos)) != std::string::npos) {
        TensorInfo info;

        // Find tensor name (go back to find the key)
        size_t key_end = header.rfind("\":", pos);
        size_t key_start = header.rfind("\"", key_end - 1);
        if (key_start != std::string::npos && key_end != std::string::npos) {
            info.name = header.substr(key_start + 1, key_end - key_start - 1);
        }

        // Parse dtype
        size_t dtype_start = header.find("\"", pos + 7);
        size_t dtype_end = header.find("\"", dtype_start + 1);
        info.dtype = header.substr(dtype_start + 1, dtype_end - dtype_start - 1);

        // Parse shape
        size_t shape_start = header.find("[", dtype_end);
        size_t shape_end = header.find("]", shape_start);
        std::string shape_str = header.substr(shape_start + 1, shape_end - shape_start - 1);

        // Extract shape values
        size_t s = 0;
        while (s < shape_str.size()) {
            while (s < shape_str.size() && !isdigit(shape_str[s])) s++;
            if (s >= shape_str.size()) break;
            size_t e = s;
            while (e < shape_str.size() && isdigit(shape_str[e])) e++;
            info.shape.push_back(std::stoll(shape_str.substr(s, e - s)));
            s = e;
        }

        // Parse data_offsets
        size_t offsets_start = header.find("data_offsets", shape_end);
        if (offsets_start != std::string::npos) {
            size_t arr_start = header.find("[", offsets_start);
            size_t arr_end = header.find("]", arr_start);
            std::string off_str = header.substr(arr_start + 1, arr_end - arr_start - 1);

            size_t comma = off_str.find(",");
            info.offset_start = std::stoull(off_str.substr(0, comma));
            info.offset_end = std::stoull(off_str.substr(comma + 1));
        }

        if (!info.name.empty() && info.name != "__metadata__") {
            tensors.push_back(info);
        }

        pos = shape_end;
    }

    return tensors;
}

// Load embedding tensor as float array
std::vector<float> load_embedding_tensor(const std::string& filepath, const TensorInfo& info) {
    std::vector<float> data;

    // Calculate expected size
    size_t num_elements = 1;
    for (auto dim : info.shape) {
        num_elements *= dim;
    }

    // Verify dtype
    if (info.dtype != "F32" && info.dtype != "F16") {
        std::cerr << "Warning: Unsupported dtype " << info.dtype << " for " << info.name << std::endl;
        return data;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return data;

    // Read header size
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Seek to tensor data
    size_t data_start = 8 + header_size + info.offset_start;
    file.seekg(data_start);

    if (info.dtype == "F32") {
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
    } else if (info.dtype == "F16") {
        // Convert F16 to F32
        std::vector<uint16_t> f16_data(num_elements);
        file.read(reinterpret_cast<char*>(f16_data.data()), num_elements * 2);

        data.resize(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            // Simple F16 to F32 conversion
            uint16_t h = f16_data[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF) << 13;

            if (exp == 0) {
                // Subnormal or zero
                data[i] = 0.0f;
            } else if (exp == 31) {
                // Inf or NaN
                uint32_t f32 = sign | 0x7F800000 | mant;
                memcpy(&data[i], &f32, 4);
            } else {
                uint32_t f32 = sign | ((exp + 112) << 23) | mant;
                memcpy(&data[i], &f32, 4);
            }
        }
    }

    return data;
}

// Compute cosine similarity using centralized SIMD implementation
float cosine_similarity(const float* a, const float* b, size_t dim) {
    return embedding::cosine_similarity(a, b, dim);
}

// Token cache: vocab index → hash + coordinates
struct TokenAtom {
    Blake3Hash hash;
    double x, y, z, m;  // 4D coordinates
    bool exists;  // Whether this token exists in DB
};
static std::unordered_map<size_t, TokenAtom> g_token_cache;

// Load or create token atoms for vocabulary
bool ensure_vocab_atoms(PGconn* conn, const std::vector<std::string>& vocab) {
    std::cerr << "Ensuring " << vocab.size() << " vocabulary tokens exist in atom table...\n";

    // First check which already exist
    size_t existing = 0;
    for (size_t i = 0; i < vocab.size(); ++i) {
        Blake3Hash hash = Blake3Hasher::hash(std::string_view(vocab[i]));

        char query[256];
        snprintf(query, sizeof(query),
            "SELECT ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom) "
            "FROM atom WHERE id = '\\x%s'::bytea",
            hash.to_hex().c_str());

        PGresult* res = PQexec(conn, query);
        if (PQntuples(res) > 0) {
            TokenAtom atom;
            atom.hash = hash;
            atom.x = std::stod(PQgetvalue(res, 0, 0));
            atom.y = std::stod(PQgetvalue(res, 0, 1));
            atom.z = std::stod(PQgetvalue(res, 0, 2));
            atom.m = std::stod(PQgetvalue(res, 0, 3));
            atom.exists = true;
            g_token_cache[i] = atom;
            existing++;
        } else {
            // Will need to create via CPE ingestion
            TokenAtom atom;
            atom.hash = hash;
            atom.exists = false;
            g_token_cache[i] = atom;
        }
        PQclear(res);
    }

    std::cerr << "  " << existing << " tokens already in DB, "
              << (vocab.size() - existing) << " need ingestion\n";

    // Batch ingest missing tokens as compositions
    std::vector<TokenData> missing_tokens;
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (!g_token_cache[i].exists) {
            TokenData token;
            token.label = vocab[i];
            token.hash = Blake3Hasher::hash(std::string_view(vocab[i]));
            token.is_atom = false;  // Vocab tokens are compositions of atoms
            // Coords will be computed from label in persist_compositions
            missing_tokens.push_back(token);
        }
    }

    if (!missing_tokens.empty()) {
        std::cerr << "Ingesting " << missing_tokens.size() << " missing vocab tokens as compositions...\n";

        // Use ProjectionPersister to insert missing tokens
        hypercube::db::PersistConfig persist_config;
        persist_config.update_existing = false;
        hypercube::db::ProjectionPersister persister(conn, persist_config);

        size_t inserted = persister.persist(missing_tokens);
        std::cerr << "Inserted " << inserted << " missing token compositions\n";

        // Update cache for newly inserted tokens
        for (const auto& token : missing_tokens) {
            // Find the index in vocab
            for (size_t i = 0; i < vocab.size(); ++i) {
                if (vocab[i] == token.label) {
                    // Check if it was inserted successfully by querying
                    char query[256];
                    snprintf(query, sizeof(query),
                        "SELECT ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom) "
                        "FROM composition WHERE id = '\\x%s'::bytea",
                        token.hash.to_hex().c_str());

                    PGresult* res = PQexec(conn, query);
                    if (PQntuples(res) > 0) {
                        TokenAtom atom;
                        atom.hash = token.hash;
                        atom.x = std::stod(PQgetvalue(res, 0, 0));
                        atom.y = std::stod(PQgetvalue(res, 0, 1));
                        atom.z = std::stod(PQgetvalue(res, 0, 2));
                        atom.m = std::stod(PQgetvalue(res, 0, 3));
                        atom.exists = true;
                        g_token_cache[i] = atom;
                        existing++;
                    }
                    PQclear(res);
                    break;
                }
            }
        }
    }

    return true;
}

// Batch insert semantic edges using COPY protocol (fast)
bool batch_insert_edges(PGconn* conn, const std::vector<SemanticEdge>& edges) {
    if (edges.empty()) return true;

    std::cerr << "Batch inserting " << edges.size() << " semantic edges...\n";

    Transaction tx(conn);

    // Create temp table for edges
    Result res = exec(conn,
        "CREATE TEMP TABLE tmp_semantic_edge ("
        "  src_id BYTEA, dst_id BYTEA, weight REAL"
        ") ON COMMIT DROP");
    if (!res.ok()) {
        std::cerr << "Failed to create temp table: " << res.error_message() << "\n";
        return false;
    }

    // COPY data in using CopyStream
    {
        CopyStream copy(conn, "COPY tmp_semantic_edge FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "COPY failed to start: " << copy.error() << "\n";
            return false;
        }

        std::string batch;
        batch.reserve(1 << 20);  // 1MB buffer

        for (const auto& e : edges) {
            copy_bytea(batch, e.source);
            copy_tab(batch);
            copy_bytea(batch, e.target);
            copy_tab(batch);
            batch += std::to_string(e.weight);
            copy_newline(batch);

            if (batch.size() > (1 << 19)) {  // 512KB chunks
                if (!copy.put(batch)) {
                    std::cerr << "COPY failed: " << copy.error() << "\n";
                    return false;
                }
                batch.clear();
            }
        }

        if (!batch.empty()) {
            if (!copy.put(batch)) {
                std::cerr << "COPY failed: " << copy.error() << "\n";
                return false;
            }
        }

        if (!copy.end()) {
            std::cerr << "COPY end failed: " << copy.error() << "\n";
            return false;
        }
    }

    // Insert edges into relation table (4-table schema)
    // Using layer=-1 and component='' as defaults for embedding similarity edges
    res = exec(conn,
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component) "
        "SELECT 'A', e.src_id, 'A', e.dst_id, 'S', e.weight, 'minilm', -1, 'embed_sim' "
        "FROM tmp_semantic_edge e "
        "WHERE EXISTS (SELECT 1 FROM atom WHERE id = e.src_id) "
        "  AND EXISTS (SELECT 1 FROM atom WHERE id = e.dst_id) "
        "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) "
        "DO UPDATE SET weight = GREATEST(relation.weight, EXCLUDED.weight), "
        "  source_count = relation.source_count + 1"
    );

    if (!res.ok()) {
        std::cerr << "Edge insert failed: " << res.error_message() << "\n";
        return false;
    }

    int inserted = cmd_tuples(res);

    tx.commit();

    std::cerr << "  Inserted " << inserted << " semantic edges\n";
    return true;
}

// Compute pairwise similarities and store as semantic edges
// Edges are stored in the unified atom table with weight in M coordinate
void compute_sparse_edges(
    const std::vector<float>& embeddings,
    size_t vocab_size,
    size_t hidden_dim,
    float threshold,
    PGconn* conn,
    const std::string& /* model_name */,
    const std::vector<std::string>& vocab  // Token strings for hash lookup
) {
    std::cerr << "Computing pairwise similarities for " << vocab_size << " tokens...\n";
    std::cerr << "Threshold: " << threshold << " (only edges >= this cosine similarity)\n";
    std::cerr << "Edges stored in unified atom table with weight in M coordinate.\n\n";

    // First ensure vocab atoms exist in DB
    ensure_vocab_atoms(conn, vocab);

    auto start = std::chrono::high_resolution_clock::now();

    std::atomic<size_t> edges_found{0};
    std::atomic<size_t> pairs_checked{0};
    std::mutex edge_mutex;

    // Collect all edges, then batch insert
    std::vector<SemanticEdge> all_edges;
    all_edges.reserve(100000);  // Pre-allocate for performance

    // Process in parallel
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<size_t> next_i{0};

    auto worker = [&]() {
        std::vector<SemanticEdge> local_edges;
        local_edges.reserve(10000);

        while (true) {
            size_t i = next_i.fetch_add(1);
            if (i >= vocab_size) break;

            // Skip if token doesn't exist in DB
            auto it_i = g_token_cache.find(i);
            if (it_i == g_token_cache.end() || !it_i->second.exists) continue;

            const float* vec_i = &embeddings[i * hidden_dim];

            for (size_t j = i + 1; j < vocab_size; j++) {
                // Skip if token doesn't exist in DB
                auto it_j = g_token_cache.find(j);
                if (it_j == g_token_cache.end() || !it_j->second.exists) continue;

                const float* vec_j = &embeddings[j * hidden_dim];

                float sim = cosine_similarity(vec_i, vec_j, hidden_dim);
                pairs_checked++;

                if (sim >= threshold) {
                    edges_found++;
                    local_edges.push_back({
                        it_i->second.hash,
                        it_j->second.hash,
                        sim
                    });
                }
            }

            // Progress every 1000 tokens
            if (i % 1000 == 0) {
                std::cerr << "\r  Processed " << i << "/" << vocab_size
                          << " tokens, found " << edges_found.load() << " edges...";
            }
        }

        // Merge local edges to global
        if (!local_edges.empty()) {
            std::lock_guard<std::mutex> lock(edge_mutex);
            all_edges.insert(all_edges.end(), local_edges.begin(), local_edges.end());
        }
    };

    for (size_t t = 0; t < num_threads; t++) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cerr << "\n\nFound " << all_edges.size() << " semantic edges above threshold.\n";

    // Batch insert all edges
    bool ok = batch_insert_edges(conn, all_edges);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cerr << "\nComplete:\n";
    std::cerr << "  Pairs checked: " << pairs_checked.load() << "\n";
    std::cerr << "  Edges found (>= " << threshold << "): " << edges_found.load() << "\n";
    std::cerr << "  Sparsity: " << (100.0 - 100.0 * edges_found.load() / std::max(pairs_checked.load(), size_t(1))) << "%\n";
    std::cerr << "  Time: " << ms << " ms\n";
    std::cerr << "  Insert status: " << (ok ? "SUCCESS" : "FAILED") << "\n";

    // Show stats from DB
    PGresult* res = PQexec(conn,
        "SELECT count(*) as total_edges, "
        "       avg(ST_M(ST_PointN(geom, 1)))::numeric(10,4) as avg_weight "
        "FROM atom WHERE depth > 0 AND children IS NOT NULL AND array_length(children, 1) = 2");
    if (PQntuples(res) > 0 && PQgetvalue(res, 0, 0)[0] != '\0') {
        std::cerr << "\nDB Statistics:\n";
        std::cerr << "  Total 2-child compositions (edges): " << PQgetvalue(res, 0, 0) << "\n";
        if (PQgetvalue(res, 0, 1)[0] != '\0') {
            std::cerr << "  Average weight (M coord): " << PQgetvalue(res, 0, 1) << "\n";
        }
    }
    PQclear(res);
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model PATH        Path to model.safetensors\n"
              << "  --vocab PATH        Path to vocab.txt (optional)\n"
              << "  --threshold FLOAT   Minimum similarity to record (default: 0.5)\n"
              << "  --tensor NAME       Tensor to use for embeddings (default: auto-detect)\n"
              << "  -d, --dbname NAME   Database name (default: hypercube)\n"
              << "  -h, --host HOST     Database host\n"
              << "  -U, --user USER     Database user\n"
              << "  --list              List tensors and exit\n"
              << "  --help              Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string vocab_path;
    std::string tensor_name;
    std::string dbname = "hypercube";
    std::string host;
    std::string user;
    float threshold = 0.5f;
    bool list_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg == "--tensor" && i + 1 < argc) {
            tensor_name = argv[++i];
        } else if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg == "--list") {
            list_only = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cerr << "=== Safetensor Embedding Extractor ===\n";
    std::cerr << "Model: " << model_path << "\n\n";

    // Parse safetensor header
    std::cerr << "Parsing safetensor header...\n";
    auto tensors = parse_safetensor_header(model_path);

    if (tensors.empty()) {
        std::cerr << "Error: No tensors found\n";
        return 1;
    }

    // List tensors
    std::cerr << "Tensors found:\n";
    for (const auto& t : tensors) {
        std::cerr << "  " << t.name << ": " << t.dtype << " [";
        for (size_t i = 0; i < t.shape.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << t.shape[i];
        }
        std::cerr << "] (" << (t.offset_end - t.offset_start) << " bytes)\n";
    }

    if (list_only) {
        return 0;
    }

    // Find embedding tensor
    TensorInfo* embedding_tensor = nullptr;
    for (auto& t : tensors) {
        if (!tensor_name.empty() && t.name == tensor_name) {
            embedding_tensor = &t;
            break;
        }
        // Auto-detect: look for [vocab_size, hidden_dim] shape
        if (t.shape.size() == 2 && t.shape[0] > 10000 && t.shape[1] > 100) {
            if (t.name.find("embed") != std::string::npos ||
                t.name.find("token") != std::string::npos ||
                t.name.find("word") != std::string::npos) {
                embedding_tensor = &t;
                break;
            }
        }
    }

    if (!embedding_tensor) {
        std::cerr << "\nError: No embedding tensor found. Use --tensor to specify.\n";
        return 1;
    }

    std::cerr << "\nUsing embedding tensor: " << embedding_tensor->name << "\n";
    std::cerr << "Shape: [" << embedding_tensor->shape[0] << ", " << embedding_tensor->shape[1] << "]\n";

    // Load embeddings
    std::cerr << "Loading embeddings...\n";
    auto embeddings = load_embedding_tensor(model_path, *embedding_tensor);

    if (embeddings.empty()) {
        std::cerr << "Error: Failed to load embeddings\n";
        return 1;
    }

    std::cerr << "Loaded " << embeddings.size() << " floats\n";

    // Connect to database (PGPASSWORD env var is picked up automatically by libpq)
    std::string conninfo = "dbname=" + dbname;
    if (!host.empty()) conninfo += " host=" + host;
    if (!user.empty()) conninfo += " user=" + user;
    
    // libpq reads PGPASSWORD from environment automatically
    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }

    // Ensure metadata table exists (the only extra table beyond atom/relation)
    PQexec(conn, R"(
        CREATE TABLE IF NOT EXISTS metadata (
            ref TEXT PRIMARY KEY,
            comp_id bytea NOT NULL,
            ref_type TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    )");

    // Extract model name from path
    std::string model_name = model_path;
    auto slash = model_name.rfind('/');
    if (slash != std::string::npos) {
        model_name = model_name.substr(slash + 1);
    }
    auto dot = model_name.rfind('.');
    if (dot != std::string::npos) {
        model_name = model_name.substr(0, dot);
    }

    // Load vocabulary file if provided
    std::vector<std::string> vocab;
    if (!vocab_path.empty()) {
        std::cerr << "\nLoading vocabulary from: " << vocab_path << "\n";
        std::ifstream vocab_file(vocab_path);
        if (vocab_file.is_open()) {
            std::string line;
            while (std::getline(vocab_file, line)) {
                vocab.push_back(line);
            }
            std::cerr << "Loaded " << vocab.size() << " tokens\n";
        } else {
            std::cerr << "Warning: Could not open vocab file\n";
        }
    }

    // If no vocab loaded, generate placeholder tokens
    if (vocab.empty()) {
        std::cerr << "No vocab file - using token indices as placeholders\n";
        for (size_t i = 0; i < static_cast<size_t>(embedding_tensor->shape[0]); i++) {
            vocab.push_back("token_" + std::to_string(i));
        }
    }

    // Compute sparse edges and store as relations
    // Edges are stored in the unified Atom/Relation Merkle DAG
    // Weight is encoded in the M coordinate of the edge centroid
    compute_sparse_edges(
        embeddings,
        static_cast<size_t>(embedding_tensor->shape[0]),
        static_cast<size_t>(embedding_tensor->shape[1]),
        threshold,
        conn,
        model_name,
        vocab
    );

    // Show final statistics - edges are now relations with weight in M coordinate
    PGresult* res = PQexec(conn,
        "SELECT count(*) as edge_relations, "
        "avg(ST_M(coords))::numeric(10,4) as avg_weight, "
        "min(ST_M(coords))::numeric(10,4) as min_weight, "
        "max(ST_M(coords))::numeric(10,4) as max_weight "
        "FROM relation WHERE child_count = 2 AND ST_M(coords) > 0");
    if (PQntuples(res) > 0 && PQgetvalue(res, 0, 0)[0] != '\0') {
        std::cerr << "\nEdge relations in Merkle DAG:\n";
        std::cerr << "  Total edge relations: " << PQgetvalue(res, 0, 0) << "\n";
        std::cerr << "  Avg weight (M coord): " << PQgetvalue(res, 0, 1) << "\n";
        std::cerr << "  Weight range: [" << PQgetvalue(res, 0, 2) << ", "
                  << PQgetvalue(res, 0, 3) << "]\n";
    }
    PQclear(res);

    std::cerr << "\nEdges stored as relations with Hilbert-indexed centroids.\n";

    PQfinish(conn);
    return 0;
}
