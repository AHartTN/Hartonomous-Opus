// =============================================================================
// hypercube CLI - Unified Command-Line Interface
// =============================================================================
//
// Single entry point for all Hypercube operations.
//
// Usage:
//   hypercube <command> [options]
//
// Commands:
//   ingest      Ingest safetensor models into the database
//   query       Query the semantic graph
//   stats       Show database statistics
//   test        Run test suite
//   backend     Show backend configuration and capabilities
//   version     Show version information
//
// Examples:
//   hypercube ingest -d hypercube "D:\Models\all-MiniLM-L6-v2"
//   hypercube query "attention mechanism"
//   hypercube stats --verbose
//   hypercube test --cpp --sql
//   hypercube backend
//
// =============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cstring>
#include <filesystem>
#include <chrono>

#include "hypercube/backend.hpp"
#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/parsing.hpp"
#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/ingest/model_manifest.hpp"
#include "hypercube/ingest/multimodal_extraction.hpp"
#include "hypercube/ingest/metadata.hpp"
#include "hypercube/ingest/metadata_db.hpp"

// Forward declarations for command modules
namespace hypercube::cli {
    int cmd_ingest(int argc, char* argv[]);
    int cmd_query(int argc, char* argv[]);
    int cmd_stats(int argc, char* argv[]);
    int cmd_test(int argc, char* argv[]);
    int cmd_backend(int argc, char* argv[]);
    int cmd_version(int argc, char* argv[]);
    int cmd_help(int argc, char* argv[]);

    // Integrated ingest functionality
    int perform_ingest(const std::string& model_path, const hypercube::ingest::IngestConfig& config);
}

// =============================================================================
// Version Info
// =============================================================================

#define HYPERCUBE_VERSION_MAJOR 1
#define HYPERCUBE_VERSION_MINOR 0
#define HYPERCUBE_VERSION_PATCH 0
#define HYPERCUBE_VERSION_STRING "1.0.0"

// =============================================================================
// Command Registry
// =============================================================================

struct Command {
    const char* name;
    const char* description;
    int (*handler)(int argc, char* argv[]);
};

static const Command g_commands[] = {
    {"ingest",  "Ingest safetensor models into the database", hypercube::cli::cmd_ingest},
    {"query",   "Query the semantic graph", hypercube::cli::cmd_query},
    {"stats",   "Show database statistics", hypercube::cli::cmd_stats},
    {"test",    "Run test suite (C++ and SQL)", hypercube::cli::cmd_test},
    {"backend", "Show backend configuration and capabilities", hypercube::cli::cmd_backend},
    {"version", "Show version information", hypercube::cli::cmd_version},
    {"help",    "Show this help message", hypercube::cli::cmd_help},
    {nullptr, nullptr, nullptr}
};

// =============================================================================
// Global Options
// =============================================================================

struct GlobalOptions {
    std::string database = "hypercube";
    std::string user = "postgres";
    std::string host = "localhost";
    std::string port = "5432";
    std::string password;  // From PGPASSWORD env
    bool verbose = false;
    bool quiet = false;
};

static GlobalOptions g_options;

std::string build_conninfo() {
    std::string conninfo = "dbname=" + g_options.database +
                           " user=" + g_options.user +
                           " host=" + g_options.host +
                           " port=" + g_options.port;
    if (!g_options.password.empty()) {
        conninfo += " password=" + g_options.password;
    }
    return conninfo;
}

// =============================================================================
// Help Command
// =============================================================================

namespace hypercube::cli {

int cmd_help([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "Hypercube - Semantic Graph Database Toolkit\n";
    std::cout << "Version " << HYPERCUBE_VERSION_STRING << "\n\n";
    std::cout << "Usage: hypercube <command> [options]\n\n";
    std::cout << "Commands:\n";
    
    for (const Command* cmd = g_commands; cmd->name; ++cmd) {
        std::cout << "  " << cmd->name;
        for (size_t i = strlen(cmd->name); i < 12; ++i) std::cout << ' ';
        std::cout << cmd->description << "\n";
    }
    
    std::cout << "\nGlobal Options:\n";
    std::cout << "  -d, --database <name>   Database name (default: hypercube)\n";
    std::cout << "  -U, --user <user>       Database user (default: postgres)\n";
    std::cout << "  -h, --host <host>       Database host (default: localhost)\n";
    std::cout << "  -p, --port <port>       Database port (default: 5432)\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  -q, --quiet             Suppress non-essential output\n";
    std::cout << "\nEnvironment:\n";
    std::cout << "  PGPASSWORD              PostgreSQL password\n";
    std::cout << "  HC_MODEL_PATHS          Colon-separated model search paths\n";
    std::cout << "\nExamples:\n";
    std::cout << "  hypercube ingest -n \"my-model\" /path/to/model.safetensors\n";
    std::cout << "  hypercube query \"transformer attention\"\n";
    std::cout << "  hypercube test --cpp --sql\n";
    std::cout << "  hypercube backend\n";
    
    return 0;
}

// =============================================================================
// Backend Command
// =============================================================================

int cmd_backend([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    auto info = hypercube::Backend::detect();
    std::cout << info.summary();
    
    // Additional hardware details if verbose
    if (g_options.verbose) {
        std::cout << "\n=== Detailed CPU Flags ===\n";
        std::cout << "Family: " << info.cpu.family << "\n";
        std::cout << "Model: " << info.cpu.model << "\n";
        std::cout << "Stepping: " << info.cpu.stepping << "\n";
    }
    
    return 0;
}

// =============================================================================
// Version Command
// =============================================================================

int cmd_version([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "Hypercube " << HYPERCUBE_VERSION_STRING << "\n";
    
    auto info = hypercube::Backend::info();
    std::cout << "Build: " << info.build_type << " (" << info.compiler << ")\n";
    std::cout << "SIMD: " << simd_level_name(info.simd_level) << "\n";
    std::cout << "Eigensolver: " << eigensolver_name(info.eigensolver) << "\n";
    std::cout << "k-NN: " << knn_backend_name(info.knn) << "\n";
    
    return 0;
}

// =============================================================================
// Stub Commands (to be implemented in separate modules)
// =============================================================================

int cmd_ingest(int argc, char* argv[]) {
    std::string model_name;
    std::string model_path;
    float threshold = 0.5f;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--name") && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg[0] != '-' && model_path.empty()) {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Usage: hypercube ingest [options] <model_path>\n";
        std::cerr << "Options:\n";
        std::cerr << "  -n, --name <name>       Model name for database\n";
        std::cerr << "  -t, --threshold <t>     Similarity threshold (default: 0.5)\n";
        return 1;
    }

    std::cout << "Ingesting: " << model_path << "\n";
    std::cout << "Model name: " << (model_name.empty() ? "(auto)" : model_name) << "\n";
    std::cout << "Threshold: " << threshold << "\n";
    std::cout << "Database: " << build_conninfo() << "\n";

    // Build ingest config
    hypercube::ingest::IngestConfig config;
    config.conninfo = build_conninfo();
    config.model_name = model_name;
    config.weight_threshold = threshold;
    config.verbose = g_options.verbose;

    return perform_ingest(model_path, config);
}

int cmd_query(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: hypercube query <query_string> [options]\n";
        return 1;
    }
    
    std::string query = argv[0];
    std::cout << "Query: " << query << "\n";
    std::cout << "Database: " << build_conninfo() << "\n";

    // Connect to database and perform semantic query
    std::string conninfo = build_conninfo();
    PGconn* conn = PQconnectdb(conninfo.c_str());

    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Database connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }

    // Query for compositions matching the search term
    std::string sql = "SELECT id, label, atom_count FROM composition WHERE label ILIKE $1 LIMIT 10";
    std::string pattern = "%" + query + "%";

    const char* params[1] = {pattern.c_str()};
    PGresult* res = PQexecParams(conn, sql.c_str(), 1, nullptr, params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Query failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQfinish(conn);
        return 1;
    }

    int nrows = PQntuples(res);
    std::cout << "\nFound " << nrows << " results:\n";
    std::cout << std::string(80, '-') << "\n";

    for (int i = 0; i < nrows; ++i) {
        const char* id = PQgetvalue(res, i, 0);
        const char* label = PQgetvalue(res, i, 1);
        const char* atom_count = PQgetvalue(res, i, 2);

        std::cout << label << " (atoms: " << atom_count << ")\n";
        std::cout << "  ID: " << id << "\n";
    }

    PQclear(res);
    PQfinish(conn);
    return 0;
}

int cmd_stats([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "Database: " << build_conninfo() << "\n";

    // Connect to database and retrieve statistics
    std::string conninfo = build_conninfo();
    PGconn* conn = PQconnectdb(conninfo.c_str());

    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Database connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }

    // Query database statistics
    std::string sql = R"SQL(
        SELECT
            (SELECT COUNT(*) FROM atom) as atoms,
            (SELECT COUNT(*) FROM composition) as compositions,
            (SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL) as compositions_with_centroid,
            (SELECT COUNT(*) FROM relation_consensus) as relations,
            (SELECT COUNT(*) FROM ml_experiment) as experiments,
            (SELECT COUNT(*) FROM ml_run) as runs,
            (SELECT COUNT(*) FROM ml_model_version) as model_versions
    )SQL";

    PGresult* res = PQexec(conn, sql.c_str());

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Stats query failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQfinish(conn);
        return 1;
    }

    std::cout << "\n=== Hypercube Database Statistics ===\n\n";
    std::cout << "Core Entities:\n";
    std::cout << "  Atoms:                       " << PQgetvalue(res, 0, 0) << "\n";
    std::cout << "  Compositions:                " << PQgetvalue(res, 0, 1) << "\n";
    std::cout << "  Compositions with centroid:  " << PQgetvalue(res, 0, 2) << "\n";
    std::cout << "  Relations:                   " << PQgetvalue(res, 0, 3) << "\n";

    std::cout << "\nML Lifecycle:\n";
    std::cout << "  Experiments:                 " << PQgetvalue(res, 0, 4) << "\n";
    std::cout << "  Training Runs:               " << PQgetvalue(res, 0, 5) << "\n";
    std::cout << "  Model Versions:              " << PQgetvalue(res, 0, 6) << "\n";

    PQclear(res);
    PQfinish(conn);
    return 0;
}

int cmd_test(int argc, char* argv[]) {
    bool run_cpp = false;
    bool run_sql = false;
    bool run_all = true;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpp") { run_cpp = true; run_all = false; }
        else if (arg == "--sql") { run_sql = true; run_all = false; }
    }

    if (run_all) {
        run_cpp = run_sql = true;
    }

    std::cout << "Running tests:\n";
    if (run_cpp) std::cout << "  - C++ unit tests\n";
    if (run_sql) std::cout << "  - SQL integration tests\n";

    int failures = 0;

    // C++ unit tests
    if (run_cpp) {
        std::cout << "\n[C++ Tests]\n";
        std::cout << "Running basic sanity checks...\n";

        // Test 1: BLAKE3 hashing
        try {
            Blake3Hasher::hash("test");
            std::cout << "  ✓ BLAKE3 hashing works\n";
        } catch (...) {
            std::cerr << "  ✗ BLAKE3 hashing failed\n";
            failures++;
        }

        // Test 2: Hilbert curve
        try {
            Point4D point{5U, 5U, 5U, 5U};
            HilbertIndex idx = HilbertCurve::coords_to_index(point);
            std::cout << "  ✓ Hilbert curve encoding works (idx=" << idx.hi << "," << idx.lo << ")\n";
        } catch (...) {
            std::cerr << "  ✗ Hilbert curve failed\n";
            failures++;
        }

        // Test 3: AtomCalculator
        try {
            AtomRecord atom = AtomCalculator::compute_atom(65);  // 'A'
            std::cout << "  ✓ AtomCalculator works (x=" << atom.coords.x << ")\n";
        } catch (...) {
            std::cerr << "  ✗ AtomCalculator failed\n";
            failures++;
        }
    }

    // SQL integration tests
    if (run_sql) {
        std::cout << "\n[SQL Tests]\n";
        std::cout << "Testing database connection...\n";

        std::string conninfo = build_conninfo();
        PGconn* conn = PQconnectdb(conninfo.c_str());

        if (PQstatus(conn) != CONNECTION_OK) {
            std::cerr << "  ✗ Database connection failed: " << PQerrorMessage(conn) << "\n";
            PQfinish(conn);
            failures++;
        } else {
            std::cout << "  ✓ Database connection successful\n";

            // Test table existence
            const char* tables[] = {"atom", "composition", "relation_consensus",
                                   "ml_experiment", "ml_run", "ml_model_version"};

            for (const char* table : tables) {
                std::string sql = "SELECT COUNT(*) FROM " + std::string(table);
                PGresult* res = PQexec(conn, sql.c_str());

                if (PQresultStatus(res) == PGRES_TUPLES_OK) {
                    std::cout << "  ✓ Table '" << table << "' exists (" << PQgetvalue(res, 0, 0) << " rows)\n";
                } else {
                    std::cerr << "  ✗ Table '" << table << "' check failed\n";
                    failures++;
                }

                PQclear(res);
            }

            PQfinish(conn);
        }
    }

    std::cout << "\n";
    if (failures == 0) {
        std::cout << "All tests passed! ✓\n";
        return 0;
    } else {
        std::cerr << failures << " test(s) failed ✗\n";
        return 1;
    }
}

// Integrated ingest functionality - extracted from ingest_safetensor_modular.cpp
int perform_ingest(const std::string& model_dir, const hypercube::ingest::IngestConfig& config) {
    namespace fs = std::filesystem;
    using namespace hypercube::ingest;

    fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << model_dir << "\n";
        return 1;
    }

    // Create ingest context
    IngestContext ctx;
    ctx.model_prefix = config.model_name + ":";

    std::cerr << "=== Integrated Safetensor Ingester ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "Threshold: " << config.weight_threshold << "\n\n";

    // Parse model manifest
    std::cerr << "[0] Parsing model manifest...\n";
    ModelManifest manifest = parse_model_manifest(dir);
    manifest.model_name = config.model_name;
    manifest.print_summary();
    ctx.manifest = manifest;

    auto total_start = std::chrono::steady_clock::now();

    // Find model files
    fs::path vocab_path, tokenizer_path, index_path;
    std::vector<fs::path> safetensor_files;

    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        std::string path_str = entry.path().string();

        // Skip hidden directories and cache folders
        if (path_str.find("\\.") != std::string::npos ||
            path_str.find("/.") != std::string::npos ||
            path_str.find(".cache") != std::string::npos) {
            continue;
        }

        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "tokenizer.json") tokenizer_path = entry.path();
        else if (name == "model.safetensors.index.json") index_path = entry.path();
        else if (name.size() >= 12 && name.substr(name.size() - 12) == ".safetensors") {
            safetensor_files.push_back(entry.path());
        }
    }

    // Parse tokenizer
    if (!tokenizer_path.empty()) {
        std::cerr << "[1] Parsing tokenizer: " << tokenizer_path << "\n";
        parse_tokenizer(ctx, tokenizer_path);
    }

    // Parse vocab.txt
    if (!vocab_path.empty()) {
        std::cerr << "[2] Parsing vocab: " << vocab_path << "\n";
        parse_vocab(ctx, vocab_path);
    }

    // Parse model metadata
    std::cerr << "\n[2.5] Parsing model metadata...\n";
    metadata::ModelMetadata model_meta;
    metadata::parse_model_metadata(dir, model_meta);

    if (!model_meta.vocab_tokens.empty() && ctx.vocab_tokens.empty()) {
        ctx.vocab_tokens.resize(model_meta.vocab_tokens.size());
        for (size_t i = 0; i < model_meta.vocab_tokens.size(); ++i) {
            const auto& vt = model_meta.vocab_tokens[i];
            TokenInfo info;
            info.text = vt.text;
            info.comp = AtomCalculator::compute_vocab_token(vt.text);
            ctx.vocab_tokens[i] = std::move(info);
            ctx.token_to_idx[vt.text] = i;
        }
        std::cerr << "[VOCAB] Transferred " << ctx.vocab_tokens.size() << " token compositions to context\n";
    }

    // Parse model tensors
    if (!index_path.empty()) {
        std::cerr << "[3] Parsing sharded model index: " << index_path << "\n";
        parse_model_index(ctx, index_path);
    } else if (!safetensor_files.empty()) {
        std::cerr << "[3] Parsing " << safetensor_files.size() << " safetensor files...\n";
        for (const auto& f : safetensor_files) {
            std::cerr << "  Parsing: " << f << "\n";
            if (!parse_safetensor_header(ctx, f)) {
                std::cerr << "  [ERROR] Failed to parse: " << f << "\n";
                return 1;
            }
        }
    }

    if (ctx.tensors.empty()) {
        std::cerr << "[ERROR] No tensors found!\n";
        return 1;
    }

    std::cerr << "[INFO] Found " << ctx.tensors.size() << " tensors\n";

    // Categorize tensors
    if (ctx.manifest.has_value()) {
        std::cerr << "[3.1] Categorizing tensors for extraction...\n";
        for (const auto& [name, meta] : ctx.tensors) {
            ctx.manifest->categorize_tensor(name, meta.shape, meta.dtype);
        }
        std::cerr << "[INFO] Created " << ctx.manifest->extraction_plans.size() << " extraction plans\n";
    }

    // Connect to database
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }

    // Insert model metadata
    std::cerr << "\n[3.5] Inserting model metadata...\n";
    if (!model_meta.model_name.empty()) {
        metadata::insert_model_metadata(conn, model_meta);
    }

    // Build composition hierarchy
    std::cerr << "\n[4] Building tensor name hierarchy...\n";
    if (!ctx.tensors.empty()) {
        hypercube::ingest::db::insert_tensor_hierarchy(conn, ctx, config);
    }

    // Insert vocab token compositions
    if (!ctx.vocab_tokens.empty()) {
        std::cerr << "\n[5] Inserting token compositions...\n";
        hypercube::ingest::db::insert_compositions(conn, ctx);
    }

    // Project embeddings
    std::cerr << "\n[5.5] Projecting token embeddings to 4D...\n";
    if (!ctx.vocab_tokens.empty()) {
        hypercube::ingest::db::project_and_update_embeddings(conn, ctx, config);
    }

    // Compute centroids
    std::cerr << "\n[6] Computing composition centroids...\n";
    {
        hypercube::db::Result res = hypercube::db::exec(conn, "SELECT recompute_composition_centroids()");
        if (!res.ok()) {
            std::cerr << "[CENTROID] Failed: " << res.error_message() << "\n";
        }
    }

    // Extract semantic relations
    std::cerr << "\n[7] Extracting semantic relations...\n";
    if (!ctx.tensors.empty()) {
        hypercube::ingest::db::extract_all_semantic_relations(conn, ctx, config);
    }

    // Extract weight-based relations (removed - duplicate of semantic extraction)
    // std::cerr << "\n[8] Extracting weight-based relations...\n";
    // hypercube::ingest::db::insert_attention_relations(conn, ctx, config);

    // Extract multimodal structures
    std::cerr << "\n[9] Extracting multimodal structures...\n";
    if (ctx.manifest.has_value()) {
        size_t multimodal_relations = extract_multimodal_structures(conn, ctx, *ctx.manifest);
        std::cerr << "[MULTIMODAL] Extracted " << multimodal_relations << " relations\n";
    }

    PQfinish(conn);

    auto total_end = std::chrono::steady_clock::now();
    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();

    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total time: " << total_secs << " seconds\n";
    std::cerr << "Tensors: " << ctx.tensors.size() << "\n";
    std::cerr << "BPE merges: " << ctx.bpe_merges.size() << "\n";
    std::cerr << "Vocab: " << ctx.vocab_tokens.size() << " tokens\n";

    return 0;
}

}  // namespace hypercube::cli

// =============================================================================
// Main Entry Point
// =============================================================================

int parse_global_options(int& argc, char**& argv) {
    // Get password from environment
#if defined(_WIN32)
    char* pw = nullptr;
    size_t len;
    if (_dupenv_s(&pw, &len, "PGPASSWORD") == 0 && pw != nullptr) {
        g_options.password = pw;
        free(pw);
    }
#else
    if (const char* pw = std::getenv("PGPASSWORD")) {
        g_options.password = pw;
    }
#endif
    
    int i = 1;  // Skip program name
    while (i < argc) {
        std::string arg = argv[i];
        
        if ((arg == "-d" || arg == "--database") && i + 1 < argc) {
            g_options.database = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            g_options.user = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            g_options.host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            g_options.port = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            g_options.verbose = true;
        } else if (arg == "-q" || arg == "--quiet") {
            g_options.quiet = true;
        } else if (arg[0] != '-') {
            // First non-option is the command
            break;
        } else {
            // Unknown global option, let command handle it
            break;
        }
        ++i;
    }
    
    // Shift argv to point to command
    argc -= i;
    argv += i;
    return 0;
}

int main(int argc, char* argv[]) {
    // Parse global options and find command
    parse_global_options(argc, argv);
    
    if (argc < 1) {
        hypercube::cli::cmd_help(0, nullptr);
        return 1;
    }
    
    const char* cmd_name = argv[0];
    ++argv;
    --argc;
    
    // Find and execute command
    for (const Command* cmd = g_commands; cmd->name; ++cmd) {
        if (strcmp(cmd->name, cmd_name) == 0) {
            return cmd->handler(argc, argv);
        }
    }
    
    std::cerr << "Unknown command: " << cmd_name << "\n";
    std::cerr << "Run 'hypercube help' for usage.\n";
    return 1;
}
