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

#include "hypercube/backend.hpp"

// Forward declarations for command modules
namespace hypercube::cli {
    int cmd_ingest(int argc, char* argv[]);
    int cmd_query(int argc, char* argv[]);
    int cmd_stats(int argc, char* argv[]);
    int cmd_test(int argc, char* argv[]);
    int cmd_backend(int argc, char* argv[]);
    int cmd_version(int argc, char* argv[]);
    int cmd_help(int argc, char* argv[]);
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

int cmd_help(int argc, char* argv[]) {
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

int cmd_backend(int argc, char* argv[]) {
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

int cmd_version(int argc, char* argv[]) {
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
    std::cerr << "Ingest command - parsing arguments...\n";
    
    std::string model_name;
    std::string model_path;
    int k_neighbors = 15;
    float threshold = 0.3f;
    
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--name") && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((arg == "-k" || arg == "--neighbors") && i + 1 < argc) {
            k_neighbors = std::stoi(argv[++i]);
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
        std::cerr << "  -k, --neighbors <k>     k-NN neighbors (default: 15)\n";
        std::cerr << "  -t, --threshold <t>     Similarity threshold (default: 0.3)\n";
        return 1;
    }
    
    std::cout << "Ingesting: " << model_path << "\n";
    std::cout << "Model name: " << (model_name.empty() ? "(auto)" : model_name) << "\n";
    std::cout << "k-neighbors: " << k_neighbors << "\n";
    std::cout << "threshold: " << threshold << "\n";
    std::cout << "Database: " << build_conninfo() << "\n";
    
    // TODO: Call actual ingest function
    std::cerr << "ERROR: Ingest not yet integrated. Use ingest_safetensor_universal directly.\n";
    return 1;
}

int cmd_query(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: hypercube query <query_string> [options]\n";
        return 1;
    }
    
    std::string query = argv[0];
    std::cout << "Query: " << query << "\n";
    std::cout << "Database: " << build_conninfo() << "\n";
    
    // TODO: Call actual query function
    std::cerr << "ERROR: Query not yet integrated.\n";
    return 1;
}

int cmd_stats(int argc, char* argv[]) {
    std::cout << "Database: " << build_conninfo() << "\n";
    
    // TODO: Call actual stats function
    std::cerr << "ERROR: Stats not yet integrated.\n";
    return 1;
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
    
    // TODO: Call actual test runner
    std::cerr << "ERROR: Test runner not yet integrated.\n";
    return 1;
}

}  // namespace hypercube::cli

// =============================================================================
// Main Entry Point
// =============================================================================

int parse_global_options(int& argc, char**& argv) {
    // Get password from environment
    if (const char* pw = std::getenv("PGPASSWORD")) {
        g_options.password = pw;
    }
    
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
