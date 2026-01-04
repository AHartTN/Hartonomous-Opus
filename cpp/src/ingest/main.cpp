/**
 * Unified Ingester - N-ary token-aware ingestion
 * 
 * Single tool for ingesting content into the Hypercube:
 * 1. Scan files to extract unique codepoints
 * 2. Load only needed atoms from database
 * 3. Build Merkle DAG via N-ary compositions (NOT binary trees)
 * 4. Batch insert compositions
 * 
 * Compositions are built at natural token boundaries:
 * - "the" = ONE composition with children [t, h, e]
 * - Sentences = composition of word compositions
 * - Documents = hierarchical composition of all content
 */

#include "hypercube/util/utf8.hpp"
#include "hypercube/db/atom_cache.hpp"
#include "hypercube/db/insert.hpp"
#include "hypercube/db/connection.hpp"
#include "hypercube/ingest/cpe.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <set>
#include <iomanip>

namespace fs = std::filesystem;
using namespace hypercube;

namespace {

bool is_text_file(const fs::path& path) {
    static const std::set<std::string> exts = {
        ".txt", ".md", ".json", ".py", ".cpp", ".hpp", ".c", ".h",
        ".js", ".ts", ".yaml", ".yml", ".xml", ".html", ".css",
        ".sql", ".sh", ".rs", ".go", ".java", ".rb"
    };
    return exts.count(path.extension().string()) > 0;
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return "";
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <path>\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  --help               Show this help\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    db::ConnectionConfig config;
    std::string target;
    
    for (int i = 1; i < argc; ++i) {
        if (config.parse_arg(argc, argv, i)) {
            continue;
        }
        std::string arg = argv[i];
        if (arg == "--help") {
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
    
    fs::path path(target);
    if (!fs::exists(path)) {
        std::cerr << "Not found: " << target << "\n";
        return 1;
    }
    
    std::cerr << "=== Hypercube Ingester (N-ary) ===\n";
    std::cerr << "Target: " << target << "\n\n";
    
    // Phase 1: Collect files and extract unique codepoints
    std::unordered_set<uint32_t> all_codepoints;
    std::vector<std::pair<fs::path, std::string>> files;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && is_text_file(entry.path())) {
                std::string content = read_file(entry.path());
                if (!content.empty()) {
                    auto unique = util::extract_unique_codepoints(content);
                    all_codepoints.insert(unique.begin(), unique.end());
                    files.emplace_back(entry.path(), std::move(content));
                }
            }
        }
    } else {
        std::string content = read_file(path);
        if (!content.empty()) {
            all_codepoints = util::extract_unique_codepoints(content);
            files.emplace_back(path, std::move(content));
        }
    }
    
    std::cerr << "[SCAN] " << files.size() << " files, " 
              << all_codepoints.size() << " unique codepoints\n";
    
    if (files.empty()) {
        std::cerr << "No content to process\n";
        return 1;
    }
    
    // Phase 2: Connect and load atoms
    db::Connection conn(config);
    if (!conn.ok()) {
        std::cerr << "Connection failed: " << conn.error() << "\n";
        return 1;
    }
    
    std::unordered_map<uint32_t, db::AtomInfo> atom_cache;
    if (!db::load_atoms_for_codepoints(conn.get(), all_codepoints, atom_cache)) {
        return 1;
    }
    
    // Phase 3: Process files with N-ary token-aware ingestion
    std::vector<ingest::CompositionRecord> all_comps;
    std::unordered_map<std::string, ingest::CompositionRecord> comp_cache;
    all_comps.reserve(1000000);
    
    for (const auto& [fpath, content] : files) {
        size_t before = all_comps.size();
        auto start_file = std::chrono::high_resolution_clock::now();
        
        // Decode UTF-8 to codepoints
        auto codepoints = util::decode_utf8(content);
        
        if (codepoints.size() > 1) {
            // Use new N-ary token-aware ingestion
            // "the" = ONE composition [t, h, e], NOT binary tree
            Blake3Hash root = ingest::ingest_text(codepoints, atom_cache, all_comps, comp_cache);
            
            if (files.size() == 1 && !root.is_zero()) {
                std::cout << root.to_hex() << "\n";
            }
        }
        
        auto end_file = std::chrono::high_resolution_clock::now();
        double file_secs = std::chrono::duration<double>(end_file - start_file).count();
        
        std::cerr << "[OK] " << content.size() << " bytes → " 
                  << (all_comps.size() - before) << " compositions (" 
                  << std::fixed << std::setprecision(2) << file_secs << "s)\n";
    }
    
    // Phase 4: Insert to database
    if (!all_comps.empty()) {
        db::insert_compositions(conn.get(), all_comps);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    
    // Stats using SQL function
    PGresult* res = PQexec(conn.get(), "SELECT * FROM db_stats()");
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        std::cerr << "\nDatabase: " << PQgetvalue(res, 0, 0) << " atoms, "
                  << PQgetvalue(res, 0, 1) << " compositions, "
                  << "depth " << PQgetvalue(res, 0, 2) << ", "
                  << PQgetvalue(res, 0, 3) << "\n";
    }
    PQclear(res);
    
    std::cerr << "Completed in " << std::fixed << std::setprecision(2) << secs << "s\n";
    
    return 0;
}
