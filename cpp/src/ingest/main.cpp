/**
 * Universal Substrate Ingester
 * 
 * Single tool for ingesting any content into the Hypercube:
 * 1. Scan files to extract integer sequence (codepoints, bytes, etc.)
 * 2. Load only needed atoms from database
 * 3. Sliding window pattern discovery (like BPE/Sequitur)
 * 4. Batch insert compositions
 * 
 * The substrate is COMPLETELY AGNOSTIC:
 * - "Hello world" = [H,e,l,l,o, ,w,o,r,l,d]
 * - "public class" = [p,u,b,l,i,c, ,c,l,a,s,s]
 * - "0.987" = [0,.,9,8,7]
 * 
 * No linguistic rules. No language detection. Pure pattern discovery.
 */

#include "hypercube/util/utf8.hpp"
#include "hypercube/db/atom_cache.hpp"
#include "hypercube/db/insert.hpp"
#include "hypercube/db/connection.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/ingest/pmi_contraction.hpp"

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
    
    std::cerr << "=== Hypercube Universal Ingester ===\n";
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
    
    // Phase 2: PMI-Based Geometric Contraction - ZERO DATABASE until final insert
    // Uses Pointwise Mutual Information for cohesion, geodesic midpoints for positioning
    ingest::PMIIngester ingester(0, 0.0);  // Auto-detect threads, PMI threshold 0
    std::vector<ingest::CompositionRecord> all_comps;
    all_comps.reserve(1000000);
    
    for (const auto& [fpath, content] : files) {
        auto start_file = std::chrono::high_resolution_clock::now();
        
        // Parallel CPE ingestion
        auto roots = ingester.ingest(content, all_comps);
        
        if (files.size() == 1 && !roots.empty() && !roots[0].is_zero()) {
            std::cout << roots[0].to_hex() << "\n";
        }
        
        auto end_file = std::chrono::high_resolution_clock::now();
        double file_secs = std::chrono::duration<double>(end_file - start_file).count();
        
        std::cerr << "[OK] " << content.size() << " bytes → " 
                  << all_comps.size() << " compositions (" 
                  << std::fixed << std::setprecision(2) << file_secs << "s)\n";
    }
    
    // Phase 3: Connect and insert (ONLY database operation)
    if (!all_comps.empty()) {
        std::cerr << "[DB] Connecting...\n";
        db::Connection conn(config);
        if (!conn.ok()) {
            std::cerr << "Connection failed: " << conn.error() << "\n";
            return 1;
        }
        db::insert_new_compositions(conn.get(), all_comps);
        
        // Stats using SQL function
        db::Result res = db::exec(conn.get(), "SELECT * FROM db_stats()");
        if (res.ok() && res.ntuples() > 0) {
            std::cerr << "\nDatabase: " << res.str(0, 0) << " atoms, "
                      << res.str(0, 1) << " compositions, "
                      << "depth " << res.str(0, 2) << ", "
                      << res.str(0, 3) << "\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    
    std::cerr << "Completed in " << std::fixed << std::setprecision(2) << secs << "s\n";
    
    return 0;
}
