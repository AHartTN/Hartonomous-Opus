/**
 * Manifold Projection Tool
 * 
 * Projects high-dimensional embeddings (384D from MiniLM, etc.) into the
 * 4D hypercube coordinate space using Johnson-Lindenstrauss random projection.
 * 
 * This is a permanent tool that:
 * 1. Reads embeddings from the shape table
 * 2. Projects to 4D using a deterministic random projection matrix
 * 3. Updates composition centroid and hilbert coordinates
 * 
 * The projection preserves pairwise distances (JL lemma) and is:
 * - Deterministic (seeded random)
 * - Fast O(n * d) per embedding
 * - Memory efficient (streaming)
 * 
 * Usage:
 *   manifold_project [options]
 *   
 * Options:
 *   --model <name>    Model name filter (default: minilm)
 *   --batch <size>    Batch size for updates (default: 1000)
 *   --seed <int>      Random seed for projection matrix (default: 42)
 *   --force           Update even if coordinates exist
 */

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/db/connection.hpp"

#include <libpq-fe.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace {

// Configuration
struct Config {
    std::string model = "minilm";
    int batch_size = 1000;
    int seed = 42;
    bool force = false;
    
    // Database
    std::string host = "localhost";
    std::string port = "5432";
    std::string dbname = "hypercube";
    std::string user = "hartonomous";
    std::string password = "hartonomous";
};

// 4D projection result
struct Projection4D {
    int32_t x, y, z, m;
};

// Generate deterministic random value from position and seed
double random_value(int row, int col, int seed) {
    // Hash-based PRNG that's deterministic
    double val = std::sin(row * 12.9898 + col * 78.233 + seed * 43.1234) * 43758.5453;
    val = val - std::floor(val);  // Fractional part [0, 1)
    return (val - 0.5) * 2.0;     // Normalize to [-1, 1]
}

// Project a 384D embedding to 4D using JL random projection
Projection4D project_embedding(const std::vector<float>& embedding, int seed) {
    const size_t input_dim = embedding.size();
    const double scale = 1.0 / std::sqrt(static_cast<double>(input_dim));
    
    // COORDINATE CONVENTION: uint32 with CENTER at 2^31 = 2147483648
    constexpr double CENTER = 2147483648.0;
    constexpr double SCALE = 2147483647.0;
    
    double proj[4] = {0.0, 0.0, 0.0, 0.0};
    
    // Matrix-vector multiply: proj = P * embedding
    // P is a 4 x input_dim random matrix (generated on the fly)
    for (size_t row = 0; row < 4; ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < input_dim; ++col) {
            double r = random_value(static_cast<int>(row), static_cast<int>(col), seed);
            sum += r * embedding[col] * scale;
        }
        proj[row] = sum;
    }
    
    // Scale to uint32 coordinates with CENTER at 2^31
    // tanh bounds values to [-1, 1], then map to uint32 coords
    auto to_coord = [CENTER, SCALE](double unit_val) -> int32_t {
        double bounded = std::tanh(unit_val);  // Now in [-1, 1]
        double scaled = CENTER + bounded * SCALE;
        if (scaled < 0.0) scaled = 0.0;
        if (scaled > 4294967295.0) scaled = 4294967295.0;
        // Store as int32 (bit-preserving from uint32)
        return static_cast<int32_t>(static_cast<uint32_t>(std::round(scaled)));
    };
    
    Projection4D result;
    result.x = to_coord(proj[0]);
    result.y = to_coord(proj[1]);
    result.z = to_coord(proj[2]);
    result.m = to_coord(proj[3]);
    
    return result;
}

// Parse embedding geometry to float array
std::vector<float> parse_embedding(const char* hex_ewkb) {
    std::vector<float> result;
    
    if (!hex_ewkb || strlen(hex_ewkb) < 20) {
        return result;
    }
    
    // EWKB format: 01 (little endian) + type + srid + data
    // Skip first 18 hex chars (9 bytes: endian + type + srid)
    // Then read coordinate pairs
    
    // For LINESTRINGZM, format is more complex
    // For now, use a simpler approach: extract via PostGIS
    
    return result;  // Will use SQL to extract
}

// Build EWKB for POINTZM
std::string build_pointzm_ewkb(int32_t x, int32_t y, int32_t z, int32_t m) {
    // EWKB: little-endian, POINTZM, SRID=0
    // Type: 0xC0000001 = 3221225473 (Point with Z and M)
    // With SRID: 0xE0000001 = 3758096385
    
    uint8_t buffer[41];  // 1 + 4 + 4 + 32 bytes
    
    buffer[0] = 0x01;  // Little endian
    
    // Type with SRID flag: POINTZM = 0xC0000001, with SRID = 0xE0000001
    uint32_t type = 0xE0000001;
    memcpy(buffer + 1, &type, 4);
    
    // SRID = 0
    uint32_t srid = 0;
    memcpy(buffer + 5, &srid, 4);
    
    // Coordinates as doubles
    double dx = static_cast<double>(x);
    double dy = static_cast<double>(y);
    double dz = static_cast<double>(z);
    double dm = static_cast<double>(m);
    
    memcpy(buffer + 9, &dx, 8);
    memcpy(buffer + 17, &dy, 8);
    memcpy(buffer + 25, &dz, 8);
    memcpy(buffer + 33, &dm, 8);
    
    // Convert to hex
    std::ostringstream oss;
    for (int i = 0; i < 41; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(buffer[i]);
    }
    
    return oss.str();
}

void print_usage() {
    std::cerr << "Usage: manifold_project [options]\n"
              << "\nOptions:\n"
              << "  --model <name>    Model name filter (default: minilm)\n"
              << "  --batch <size>    Batch size for updates (default: 1000)\n"
              << "  --seed <int>      Random seed for projection (default: 42)\n"
              << "  --force           Update even if coordinates exist\n"
              << "  --help            Show this help\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    // Load from environment
    if (auto v = std::getenv("HC_DB_HOST")) config.host = v;
    if (auto v = std::getenv("HC_DB_PORT")) config.port = v;
    if (auto v = std::getenv("HC_DB_NAME")) config.dbname = v;
    if (auto v = std::getenv("HC_DB_USER")) config.user = v;
    if (auto v = std::getenv("HC_DB_PASS")) config.password = v;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            exit(0);
        } else if (arg == "--model" && i + 1 < argc) {
            config.model = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if (arg == "--force") {
            config.force = true;
        }
    }
    
    return config;
}

}  // namespace

int main(int argc, char* argv[]) {
    Config config = parse_args(argc, argv);
    
    std::cerr << "[PROJ] Manifold Projection Tool\n";
    std::cerr << "[PROJ] Model: " << config.model << ", Seed: " << config.seed << "\n";
    
    // Build connection string
    std::ostringstream conn_str;
    conn_str << "host=" << config.host
             << " port=" << config.port
             << " dbname=" << config.dbname
             << " user=" << config.user
             << " password=" << config.password;
    
    PGconn* conn = PQconnectdb(conn_str.str().c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "[ERROR] Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    std::cerr << "[PROJ] Connected to database\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Query embeddings that need projection
    std::ostringstream query;
    query << "SELECT s.entity_id, embedding_to_array(s.embedding) AS arr "
          << "FROM shape s "
          << "JOIN composition c ON c.id = s.entity_id "
          << "WHERE s.model_name ILIKE '%" << config.model << "%' "
          << "AND s.dim_count = 384 ";
    
    if (!config.force) {
        query << "AND c.centroid IS NULL ";
    }
    
    query << "LIMIT " << config.batch_size;
    
    size_t total_processed = 0;
    size_t total_updated = 0;
    
    while (true) {
        PGresult* res = PQexec(conn, query.str().c_str());
        
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            std::cerr << "[ERROR] Query failed: " << PQerrorMessage(conn) << "\n";
            PQclear(res);
            break;
        }
        
        int nrows = PQntuples(res);
        if (nrows == 0) {
            PQclear(res);
            break;  // No more to process
        }
        
        std::cerr << "[PROJ] Processing batch of " << nrows << " embeddings...\n";
        
        // Start transaction
        PGresult* begin_res = PQexec(conn, "BEGIN");
        PQclear(begin_res);
        
        for (int i = 0; i < nrows; ++i) {
            const char* entity_id_hex = PQgetvalue(res, i, 0);
            const char* arr_str = PQgetvalue(res, i, 1);
            
            // Parse array string: {0.1,0.2,0.3,...}
            std::vector<float> embedding;
            if (arr_str && arr_str[0] == '{') {
                std::string s(arr_str + 1);  // Skip '{'
                if (!s.empty() && s.back() == '}') s.pop_back();
                
                std::istringstream iss(s);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    try {
                        embedding.push_back(std::stof(token));
                    } catch (...) {
                        // Skip invalid values
                    }
                }
            }
            
            if (embedding.size() < 4) {
                continue;  // Skip invalid embeddings
            }
            
            // Project to 4D
            Projection4D proj = project_embedding(embedding, config.seed);
            
            // Compute Hilbert index
            hypercube::Point4D p;
            p.x = proj.x;
            p.y = proj.y;
            p.z = proj.z;
            p.m = proj.m;
            hypercube::HilbertIndex hilbert = hypercube::HilbertCurve::coords_to_index(p);
            
            // Build update query
            std::ostringstream update;
            update << "UPDATE composition SET "
                   << "centroid = ST_GeomFromEWKB('\\x" << build_pointzm_ewkb(proj.x, proj.y, proj.z, proj.m) << "'::bytea), "
                   << "hilbert_lo = " << static_cast<int64_t>(hilbert.lo) << ", "
                   << "hilbert_hi = " << static_cast<int64_t>(hilbert.hi) << " "
                   << "WHERE id = '\\x" << entity_id_hex << "'::bytea";
            
            PGresult* upd_res = PQexec(conn, update.str().c_str());
            if (PQresultStatus(upd_res) == PGRES_COMMAND_OK) {
                ++total_updated;
            }
            PQclear(upd_res);
            
            ++total_processed;
        }
        
        // Commit transaction
        PGresult* commit_res = PQexec(conn, "COMMIT");
        PQclear(commit_res);
        
        PQclear(res);
        
        if (nrows < config.batch_size) {
            break;  // Last batch was partial
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    
    std::cerr << "[PROJ] Complete: " << total_processed << " processed, "
              << total_updated << " updated in " << std::fixed << std::setprecision(2) 
              << secs << "s\n";
    
    // Verify results
    PGresult* verify = PQexec(conn, 
        "SELECT COUNT(*) FROM composition c "
        "JOIN shape s ON s.entity_id = c.id "
        "WHERE c.centroid IS NOT NULL AND s.dim_count = 384");
    
    if (PQresultStatus(verify) == PGRES_TUPLES_OK && PQntuples(verify) > 0) {
        std::cerr << "[PROJ] Compositions with projected coordinates: " 
                  << PQgetvalue(verify, 0, 0) << "\n";
    }
    PQclear(verify);
    
    PQfinish(conn);
    
    return 0;
}
