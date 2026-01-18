/**
 * Hypercube Integration Test Suite
 * 
 * Comprehensive tests for the Hartonomous-Opus semantic web system.
 * Tests database connectivity, schema integrity, query functions,
 * spatial operations, and C++ offloading.
 * 
 * Build: g++ -std=c++20 -O2 -I../cpp/include -lpq -o test_integration test_integration.cpp
 * Run: ./test_integration -d hypercube -h localhost
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <sstream>
#include <libpq-fe.h>

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double elapsed_ms;
};

class TestSuite {
private:
    PGconn* conn_;
    std::vector<TestResult> results_;
    int passed_ = 0;
    int failed_ = 0;

public:
    TestSuite(const std::string& conninfo) {
        conn_ = PQconnectdb(conninfo.c_str());
        if (PQstatus(conn_) != CONNECTION_OK) {
            std::cerr << "Connection failed: " << PQerrorMessage(conn_) << "\n";
            exit(1);
        }
    }
    
    ~TestSuite() {
        if (conn_) PQfinish(conn_);
    }
    
    void run_test(const std::string& name, bool (*test_fn)(PGconn*, std::string&)) {
        std::string message;
        auto start = std::chrono::high_resolution_clock::now();
        
        bool passed = false;
        try {
            passed = test_fn(conn_, message);
        } catch (const std::exception& e) {
            message = std::string("Exception: ") + e.what();
            passed = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        results_.push_back({name, passed, message, elapsed});
        
        if (passed) {
            passed_++;
            std::cout << "PASS: " << name << " (" << std::fixed << std::setprecision(1) << elapsed << "ms)\n";
        } else {
            failed_++;
            std::cout << "FAIL: " << name << "\n";
            std::cout << "  " << message << "\n";
        }
    }
    
    void print_summary() {
        std::cout << "\n===================================================================\n";
        std::cout << "Results: " << passed_ << " passed, " << failed_ << " failed, " << (passed_ + failed_) << " total\n";
        
        if (failed_ > 0) {
            std::cout << "\nFailed tests:\n";
            for (const auto& r : results_) {
                if (!r.passed) {
                    std::cout << "  - " << r.name << ": " << r.message << "\n";
                }
            }
        }
    }
    
    int exit_code() { return failed_ > 0 ? 1 : 0; }
};

// =============================================================================
// Test Helper Functions
// =============================================================================

static bool exec_query(PGconn* conn, const char* query, PGresult** result) {
    *result = PQexec(conn, query);
    return PQresultStatus(*result) == PGRES_TUPLES_OK || 
           PQresultStatus(*result) == PGRES_COMMAND_OK;
}

static int64_t get_int64(PGresult* res, int row, int col) {
    return std::stoll(PQgetvalue(res, row, col));
}

static double get_double(PGresult* res, int row, int col) {
    return std::stod(PQgetvalue(res, row, col));
}

// =============================================================================
// Schema Tests
// =============================================================================

bool test_atom_table_exists(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, "SELECT 1 FROM atom LIMIT 1", &res)) {
        msg = "atom table does not exist";
        PQclear(res);
        return false;
    }
    PQclear(res);
    return true;
}

bool test_atom_indexes(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT indexname FROM pg_indexes WHERE tablename = 'atom'", &res)) {
        msg = "Failed to query indexes";
        PQclear(res);
        return false;
    }
    
    int count = PQntuples(res);
    if (count < 4) {
        msg = "Expected at least 4 indexes, found " + std::to_string(count);
        PQclear(res);
        return false;
    }
    
    PQclear(res);
    return true;
}

bool test_postgis_extension(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, "SELECT PostGIS_Version()", &res)) {
        msg = "PostGIS not installed";
        PQclear(res);
        return false;
    }
    
    msg = std::string("PostGIS ") + PQgetvalue(res, 0, 0);
    PQclear(res);
    return true;
}

// =============================================================================
// Data Integrity Tests
// =============================================================================

bool test_leaf_atoms_have_codepoints(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT COUNT(*) FROM atom WHERE depth = 0 AND codepoint IS NULL", &res)) {
        msg = "Query failed";
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count > 0) {
        msg = std::to_string(count) + " leaf atoms missing codepoint";
        return false;
    }
    return true;
}

bool test_compositions_have_children(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT COUNT(*) FROM atom WHERE depth > 0 AND children IS NULL", &res)) {
        msg = "Query failed";
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count > 0) {
        msg = std::to_string(count) + " compositions missing children";
        return false;
    }
    return true;
}

bool test_geometry_srid_is_zero(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0", &res)) {
        msg = "Query failed";
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count > 0) {
        msg = std::to_string(count) + " atoms have non-zero SRID";
        return false;
    }
    return true;
}

bool test_coordinates_full_precision(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Check that coordinates use full 32-bit range (not normalized 0-1)
    if (!exec_query(conn, 
        "SELECT MAX(ST_X(geom)), MAX(ST_Y(geom)), MAX(ST_Z(geom)), MAX(ST_M(geom)) "
        "FROM atom WHERE depth = 0", &res)) {
        msg = "Query failed";
        PQclear(res);
        return false;
    }
    
    double max_x = get_double(res, 0, 0);
    double max_y = get_double(res, 0, 1);
    double max_z = get_double(res, 0, 2);
    double max_m = get_double(res, 0, 3);
    PQclear(res);
    
    // If max values are < 256, coordinates are probably normalized (wrong)
    if (max_x < 256 && max_y < 256 && max_z < 256 && max_m < 256) {
        msg = "Coordinates appear to be normalized (max values too small)";
        return false;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0);
    oss << "max coords: (" << max_x << ", " << max_y << ", " << max_z << ", " << max_m << ")";
    msg = oss.str();
    return true;
}

bool test_hilbert_values_populated(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT COUNT(*) FROM atom WHERE hilbert_lo = 0 AND hilbert_hi = 0", &res)) {
        msg = "Query failed";
        PQclear(res);
        return false;
    }
    
    int64_t zeros = get_int64(res, 0, 0);
    PQclear(res);
    
    // A few zeros are OK (the origin point), but not too many
    if (!exec_query(conn, "SELECT COUNT(*) FROM atom", &res)) {
        msg = "Count query failed";
        PQclear(res);
        return false;
    }
    
    int64_t total = get_int64(res, 0, 0);
    PQclear(res);
    
    double zero_pct = 100.0 * zeros / total;
    if (zero_pct > 1.0) {
        msg = std::to_string(zeros) + " atoms (" + std::to_string(zero_pct) + "%) have zero Hilbert values";
        return false;
    }
    
    return true;
}

// =============================================================================
// Function Tests
// =============================================================================

bool test_stats_function(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, "SELECT * FROM stats()", &res)) {
        msg = "stats() function failed";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) != 1) {
        msg = "stats() should return exactly 1 row";
        PQclear(res);
        return false;
    }
    
    int64_t total = get_int64(res, 0, 0);
    msg = "total_atoms = " + std::to_string(total);
    PQclear(res);
    return total > 0;
}

bool test_depth_distribution_view(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, "SELECT * FROM v_depth_distribution ORDER BY depth", &res)) {
        msg = "v_depth_distribution view failed";
        PQclear(res);
        return false;
    }
    
    int rows = PQntuples(res);
    if (rows < 2) {
        msg = "Expected at least 2 depth levels";
        PQclear(res);
        return false;
    }
    
    msg = std::to_string(rows) + " depth levels";
    PQclear(res);
    return true;
}

bool test_get_atom_by_codepoint(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Test with ASCII 'A' (65)
    if (!exec_query(conn, "SELECT id FROM get_atom_by_codepoint(65)", &res)) {
        msg = "get_atom_by_codepoint(65) failed";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) != 1) {
        msg = "Expected exactly 1 result for codepoint 65 ('A')";
        PQclear(res);
        return false;
    }
    
    PQclear(res);
    return true;
}

bool test_knn_hilbert(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a random atom and find its neighbors
    if (!exec_query(conn, 
        "SELECT id FROM atom WHERE depth = 0 LIMIT 1", &res)) {
        msg = "Failed to get test atom";
        PQclear(res);
        return false;
    }
    
    std::string atom_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM knn_hilbert('" + atom_id + "'::bytea, 10)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "knn_hilbert failed";
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count != 10) {
        msg = "Expected 10 neighbors, got " + std::to_string(count);
        return false;
    }
    
    return true;
}

bool test_content_exists(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a known atom ID
    if (!exec_query(conn, "SELECT id FROM atom LIMIT 1", &res)) {
        msg = "Failed to get test atom";
        PQclear(res);
        return false;
    }
    
    std::string atom_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT content_exists('" + atom_id + "'::bytea)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "content_exists failed";
        PQclear(res);
        return false;
    }
    
    std::string result = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    if (result != "t") {
        msg = "content_exists returned false for existing atom";
        return false;
    }
    
    // Test with non-existent hash
    if (!exec_query(conn, 
        "SELECT content_exists('\\x0000000000000000000000000000000000000000000000000000000000000000'::bytea)", 
        &res)) {
        msg = "content_exists (negative test) failed";
        PQclear(res);
        return false;
    }
    
    result = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    if (result != "f") {
        msg = "content_exists returned true for non-existent atom";
        return false;
    }
    
    return true;
}

// =============================================================================
// Spatial Query Tests
// =============================================================================

bool test_gist_index_used(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "EXPLAIN SELECT id FROM atom WHERE geom && ST_MakeEnvelope(0, 0, 1000000, 1000000, 0)", 
        &res)) {
        msg = "EXPLAIN query failed";
        PQclear(res);
        return false;
    }
    
    std::string plan;
    for (int i = 0; i < PQntuples(res); i++) {
        plan += PQgetvalue(res, i, 0);
    }
    PQclear(res);
    
    if (plan.find("Index") == std::string::npos && plan.find("idx_atom_geom") == std::string::npos) {
        msg = "GIST index not used in spatial query";
        return false;
    }
    
    return true;
}

bool test_hilbert_index_used(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "EXPLAIN SELECT id FROM atom ORDER BY hilbert_hi, hilbert_lo LIMIT 10", 
        &res)) {
        msg = "EXPLAIN query failed";
        PQclear(res);
        return false;
    }
    
    std::string plan;
    for (int i = 0; i < PQntuples(res); i++) {
        plan += PQgetvalue(res, i, 0);
    }
    PQclear(res);
    
    if (plan.find("Index") == std::string::npos) {
        msg = "Hilbert index not used in ORDER BY query";
        return false;
    }
    
    return true;
}

// =============================================================================
// Model Infrastructure Tests  
// =============================================================================

bool test_model_registry_table(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, "SELECT 1 FROM model_registry LIMIT 1", &res)) {
        msg = "model_registry table does not exist";
        PQclear(res);
        return false;
    }
    PQclear(res);
    return true;
}

bool test_register_model_function(PGconn* conn, std::string& msg) {
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT register_model('test_model', 'test', NULL, '{}'::jsonb)", &res)) {
        msg = "register_model function failed";
        PQclear(res);
        return false;
    }
    
    int model_id = std::stoi(PQgetvalue(res, 0, 0));
    PQclear(res);
    
    // Cleanup
    std::string cleanup = "DELETE FROM model_registry WHERE name = 'test_model'";
    res = PQexec(conn, cleanup.c_str());
    PQclear(res);
    
    msg = "model_id = " + std::to_string(model_id);
    return model_id > 0;
}

// =============================================================================
// AI Operations Tests
// =============================================================================

bool test_attention_self(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a composition to test with
    if (!exec_query(conn, "SELECT id FROM atom WHERE depth = 2 LIMIT 1", &res)) {
        msg = "Failed to get test composition";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) == 0) {
        msg = "No depth-2 compositions found";
        PQclear(res);
        return false;
    }
    
    std::string comp_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM attention_self('" + comp_id + "'::bytea, 3, 10)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "attention_self function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count == 0) {
        msg = "attention_self returned no results";
        return false;
    }
    
    msg = std::to_string(count) + " attention results";
    return true;
}

bool test_attention_cross(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get two compositions to test cross-attention
    if (!exec_query(conn, "SELECT array_agg(id) FROM (SELECT id FROM atom WHERE depth = 1 LIMIT 3) t", &res)) {
        msg = "Failed to get test compositions";
        PQclear(res);
        return false;
    }
    
    std::string ids_array = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM attention_cross('" + ids_array + "'::bytea[], 10)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "attention_cross function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    msg = std::to_string(count) + " cross-attention results";
    return count > 0;
}

bool test_transform_analogy(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get three different compositions
    if (!exec_query(conn, "SELECT id FROM atom WHERE depth = 1 LIMIT 3", &res)) {
        msg = "Failed to get test compositions";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) < 3) {
        msg = "Need at least 3 depth-1 compositions";
        PQclear(res);
        return false;
    }
    
    std::string id_a = PQgetvalue(res, 0, 0);
    std::string id_b = PQgetvalue(res, 1, 0);
    std::string id_c = PQgetvalue(res, 2, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM transform_analogy('" + id_a + "'::bytea, '" + 
                        id_b + "'::bytea, '" + id_c + "'::bytea, 5)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "transform_analogy function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    msg = std::to_string(count) + " analogy results";
    return count > 0;
}

bool test_infer_related(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a composition with children
    if (!exec_query(conn, "SELECT id FROM atom WHERE depth = 3 LIMIT 1", &res)) {
        msg = "Failed to get test composition";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) == 0) {
        msg = "No depth-3 compositions found";
        PQclear(res);
        return false;
    }
    
    std::string comp_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM infer_related('" + comp_id + "'::bytea, 1, 20)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "infer_related function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    msg = std::to_string(count) + " related compositions";
    return true;  // Zero results is OK if composition has unique children
}

bool test_generate_random_walk(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a seed composition
    if (!exec_query(conn, "SELECT id FROM atom WHERE depth = 2 LIMIT 1", &res)) {
        msg = "Failed to get seed composition";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) == 0) {
        msg = "No depth-2 compositions found";
        PQclear(res);
        return false;
    }
    
    std::string seed_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM generate_random_walk('" + seed_id + "'::bytea, 5, 0.5)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "generate_random_walk function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (count == 0) {
        msg = "generate_random_walk returned no steps";
        return false;
    }
    
    msg = std::to_string(count) + " walk steps";
    return true;
}

bool test_generate_continuation(PGconn* conn, std::string& msg) {
    PGresult* res;
    // Get a composition to continue from
    if (!exec_query(conn, "SELECT id FROM atom WHERE depth = 2 LIMIT 1", &res)) {
        msg = "Failed to get prefix composition";
        PQclear(res);
        return false;
    }
    
    if (PQntuples(res) == 0) {
        msg = "No depth-2 compositions found";
        PQclear(res);
        return false;
    }
    
    std::string prefix_id = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    std::string query = "SELECT COUNT(*) FROM generate_continuation('" + prefix_id + "'::bytea, 5)";
    if (!exec_query(conn, query.c_str(), &res)) {
        msg = "generate_continuation function failed: " + std::string(PQerrorMessage(conn));
        PQclear(res);
        return false;
    }
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    msg = std::to_string(count) + " continuation items";
    return count > 0;
}

// =============================================================================
// Performance Tests
// =============================================================================

bool test_count_performance(PGconn* conn, std::string& msg) {
    auto start = std::chrono::high_resolution_clock::now();
    
    PGresult* res;
    if (!exec_query(conn, "SELECT COUNT(*) FROM atom", &res)) {
        msg = "COUNT query failed";
        PQclear(res);
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    int64_t count = get_int64(res, 0, 0);
    PQclear(res);
    
    if (elapsed > 5000) {
        msg = "COUNT took too long: " + std::to_string(elapsed) + "ms";
        return false;
    }
    
    std::ostringstream oss;
    oss << count << " atoms in " << std::fixed << std::setprecision(1) << elapsed << "ms";
    msg = oss.str();
    return true;
}

bool test_depth_query_performance(PGconn* conn, std::string& msg) {
    auto start = std::chrono::high_resolution_clock::now();
    
    PGresult* res;
    if (!exec_query(conn, 
        "SELECT id FROM atom WHERE depth = 1 ORDER BY hilbert_hi, hilbert_lo LIMIT 100", 
        &res)) {
        msg = "Depth query failed";
        PQclear(res);
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    int count = PQntuples(res);
    PQclear(res);
    
    if (elapsed > 1000) {
        msg = "Depth query took too long: " + std::to_string(elapsed) + "ms";
        return false;
    }
    
    std::ostringstream oss;
    oss << count << " results in " << std::fixed << std::setprecision(1) << elapsed << "ms";
    msg = oss.str();
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::string conninfo = "dbname=hypercube host=hart-server";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            conninfo = "dbname=" + std::string(argv[++i]) + " host=hart-server";
        } else if (arg == "-h" && i + 1 < argc) {
            size_t pos = conninfo.find("host=");
            if (pos != std::string::npos) {
                size_t end = conninfo.find(' ', pos);
                conninfo.replace(pos, end - pos, "host=" + std::string(argv[++i]));
            } else {
                conninfo += " host=" + std::string(argv[++i]);
            }
        } else if (arg == "-U" && i + 1 < argc) {
            conninfo += " user=" + std::string(argv[++i]);
        }
    }
    
    std::cout << "===================================================================\n";
    std::cout << "Hypercube Integration Test Suite\n";
    std::cout << "===================================================================\n\n";
    
    TestSuite suite(conninfo);
    
    std::cout << "Schema Tests:\n";
    suite.run_test("atom table exists", test_atom_table_exists);
    suite.run_test("atom indexes exist", test_atom_indexes);
    suite.run_test("PostGIS extension", test_postgis_extension);
    suite.run_test("model registry table", test_model_registry_table);
    
    std::cout << "\nData Integrity Tests:\n";
    suite.run_test("leaf atoms have codepoints", test_leaf_atoms_have_codepoints);
    suite.run_test("compositions have children", test_compositions_have_children);
    suite.run_test("geometry SRID is 0", test_geometry_srid_is_zero);
    suite.run_test("coordinates full precision", test_coordinates_full_precision);
    suite.run_test("Hilbert values populated", test_hilbert_values_populated);
    
    std::cout << "\nFunction Tests:\n";
    suite.run_test("stats() function", test_stats_function);
    suite.run_test("v_depth_distribution view", test_depth_distribution_view);
    suite.run_test("get_atom_by_codepoint()", test_get_atom_by_codepoint);
    suite.run_test("knn_hilbert()", test_knn_hilbert);
    suite.run_test("content_exists()", test_content_exists);
    suite.run_test("register_model()", test_register_model_function);
    
    std::cout << "\nIndex Usage Tests:\n";
    suite.run_test("GIST index used for spatial", test_gist_index_used);
    suite.run_test("Hilbert index used for ordering", test_hilbert_index_used);
    
    std::cout << "\nAI Operations Tests:\n";
    suite.run_test("attention_self()", test_attention_self);
    suite.run_test("attention_cross()", test_attention_cross);
    suite.run_test("transform_analogy()", test_transform_analogy);
    suite.run_test("infer_related()", test_infer_related);
    suite.run_test("generate_random_walk()", test_generate_random_walk);
    suite.run_test("generate_continuation()", test_generate_continuation);
    
    std::cout << "\nPerformance Tests:\n";
    suite.run_test("COUNT performance", test_count_performance);
    suite.run_test("Depth query performance", test_depth_query_performance);
    
    suite.print_summary();
    
    return suite.exit_code();
}
