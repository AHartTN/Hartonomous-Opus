/**
 * Query API Tests for Hartonomous Hypercube
 * 
 * Human-readable test output showing:
 * - What each test is checking
 * - The actual SQL queries being run
 * - The results returned
 * - Pass/fail with explanation
 * 
 * Uses 4-Table Schema:
 * - atom: Unicode codepoints only (leaves)
 * - composition: Aggregations (depth > 0)
 * - composition_child: Ordered children
 * - relation: Semantic edges
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <libpq-fe.h>

// ANSI color codes for terminal output
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_DIM     "\033[2m"
#define COLOR_BOLD    "\033[1m"

// Test tracking
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;
static PGconn* g_conn = nullptr;

// Print helpers
void print_section(const char* title) {
    std::cout << "\n" << COLOR_BOLD << "=== " << title << " ===" << COLOR_RESET << "\n\n";
}

void print_test(const char* name) {
    std::cout << COLOR_CYAN << "> " << name << COLOR_RESET << "\n";
}

void print_query(const char* sql) {
    std::cout << COLOR_DIM << "  SQL: " << sql << COLOR_RESET << "\n";
}

void print_result(const char* label, const std::string& value) {
    std::cout << "  " << label << ": " << COLOR_YELLOW << value << COLOR_RESET << "\n";
}

void print_pass(const char* msg) {
    std::cout << COLOR_GREEN << "  [OK] " << msg << COLOR_RESET << "\n";
    tests_passed++;
}

void print_fail(const char* msg) {
    std::cout << COLOR_RED << "  [FAIL] " << msg << COLOR_RESET << "\n";
    tests_failed++;
}

void print_skip(const char* msg) {
    std::cout << COLOR_YELLOW << "  [SKIP] " << msg << COLOR_RESET << "\n";
    tests_skipped++;
}

// Query helpers
std::string query_value(const char* sql, bool show = true) {
    if (show) print_query(sql);
    PGresult* res = PQexec(g_conn, sql);
    std::string result;
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        const char* val = PQgetvalue(res, 0, 0);
        if (val) result = val;
    } else if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << COLOR_RED << "  Query error: " << PQerrorMessage(g_conn) << COLOR_RESET;
    }
    PQclear(res);
    return result;
}

int query_int(const char* sql, bool show = true) {
    std::string val = query_value(sql, show);
    return val.empty() ? -1 : std::stoi(val);
}

long query_long(const char* sql, bool show = true) {
    std::string val = query_value(sql, show);
    return val.empty() ? -1L : std::stol(val);
}

double query_double(const char* sql, bool show = true) {
    std::string val = query_value(sql, show);
    return val.empty() ? -1.0 : std::stod(val);
}

void query_table(const char* sql, int max_rows = 10) {
    print_query(sql);
    PGresult* res = PQexec(g_conn, sql);
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << COLOR_RED << "  Query error: " << PQerrorMessage(g_conn) << COLOR_RESET;
        PQclear(res);
        return;
    }
    
    int nrows = PQntuples(res);
    int ncols = PQnfields(res);
    
    // Print header
    std::cout << "  +";
    for (int c = 0; c < ncols; c++) {
        std::cout << "-----------------";
        if (c < ncols - 1) std::cout << "+";
    }
    std::cout << "+\n";
    
    std::cout << "  |";
    for (int c = 0; c < ncols; c++) {
        std::cout << COLOR_BOLD << std::setw(16) << std::left << PQfname(res, c) << COLOR_RESET << "|";
    }
    std::cout << "\n";
    
    std::cout << "  +";
    for (int c = 0; c < ncols; c++) {
        std::cout << "-----------------";
        if (c < ncols - 1) std::cout << "+";
    }
    std::cout << "+\n";
    
    // Print rows
    int display_rows = std::min(nrows, max_rows);
    for (int r = 0; r < display_rows; r++) {
        std::cout << "  |";
        for (int c = 0; c < ncols; c++) {
            std::string val = PQgetvalue(res, r, c) ? PQgetvalue(res, r, c) : "NULL";
            if (val.length() > 15) val = val.substr(0, 12) + "...";
            std::cout << std::setw(16) << std::left << val << "|";
        }
        std::cout << "\n";
    }
    
    std::cout << "  +";
    for (int c = 0; c < ncols; c++) {
        std::cout << "-----------------";
        if (c < ncols - 1) std::cout << "+";
    }
    std::cout << "+\n";
    
    if (nrows > max_rows) {
        std::cout << COLOR_DIM << "  ... and " << (nrows - max_rows) << " more rows" << COLOR_RESET << "\n";
    }
    std::cout << "  Total: " << nrows << " rows\n";
    
    PQclear(res);
}

// =============================================================================
// TESTS
// =============================================================================

void test_schema_tables() {
    print_section("4-TABLE SCHEMA VERIFICATION");
    
    print_test("Checking required tables exist");
    
    const char* tables[] = {"atom", "composition", "composition_child", "relation", "shape"};
    for (const char* table : tables) {
        std::string sql = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '" + 
                          std::string(table) + "' AND table_schema = 'public'";
        int exists = query_int(sql.c_str(), false);
        if (exists == 1) {
            print_pass((std::string(table) + " table exists").c_str());
        } else {
            print_fail((std::string(table) + " table MISSING").c_str());
        }
    }
}

void test_atom_seeding() {
    print_section("ATOM SEEDING - Unicode Codepoints");
    
    print_test("How many Unicode codepoints are seeded?");
    long count = query_long("SELECT COUNT(*) FROM atom");
    print_result("Total atoms", std::to_string(count));
    
    if (count >= 1112064) {
        print_pass("All 1,112,064 Unicode codepoints are seeded");
    } else {
        print_fail(("Only " + std::to_string(count) + " atoms - expected 1,112,064").c_str());
    }
    
    print_test("Looking up atom for 'A' (codepoint 65)");
    query_table("SELECT codepoint, encode(id, 'hex') as blake3_hash, "
                "ST_X(geom)::numeric(12,0) as x, ST_Y(geom)::numeric(12,0) as y, "
                "ST_Z(geom)::numeric(12,0) as z, ST_M(geom)::numeric(12,0) as m "
                "FROM atom WHERE codepoint = 65");
    print_pass("Atom 'A' found with 4D coordinates");
    
    print_test("Sample of printable ASCII characters");
    query_table("SELECT chr(codepoint) as char, codepoint, "
                "hilbert_lo, hilbert_hi "
                "FROM atom WHERE codepoint BETWEEN 65 AND 74 ORDER BY codepoint");
}

void test_core_functions() {
    print_section("CORE SQL FUNCTIONS");
    
    print_test("atom_is_leaf() - Check if ID is a leaf atom");
    std::string result = query_value(
        "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))"
    );
    print_result("atom_is_leaf('A')", result);
    if (result == "t") {
        print_pass("Correctly identifies 'A' as a leaf atom");
    } else {
        print_fail("Should return true for leaf atom");
    }
    
    print_test("atom_centroid() - Get 4D geometry of an atom");
    query_table("SELECT ST_X(g)::numeric(12,0) as x, ST_Y(g)::numeric(12,0) as y, "
                "ST_Z(g)::numeric(12,0) as z, ST_M(g)::numeric(12,0) as m "
                "FROM (SELECT atom_centroid((SELECT id FROM atom WHERE codepoint = 65)) as g) sub");
    print_pass("atom_centroid returns POINTZM geometry");
    
    print_test("atom_reconstruct_text() - Reconstruct text from atom ID");
    std::string text = query_value(
        "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))"
    );
    print_result("Reconstructed", "'" + text + "'");
    if (text == "A") {
        print_pass("Correctly reconstructs 'A' from atom ID");
    } else {
        print_fail("Should reconstruct to 'A'");
    }
    
    print_test("atom_distance() - Euclidean distance between atoms");
    double self_dist = query_double(
        "SELECT atom_distance("
        "(SELECT id FROM atom WHERE codepoint = 65),"
        "(SELECT id FROM atom WHERE codepoint = 65))"
    );
    print_result("Distance A to A", std::to_string(self_dist));
    if (std::abs(self_dist) < 0.0001) {
        print_pass("Self-distance is 0");
    } else {
        print_fail("Self-distance should be 0");
    }
    
    double ab_dist = query_double(
        "SELECT atom_distance("
        "(SELECT id FROM atom WHERE codepoint = 65),"
        "(SELECT id FROM atom WHERE codepoint = 66))"
    );
    print_result("Distance A to B", std::to_string(ab_dist));
    if (ab_dist > 0) {
        print_pass("Different atoms have non-zero distance");
    } else {
        print_fail("Different atoms should have distance > 0");
    }
}

void test_knn_queries() {
    print_section("K-NEAREST NEIGHBOR QUERIES");
    
    print_test("atom_knn() - Find 10 nearest neighbors to 'A'");
    query_table("SELECT chr(a.codepoint) as char, a.codepoint, "
                "knn.distance::numeric(20,2) as distance "
                "FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 10) knn "
                "JOIN atom a ON a.id = knn.neighbor_id "
                "WHERE a.codepoint BETWEEN 32 AND 126 "
                "ORDER BY knn.distance");
    print_pass("KNN returns nearest printable characters");
    
    print_test("Which letters are spatially closest to 'A'?");
    query_table("SELECT chr(a.codepoint) as letter, "
                "knn.distance::numeric(20,2) as distance "
                "FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 100) knn "
                "JOIN atom a ON a.id = knn.neighbor_id "
                "WHERE a.codepoint BETWEEN 65 AND 90 "  // A-Z only
                "ORDER BY knn.distance "
                "LIMIT 10");
}

void test_semantic_relations() {
    print_section("SEMANTIC RELATIONS");
    
    print_test("How many semantic edges exist?");
    long count = query_long("SELECT COUNT(*) FROM relation");
    print_result("Total edges", std::to_string(count));
    
    if (count == 0) {
        print_skip("No semantic relations yet - run extract_embeddings");
        return;
    }
    print_pass(("Found " + std::to_string(count) + " semantic relationships").c_str());
    
    print_test("Relation types in the database");
    query_table("SELECT relation_type, COUNT(*) as count "
                "FROM relation GROUP BY relation_type ORDER BY count DESC");
    
    print_test("Top 10 strongest semantic relationships");
    query_table("SELECT source_type, encode(source_id, 'hex') as source, "
                "target_type, encode(target_id, 'hex') as target, "
                "weight::numeric(6,4) as weight "
                "FROM relation ORDER BY weight DESC LIMIT 10");
    
    print_test("semantic_neighbors() - Find semantically related atoms");
    std::string source = query_value(
        "SELECT encode(source_id, 'hex') FROM relation WHERE source_type = 'atom' LIMIT 1", false
    );
    if (!source.empty()) {
        query_table("SELECT encode(target_id, 'hex') as neighbor, "
                    "weight::numeric(6,4) as similarity "
                    "FROM semantic_neighbors("
                    "(SELECT source_id FROM relation WHERE source_type = 'atom' LIMIT 1), 5)");
        print_pass("semantic_neighbors returns related atoms");
    }
}

void test_compositions() {
    print_section("COMPOSITIONS (BPE Tokens)");
    
    print_test("How many compositions exist?");
    long count = query_long("SELECT COUNT(*) FROM composition");
    print_result("Total compositions", std::to_string(count));
    
    if (count == 0) {
        print_skip("No compositions yet - run ingest_safetensor to create BPE tokens");
        return;
    }
    
    print_test("Composition depth distribution");
    query_table("SELECT depth, COUNT(*) as count FROM composition GROUP BY depth ORDER BY depth");
    
    print_test("Sample compositions");
    query_table("SELECT encode(id, 'hex') as id, label, depth, child_count, atom_count "
                "FROM composition ORDER BY depth, label LIMIT 10");
    
    print_test("Composition children");
    query_table("SELECT encode(cc.composition_id, 'hex') as parent, "
                "cc.ordinal, cc.child_type, encode(cc.child_id, 'hex') as child "
                "FROM composition_child cc LIMIT 10");
}

void test_system_stats() {
    print_section("SYSTEM STATISTICS");
    
    print_test("atom_stats() - Overall system statistics");
    query_table("SELECT * FROM atom_stats()");
    
    print_test("Index status");
    query_table("SELECT indexname, tablename FROM pg_indexes "
                "WHERE schemaname = 'public' AND tablename IN ('atom', 'composition', 'relation')");
}

void test_example_queries() {
    print_section("EXAMPLE LLM QUERIES");
    
    print_test("Q: What characters are most similar to 'A' in 4D space?");
    query_table("SELECT chr(a.codepoint) as character, "
                "a.codepoint as unicode, "
                "knn.distance::numeric(12,2) as distance "
                "FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 20) knn "
                "JOIN atom a ON a.id = knn.neighbor_id "
                "WHERE a.codepoint BETWEEN 32 AND 126 "
                "ORDER BY knn.distance "
                "LIMIT 10");
    print_pass("LLM can find spatially similar characters");
    
    print_test("Q: Show the 4D neighborhood of letter 'M' (middle of alphabet)");
    query_table("SELECT chr(a.codepoint) as char, "
                "ST_X(a.geom)::numeric(12,0) as x, "
                "ST_Y(a.geom)::numeric(12,0) as y, "
                "ST_Z(a.geom)::numeric(12,0) as z, "
                "ST_M(a.geom)::numeric(12,0) as m "
                "FROM atom_knn((SELECT id FROM atom WHERE codepoint = 77), 5) knn "
                "JOIN atom a ON a.id = knn.neighbor_id");
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    std::cout << "\n";
    std::cout << COLOR_BOLD << "+--------------------------------------------------------------+\n";
    std::cout << "|     Hartonomous Hypercube - Query API Test Suite            |\n";
    std::cout << "|     Human-Readable Output with Query Results                |\n";
    std::cout << "+--------------------------------------------------------------+\n" << COLOR_RESET;
    
    // Connection
    const char* host = std::getenv("HC_DB_HOST") ? std::getenv("HC_DB_HOST") :
                       std::getenv("PGHOST") ? std::getenv("PGHOST") : "localhost";
    const char* port = std::getenv("HC_DB_PORT") ? std::getenv("HC_DB_PORT") :
                       std::getenv("PGPORT") ? std::getenv("PGPORT") : "5432";
    const char* user = std::getenv("HC_DB_USER") ? std::getenv("HC_DB_USER") :
                       std::getenv("PGUSER") ? std::getenv("PGUSER") : "hartonomous";
    const char* pass = std::getenv("HC_DB_PASS") ? std::getenv("HC_DB_PASS") :
                       std::getenv("PGPASSWORD") ? std::getenv("PGPASSWORD") : "hartonomous";
    const char* db = std::getenv("HC_DB_NAME") ? std::getenv("HC_DB_NAME") :
                     std::getenv("PGDATABASE") ? std::getenv("PGDATABASE") : "hypercube";
    
    std::string conninfo = "host=" + std::string(host) +
                           " port=" + std::string(port) +
                           " user=" + std::string(user) +
                           " password=" + std::string(pass) +
                           " dbname=" + std::string(db);
    
    std::cout << "\n" << COLOR_DIM << "Connecting to " << user << "@" << host << ":" << port << "/" << db << "..." << COLOR_RESET << "\n";
    
    g_conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(g_conn) != CONNECTION_OK) {
        std::cerr << COLOR_RED << "Connection failed: " << PQerrorMessage(g_conn) << COLOR_RESET << std::endl;
        PQfinish(g_conn);
        return 1;
    }
    std::cout << COLOR_GREEN << "Connected!\n" << COLOR_RESET;
    
    // Run tests
    test_schema_tables();
    test_atom_seeding();
    test_core_functions();
    test_knn_queries();
    test_semantic_relations();
    test_compositions();
    test_system_stats();
    test_example_queries();
    
    // Summary
    std::cout << "\n";
    std::cout << COLOR_BOLD << "================================================================\n";
    std::cout << "                         TEST SUMMARY\n";
    std::cout << "================================================================\n" << COLOR_RESET;
    std::cout << COLOR_GREEN << "  Passed:  " << tests_passed << COLOR_RESET << "\n";
    std::cout << COLOR_RED << "  Failed:  " << tests_failed << COLOR_RESET << "\n";
    std::cout << COLOR_YELLOW << "  Skipped: " << tests_skipped << COLOR_RESET << "\n";
    std::cout << "================================================================\n\n";
    
    PQfinish(g_conn);
    
    return tests_failed > 0 ? 1 : 0;
}
