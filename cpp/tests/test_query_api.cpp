/**
 * Query API Tests for Hartonomous Hypercube
 * 
 * Tests the reusable SQL functions that form the query API.
 * These are the functions an LLM would call to ask questions.
 * 
 * Uses 4-Table Schema:
 * - atom: Unicode codepoints only (leaves)
 * - composition: Aggregations (depth > 0)
 * - composition_child: Ordered children
 * - relation: Semantic edges
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <libpq-fe.h>

// Test framework
static int tests_passed = 0;
static int tests_failed = 0;
typedef void (*TestFunc)();
struct TestEntry { const char* name; TestFunc func; };
static std::vector<TestEntry> g_tests;

#define TEST(name) \
    void name(); \
    static struct name##_register { \
        name##_register() { g_tests.push_back({#name, name}); } \
    } name##_instance; \
    void name()

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAILED: " << msg << std::endl; \
        tests_failed++; \
    } else { \
        std::cout << "PASSED: " << msg << std::endl; \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "FAILED: " << msg << " (expected: " << (b) << ", got: " << (a) << ")" << std::endl; \
        tests_failed++; \
    } else { \
        std::cout << "PASSED: " << msg << std::endl; \
        tests_passed++; \
    } \
} while(0)

static PGconn* g_conn = nullptr;

std::string query_value(const char* sql) {
    PGresult* res = PQexec(g_conn, sql);
    std::string result;
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        const char* val = PQgetvalue(res, 0, 0);
        if (val) result = val;
    } else {
        std::cerr << "Query failed: " << PQerrorMessage(g_conn) << std::endl;
        std::cerr << "SQL: " << sql << std::endl;
    }
    PQclear(res);
    return result;
}

int query_int(const char* sql) {
    std::string val = query_value(sql);
    return val.empty() ? -1 : std::stoi(val);
}

double query_double(const char* sql) {
    std::string val = query_value(sql);
    return val.empty() ? -1.0 : std::stod(val);
}

// =============================================================================
// 4-TABLE SCHEMA VERIFICATION
// =============================================================================
TEST(test_four_table_schema) {
    // Verify all 4 tables exist
    int atom_exists = query_int(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'atom' AND table_schema = 'public'"
    );
    ASSERT_EQ(atom_exists, 1, "atom table exists");
    
    int comp_exists = query_int(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'composition' AND table_schema = 'public'"
    );
    ASSERT_EQ(comp_exists, 1, "composition table exists");
    
    int rel_exists = query_int(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'relation' AND table_schema = 'public'"
    );
    ASSERT_EQ(rel_exists, 1, "relation table exists");
    
    int shape_exists = query_int(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'shape' AND table_schema = 'public'"
    );
    ASSERT_EQ(shape_exists, 1, "shape table exists");
}

// =============================================================================
// API FUNCTION: atom_lookup - Find atoms by various criteria
// =============================================================================
TEST(test_atom_lookup_by_codepoint) {
    // Find atom by codepoint
    std::string id = query_value(
        "SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65"
    );
    ASSERT_TRUE(!id.empty(), "atom_lookup by codepoint returns hash");
    ASSERT_EQ(id.length(), size_t(64), "Hash is 32 bytes (64 hex chars)");
}

TEST(test_atom_lookup_by_char) {
    // Find atom by character using convert_to function
    std::string codepoint = query_value(
        "SELECT codepoint FROM atom WHERE value = convert_to('A', 'UTF8')"
    );
    ASSERT_EQ(codepoint, "65", "Can find atom by UTF8 value");
}

// =============================================================================
// API FUNCTION: atom_is_leaf - Check if atom is a leaf (Unicode codepoint)
// In 4-table schema: all atoms ARE leaves, compositions are separate
// =============================================================================
TEST(test_atom_is_leaf) {
    int is_leaf = query_int(
        "SELECT CASE WHEN atom_is_leaf("
        "  (SELECT id FROM atom WHERE codepoint = 65)"
        ") THEN 1 ELSE 0 END"
    );
    ASSERT_EQ(is_leaf, 1, "atom_is_leaf returns true for codepoint");
    
    // Check composition is not a leaf (if any exist)
    int comp_count = query_int("SELECT COUNT(*) FROM composition");
    if (comp_count > 0) {
        int comp_is_leaf = query_int(
            "SELECT CASE WHEN atom_is_leaf("
            "  (SELECT id FROM composition LIMIT 1)"
            ") THEN 1 ELSE 0 END"
        );
        ASSERT_EQ(comp_is_leaf, 0, "atom_is_leaf returns false for composition");
    }
}

// =============================================================================
// API FUNCTION: atom_centroid - Get 4D centroid of any atom
// Returns GEOMETRY - use ST_X, ST_Y, ST_Z, ST_M to extract
// =============================================================================
TEST(test_atom_centroid) {
    // Get centroid of a leaf using PostGIS ST_* functions
    std::string x = query_value(
        "SELECT ST_X(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    std::string y = query_value(
        "SELECT ST_Y(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    std::string z = query_value(
        "SELECT ST_Z(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    std::string m = query_value(
        "SELECT ST_M(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    
    ASSERT_TRUE(!x.empty() && !y.empty() && !z.empty() && !m.empty(),
                "atom_centroid returns all 4 coordinates");
    
    // Verify coordinates are in valid range [0, 2^32-1] (raw uint32 values)
    double xd = std::stod(x);
    double yd = std::stod(y);
    double zd = std::stod(z);
    double md = std::stod(m);
    
    constexpr double MAX_COORD = 4294967295.0;
    ASSERT_TRUE(xd >= 0 && xd <= MAX_COORD && yd >= 0 && yd <= MAX_COORD && 
                zd >= 0 && zd <= MAX_COORD && md >= 0 && md <= MAX_COORD,
                "Centroid coordinates in valid range [0, 2^32-1]");
}

// =============================================================================
// API FUNCTION: atom_distance - Euclidean distance between two atoms
// =============================================================================
TEST(test_atom_distance) {
    // Same atom has zero distance
    double self_dist = query_double(
        "SELECT atom_distance("
        "  (SELECT id FROM atom WHERE codepoint = 65),"
        "  (SELECT id FROM atom WHERE codepoint = 65)"
        ")"
    );
    ASSERT_TRUE(std::abs(self_dist) < 0.0001, "atom_distance to self is 0");
    
    // Different atoms have non-zero distance
    double diff_dist = query_double(
        "SELECT atom_distance("
        "  (SELECT id FROM atom WHERE codepoint = 65),"
        "  (SELECT id FROM atom WHERE codepoint = 90)"
        ")"
    );
    ASSERT_TRUE(diff_dist > 0, "atom_distance between different atoms > 0");
    
    // Distance is symmetric
    double dist_ab = query_double(
        "SELECT atom_distance("
        "  (SELECT id FROM atom WHERE codepoint = 65),"
        "  (SELECT id FROM atom WHERE codepoint = 66)"
        ")"
    );
    double dist_ba = query_double(
        "SELECT atom_distance("
        "  (SELECT id FROM atom WHERE codepoint = 66),"
        "  (SELECT id FROM atom WHERE codepoint = 65)"
        ")"
    );
    ASSERT_TRUE(std::abs(dist_ab - dist_ba) < 0.0001, "atom_distance is symmetric");
}

// =============================================================================
// API FUNCTION: atom_knn - Find k-nearest neighbors by distance
// =============================================================================
TEST(test_atom_knn) {
    // Find 5 nearest neighbors to 'A'
    int count = query_int(
        "SELECT COUNT(*) FROM atom_knn("
        "  (SELECT id FROM atom WHERE codepoint = 65), 5"
        ")"
    );
    ASSERT_EQ(count, 5, "atom_knn returns k neighbors");
    
    // Nearest neighbor should be close
    double nearest_dist = query_double(
        "SELECT distance FROM atom_knn("
        "  (SELECT id FROM atom WHERE codepoint = 65), 1"
        ")"
    );
    ASSERT_TRUE(nearest_dist >= 0, "Nearest neighbor has valid distance");
}

// =============================================================================
// API FUNCTION: atom_nearest_spatial - Compatibility wrapper for atom_knn
// =============================================================================
TEST(test_atom_nearest_spatial) {
    // Find 5 nearest neighbors using spatial index
    int count = query_int(
        "SELECT COUNT(*) FROM atom_nearest_spatial("
        "  (SELECT id FROM atom WHERE codepoint = 65), 5"
        ")"
    );
    ASSERT_EQ(count, 5, "atom_nearest_spatial returns k neighbors");
}

// =============================================================================
// API FUNCTION: semantic_neighbors - Find related atoms via relation table
// =============================================================================
TEST(test_semantic_neighbors) {
    // Check if we have any relations
    int rel_count = query_int("SELECT COUNT(*) FROM relation");
    if (rel_count == 0) {
        std::cout << "SKIPPED: No semantic relations (run extract_embeddings)" << std::endl;
        return;
    }
    
    // Find semantic neighbors of a token that has relations
    std::string token = query_value(
        "SELECT encode(source_id, 'hex') FROM relation WHERE source_type = 'atom' LIMIT 1"
    );
    if (!token.empty()) {
        int count = query_int(
            "SELECT COUNT(*) FROM semantic_neighbors("
            "  (SELECT source_id FROM relation WHERE source_type = 'atom' LIMIT 1), 5"
            ")"
        );
        ASSERT_TRUE(count >= 0, "semantic_neighbors returns results");
    }
}

// =============================================================================
// API FUNCTION: atom_reconstruct_text - Reconstruct original text
// =============================================================================
TEST(test_atom_reconstruct_text) {
    // For leaf atoms, reconstruct should return the character
    std::string text = query_value(
        "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))"
    );
    ASSERT_EQ(text, "A", "atom_reconstruct_text returns 'A' for codepoint 65");
    
    // Check compositions if any exist
    int comp_count = query_int("SELECT COUNT(*) FROM composition");
    if (comp_count > 0) {
        std::string comp_text = query_value(
            "SELECT atom_reconstruct_text((SELECT id FROM composition LIMIT 1))"
        );
        ASSERT_TRUE(!comp_text.empty(), "atom_reconstruct_text works for compositions");
    }
}

// =============================================================================
// API FUNCTION: atom_stats - System statistics (4-table version)
// Returns TABLE(atoms, compositions, relations, shapes, max_depth)
// =============================================================================
TEST(test_atom_stats) {
    std::string total_atoms = query_value("SELECT atoms FROM atom_stats()");
    ASSERT_TRUE(!total_atoms.empty(), "atom_stats returns atoms count");
    
    long atoms = std::stol(total_atoms);
    ASSERT_TRUE(atoms >= 1112064, "System has all Unicode atoms seeded");
}

// =============================================================================
// QUERY PATTERN: Find similar atoms to a given character
// =============================================================================
TEST(test_similarity_query) {
    // Pattern: "What characters are most similar to 'A'?"
    std::string similar = query_value(
        "SELECT string_agg(chr(a.codepoint), '' ORDER BY knn.distance) "
        "FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 10) knn "
        "JOIN atom a ON a.id = knn.neighbor_id "
        "WHERE a.codepoint BETWEEN 32 AND 126"
    );
    ASSERT_TRUE(!similar.empty(), "Similarity query returns printable chars");
    std::cout << "  Similar to 'A': " << similar << std::endl;
}

// =============================================================================
// QUERY PATTERN: Find atoms in a bounding box (4D range query)
// =============================================================================
TEST(test_bounding_box_query) {
    // Get centroid of 'A' and find atoms nearby
    // Radius is ~1% of coordinate space
    int count = query_int(
        "WITH target AS ("
        "  SELECT ST_X(geom) as x, ST_Y(geom) as y, ST_Z(geom) as z, ST_M(geom) as m "
        "  FROM atom WHERE codepoint = 65"
        ") "
        "SELECT COUNT(*) FROM atom a, target t "
        "WHERE ABS(ST_X(a.geom) - t.x) < 50000000 "
        "  AND ABS(ST_Y(a.geom) - t.y) < 50000000 "
        "  AND ABS(ST_Z(a.geom) - t.z) < 50000000 "
        "  AND ABS(ST_M(a.geom) - t.m) < 50000000"
    );
    ASSERT_TRUE(count > 0, "Bounding box query finds nearby atoms");
    std::cout << "  Atoms near 'A' (50M radius): " << count << std::endl;
}

// =============================================================================
// QUERY PATTERN: Relation table stats
// =============================================================================
TEST(test_relation_stats) {
    // Count semantic edges
    int edges = query_int("SELECT COUNT(*) FROM relation");
    std::cout << "  Total semantic edges: " << edges << std::endl;
    ASSERT_TRUE(edges >= 0, "Can query relation table");
    
    // Count distinct relation types
    int types = query_int("SELECT COUNT(DISTINCT relation_type) FROM relation");
    std::cout << "  Distinct relation types: " << types << std::endl;
}

// =============================================================================
// SRID VALIDATION: All geometries have SRID 0
// =============================================================================
TEST(test_srid_zero) {
    int wrong_srid = query_int("SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0");
    ASSERT_EQ(wrong_srid, 0, "All atoms have SRID 0 (raw 4D space)");
}

// =============================================================================
// COMPOSITION TABLE (if populated)
// =============================================================================
TEST(test_composition_table) {
    int comp_count = query_int("SELECT COUNT(*) FROM composition");
    std::cout << "  Compositions: " << comp_count << std::endl;
    
    if (comp_count > 0) {
        // Check that compositions have depth > 0
        int zero_depth = query_int("SELECT COUNT(*) FROM composition WHERE depth = 0");
        ASSERT_EQ(zero_depth, 0, "All compositions have depth > 0");
        
        // Check composition_child links
        int child_links = query_int("SELECT COUNT(*) FROM composition_child");
        ASSERT_TRUE(child_links > 0, "composition_child has links");
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    std::cout << "=== Hartonomous Query API Tests (4-Table Schema) ===" << std::endl;
    
    // Check HC_* (new) first, then PG* (legacy) for backwards compatibility
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
    
    std::cout << "Connecting to: " << user << "@" << host << ":" << port << "/" << db << std::endl;
    
    g_conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(g_conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(g_conn) << std::endl;
        PQfinish(g_conn);
        return 1;
    }
    
    std::cout << "Connected\n" << std::endl;
    
    for (const auto& test : g_tests) {
        std::cout << "\n--- " << test.name << " ---" << std::endl;
        test.func();
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    PQfinish(g_conn);
    
    return tests_failed > 0 ? 1 : 0;
}
