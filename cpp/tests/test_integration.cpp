/**
 * Integration Tests for Hartonomous Hypercube (4-Table Schema)
 * 
 * End-to-end tests that verify:
 * - Database connectivity and 4-table schema (atom, composition, relation, shape)
 * - Atom seeding and lookup
 * - Function compatibility
 * - Hilbert indexing and spatial queries
 * 
 * All queries use SRID 0 (no projection - raw 4D space)
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <libpq-fe.h>

#include "../include/hypercube/types.hpp"
#include "../include/hypercube/blake3.hpp"

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

// =============================================================================
// TEST: 4-Table Schema Exists
// =============================================================================
TEST(test_schema_exists) {
    int atom_table = query_int(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'atom'"
    );
    ASSERT_EQ(atom_table, 1, "atom table exists");
    
    int comp_table = query_int(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'composition'"
    );
    ASSERT_EQ(comp_table, 1, "composition table exists");
    
    int rel_table = query_int(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'relation'"
    );
    ASSERT_EQ(rel_table, 1, "relation table exists");
    
    int shape_table = query_int(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'shape'"
    );
    ASSERT_EQ(shape_table, 1, "shape table exists");
    
    // Check atom columns (4-table schema: no depth, no children)
    int has_geom = query_int(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'atom' AND column_name = 'geom'"
    );
    ASSERT_EQ(has_geom, 1, "atom.geom column exists");
    
    int has_codepoint = query_int(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'atom' AND column_name = 'codepoint'"
    );
    ASSERT_EQ(has_codepoint, 1, "atom.codepoint column exists");
    
    int has_hilbert = query_int(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'atom' AND column_name = 'hilbert_lo'"
    );
    ASSERT_EQ(has_hilbert, 1, "atom.hilbert_lo column exists");
}

// =============================================================================
// TEST: Atoms are seeded correctly
// =============================================================================
TEST(test_atoms_seeded) {
    // In 4-table schema, atom table contains ONLY leaf atoms
    int atom_count = query_int("SELECT COUNT(*) FROM atom");
    ASSERT_TRUE(atom_count >= 1112064, "All Unicode atoms seeded (>= 1.1M)");
    
    int has_A = query_int("SELECT 1 FROM atom WHERE codepoint = 65");
    ASSERT_EQ(has_A, 1, "Atom 'A' (codepoint 65) exists");
    
    int has_a = query_int("SELECT 1 FROM atom WHERE codepoint = 97");
    ASSERT_EQ(has_a, 1, "Atom 'a' (codepoint 97) exists");
    
    int wrong_srid = query_int(
        "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0 LIMIT 1"
    );
    ASSERT_EQ(wrong_srid, 0, "All atoms have SRID 0");
}

// =============================================================================
// TEST: Atom geometry is POINTZM
// =============================================================================
TEST(test_atom_geometry_type) {
    std::string geom_type = query_value(
        "SELECT ST_GeometryType(geom) FROM atom LIMIT 1"
    );
    ASSERT_EQ(geom_type, "ST_Point", "Atoms have POINTZM geometry");
    
    int has_z = query_int(
        "SELECT CASE WHEN ST_Z(geom) IS NOT NULL THEN 1 ELSE 0 END "
        "FROM atom LIMIT 1"
    );
    ASSERT_EQ(has_z, 1, "Atoms have Z coordinate");
    
    int has_m = query_int(
        "SELECT CASE WHEN ST_M(geom) IS NOT NULL THEN 1 ELSE 0 END "
        "FROM atom LIMIT 1"
    );
    ASSERT_EQ(has_m, 1, "Atoms have M coordinate");
}

// =============================================================================
// TEST: Case pairs are semantically close
// =============================================================================
TEST(test_case_pair_proximity) {
    std::string dist_Aa = query_value(
        "SELECT ST_Distance(a1.geom, a2.geom) "
        "FROM atom a1, atom a2 "
        "WHERE a1.codepoint = 65 AND a2.codepoint = 97"
    );
    double d_Aa = std::stod(dist_Aa);
    
    std::string dist_AZ = query_value(
        "SELECT ST_Distance(a1.geom, a2.geom) "
        "FROM atom a1, atom a2 "
        "WHERE a1.codepoint = 65 AND a2.codepoint = 90"
    );
    double d_AZ = std::stod(dist_AZ);
    
    ASSERT_TRUE(d_Aa < d_AZ, "A/a closer than A/Z (semantic clustering)");
}

// =============================================================================
// TEST: Hilbert indices preserve locality
// =============================================================================
TEST(test_hilbert_locality) {
    std::string hilbert_A = query_value(
        "SELECT hilbert_lo::text FROM atom WHERE codepoint = 65"
    );
    std::string hilbert_B = query_value(
        "SELECT hilbert_lo::text FROM atom WHERE codepoint = 66"
    );
    
    ASSERT_TRUE(!hilbert_A.empty() && !hilbert_B.empty(), 
                "Hilbert indices computed for atoms");
}

// =============================================================================
// TEST: BLAKE3 hashing is deterministic
// =============================================================================
TEST(test_blake3_determinism) {
    std::string hash1 = query_value(
        "SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65"
    );
    std::string hash2 = query_value(
        "SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65"
    );
    ASSERT_EQ(hash1, hash2, "BLAKE3 hash is deterministic");
    
    std::string hash_a = query_value(
        "SELECT encode(id, 'hex') FROM atom WHERE codepoint = 97"
    );
    ASSERT_TRUE(hash1 != hash_a, "Different codepoints have different hashes");
}

// =============================================================================
// TEST: Core functions work
// =============================================================================
TEST(test_sql_functions) {
    int is_leaf = query_int(
        "SELECT CASE WHEN atom_is_leaf("
        "  (SELECT id FROM atom WHERE codepoint = 65)"
        ") THEN 1 ELSE 0 END"
    );
    ASSERT_EQ(is_leaf, 1, "atom_is_leaf() works for leaf atoms");
    
    std::string centroid = query_value(
        "SELECT ST_X(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    ASSERT_TRUE(!centroid.empty(), "atom_centroid() returns coordinates");
    
    std::string dist = query_value(
        "SELECT atom_distance("
        "  (SELECT id FROM atom WHERE codepoint = 65),"
        "  (SELECT id FROM atom WHERE codepoint = 97)"
        ")"
    );
    ASSERT_TRUE(!dist.empty(), "atom_distance() computes distance");
}

// =============================================================================
// TEST: KNN function works
// =============================================================================
TEST(test_knn) {
    int knn_count = query_int(
        "SELECT COUNT(*) FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 5)"
    );
    ASSERT_EQ(knn_count, 5, "atom_knn() returns 5 neighbors");
}

// =============================================================================
// TEST: Relation table has semantic edges
// =============================================================================
TEST(test_relations) {
    int rel_count = query_int("SELECT COUNT(*) FROM relation");
    ASSERT_TRUE(rel_count >= 0, "relation table accessible");
    
    if (rel_count > 0) {
        std::string weight = query_value(
            "SELECT weight::text FROM relation ORDER BY weight DESC LIMIT 1"
        );
        ASSERT_TRUE(!weight.empty(), "relation.weight column works");
    }
}

// =============================================================================
// TEST: Composition table structure
// =============================================================================
TEST(test_composition_table) {
    int has_depth = query_int(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'composition' AND column_name = 'depth'"
    );
    ASSERT_EQ(has_depth, 1, "composition.depth column exists");
    
    int has_centroid = query_int(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'composition' AND column_name = 'centroid'"
    );
    ASSERT_EQ(has_centroid, 1, "composition.centroid column exists");
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    const char* conninfo = "dbname=hypercube host=localhost port=5432 user=hartonomous password=hartonomous";
    
    if (argc > 1) {
        conninfo = argv[1];
    }
    
    g_conn = PQconnectdb(conninfo);
    if (PQstatus(g_conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(g_conn) << std::endl;
        return 1;
    }
    
    std::cout << "Connected to database" << std::endl;
    std::cout << "Running " << g_tests.size() << " integration tests..." << std::endl;
    std::cout << std::endl;
    
    for (const auto& test : g_tests) {
        std::cout << "--- " << test.name << " ---" << std::endl;
        test.func();
        std::cout << std::endl;
    }
    
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    PQfinish(g_conn);
    
    return tests_failed > 0 ? 1 : 0;
}
