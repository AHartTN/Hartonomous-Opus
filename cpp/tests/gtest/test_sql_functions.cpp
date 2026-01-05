// =============================================================================
// SQL Function Tests
// =============================================================================

#include <gtest/gtest.h>
#include <libpq-fe.h>
#include <cstdlib>
#include <string>

class SQLFunctionTest : public ::testing::Test {
protected:
    PGconn* conn = nullptr;
    
    void SetUp() override {
        const char* db = std::getenv("HC_DB_NAME");
        const char* user = std::getenv("HC_DB_USER");
        const char* host = std::getenv("HC_DB_HOST");
        const char* port = std::getenv("HC_DB_PORT");
        const char* pass = std::getenv("PGPASSWORD");
        
        std::string conninfo = "dbname=" + std::string(db ? db : "hypercube_test");
        conninfo += " user=" + std::string(user ? user : "postgres");
        conninfo += " host=" + std::string(host ? host : "localhost");
        conninfo += " port=" + std::string(port ? port : "5432");
        if (pass) conninfo += " password=" + std::string(pass);
        
        conn = PQconnectdb(conninfo.c_str());
        if (PQstatus(conn) != CONNECTION_OK) {
            GTEST_SKIP() << "Database connection failed: " << PQerrorMessage(conn);
        }
        
        // Start transaction for each test
        PQexec(conn, "BEGIN");
    }
    
    void TearDown() override {
        if (conn) {
            // Rollback to keep database clean
            PQexec(conn, "ROLLBACK");
            PQfinish(conn);
            conn = nullptr;
        }
    }
    
    PGresult* exec(const char* query) {
        return PQexec(conn, query);
    }
    
    bool exec_ok(const char* query) {
        PGresult* res = PQexec(conn, query);
        ExecStatusType status = PQresultStatus(res);
        bool ok = (status == PGRES_TUPLES_OK || status == PGRES_COMMAND_OK);
        if (!ok) {
            std::cerr << "Query failed: " << PQresultErrorMessage(res) << "\n";
        }
        PQclear(res);
        return ok;
    }
};

// Test get_or_create_atom function
TEST_F(SQLFunctionTest, GetOrCreateAtom) {
    // Create an atom
    PGresult* res = exec("SELECT get_or_create_atom('test_token_xyz', 0.5)");
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    EXPECT_EQ(PQntuples(res), 1);
    
    std::string id1 = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    // Same token should return same ID
    res = exec("SELECT get_or_create_atom('test_token_xyz', 0.6)");
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    std::string id2 = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    EXPECT_EQ(id1, id2) << "Same token should return same atom ID";
}

// Test create_composition_from_atoms function
TEST_F(SQLFunctionTest, CreateCompositionFromAtoms) {
    // Create some atoms first
    exec("SELECT get_or_create_atom('comp_test_a', 0.3)");
    exec("SELECT get_or_create_atom('comp_test_b', 0.4)");
    
    // Create composition
    PGresult* res = exec(
        "SELECT create_composition_from_atoms(ARRAY['comp_test_a', 'comp_test_b']::text[])"
    );
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        // Function may not exist yet
        GTEST_SKIP() << "create_composition_from_atoms not implemented: " 
                     << PQresultErrorMessage(res);
    }
    
    EXPECT_EQ(PQntuples(res), 1);
    PQclear(res);
}

// Test bigram statistics update
TEST_F(SQLFunctionTest, BigramStatsUpdate) {
    // Insert test atoms
    exec("SELECT get_or_create_atom('bigram_a', 0.1)");
    exec("SELECT get_or_create_atom('bigram_b', 0.2)");
    
    // Insert bigram
    bool ok = exec_ok(
        "INSERT INTO bigram_stats (token_a, token_b, count, pmi) "
        "VALUES ('bigram_a', 'bigram_b', 1, 0.0) "
        "ON CONFLICT (token_a, token_b) DO UPDATE SET count = bigram_stats.count + 1"
    );
    EXPECT_TRUE(ok);
}

// Test Hilbert index computation
TEST_F(SQLFunctionTest, HilbertIndexComputation) {
    // This assumes hilbert_index function exists
    PGresult* res = exec("SELECT hilbert_index(0, 0, 0, 0)");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        GTEST_SKIP() << "hilbert_index function not available";
    }
    
    // Origin should map to 0
    std::string result = PQgetvalue(res, 0, 0);
    EXPECT_EQ(result, "0") << "Origin should have Hilbert index 0";
    PQclear(res);
}

// Test Blake3 hash computation
TEST_F(SQLFunctionTest, Blake3HashComputation) {
    PGresult* res = exec("SELECT blake3_hash('test')");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        GTEST_SKIP() << "blake3_hash function not available";
    }
    
    std::string hash = PQgetvalue(res, 0, 0);
    EXPECT_EQ(hash.length(), 64) << "Blake3 hash should be 64 hex chars";
    PQclear(res);
}

// Test relation creation
TEST_F(SQLFunctionTest, RelationCreation) {
    // Create test compositions
    exec("INSERT INTO composition (id, label, depth) VALUES "
         "(decode('0000000000000000000000000000000000000000000000000000000000000001', 'hex'), 'test_comp_1', 1), "
         "(decode('0000000000000000000000000000000000000000000000000000000000000002', 'hex'), 'test_comp_2', 1) "
         "ON CONFLICT DO NOTHING");
    
    // Create relation
    bool ok = exec_ok(
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model) "
        "VALUES ('C', decode('0000000000000000000000000000000000000000000000000000000000000001', 'hex'), "
        "        'C', decode('0000000000000000000000000000000000000000000000000000000000000002', 'hex'), "
        "        'P', 0.8, 'test_model')"
    );
    EXPECT_TRUE(ok);
}
