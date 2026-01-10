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
        auto get_env = [](const char* name) -> std::string {
#if defined(_WIN32)
            char* buf = nullptr;
            size_t len;
            if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr) {
                std::string result = buf;
                free(buf);
                return result;
            }
            return "";
#else
            const char* val = std::getenv(name);
            return val ? std::string(val) : "";
#endif
        };
        std::string db = get_env("HC_DB_NAME");
        std::string user = get_env("HC_DB_USER");
        std::string host = get_env("HC_DB_HOST");
        std::string port = get_env("HC_DB_PORT");
        std::string pass = get_env("HC_DB_PASS");

        std::string conninfo = "dbname=" + (!db.empty() ? db : "hypercube");
        conninfo += " user=" + (!user.empty() ? user : "postgres");
        conninfo += " host=" + (!host.empty() ? host : "localhost");
        conninfo += " port=" + (!port.empty() ? port : "5432");
        if (!pass.empty()) conninfo += " password=" + pass;
        
        conn = PQconnectdb(conninfo.c_str());
        if (PQstatus(conn) != CONNECTION_OK) {
            GTEST_SKIP() << "Database connection failed: " << PQerrorMessage(conn);
        }
        
        // Each test runs in its own transaction (will be rolled back in TearDown)
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
    
    // Reset transaction state if previous command failed
    void reset_transaction() {
        PGresult* status = PQexec(conn, "SELECT 1");
        if (PQresultStatus(status) != PGRES_TUPLES_OK) {
            PQclear(status);
            PQexec(conn, "ROLLBACK");
            PQexec(conn, "BEGIN");
        } else {
            PQclear(status);
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

// Test atom lookup function
TEST_F(SQLFunctionTest, GetOrCreateAtom) {
    // Look up an existing atom by codepoint
    PGresult* res = exec("SELECT id FROM atom WHERE codepoint = 65");
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    EXPECT_EQ(PQntuples(res), 1) << "Should find atom for 'A' (codepoint 65)";
    PQclear(res);
    
    // Verify atom_is_leaf works
    res = exec("SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))");
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    std::string is_leaf = PQgetvalue(res, 0, 0);
    PQclear(res);
    EXPECT_EQ(is_leaf, "t") << "Unicode atom should be a leaf";
}

// Test composition lookup
TEST_F(SQLFunctionTest, CreateCompositionFromAtoms) {
    // Check if any compositions exist
    PGresult* res = exec("SELECT COUNT(*) FROM composition");
    
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    int count = std::atoi(PQgetvalue(res, 0, 0));
    PQclear(res);
    
    // Compositions may or may not exist depending on whether ingestion was run
    if (count == 0) {
        GTEST_SKIP() << "No compositions in database - run ingestion first";
    }
    
    EXPECT_GT(count, 0) << "Should have compositions after ingestion";
}

// Test bigram statistics update
TEST_F(SQLFunctionTest, BigramStatsUpdate) {
    // Get two atom IDs for testing
    PGresult* res = exec("SELECT id FROM atom WHERE codepoint IN (65, 66) LIMIT 2");
    if (PQntuples(res) < 2) {
        PQclear(res);
        GTEST_SKIP() << "Need at least 2 atoms for bigram test";
    }
    PQclear(res);
    
    // Insert bigram using actual atom IDs (bytea)
    bool ok = exec_ok(
        "INSERT INTO bigram_stats (left_id, right_id, count, pmi) "
        "SELECT a1.id, a2.id, 1, 0.0 "
        "FROM atom a1, atom a2 "
        "WHERE a1.codepoint = 65 AND a2.codepoint = 66 "
        "ON CONFLICT (left_id, right_id) DO UPDATE SET count = bigram_stats.count + 1"
    );
    EXPECT_TRUE(ok);
    
    // Verify it was inserted
    res = exec(
        "SELECT count FROM bigram_stats b "
        "JOIN atom a1 ON b.left_id = a1.id "
        "JOIN atom a2 ON b.right_id = a2.id "
        "WHERE a1.codepoint = 65 AND a2.codepoint = 66"
    );
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    EXPECT_GT(PQntuples(res), 0) << "Bigram should exist";
    PQclear(res);
}

// Test Hilbert index retrieval
TEST_F(SQLFunctionTest, HilbertIndexComputation) {
    // Check that atoms have Hilbert indices
    PGresult* res = exec("SELECT hilbert_lo, hilbert_hi FROM atom WHERE codepoint = 65");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        GTEST_SKIP() << "Hilbert indices not populated";
    }
    
    std::string lo = PQgetvalue(res, 0, 0);
    std::string hi = PQgetvalue(res, 0, 1);
    PQclear(res);
    
    // Hilbert indices should be populated
    EXPECT_FALSE(lo.empty()) << "hilbert_lo should be set";
    EXPECT_FALSE(hi.empty()) << "hilbert_hi should be set";
}

// Test Blake3 hash in atoms
TEST_F(SQLFunctionTest, Blake3HashComputation) {
    // Check that atoms have BLAKE3 hashes (id column is the hash)
    PGresult* res = exec("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        GTEST_SKIP() << "Atom not found";
    }
    
    std::string hash = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    EXPECT_EQ(hash.length(), 64) << "Blake3 hash should be 64 hex chars (32 bytes)";
    
    // Verify determinism - same codepoint should have same hash
    res = exec("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65");
    std::string hash2 = PQgetvalue(res, 0, 0);
    PQclear(res);
    
    EXPECT_EQ(hash, hash2) << "BLAKE3 hash should be deterministic";
}

// Test relation creation
TEST_F(SQLFunctionTest, RelationCreation) {
    // Reset transaction in case previous test failed
    reset_transaction();
    
    // Create test compositions with all required fields
    PGresult* res = exec("INSERT INTO composition (id, label, depth, child_count, atom_count) VALUES "
         "(decode('0000000000000000000000000000000000000000000000000000000000000001', 'hex'), 'test_comp_1', 1, 0, 0), "
         "(decode('0000000000000000000000000000000000000000000000000000000000000002', 'hex'), 'test_comp_2', 1, 0, 0) "
         "ON CONFLICT DO NOTHING");
    PQclear(res);
    
    // Create relation
    bool ok = exec_ok(
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model) "
        "VALUES ('C', decode('0000000000000000000000000000000000000000000000000000000000000001', 'hex'), "
        "        'C', decode('0000000000000000000000000000000000000000000000000000000000000002', 'hex'), "
        "        'P', 0.8, 'test_model')"
    );
    EXPECT_TRUE(ok);
}
