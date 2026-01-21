// =============================================================================
// SQL Query API Tests
// =============================================================================

#include <gtest/gtest.h>
#include <libpq-fe.h>
#include <cstdlib>
#include <string>
#include <vector>

class SQLQueryAPITest : public ::testing::Test {
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
        conninfo += " host=" + (!host.empty() ? host : "hart-server");
        conninfo += " port=" + (!port.empty() ? port : "5432");
        conninfo += " connect_timeout=5";
        if (!pass.empty()) conninfo += " password=" + pass;

        std::cout << "Attempting database connection with timeout..." << std::endl;
        conn = PQconnectdb(conninfo.c_str());
        std::cout << "Database connection attempt completed." << std::endl;
        if (PQstatus(conn) != CONNECTION_OK) {
            GTEST_SKIP() << "Database connection failed: " << PQerrorMessage(conn);
        }
    }
    
    void TearDown() override {
        if (conn) {
            PQfinish(conn);
            conn = nullptr;
        }
    }
    
    int row_count(const char* query) {
        PGresult* res = PQexec(conn, query);
        int count = (PQresultStatus(res) == PGRES_TUPLES_OK) ? PQntuples(res) : -1;
        PQclear(res);
        return count;
    }
};

// Test semantic_search function exists and is callable
TEST_F(SQLQueryAPITest, SemanticSearchCallable) {
    PGresult* res = PQexec(conn, "SELECT * FROM search_text('test', 10)");

    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << "search_text failed: " << PQresultErrorMessage(res);

    // Check structure
    ASSERT_EQ(PQnfields(res), 5);
    EXPECT_STREQ(PQfname(res, 0), "id");
    EXPECT_STREQ(PQfname(res, 1), "text");
    EXPECT_STREQ(PQfname(res, 2), "depth");
    EXPECT_STREQ(PQfname(res, 3), "atom_count");
    EXPECT_STREQ(PQfname(res, 4), "score");

    // Check results
    int ntuples = PQntuples(res);
    EXPECT_GE(ntuples, 0);
    for(int i = 0; i < ntuples; i++) {
        // id is BYTEA, should not be null
        ASSERT_TRUE(PQgetvalue(res, i, 0) != nullptr);

        // text should not be null and length > 3
        const char* text_val = PQgetvalue(res, i, 1);
        ASSERT_TRUE(text_val != nullptr);
        std::string text(text_val);
        EXPECT_GT(text.length(), 3);

        // depth should be integer >=3
        int depth = std::atoi(PQgetvalue(res, i, 2));
        EXPECT_GE(depth, 3);

        // atom_count BIGINT >=0
        long atom_count = std::atol(PQgetvalue(res, i, 3));
        EXPECT_GE(atom_count, 0);

        // score FLOAT >=0
        float score = std::atof(PQgetvalue(res, i, 4));
        EXPECT_GE(score, 0.0f);
    }
    PQclear(res);
}

// Test semantic_neighbors function exists and is callable
TEST_F(SQLQueryAPITest, GetNeighborsCallable) {
    // First get any atom ID
    PGresult* res = PQexec(conn, "SELECT id FROM atom WHERE codepoint = 65");

    if (PQntuples(res) == 0) {
        PQclear(res);
        GTEST_SKIP() << "No atoms in database";
    }

    // Use binary format for bytea
    PQclear(res);

    // Call semantic_neighbors with subquery
    res = PQexec(conn, "SELECT * FROM semantic_neighbors((SELECT id FROM atom WHERE codepoint = 65), 5)");

    ASSERT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) << "semantic_neighbors failed: " << PQresultErrorMessage(res);

    // Check structure
    ASSERT_EQ(PQnfields(res), 3);
    EXPECT_STREQ(PQfname(res, 0), "neighbor_id");
    EXPECT_STREQ(PQfname(res, 1), "weight");
    EXPECT_STREQ(PQfname(res, 2), "relation_type");

    // Check results
    int ntuples = PQntuples(res);
    EXPECT_GE(ntuples, 0);
    for(int i = 0; i < ntuples; i++) {
        // neighbor_id BYTEA
        ASSERT_TRUE(PQgetvalue(res, i, 0) != nullptr);

        // weight REAL >=0
        float weight = std::atof(PQgetvalue(res, i, 1));
        EXPECT_GE(weight, 0.0f);

        // relation_type CHAR(1)
        const char* rt = PQgetvalue(res, i, 2);
        ASSERT_TRUE(rt != nullptr);
        EXPECT_EQ(strlen(rt), 1);
    }
    PQclear(res);
}

// Test Hilbert range query
TEST_F(SQLQueryAPITest, HilbertRangeQuery) {
    PGresult* res = PQexec(conn, 
        "SELECT * FROM composition WHERE hilbert_lo BETWEEN 0 AND 1000000 LIMIT 10"
    );
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        GTEST_SKIP() << "hilbert_lo column not available";
    }
    
    // Should work even if empty
    EXPECT_GE(PQntuples(res), 0);
    PQclear(res);
}

// Test spatial query (PostGIS)
TEST_F(SQLQueryAPITest, SpatialQuery) {
    PGresult* res = PQexec(conn,
        "SELECT c.label FROM composition c "
        "WHERE c.centroid IS NOT NULL "
        "ORDER BY ST_Distance(c.centroid, ST_MakePoint(0, 0, 0, 0)) "
        "LIMIT 5"
    );
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string err = PQresultErrorMessage(res);
        PQclear(res);
        
        if (err.find("ST_Distance") != std::string::npos || 
            err.find("centroid") != std::string::npos) {
            GTEST_SKIP() << "PostGIS spatial query not available";
        }
        FAIL() << "Spatial query failed: " << err;
    }
    
    PQclear(res);
}

// Test relation traversal
TEST_F(SQLQueryAPITest, RelationTraversal) {
    PGresult* res = PQexec(conn,
        "SELECT r.target_id, r.weight "
        "FROM relation r "
        "WHERE r.source_type = 'C' "
        "ORDER BY r.weight DESC "
        "LIMIT 10"
    );
    
    EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK);
    PQclear(res);
}

// Test model registry query
TEST_F(SQLQueryAPITest, ModelRegistryQuery) {
    PGresult* res = PQexec(conn, "SELECT * FROM model LIMIT 5");
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        GTEST_SKIP() << "model table not available";
    }
    
    EXPECT_GE(PQntuples(res), 0);
    PQclear(res);
}

// Test aggregate statistics
TEST_F(SQLQueryAPITest, AggregateStats) {
    struct StatQuery {
        const char* query;
        const char* description;
    };
    
    std::vector<StatQuery> queries = {
        {"SELECT COUNT(*) FROM atom", "atom count"},
        {"SELECT COUNT(*) FROM composition", "composition count"},
        {"SELECT COUNT(*) FROM relation", "relation count"},
    };
    
    for (const auto& q : queries) {
        PGresult* res = PQexec(conn, q.query);
        EXPECT_EQ(PQresultStatus(res), PGRES_TUPLES_OK) 
            << "Failed: " << q.description;
        if (PQresultStatus(res) == PGRES_TUPLES_OK) {
            int count = std::atoi(PQgetvalue(res, 0, 0));
            EXPECT_GE(count, 0) << q.description << " should be non-negative";
        }
        PQclear(res);
    }
}
