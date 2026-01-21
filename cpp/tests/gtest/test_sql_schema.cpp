// =============================================================================
// SQL Schema Tests
// =============================================================================

#include <gtest/gtest.h>
#include <libpq-fe.h>
#include <cstdlib>
#include <string>

class SQLSchemaTest : public ::testing::Test {
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
    
    bool table_exists(const char* table) {
        std::string query = "SELECT 1 FROM information_schema.tables WHERE table_name = '" 
                          + std::string(table) + "'";
        PGresult* res = PQexec(conn, query.c_str());
        bool exists = PQntuples(res) > 0;
        PQclear(res);
        return exists;
    }
    
    bool function_exists(const char* func) {
        std::string query = "SELECT 1 FROM pg_proc WHERE proname = '" 
                          + std::string(func) + "'";
        PGresult* res = PQexec(conn, query.c_str());
        bool exists = PQntuples(res) > 0;
        PQclear(res);
        return exists;
    }
};

// Test core tables exist
TEST_F(SQLSchemaTest, CoreTablesExist) {
    EXPECT_TRUE(table_exists("atom")) << "atom table should exist";
    EXPECT_TRUE(table_exists("composition")) << "composition table should exist";
    EXPECT_TRUE(table_exists("relation")) << "relation table should exist";
}

// Test model registry table
TEST_F(SQLSchemaTest, ModelRegistryExists) {
    EXPECT_TRUE(table_exists("model")) << "model table should exist";
}

// Test bigram stats table
TEST_F(SQLSchemaTest, BigramStatsExists) {
    EXPECT_TRUE(table_exists("bigram_stats")) << "bigram_stats table should exist";
}

// Test core functions exist
TEST_F(SQLSchemaTest, CoreFunctionsExist) {
    EXPECT_TRUE(function_exists("atom_knn")) << "atom_knn should exist";
    EXPECT_TRUE(function_exists("atom_reconstruct_text")) << "atom_reconstruct_text should exist";
}

// Test query API functions exist
TEST_F(SQLSchemaTest, QueryAPIFunctionsExist) {
    EXPECT_TRUE(function_exists("search_text")) << "search_text should exist";
    EXPECT_TRUE(function_exists("semantic_neighbors")) << "semantic_neighbors should exist";
}

// Test PostGIS extension
TEST_F(SQLSchemaTest, PostGISEnabled) {
    PGresult* res = PQexec(conn, "SELECT PostGIS_Version()");
    bool has_postgis = PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0;
    PQclear(res);
    EXPECT_TRUE(has_postgis) << "PostGIS should be installed";
}

// Test geometry columns
TEST_F(SQLSchemaTest, GeometryColumns) {
    PGresult* res = PQexec(conn, 
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'composition' AND column_name = 'centroid'");
    bool has_centroid = PQntuples(res) > 0;
    PQclear(res);
    EXPECT_TRUE(has_centroid) << "composition should have centroid column";
}
