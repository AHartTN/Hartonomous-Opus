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
    EXPECT_TRUE(table_exists("model_registry")) << "model_registry table should exist";
}

// Test bigram stats table
TEST_F(SQLSchemaTest, BigramStatsExists) {
    EXPECT_TRUE(table_exists("bigram_stats")) << "bigram_stats table should exist";
}

// Test core functions exist
TEST_F(SQLSchemaTest, CoreFunctionsExist) {
    EXPECT_TRUE(function_exists("get_or_create_atom")) << "get_or_create_atom should exist";
    EXPECT_TRUE(function_exists("create_composition_from_atoms")) << "create_composition_from_atoms should exist";
}

// Test query API functions exist
TEST_F(SQLSchemaTest, QueryAPIFunctionsExist) {
    EXPECT_TRUE(function_exists("semantic_search")) << "semantic_search should exist";
    EXPECT_TRUE(function_exists("get_neighbors")) << "get_neighbors should exist";
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
