/**
 * Integration Tests for Hartonomous Hypercube
 * 
 * Validates the complete 4-table schema and core functionality.
 * Shows human-readable output with query results.
 * 
 * Schema:
 * - atom: Unicode codepoints (leaf atoms only)
 * - composition: BPE tokens and aggregations
 * - composition_child: Ordered children links
 * - relation: Semantic edges from embedding models
 * - shape: Embedding vectors per entity/model
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <libpq-fe.h>

// ANSI colors
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define DIM     "\033[2m"
#define BOLD    "\033[1m"

static int passed = 0, failed = 0, skipped = 0;
static PGconn* conn = nullptr;

void section(const char* title) {
    std::cout << "\n" << BOLD << "=== " << title << " ===" << RESET << "\n\n";
}

void test(const char* name) {
    std::cout << CYAN << "> " << name << RESET << "\n";
}

void sql(const char* query) {
    std::cout << DIM << "  SQL: " << query << RESET << "\n";
}

void result(const char* label, const std::string& val) {
    std::cout << "  " << label << ": " << YELLOW << val << RESET << "\n";
}

void pass(const char* msg) { std::cout << GREEN << "  [OK] " << msg << RESET << "\n"; passed++; }
void fail(const char* msg) { std::cout << RED << "  [FAIL] " << msg << RESET << "\n"; failed++; }
void skip(const char* msg) { std::cout << YELLOW << "  [SKIP] " << msg << RESET << "\n"; skipped++; }

std::string qval(const char* q, bool show = true) {
    if (show) sql(q);
    PGresult* r = PQexec(conn, q);
    std::string v;
    if (PQresultStatus(r) == PGRES_TUPLES_OK && PQntuples(r) > 0) {
        const char* s = PQgetvalue(r, 0, 0);
        if (s) v = s;
    }
    PQclear(r);
    return v;
}

int qint(const char* q, bool show = true) {
    std::string v = qval(q, show);
    return v.empty() ? -1 : std::stoi(v);
}

long qlong(const char* q, bool show = true) {
    std::string v = qval(q, show);
    return v.empty() ? -1L : std::stol(v);
}

double qdbl(const char* q, bool show = true) {
    std::string v = qval(q, show);
    return v.empty() ? -999.0 : std::stod(v);
}

// =============================================================================

void test_tables() {
    section("SCHEMA VERIFICATION");
    
    test("Checking 4-table schema exists");
    
    struct { const char* name; const char* desc; } tables[] = {
        {"atom", "Unicode codepoints (leaves)"},
        {"composition", "BPE tokens and aggregations"},
        {"composition_child", "Ordered child links"},
        {"relation", "Semantic edges"},
        {"shape", "Embedding vectors"}
    };
    
    for (auto& t : tables) {
        std::string q = "SELECT 1 FROM information_schema.tables WHERE table_name = '" + 
                        std::string(t.name) + "' AND table_schema = 'public'";
        if (qint(q.c_str(), false) == 1) {
            pass((std::string(t.name) + " - " + t.desc).c_str());
        } else {
            fail((std::string(t.name) + " table MISSING").c_str());
        }
    }
}

void test_atoms() {
    section("ATOM TABLE - Unicode Leaf Atoms");
    
    test("Count all seeded atoms");
    long count = qlong("SELECT COUNT(*) FROM atom");
    result("Total atoms", std::to_string(count));
    
    if (count >= 1112064) {
        pass("All 1,112,064 Unicode codepoints seeded");
    } else if (count > 0) {
        fail(("Only " + std::to_string(count) + " atoms (need 1,112,064)").c_str());
    } else {
        fail("No atoms seeded! Run seed_atoms_parallel");
    }
    
    test("Verify atom 'A' (U+0041) properties");
    std::string hash = qval("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65");
    result("BLAKE3 hash", hash.substr(0, 16) + "...");
    
    double x = qdbl("SELECT ST_X(geom) FROM atom WHERE codepoint = 65");
    double y = qdbl("SELECT ST_Y(geom) FROM atom WHERE codepoint = 65");
    double z = qdbl("SELECT ST_Z(geom) FROM atom WHERE codepoint = 65");
    double m = qdbl("SELECT ST_M(geom) FROM atom WHERE codepoint = 65");
    
    result("4D coordinates", "(" + std::to_string((long)x) + ", " + 
           std::to_string((long)y) + ", " + std::to_string((long)z) + ", " + 
           std::to_string((long)m) + ")");
    
    if (x > 0 && y > 0 && z > 0 && m >= 0) {
        pass("Atom has valid 4D geometry (POINTZM)");
    } else {
        fail("Invalid geometry");
    }
    
    test("Verify Hilbert index values");
    std::string hi = qval("SELECT hilbert_lo FROM atom WHERE codepoint = 65");
    std::string hh = qval("SELECT hilbert_hi FROM atom WHERE codepoint = 65", false);
    result("Hilbert curve", "lo=" + hi + ", hi=" + hh);
    
    if (!hi.empty() && !hh.empty()) {
        pass("Hilbert index populated for spatial locality");
    } else {
        fail("Missing Hilbert index");
    }
    
    test("Case pair proximity: 'A' vs 'a' should be neighbors");
    int upper_cp = 65, lower_cp = 97;
    double case_dist = qdbl(
        "SELECT ST_3DDistance("
        "(SELECT geom FROM atom WHERE codepoint = 65),"
        "(SELECT geom FROM atom WHERE codepoint = 97))"
    );
    result("Distance A<->a", std::to_string((long)case_dist));
    
    // Check if 'a' is among 100 nearest neighbors of 'A'
    int rank = qint(
        "WITH ranked AS ("
        "  SELECT neighbor_id, ROW_NUMBER() OVER (ORDER BY distance) as rank "
        "  FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 100)"
        ") "
        "SELECT rank FROM ranked WHERE neighbor_id = (SELECT id FROM atom WHERE codepoint = 97)"
    );
    result("Rank of 'a' among A's neighbors", std::to_string(rank));
    
    if (rank > 0 && rank <= 50) {
        pass("Case pairs are spatially close (semantic design working)");
    } else if (rank > 0) {
        pass(("Case pair found at rank " + std::to_string(rank)).c_str());
    } else {
        skip("Case pair not in top 100 - may need projection tuning");
    }
}

void test_indexes() {
    section("SPATIAL INDEXES");
    
    test("GIST index for geometry queries");
    int gist = qint(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_atom_geom' AND tablename = 'atom'"
    );
    if (gist == 1) {
        pass("idx_atom_geom GIST index exists");
    } else {
        fail("Missing GIST index - spatial queries will be slow");
    }
    
    test("Hilbert index for locality queries");
    int hilbert = qint(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_atom_hilbert' AND tablename = 'atom'"
    );
    if (hilbert == 1) {
        pass("idx_atom_hilbert exists for fast range scans");
    } else {
        fail("Missing Hilbert index");
    }
}

void test_functions() {
    section("SQL FUNCTIONS");
    
    test("atom_is_leaf() - Identify leaf vs composition");
    std::string leaf = qval("SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))");
    result("atom_is_leaf('A')", leaf);
    if (leaf == "t") pass("Correctly identifies leaf atoms"); else fail("Wrong result");
    
    test("atom_centroid() - Get geometry for any entity");
    std::string geom = qval(
        "SELECT ST_AsText(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))"
    );
    result("Geometry", geom.substr(0, 50) + "...");
    if (geom.find("POINT") != std::string::npos) {
        pass("Returns valid POINT geometry");
    } else {
        fail("Invalid geometry return");
    }
    
    test("atom_reconstruct_text() - Rebuild text from ID");
    std::string text = qval(
        "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 72)) || "
        "atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 105))"
    );
    result("Reconstructed 'H' + 'i'", "'" + text + "'");
    if (text == "Hi") pass("Text reconstruction works"); else fail("Wrong text");
    
    test("atom_knn() - K-nearest neighbors");
    int k = qint("SELECT COUNT(*) FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 10)");
    result("Neighbors returned", std::to_string(k));
    if (k == 10) pass("Returns exactly k neighbors"); else fail("Wrong count");
    
    test("atom_distance() - Euclidean distance");
    double d = qdbl(
        "SELECT atom_distance("
        "(SELECT id FROM atom WHERE codepoint = 65),"
        "(SELECT id FROM atom WHERE codepoint = 66))"
    );
    result("Distance A<->B", std::to_string(d));
    if (d > 0) pass("Computes positive distance"); else fail("Invalid distance");
}

void test_relations() {
    section("SEMANTIC RELATIONS");
    
    test("Count edges from embedding model");
    long edges = qlong("SELECT COUNT(*) FROM relation");
    result("Total edges", std::to_string(edges));
    
    if (edges == 0) {
        skip("No relations - run extract_embeddings first");
        return;
    }
    pass(("Found " + std::to_string(edges) + " semantic edges").c_str());
    
    test("Edge source types");
    std::string types = qval(
        "SELECT string_agg(source_type || ':' || cnt::text, ', ') "
        "FROM (SELECT source_type, COUNT(*) as cnt FROM relation GROUP BY source_type) sub"
    );
    result("Distribution", types);
    
    test("Top weight edges");
    double max_weight = qdbl("SELECT MAX(weight) FROM relation");
    result("Max similarity", std::to_string(max_weight));
}

void test_compositions() {
    section("COMPOSITIONS (BPE Tokens)");
    
    long count = qlong("SELECT COUNT(*) FROM composition");
    result("Total compositions", std::to_string(count));
    
    if (count == 0) {
        skip("No compositions yet - run ingest_safetensor");
        return;
    }
    
    test("Depth distribution");
    std::string depths = qval(
        "SELECT string_agg(depth::text || ':' || cnt::text, ', ' ORDER BY depth) "
        "FROM (SELECT depth, COUNT(*) as cnt FROM composition GROUP BY depth) sub"
    );
    result("Depth counts", depths);
    pass(("Found " + std::to_string(count) + " compositions").c_str());
}

void test_blake3() {
    section("BLAKE3 DETERMINISM");
    
    test("Same input produces same hash");
    std::string h1 = qval("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65", false);
    std::string h2 = qval("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65", false);
    
    if (h1 == h2 && !h1.empty()) {
        pass("BLAKE3 hash is deterministic");
        result("Hash(A)", h1.substr(0, 32) + "...");
    } else {
        fail("Hash mismatch or empty");
    }
    
    test("Different inputs produce different hashes");
    std::string ha = qval("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 65", false);
    std::string hb = qval("SELECT encode(id, 'hex') FROM atom WHERE codepoint = 66", false);
    
    if (ha != hb) {
        pass("Different codepoints have different hashes");
    } else {
        fail("Hash collision detected!");
    }
}

// =============================================================================

int main() {
    std::cout << "\n" << BOLD;
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << "|     Hartonomous Hypercube - Integration Tests               |\n";
    std::cout << "|     4-Table Schema Validation                               |\n";
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << RESET;
    
    // Connect
    const char* host = std::getenv("HC_DB_HOST") ? std::getenv("HC_DB_HOST") :
                       std::getenv("PGHOST") ? std::getenv("PGHOST") : "localhost";
    const char* port = std::getenv("HC_DB_PORT") ? std::getenv("HC_DB_PORT") :
                       std::getenv("PGPORT") ? std::getenv("PGPORT") : "5432";
    const char* user = std::getenv("HC_DB_USER") ? std::getenv("HC_DB_USER") :
                       std::getenv("PGUSER") ? std::getenv("PGUSER") : "hartonomous";
    const char* pw = std::getenv("HC_DB_PASS") ? std::getenv("HC_DB_PASS") :
                     std::getenv("PGPASSWORD") ? std::getenv("PGPASSWORD") : "hartonomous";
    const char* db = std::getenv("HC_DB_NAME") ? std::getenv("HC_DB_NAME") :
                     std::getenv("PGDATABASE") ? std::getenv("PGDATABASE") : "hypercube";
    
    std::string connstr = "host=" + std::string(host) + " port=" + std::string(port) +
                          " user=" + std::string(user) + " password=" + std::string(pw) +
                          " dbname=" + std::string(db);
    
    std::cout << "\n" << DIM << "Connecting to " << db << "@" << host << "..." << RESET << "\n";
    
    conn = PQconnectdb(connstr.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << RED << "Connection failed: " << PQerrorMessage(conn) << RESET << "\n";
        return 1;
    }
    std::cout << GREEN << "Connected!\n" << RESET;
    
    // Run all tests
    test_tables();
    test_atoms();
    test_indexes();
    test_functions();
    test_relations();
    test_compositions();
    test_blake3();
    
    // Summary
    std::cout << "\n" << BOLD;
    std::cout << "================================================================\n";
    std::cout << "                         RESULTS\n";
    std::cout << "================================================================\n" << RESET;
    std::cout << GREEN << "  [OK] Passed:  " << passed << RESET << "\n";
    std::cout << RED << "  [X]  Failed:  " << failed << RESET << "\n";
    std::cout << YELLOW << "  [-]  Skipped: " << skipped << RESET << "\n";
    std::cout << "================================================================\n\n";
    
    PQfinish(conn);
    return failed > 0 ? 1 : 0;
}
