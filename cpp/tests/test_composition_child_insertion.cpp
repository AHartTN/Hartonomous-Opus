/**
 * Unit Tests for Composition Child Insertion
 *
 * Tests the insert_compositions() function to verify correct handling of:
 * - Normal insertion when all dependencies exist
 * - Insertion when some child atoms don't exist
 * - Child count validation and mismatch detection
 * - Error handling and partial success scenarios
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <libpq-fe.h>
#include "hypercube/types.hpp"
#include "hypercube/ingest/cpe.hpp"
#include "hypercube/db/insert.hpp"

// ANSI colors - disabled for clean logs
#define RESET   ""
#define GREEN   ""
#define RED     ""
#define YELLOW  ""
#define CYAN    ""
#define DIM     ""
#define BOLD    ""

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

void pass(const char* msg) { std::cout << "  PASS: " << msg << "\n"; passed++; }
void fail(const char* msg) { std::cout << "  FAIL: " << msg << "\n"; failed++; }
void skip(const char* msg) { std::cout << "  SKIP: " << msg << "\n"; skipped++; }

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

// Helper to create and insert a test atom
void insert_test_atom(const Blake3Hash& hash, uint32_t codepoint, int32_t x, int32_t y, int32_t z, int32_t m) {
    std::string hex = hash.to_hex();
    std::string query = "INSERT INTO atom (id, codepoint, value, geom, hilbert_lo, hilbert_hi) VALUES "
                       "('\\\\x" + hex + "', " + std::to_string(codepoint) + ", '\\\\x00', "
                       "'POINTZM(" + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) + " " + std::to_string(m) + ")', "
                       "0, 0) ON CONFLICT (id) DO NOTHING";
    PQexec(conn, query.c_str());
}

// Helper to clean test data
void cleanup_test_data() {
    PQexec(conn, "DELETE FROM composition_child WHERE composition_id IN (SELECT id FROM composition WHERE label LIKE 'test_%')");
    PQexec(conn, "DELETE FROM composition WHERE label LIKE 'test_%'");
    PQexec(conn, "DELETE FROM atom WHERE codepoint >= 1000000"); // Test codepoints
}

// =============================================================================

// Test normal insertion when all child atoms exist
void test_normal_insertion() {
    section("NORMAL INSERTION - All Dependencies Exist");

    test("Insert composition with existing children");
    PQexec(conn, "BEGIN");

    // Insert test atoms
    Blake3Hash atom1 = Blake3Hash::from_hex("1111111111111111111111111111111111111111111111111111111111111111");
    Blake3Hash atom2 = Blake3Hash::from_hex("2222222222222222222222222222222222222222222222222222222222222222");
    insert_test_atom(atom1, 1000000, 1000000, 1000000, 1000000, 1000000);
    insert_test_atom(atom2, 1000001, 1000001, 1000001, 1000001, 1000001);

    // Create composition
    hypercube::ingest::CompositionRecord comp;
    comp.hash = Blake3Hash::from_hex("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    comp.label = "test_normal";
    comp.depth = 1;
    comp.atom_count = 2;
    comp.coord_x = 1000000;
    comp.coord_y = 1000000;
    comp.coord_z = 1000000;
    comp.coord_m = 1000000;
    comp.hilbert_lo = 1;
    comp.hilbert_hi = 1;

    hypercube::ingest::ChildInfo child1;
    child1.hash = atom1;
    child1.is_atom = true;
    child1.x = 1000000;
    child1.y = 1000000;
    child1.z = 1000000;
    child1.m = 1000000;
    child1.codepoint = 1000000;
    child1.label = "test_atom1";

    hypercube::ingest::ChildInfo child2;
    child2.hash = atom2;
    child2.is_atom = true;
    child2.x = 1000001;
    child2.y = 1000001;
    child2.z = 1000001;
    child2.m = 1000001;
    child2.codepoint = 1000001;
    child2.label = "test_atom2";

    comp.children = {child1, child2};

    std::vector<hypercube::ingest::CompositionRecord> comps = {comp};

    bool success = hypercube::db::insert_compositions(conn, comps);
    if (success) {
        pass("insert_compositions returned true");

        // Verify composition inserted
        int comp_count = qint("SELECT COUNT(*) FROM composition WHERE id = '\\xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'");
        if (comp_count == 1) {
            pass("Composition inserted");
        } else {
            fail("Composition not inserted");
        }

        // Verify children inserted
        int child_count = qint("SELECT COUNT(*) FROM composition_child WHERE composition_id = '\\xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'");
        if (child_count == 2) {
            pass("Both children inserted");
        } else {
            fail(("Expected 2 children, got " + std::to_string(child_count)).c_str());
        }
    } else {
        fail("insert_compositions failed");
    }

    PQexec(conn, "ROLLBACK");
}

// Test insertion when some child atoms don't exist
void test_missing_dependencies() {
    section("MISSING DEPENDENCIES - Some Children Don't Exist");

    test("Insert composition with missing child atoms");
    PQexec(conn, "BEGIN");

    // Insert only one atom
    Blake3Hash atom1 = Blake3Hash::from_hex("3333333333333333333333333333333333333333333333333333333333333333");
    insert_test_atom(atom1, 1000002, 1000002, 1000002, 1000002, 1000002);

    // Missing atom
    Blake3Hash atom_missing = Blake3Hash::from_hex("4444444444444444444444444444444444444444444444444444444444444444");

    // Create composition
    hypercube::ingest::CompositionRecord comp;
    comp.hash = Blake3Hash::from_hex("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    comp.label = "test_missing";
    comp.depth = 1;
    comp.atom_count = 2;
    comp.coord_x = 1000002;
    comp.coord_y = 1000002;
    comp.coord_z = 1000002;
    comp.coord_m = 1000002;
    comp.hilbert_lo = 2;
    comp.hilbert_hi = 2;

    hypercube::ingest::ChildInfo child1;
    child1.hash = atom1;
    child1.is_atom = true;
    child1.x = 1000002;
    child1.y = 1000002;
    child1.z = 1000002;
    child1.m = 1000002;
    child1.codepoint = 1000002;
    child1.label = "test_atom3";

    hypercube::ingest::ChildInfo child2;
    child2.hash = atom_missing;
    child2.is_atom = true;
    child2.x = 1000003;
    child2.y = 1000003;
    child2.z = 1000003;
    child2.m = 1000003;
    child2.codepoint = 1000003;
    child2.label = "missing_atom";

    comp.children = {child1, child2};

    std::vector<hypercube::ingest::CompositionRecord> comps = {comp};

    bool success = hypercube::db::insert_compositions(conn, comps);
    if (success) {
        pass("insert_compositions returned true despite missing dependencies");

        // Verify composition inserted
        int comp_count = qint("SELECT COUNT(*) FROM composition WHERE id = '\\xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'");
        if (comp_count == 1) {
            pass("Composition inserted despite missing children");
        } else {
            fail("Composition not inserted");
        }

        // Verify children inserted (should insert links even for missing children)
        int child_count = qint("SELECT COUNT(*) FROM composition_child WHERE composition_id = '\\xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'");
        if (child_count == 2) {
            pass("Both child links inserted even for missing atoms");
        } else {
            fail(("Expected 2 child links, got " + std::to_string(child_count)).c_str());
        }
    } else {
        fail("insert_compositions failed due to missing dependencies");
    }

    PQexec(conn, "ROLLBACK");
}

// Test partial insertion when some compositions already exist
void test_partial_insertion() {
    section("PARTIAL INSERTION - Some Compositions Already Exist");

    test("Insert compositions where some already exist");
    PQexec(conn, "BEGIN");

    // Insert test atoms
    Blake3Hash atom1 = Blake3Hash::from_hex("5555555555555555555555555555555555555555555555555555555555555555");
    insert_test_atom(atom1, 1000004, 1000004, 1000004, 1000004, 1000004);

    // Insert existing composition
    hypercube::ingest::CompositionRecord existing_comp;
    existing_comp.hash = Blake3Hash::from_hex("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    existing_comp.label = "test_existing";
    existing_comp.depth = 1;
    existing_comp.atom_count = 1;
    existing_comp.coord_x = 1000004;
    existing_comp.coord_y = 1000004;
    existing_comp.coord_z = 1000004;
    existing_comp.coord_m = 1000004;
    existing_comp.hilbert_lo = 3;
    existing_comp.hilbert_hi = 3;

    hypercube::ingest::ChildInfo child1;
    child1.hash = atom1;
    child1.is_atom = true;
    child1.x = 1000004;
    child1.y = 1000004;
    child1.z = 1000004;
    child1.m = 1000004;
    child1.codepoint = 1000004;
    child1.label = "test_atom4";

    existing_comp.children = {child1};

    // Insert the existing one first
    std::vector<hypercube::ingest::CompositionRecord> first_batch = {existing_comp};
    hypercube::db::insert_compositions(conn, first_batch);

    // Now insert again with new composition
    Blake3Hash atom2 = Blake3Hash::from_hex("6666666666666666666666666666666666666666666666666666666666666666");
    insert_test_atom(atom2, 1000005, 1000005, 1000005, 1000005, 1000005);

    hypercube::ingest::CompositionRecord new_comp;
    new_comp.hash = Blake3Hash::from_hex("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
    new_comp.label = "test_new";
    new_comp.depth = 1;
    new_comp.atom_count = 1;
    new_comp.coord_x = 1000005;
    new_comp.coord_y = 1000005;
    new_comp.coord_z = 1000005;
    new_comp.coord_m = 1000005;
    new_comp.hilbert_lo = 4;
    new_comp.hilbert_hi = 4;

    hypercube::ingest::ChildInfo child2;
    child2.hash = atom2;
    child2.is_atom = true;
    child2.x = 1000005;
    child2.y = 1000005;
    child2.z = 1000005;
    child2.m = 1000005;
    child2.codepoint = 1000005;
    child2.label = "test_atom5";

    new_comp.children = {child2};

    std::vector<hypercube::ingest::CompositionRecord> second_batch = {existing_comp, new_comp}; // existing one first

    bool success = hypercube::db::insert_compositions(conn, second_batch);
    if (success) {
        pass("insert_compositions handled partial insertion");

        // Check existing composition still there
        int existing_count = qint("SELECT COUNT(*) FROM composition WHERE id = '\\xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'");
        if (existing_count == 1) {
            pass("Existing composition preserved");
        } else {
            fail("Existing composition missing");
        }

        // Check new composition inserted
        int new_count = qint("SELECT COUNT(*) FROM composition WHERE id = '\\xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'");
        if (new_count == 1) {
            pass("New composition inserted");
        } else {
            fail("New composition not inserted");
        }

        // Check children: only for new composition
        int total_children = qint("SELECT COUNT(*) FROM composition_child");
        int new_children = qint("SELECT COUNT(*) FROM composition_child WHERE composition_id = '\\xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'");
        if (total_children == 2 && new_children == 1) { // 1 from first insert, 1 from new
            pass("Children inserted only for new compositions");
        } else {
            fail(("Unexpected child counts: total=" + std::to_string(total_children) + ", new=" + std::to_string(new_children)).c_str());
        }
    } else {
        fail("insert_compositions failed on partial insertion");
    }

    PQexec(conn, "ROLLBACK");
}

// Test empty input
void test_empty_input() {
    section("EMPTY INPUT");

    test("Insert empty composition list");
    PQexec(conn, "BEGIN");

    std::vector<hypercube::ingest::CompositionRecord> comps;

    bool success = hypercube::db::insert_compositions(conn, comps);
    if (success) {
        pass("insert_compositions handles empty input");
    } else {
        fail("insert_compositions failed on empty input");
    }

    PQexec(conn, "ROLLBACK");
}

// Test composition with no children
void test_no_children() {
    section("COMPOSITION WITH NO CHILDREN");

    test("Insert composition with no children");
    PQexec(conn, "BEGIN");

    hypercube::ingest::CompositionRecord comp;
    comp.hash = Blake3Hash::from_hex("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    comp.label = "test_no_children";
    comp.depth = 1;
    comp.atom_count = 0;
    comp.coord_x = 0;
    comp.coord_y = 0;
    comp.coord_z = 0;
    comp.coord_m = 0;
    comp.hilbert_lo = 5;
    comp.hilbert_hi = 5;
    comp.children = {}; // No children

    std::vector<hypercube::ingest::CompositionRecord> comps = {comp};

    bool success = hypercube::db::insert_compositions(conn, comps);
    if (success) {
        pass("insert_compositions handles composition with no children");

        int comp_count = qint("SELECT COUNT(*) FROM composition WHERE id = '\\xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'");
        if (comp_count == 1) {
            pass("Composition with no children inserted");
        } else {
            fail("Composition with no children not inserted");
        }

        int child_count = qint("SELECT COUNT(*) FROM composition_child WHERE composition_id = '\\xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'");
        if (child_count == 0) {
            pass("No children inserted for composition with no children");
        } else {
            fail("Unexpected children for composition with no children");
        }
    } else {
        fail("insert_compositions failed for composition with no children");
    }

    PQexec(conn, "ROLLBACK");
}

// =============================================================================

int main() {
    std::cout << "\n" << BOLD;
    std::cout << "+----------------------------------------------------------------+\n";
    std::cout << "|     Composition Child Insertion Unit Tests                     |\n";
    std::cout << "|     Testing insert_compositions() function                    |\n";
    std::cout << "+----------------------------------------------------------------+\n";
    std::cout << RESET;

    // Connect
    auto get_env = [](const char* name) -> std::string {
#if defined(_WIN32)
        char* val = nullptr;
        size_t len;
        if (_dupenv_s(&val, &len, name) == 0 && val != nullptr) {
            std::string result(val);
            free(val);
            return result;
        }
        return "";
#else
        const char* val = std::getenv(name);
        return val ? val : "";
#endif
    };

    std::string host_str = get_env("HC_DB_HOST");
    if (host_str.empty()) host_str = get_env("PGHOST");
    if (host_str.empty()) host_str = "hart-server";
    const char* host = host_str.c_str();

    std::string port_str = get_env("HC_DB_PORT");
    if (port_str.empty()) port_str = get_env("PGPORT");
    if (port_str.empty()) port_str = "5432";
    const char* port = port_str.c_str();

    std::string user_str = get_env("HC_DB_USER");
    if (user_str.empty()) user_str = get_env("PGUSER");
    if (user_str.empty()) user_str = "postgres";
    const char* user = user_str.c_str();

    std::string pw_str = get_env("HC_DB_PASS");
    if (pw_str.empty()) pw_str = get_env("PGPASSWORD");
    if (pw_str.empty()) pw_str = "postgres";
    const char* pw = pw_str.c_str();

    std::string db_str = get_env("HC_DB_NAME");
    if (db_str.empty()) db_str = get_env("PGDATABASE");
    if (db_str.empty()) db_str = "hypercube";
    const char* db = db_str.c_str();

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

    // Clean up any leftover test data
    cleanup_test_data();

    // Run all tests
    test_normal_insertion();
    test_missing_dependencies();
    test_partial_insertion();
    test_empty_input();
    test_no_children();

    // Summary
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "                         RESULTS\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed:  " << passed << "\n";
    std::cout << "  Failed:  " << failed << "\n";
    std::cout << "  Skipped: " << skipped << "\n";
    std::cout << "================================================================\n\n";

    PQfinish(conn);
    return failed > 0 ? 1 : 0;
}