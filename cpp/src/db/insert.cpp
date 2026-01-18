#include "hypercube/db/insert.hpp"
#include "hypercube/db/geometry.hpp"
#include <iostream>
#include <chrono>
#include <sstream>

namespace hypercube::db {

// ============================================================================
// ARCHITECTURE: C++ computes everything, SQL just stores
// - Labels computed in C++ during PMI contraction
// - No round-trips: use ON CONFLICT DO NOTHING for deduplication
// - All data written in single batch COPY
// ============================================================================

// Helper to escape string for PostgreSQL COPY TEXT format
static std::string escape_copy_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '\\': result += "\\\\"; break;
            case '\t': result += "\\t"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            default: result += c; break;
        }
    }
    return result;
}

size_t insert_new_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps) {
    // No round-trip! Just insert directly - DB handles dedup via ON CONFLICT
    if (comps.empty()) return 0;
    return insert_compositions(conn, comps) ? comps.size() : 0;
}

bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps) {
    if (comps.empty()) return true;

    auto start = std::chrono::high_resolution_clock::now();

    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);

    // =========================================================================
    // Create temp table, COPY into it, then INSERT with ON CONFLICT DO NOTHING
    // This avoids round-trips while handling duplicates efficiently
    // =========================================================================

    // Create temp table for compositions
    res = PQexec(conn, R"(
        CREATE TEMP TABLE temp_composition (
            id bytea PRIMARY KEY,
            label text,
            depth integer,
            child_count integer,
            atom_count bigint,
            geom geometry(LINESTRINGZM, 0),
            centroid geometry(POINTZM, 0),
            hilbert_lo bigint,
            hilbert_hi bigint
        ) ON COMMIT DROP
    )");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "CREATE TEMP TABLE composition failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Create temp table for children
    res = PQexec(conn, R"(
        CREATE TEMP TABLE temp_composition_child (
            composition_id bytea,
            ordinal integer,
            child_type char(1),
            child_id bytea
        ) ON COMMIT DROP
    )");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "CREATE TEMP TABLE child failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Step 1: COPY compositions into temp table (labels computed in C++)
    res = PQexec(conn, "COPY temp_composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY temp_composition failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Send composition rows with labels computed in C++
    for (const auto& c : comps) {
        std::ostringstream line;

        // id (bytea hex format)
        line << "\\\\x" << c.hash.to_hex() << "\t";

        // label (computed in C++, not SQL!)
        if (c.label.empty()) {
            line << "\\N\t";
        } else {
            line << escape_copy_string(c.label) << "\t";
        }

        // depth, child_count, atom_count
        line << c.depth << "\t"
             << c.children.size() << "\t"
             << c.atom_count << "\t";

        // geom (LINESTRINGZM from child coordinates)
        if (c.children.size() >= 2) {
            std::vector<std::array<int32_t, 4>> points;
            points.reserve(c.children.size());
            for (const auto& child : c.children) {
                points.push_back({child.x, child.y, child.z, child.m});
            }
            line << build_linestringzm_ewkb(points) << "\t";
        } else {
            line << "\\N\t";
        }

        // centroid (POINTZM)
        line << build_pointzm_ewkb(c.coord_x, c.coord_y, c.coord_z, c.coord_m) << "\t";

        // hilbert_lo, hilbert_hi
        line << c.hilbert_lo << "\t" << c.hilbert_hi << "\n";

        std::string data = line.str();
        if (PQputCopyData(conn, data.c_str(), static_cast<int>(data.size())) != 1) {
            std::cerr << "PQputCopyData composition failed\n";
            PQputCopyEnd(conn, "error");
            PQexec(conn, "ROLLBACK");
            return false;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "PQputCopyEnd composition failed\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }

    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY temp_composition result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Step 2: COPY children into temp table
    res = PQexec(conn, "COPY temp_composition_child (composition_id, ordinal, child_type, child_id) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY temp_composition_child failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Send child rows
    size_t child_count = 0;
    for (const auto& c : comps) {
        for (size_t i = 0; i < c.children.size(); ++i) {
            char child_type = c.children[i].is_atom ? 'A' : 'C';

            std::ostringstream line;
            line << "\\\\x" << c.hash.to_hex() << "\t"
                 << i << "\t"
                 << child_type << "\t"
                 << "\\\\x" << c.children[i].hash.to_hex() << "\n";

            std::string data = line.str();
            if (PQputCopyData(conn, data.c_str(), static_cast<int>(data.size())) != 1) {
                std::cerr << "PQputCopyData child failed\n";
                PQputCopyEnd(conn, "error");
                PQexec(conn, "ROLLBACK");
                return false;
            }
            child_count++;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "PQputCopyEnd child failed\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }

    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY temp_composition_child result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Step 3: INSERT from temp tables with ON CONFLICT DO NOTHING (dedup at DB level)
    res = PQexec(conn, R"(
        INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi)
        SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi
        FROM temp_composition
        ON CONFLICT (id) DO NOTHING
    )");
    int comps_inserted = 0;
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        comps_inserted = atoi(PQcmdTuples(res));
    } else {
        std::cerr << "INSERT composition failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);

    // Calculate expected children for newly inserted compositions
    int expected_children = 0;
    res = PQexec(conn, "SELECT COALESCE(SUM(child_count), 0)::text FROM temp_composition WHERE id IN (SELECT id FROM composition)");
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        expected_children = atoi(PQgetvalue(res, 0, 0));
    } else {
        std::cerr << "Failed to calculate expected children: " << PQerrorMessage(conn) << std::endl;
        // Continue, but warn
    }
    PQclear(res);

    // Insert children only for newly inserted compositions
    res = PQexec(conn, R"(
        INSERT INTO composition_child (composition_id, ordinal, child_type, child_id)
        SELECT tc.composition_id, tc.ordinal, tc.child_type, tc.child_id
        FROM temp_composition_child tc
        WHERE EXISTS (SELECT 1 FROM composition c WHERE c.id = tc.composition_id)
        ON CONFLICT DO NOTHING
    )");
    int children_inserted = 0;
    bool children_success = false;
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        children_inserted = atoi(PQcmdTuples(res));
        children_success = true;
    } else {
        std::cerr << "INSERT children failed: " << PQerrorMessage(conn) << std::endl;
        if (comps_inserted > 0) {
            std::cerr << "WARNING: Child insertion failed after successfully inserting " << comps_inserted << " compositions. Committing transaction for later retry. Error: " << PQerrorMessage(conn) << std::endl;
            // Commit the transaction
            PGresult* commit_res = PQexec(conn, "COMMIT");
            PQclear(commit_res);
            return true; // Partial success
        } else {
            PQclear(res);
            PQexec(conn, "ROLLBACK");
            return false;
        }
    }
    PQclear(res);

    // If children insertion succeeded, validate count and commit
    if (children_success) {
        if (children_inserted != expected_children) {
            std::cerr << "WARNING: Expected " << expected_children << " children for inserted compositions, but inserted " << children_inserted << std::endl;
        }
        res = PQexec(conn, "COMMIT");
        PQclear(res);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        size_t skipped = comps.size() - comps_inserted;
        std::cerr << "[DB] Inserted " << comps_inserted << " compositions, "
                  << children_inserted << " children";
        if (skipped > 0) {
            std::cerr << " (skipped " << skipped << " existing)";
        }
        if (children_inserted != expected_children) {
            std::cerr << " (expected " << expected_children << " children)";
        }
        std::cerr << " (" << ms << " ms)\n";

        return true;
    }

    // If we reach here, children failed but no compositions were inserted (already handled above)
}

} // namespace hypercube::db
