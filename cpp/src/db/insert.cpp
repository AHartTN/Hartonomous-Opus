#include "hypercube/db/insert.hpp"
#include "hypercube/db/geometry.hpp"
#include <iostream>
#include <chrono>
#include <sstream>

namespace hypercube::db {

std::unordered_set<std::string> check_existing_compositions(
    PGconn* conn,
    const std::vector<ingest::CompositionRecord>& comps) {

    std::unordered_set<std::string> existing;
    if (comps.empty()) return existing;

    // Build array of hashes to check (batch query)
    // Use VALUES list for efficient bulk lookup
    std::ostringstream query;
    query << "SELECT encode(id, 'hex') FROM atom WHERE id IN (";

    for (size_t i = 0; i < comps.size(); ++i) {
        if (i > 0) query << ",";
        query << "'\\x" << comps[i].hash.to_hex() << "'::bytea";
    }
    query << ")";

    PGresult* res = PQexec(conn, query.str().c_str());
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        int nrows = PQntuples(res);
        for (int i = 0; i < nrows; ++i) {
            existing.insert(PQgetvalue(res, i, 0));
        }
    }
    PQclear(res);

    return existing;
}

std::vector<ingest::CompositionRecord> filter_new_compositions(
    PGconn* conn,
    const std::vector<ingest::CompositionRecord>& comps) {

    if (comps.empty()) return {};

    auto existing = check_existing_compositions(conn, comps);

    std::vector<ingest::CompositionRecord> new_comps;
    new_comps.reserve(comps.size() - existing.size());

    for (const auto& c : comps) {
        if (existing.find(c.hash.to_hex()) == existing.end()) {
            new_comps.push_back(c);
        }
    }

    return new_comps;
}

size_t insert_new_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps) {
    if (comps.empty()) return 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Filter to only new compositions
    auto new_comps = filter_new_compositions(conn, comps);

    auto filter_end = std::chrono::high_resolution_clock::now();
    auto filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(filter_end - start).count();

    size_t skipped = comps.size() - new_comps.size();
    if (skipped > 0) {
        std::cerr << "[DEDUP] Skipped " << skipped << " existing compositions ("
                  << filter_ms << " ms to check)\n";
    }

    if (new_comps.empty()) return 0;

    // Insert only new compositions
    if (insert_compositions(conn, new_comps)) {
        return new_comps.size();
    }
    return 0;
}

bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps) {
    if (comps.empty()) return true;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Use SQL function to prepare staging table
    res = PQexec(conn, "SELECT batch_insert_prepare()");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to prepare batch: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY into staging table - use TEXT format (tab delimited)
    res = PQexec(conn, "COPY tmp_atom FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // Send data rows (tab-delimited, \N for NULL)
    for (const auto& c : comps) {
        std::ostringstream line;
        
        // id (bytea hex format: \x followed by hex)
        line << "\\\\x" << c.hash.to_hex() << "\t";
        
        // geom (EWKB hex) - build LINESTRINGZM from all N children
        std::vector<std::array<int32_t, 4>> points;
        points.reserve(c.children.size());
        for (const auto& child : c.children) {
            points.push_back({child.x, child.y, child.z, child.m});
        }
        line << build_linestringzm_ewkb(points) << "\t";
        
        // children array - PostgreSQL array literal format for bytea[]
        // Format: {"\\x<hex>","\\x<hex>",...}
        line << "{";
        for (size_t i = 0; i < c.children.size(); ++i) {
            if (i > 0) line << ",";
            line << "\"\\\\\\\\x" << c.children[i].hash.to_hex() << "\"";
        }
        line << "}\t";
        
        // value (NULL for compositions)
        line << "\\N\t";
        
        // hilbert_lo, hilbert_hi, depth, atom_count
        line << c.hilbert_lo << "\t" << c.hilbert_hi << "\t"
             << c.depth << "\t" << c.atom_count << "\n";
        
        std::string data = line.str();
        if (PQputCopyData(conn, data.c_str(), static_cast<int>(data.size())) != 1) {
            std::cerr << "PQputCopyData failed\n";
            PQputCopyEnd(conn, "error");
            PQexec(conn, "ROLLBACK");
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "PQputCopyEnd failed\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // Use SQL function to finalize (upsert from staging to atom)
    res = PQexec(conn, "SELECT batch_insert_finalize()");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Finalize failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    int inserted = std::atoi(PQgetvalue(res, 0, 0));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[DB] Inserted " << inserted << " of " << comps.size() 
              << " compositions in " << ms << " ms\n";
    
    return true;
}

} // namespace hypercube::db
