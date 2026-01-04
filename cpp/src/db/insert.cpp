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
    // Look in COMPOSITION table, not atom
    std::ostringstream query;
    query << "SELECT encode(id, 'hex') FROM composition WHERE id IN (";

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
    
    // =========================================================================
    // NEW 4-TABLE SCHEMA: Insert into composition + composition_child tables
    // =========================================================================
    
    // Step 1: COPY compositions into composition table
    res = PQexec(conn, "COPY composition (id, label, depth, child_count, atom_count) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY composition failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // Send composition rows
    for (const auto& c : comps) {
        std::ostringstream line;
        
        // id (bytea hex format)
        line << "\\\\x" << c.hash.to_hex() << "\t";
        
        // label (NULL for auto-discovered patterns)
        line << "\\N\t";
        
        // depth, child_count, atom_count
        line << c.depth << "\t"
             << c.children.size() << "\t"
             << c.atom_count << "\n";
        
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
        std::cerr << "COPY composition result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    size_t comp_inserted = comps.size();
    
    // Step 2: COPY children into composition_child table
    res = PQexec(conn, "COPY composition_child (composition_id, ordinal, child_type, child_id) FROM STDIN WITH (FORMAT text)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY composition_child failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // Send child rows
    size_t child_count = 0;
    for (const auto& c : comps) {
        // If composition depth == 1, all children are atoms ('A')
        // If depth > 1, children are compositions ('C')
        char child_type = (c.depth == 1) ? 'A' : 'C';
        
        for (size_t i = 0; i < c.children.size(); ++i) {
            std::ostringstream line;
            
            // composition_id
            line << "\\\\x" << c.hash.to_hex() << "\t";
            
            // ordinal (0-based)
            line << i << "\t";
            
            // child_type
            line << child_type << "\t";
            
            // child_id
            line << "\\\\x" << c.children[i].hash.to_hex() << "\n";
            
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
        std::cerr << "COPY composition_child result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[DB] Inserted " << comp_inserted << " compositions with " 
              << child_count << " children in " << ms << " ms\n";
    
    return true;
}

} // namespace hypercube::db
