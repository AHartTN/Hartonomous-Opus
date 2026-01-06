#include "hypercube/db/atom_cache.hpp"
#include "hypercube/db/helpers.hpp"
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

namespace hypercube::db {

bool load_atoms_for_codepoints(
    PGconn* conn,
    const std::unordered_set<uint32_t>& codepoints,
    std::unordered_map<uint32_t, AtomInfo>& cache
) {
    if (codepoints.empty()) return true;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Build PostgreSQL array literal: ARRAY[1,2,3]
    std::ostringstream arr;
    arr << "ARRAY[";
    bool first = true;
    for (uint32_t cp : codepoints) {
        if (!first) arr << ",";
        arr << cp;
        first = false;
    }
    arr << "]::INTEGER[]";
    
    // Use SQL function: get_atoms_by_codepoints(INTEGER[])
    std::string query = "SELECT * FROM get_atoms_by_codepoints(" + arr.str() + ")";
    
    PGresult* res = PQexec(conn, query.c_str());
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    
    int rows = PQntuples(res);
    if (rows == 0) {
        std::cerr << "ERROR: No atoms found for requested codepoints\n";
        PQclear(res);
        return false;
    }
    
    for (int i = 0; i < rows; ++i) {
        uint32_t cp = get_uint32(res, i, 0);
        
        AtomInfo info;
        info.hash = get_hash_from_hex(res, i, 1);
        info.coord_x = get_int64(res, i, 2);
        info.coord_y = get_int64(res, i, 3);
        info.coord_z = get_int64(res, i, 4);
        info.coord_m = get_int64(res, i, 5);
        
        cache[cp] = info;
    }
    
    PQclear(res);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[CACHE] Loaded " << rows << " atoms for " << codepoints.size() 
              << " unique codepoints in " << ms << " ms\n";
    
    return true;
}

const AtomInfo* get_cached_atom(
    const std::unordered_map<uint32_t, AtomInfo>& cache,
    uint32_t codepoint
) {
    auto it = cache.find(codepoint);
    return (it != cache.end()) ? &it->second : nullptr;
}

} // namespace hypercube::db
