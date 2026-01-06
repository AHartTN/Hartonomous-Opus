/**
 * @file helpers.hpp
 * @brief PostgreSQL helper functions for consistent data access
 * 
 * Consolidates common patterns for:
 * - Result value extraction (with null/type handling)
 * - Hash conversion (bytea ↔ Blake3Hash)
 * - Coordinate parsing
 * - Query execution with error handling
 * 
 * DESIGN PRINCIPLE: Keep bytea as bytea in the database, convert at boundaries.
 * Avoid hex encoding in SQL - it doubles the size and adds overhead.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <stdexcept>
#include <libpq-fe.h>

#include "hypercube/types.hpp"

namespace hypercube::db {

// =============================================================================
// Result Value Extraction Helpers
// =============================================================================

/**
 * Safe extraction of string value from PGresult.
 * Returns empty string if null or out of bounds.
 */
inline std::string get_string(PGresult* res, int row, int col) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return {};
    }
    if (PQgetisnull(res, row, col)) {
        return {};
    }
    const char* val = PQgetvalue(res, row, col);
    return val ? val : "";
}

/**
 * Safe extraction of integer value from PGresult.
 * Returns default_val if null, empty, or parse error.
 */
inline int get_int(PGresult* res, int row, int col, int default_val = 0) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return default_val;
    }
    if (PQgetisnull(res, row, col)) {
        return default_val;
    }
    const char* val = PQgetvalue(res, row, col);
    if (!val || *val == '\0') {
        return default_val;
    }
    try {
        return std::stoi(val);
    } catch (...) {
        return default_val;
    }
}

/**
 * Safe extraction of int64 value from PGresult.
 */
inline int64_t get_int64(PGresult* res, int row, int col, int64_t default_val = 0) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return default_val;
    }
    if (PQgetisnull(res, row, col)) {
        return default_val;
    }
    const char* val = PQgetvalue(res, row, col);
    if (!val || *val == '\0') {
        return default_val;
    }
    try {
        return std::stoll(val);
    } catch (...) {
        return default_val;
    }
}

/**
 * Safe extraction of uint32 value from PGresult.
 */
inline uint32_t get_uint32(PGresult* res, int row, int col, uint32_t default_val = 0) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return default_val;
    }
    if (PQgetisnull(res, row, col)) {
        return default_val;
    }
    const char* val = PQgetvalue(res, row, col);
    if (!val || *val == '\0') {
        return default_val;
    }
    try {
        return static_cast<uint32_t>(std::stoul(val));
    } catch (...) {
        return default_val;
    }
}

/**
 * Safe extraction of double value from PGresult.
 */
inline double get_double(PGresult* res, int row, int col, double default_val = 0.0) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return default_val;
    }
    if (PQgetisnull(res, row, col)) {
        return default_val;
    }
    const char* val = PQgetvalue(res, row, col);
    if (!val || *val == '\0') {
        return default_val;
    }
    try {
        return std::stod(val);
    } catch (...) {
        return default_val;
    }
}

/**
 * Safe extraction of float value from PGresult.
 */
inline float get_float(PGresult* res, int row, int col, float default_val = 0.0f) {
    return static_cast<float>(get_double(res, row, col, default_val));
}

/**
 * Safe extraction of boolean value from PGresult.
 * Handles 't'/'f', 'true'/'false', '1'/'0'.
 */
inline bool get_bool(PGresult* res, int row, int col, bool default_val = false) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return default_val;
    }
    if (PQgetisnull(res, row, col)) {
        return default_val;
    }
    const char* val = PQgetvalue(res, row, col);
    if (!val || *val == '\0') {
        return default_val;
    }
    return val[0] == 't' || val[0] == 'T' || val[0] == '1';
}

// =============================================================================
// Blake3Hash ↔ bytea Conversion
// =============================================================================

/**
 * Extract Blake3Hash from bytea column (binary format).
 * Use with PQgetvalue after selecting with FORMAT_BINARY or from raw bytea.
 * 
 * For text format (default), use get_hash_from_hex or select with encode(col, 'hex').
 */
inline Blake3Hash get_hash_binary(PGresult* res, int row, int col) {
    if (!res || row >= PQntuples(res) || col >= PQnfields(res)) {
        return {};
    }
    if (PQgetisnull(res, row, col)) {
        return {};
    }
    
    // Check if binary format
    int format = PQfformat(res, col);
    if (format == 1) {
        // Binary format - direct copy
        int len = PQgetlength(res, row, col);
        if (len == 32) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(PQgetvalue(res, row, col));
            return Blake3Hash(data);
        }
    }
    
    // Text format - should be \x prefixed bytea escape
    const char* val = PQgetvalue(res, row, col);
    if (val && val[0] == '\\' && val[1] == 'x' && std::strlen(val) == 66) {
        return Blake3Hash::from_hex(std::string_view(val + 2, 64));
    }
    
    return {};
}

/**
 * Extract Blake3Hash from hex-encoded string column.
 * Use when selecting with: encode(id, 'hex') as id_hex
 */
inline Blake3Hash get_hash_from_hex(PGresult* res, int row, int col) {
    std::string hex = get_string(res, row, col);
    if (hex.size() == 64) {
        return Blake3Hash::from_hex(hex);
    }
    // Handle \x prefix (bytea escape format)
    if (hex.size() == 66 && hex[0] == '\\' && hex[1] == 'x') {
        return Blake3Hash::from_hex(std::string_view(hex).substr(2));
    }
    return {};
}

/**
 * Format Blake3Hash as PostgreSQL bytea literal.
 * Use in SQL: WHERE id = E'\\x...'
 */
inline std::string to_bytea_literal(const Blake3Hash& hash) {
    std::string result = "E'\\\\x";
    result += hash.to_hex();
    result += "'";
    return result;
}

/**
 * Format Blake3Hash for parameterized query.
 * Returns raw binary data for use with PQexecParams.
 */
inline const char* to_bytea_param(const Blake3Hash& hash) {
    return reinterpret_cast<const char*>(hash.data());
}

// =============================================================================
// Point4D ↔ PostGIS Conversion
// =============================================================================

/**
 * Extract Point4D from PostGIS POINTZM.
 * Use when selecting with: ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom)
 * Columns should be at col, col+1, col+2, col+3.
 */
inline Point4D get_point4d(PGresult* res, int row, int col) {
    return Point4D{
        static_cast<uint32_t>(get_double(res, row, col)),
        static_cast<uint32_t>(get_double(res, row, col + 1)),
        static_cast<uint32_t>(get_double(res, row, col + 2)),
        static_cast<uint32_t>(get_double(res, row, col + 3))
    };
}

// =============================================================================
// Query Execution Helpers
// =============================================================================

/**
 * RAII wrapper for PGresult.
 */
class Result {
public:
    Result() : res_(nullptr) {}
    explicit Result(PGresult* res) : res_(res) {}
    ~Result() { if (res_) PQclear(res_); }
    
    // Move only
    Result(Result&& other) noexcept : res_(other.res_) { other.res_ = nullptr; }
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            if (res_) PQclear(res_);
            res_ = other.res_;
            other.res_ = nullptr;
        }
        return *this;
    }
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;
    
    PGresult* get() const { return res_; }
    operator PGresult*() const { return res_; }
    PGresult* operator->() const { return res_; }
    
    bool ok() const {
        ExecStatusType status = PQresultStatus(res_);
        return status == PGRES_TUPLES_OK || status == PGRES_COMMAND_OK;
    }
    
    bool has_rows() const {
        return res_ && PQresultStatus(res_) == PGRES_TUPLES_OK && PQntuples(res_) > 0;
    }
    
    int ntuples() const { return res_ ? PQntuples(res_) : 0; }
    int nfields() const { return res_ ? PQnfields(res_) : 0; }
    
    bool is_null(int row, int col) const {
        if (!res_) return true;
        if (PQgetisnull(res_, row, col)) return true;
        const char* val = PQgetvalue(res_, row, col);
        return val == nullptr || val[0] == '\0';
    }
    
    std::string error_message() const {
        return res_ ? PQresultErrorMessage(res_) : "null result";
    }
    
    // Convenience accessors using the helper functions
    std::string str(int row, int col) const { return get_string(res_, row, col); }
    int integer(int row, int col, int def = 0) const { return get_int(res_, row, col, def); }
    int64_t int64(int row, int col, int64_t def = 0) const { return get_int64(res_, row, col, def); }
    double dbl(int row, int col, double def = 0.0) const { return get_double(res_, row, col, def); }
    float flt(int row, int col, float def = 0.0f) const { return get_float(res_, row, col, def); }
    bool boolean(int row, int col, bool def = false) const { return get_bool(res_, row, col, def); }
    Blake3Hash hash_hex(int row, int col) const { return get_hash_from_hex(res_, row, col); }
    Point4D point4d(int row, int col) const { return get_point4d(res_, row, col); }
    
private:
    PGresult* res_;
};

/**
 * Execute query and return RAII Result wrapper.
 */
inline Result exec(PGconn* conn, const char* sql) {
    return Result(PQexec(conn, sql));
}

inline Result exec(PGconn* conn, const std::string& sql) {
    return exec(conn, sql.c_str());
}

/**
 * Execute query and return single scalar value.
 */
inline std::optional<std::string> query_scalar(PGconn* conn, const char* sql) {
    Result res = exec(conn, sql);
    if (res.has_rows()) {
        return res.str(0, 0);
    }
    return std::nullopt;
}

inline std::optional<int> query_int(PGconn* conn, const char* sql) {
    Result res = exec(conn, sql);
    if (res.has_rows()) {
        return res.integer(0, 0);
    }
    return std::nullopt;
}

inline std::optional<int64_t> query_int64(PGconn* conn, const char* sql) {
    Result res = exec(conn, sql);
    if (res.has_rows()) {
        return res.int64(0, 0);
    }
    return std::nullopt;
}

/**
 * Get count of rows affected by last command.
 * Use after INSERT/UPDATE/DELETE.
 */
inline int cmd_tuples(PGresult* res) {
    if (!res) return 0;
    const char* val = PQcmdTuples(res);
    if (!val || *val == '\0') return 0;
    try {
        return std::stoi(val);
    } catch (...) {
        return 0;
    }
}

// =============================================================================
// Entity Type Helpers (avoid string comparisons)
// =============================================================================

/**
 * Entity type enum for relation source/target.
 * Stored as char in DB: 'A' = atom, 'C' = composition.
 */
enum class EntityType : char {
    Atom = 'A',
    Composition = 'C',
    Unknown = '?'
};

inline EntityType parse_entity_type(const char* val) {
    if (!val || *val == '\0') return EntityType::Unknown;
    switch (val[0]) {
        case 'A': case 'a': return EntityType::Atom;
        case 'C': case 'c': return EntityType::Composition;
        default: return EntityType::Unknown;
    }
}

inline EntityType get_entity_type(PGresult* res, int row, int col) {
    return parse_entity_type(PQgetvalue(res, row, col));
}

inline char entity_type_char(EntityType t) {
    return static_cast<char>(t);
}

inline const char* entity_type_str(EntityType t) {
    switch (t) {
        case EntityType::Atom: return "atom";
        case EntityType::Composition: return "composition";
        default: return "unknown";
    }
}

/**
 * Relation type enum.
 * 'E' = Embedding k-NN, 'R' = Router, 'W' = Weight similarity,
 * 'D' = Dimension activation, 'C' = BPE composition, 'P' = Proximity.
 */
enum class RelationType : char {
    Embedding = 'E',
    Router = 'R',
    Weight = 'W',
    Dimension = 'D',
    Composition = 'C',
    Proximity = 'P',
    Unknown = '?'
};

inline RelationType parse_relation_type(const char* val) {
    if (!val || *val == '\0') return RelationType::Unknown;
    switch (val[0]) {
        case 'E': case 'e': return RelationType::Embedding;
        case 'R': case 'r': return RelationType::Router;
        case 'W': case 'w': return RelationType::Weight;
        case 'D': case 'd': return RelationType::Dimension;
        case 'C': case 'c': return RelationType::Composition;
        case 'P': case 'p': return RelationType::Proximity;
        default: return RelationType::Unknown;
    }
}

inline RelationType get_relation_type(PGresult* res, int row, int col) {
    return parse_relation_type(PQgetvalue(res, row, col));
}

inline char relation_type_char(RelationType t) {
    return static_cast<char>(t);
}

} // namespace hypercube::db
