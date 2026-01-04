#pragma once

#include <string>
#include <cstdlib>
#include <libpq-fe.h>

namespace hypercube::db {

// Database connection configuration
struct ConnectionConfig {
    std::string dbname;
    std::string host;
    std::string port;
    std::string user;
    std::string password;
    
    ConnectionConfig() {
        // Read from HC_DB_* env vars with defaults
        auto get_env = [](const char* name, const char* def) -> std::string {
            const char* val = std::getenv(name);
            return val ? val : def;
        };
        dbname = get_env("HC_DB_NAME", "hypercube");
        host = get_env("HC_DB_HOST", "localhost");
        port = get_env("HC_DB_PORT", "5432");
        user = get_env("HC_DB_USER", "hartonomous");
        password = get_env("HC_DB_PASS", "hartonomous");
    }
    
    // Build libpq connection string
    std::string to_conninfo() const {
        std::string conninfo = "dbname=" + dbname;
        if (!host.empty()) conninfo += " host=" + host;
        if (!port.empty()) conninfo += " port=" + port;
        if (!user.empty()) conninfo += " user=" + user;
        if (!password.empty()) conninfo += " password=" + password;
        return conninfo;
    }
    
    // Parse from command line args (modifies index)
    // Returns false if unknown arg
    bool parse_arg(int argc, char** argv, int& i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
            return true;
        }
        if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
            return true;
        }
        if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
            return true;
        }
        if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
            return true;
        }
        if ((arg == "-W" || arg == "--password") && i + 1 < argc) {
            password = argv[++i];
            return true;
        }
        return false;
    }
};

// RAII wrapper for PGconn
class Connection {
public:
    Connection() : conn_(nullptr) {}
    
    explicit Connection(const std::string& conninfo) {
        conn_ = PQconnectdb(conninfo.c_str());
    }
    
    explicit Connection(const ConnectionConfig& config) 
        : Connection(config.to_conninfo()) {}
    
    ~Connection() {
        if (conn_) {
            PQfinish(conn_);
        }
    }
    
    // Move only
    Connection(Connection&& other) noexcept : conn_(other.conn_) {
        other.conn_ = nullptr;
    }
    
    Connection& operator=(Connection&& other) noexcept {
        if (this != &other) {
            if (conn_) PQfinish(conn_);
            conn_ = other.conn_;
            other.conn_ = nullptr;
        }
        return *this;
    }
    
    // No copy
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    
    // Access underlying connection
    PGconn* get() const { return conn_; }
    operator PGconn*() const { return conn_; }
    
    // Check connection status
    bool ok() const {
        return conn_ && PQstatus(conn_) == CONNECTION_OK;
    }
    
    const char* error() const {
        return conn_ ? PQerrorMessage(conn_) : "No connection";
    }
    
private:
    PGconn* conn_;
};

// Execute query and check result
inline bool exec_ok(PGconn* conn, const char* query, PGresult** out = nullptr) {
    PGresult* res = PQexec(conn, query);
    ExecStatusType status = PQresultStatus(res);
    bool ok = (status == PGRES_COMMAND_OK || status == PGRES_TUPLES_OK);
    if (out) {
        *out = res;
    } else {
        PQclear(res);
    }
    return ok;
}

} // namespace hypercube::db
