#pragma once

#include <string>
#include <cstdlib>
#include <libpq-fe.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

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
#if defined(_WIN32)
            char* val = nullptr;
            size_t len;
            if (_dupenv_s(&val, &len, name) == 0 && val != nullptr) {
                std::string result(val);
                free(val);
                return result;
            }
            return def;
#else
            const char* val = std::getenv(name);
            return val ? val : def;
#endif
        };
        dbname = get_env("HC_DB_NAME", "hypercube");
        host = get_env("HC_DB_HOST", "HART-SERVER");
        port = get_env("HC_DB_PORT", "5432");
        user = get_env("HC_DB_USER", "postgres");
        password = get_env("HC_DB_PASS", "postgres");
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

// Connection pool for efficient database connections
class ConnectionPool {
private:
    std::queue<std::unique_ptr<Connection>> pool_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const std::string conninfo_;
    size_t max_size_;
    std::atomic<size_t> active_connections_{0};

public:
    /**
     * @brief Create connection pool
     * @param config Database configuration
     * @param max_size Maximum number of connections
     */
    ConnectionPool(const ConnectionConfig& config, size_t max_size = 10)
        : conninfo_(config.to_conninfo()), max_size_(max_size) {}

    /**
     * @brief Get connection from pool (blocks if none available)
     * @return RAII connection wrapper
     */
    std::unique_ptr<Connection> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for available connection or create new one
        while (pool_.empty() && active_connections_.load() >= max_size_) {
            cv_.wait(lock);
        }

        if (!pool_.empty()) {
            auto conn = std::move(pool_.front());
            pool_.pop();
            return conn;
        }

        // Create new connection
        active_connections_.fetch_add(1);
        auto conn = std::make_unique<Connection>(conninfo_);
        if (!conn->ok()) {
            active_connections_.fetch_sub(1);
            throw std::runtime_error(std::string("Failed to connect: ") + conn->error());
        }
        return conn;
    }

    /**
     * @brief Return connection to pool
     * @param conn Connection to return (moves ownership)
     */
    void release(std::unique_ptr<Connection> conn) {
        if (!conn || !conn->ok()) {
            // Bad connection, don't reuse
            active_connections_.fetch_sub(1);
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push(std::move(conn));
        cv_.notify_one();
    }

    /**
     * @brief Get current pool statistics
     */
    std::tuple<size_t, size_t> stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {pool_.size(), active_connections_.load()};
    }

    /**
     * @brief Drain all connections (for shutdown)
     */
    void drain() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pool_.empty()) {
            pool_.pop();
        }
        // Active connections will be cleaned up when released
    }
};

// RAII wrapper for pooled connections
class PooledConnection {
private:
    ConnectionPool* pool_;
    std::unique_ptr<Connection> conn_;

public:
    PooledConnection(ConnectionPool& pool, std::unique_ptr<Connection> conn)
        : pool_(&pool), conn_(std::move(conn)) {}

    ~PooledConnection() {
        if (conn_) {
            pool_->release(std::move(conn_));
        }
    }

    // No copy/move - connections must be managed by pool
    PooledConnection(const PooledConnection&) = delete;
    PooledConnection& operator=(const PooledConnection&) = delete;
    PooledConnection(PooledConnection&&) = delete;
    PooledConnection& operator=(PooledConnection&&) = delete;

    // Access underlying connection
    Connection* operator->() const { return conn_.get(); }
    Connection& operator*() const { return *conn_; }
    PGconn* get() const { return conn_->get(); }
    operator PGconn*() const { return conn_->get(); }
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
