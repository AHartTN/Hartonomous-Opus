/**
 * @file operations.hpp
 * @brief High-level database operations - COPY streaming, connection pooling, transactions
 * 
 * Consolidates common patterns:
 * - Transaction RAII wrapper
 * - COPY protocol streaming
 * - Connection pooling for parallel operations
 * - Batch builders with consistent escaping
 * 
 * DESIGN: Composable operations that hide libpq details.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <libpq-fe.h>

#include "hypercube/db/connection.hpp"
#include "hypercube/db/helpers.hpp"

namespace hypercube::db {

// =============================================================================
// Transaction RAII
// =============================================================================

/**
 * RAII transaction wrapper. Commits on success, rolls back on exception/early exit.
 * 
 * Usage:
 *   {
 *       Transaction tx(conn);
 *       exec(conn, "INSERT ...");
 *       exec(conn, "UPDATE ...");
 *       tx.commit();  // Explicit commit
 *   }  // Rolls back if commit() not called
 */
class Transaction {
public:
    explicit Transaction(PGconn* conn) : conn_(conn), committed_(false) {
        PGresult* res = PQexec(conn_, "BEGIN");
        PQclear(res);
    }
    
    ~Transaction() {
        if (!committed_) {
            PGresult* res = PQexec(conn_, "ROLLBACK");
            PQclear(res);
        }
    }
    
    void commit() {
        if (!committed_) {
            PGresult* res = PQexec(conn_, "COMMIT");
            PQclear(res);
            committed_ = true;
        }
    }
    
    void rollback() {
        if (!committed_) {
            PGresult* res = PQexec(conn_, "ROLLBACK");
            PQclear(res);
            committed_ = true;  // Mark as handled
        }
    }
    
    // Non-copyable, non-movable
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    
private:
    PGconn* conn_;
    bool committed_;
};

// =============================================================================
// COPY Protocol Streaming
// =============================================================================

/**
 * Escape a string for COPY protocol (tab-delimited text format).
 */
inline void copy_escape(std::string& dest, std::string_view src) {
    for (char ch : src) {
        switch (ch) {
            case '\t': dest += "\\t"; break;
            case '\n': dest += "\\n"; break;
            case '\r': dest += "\\r"; break;
            case '\\': dest += "\\\\"; break;
            default: dest += ch; break;
        }
    }
}

/**
 * Append a bytea value as hex for COPY protocol.
 * Format: \\x followed by hex bytes
 */
inline void copy_bytea(std::string& dest, const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    dest += "\\\\x";
    for (size_t i = 0; i < len; ++i) {
        dest += hex[(data[i] >> 4) & 0xF];
        dest += hex[data[i] & 0xF];
    }
}

inline void copy_bytea(std::string& dest, const Blake3Hash& hash) {
    copy_bytea(dest, hash.data(), 32);
}

/**
 * Append NULL value for COPY protocol.
 */
inline void copy_null(std::string& dest) {
    dest += "\\N";
}

/**
 * Append column delimiter (tab).
 */
inline void copy_tab(std::string& dest) {
    dest += '\t';
}

/**
 * Append row delimiter (newline).
 */
inline void copy_newline(std::string& dest) {
    dest += '\n';
}

/**
 * RAII wrapper for COPY protocol streaming.
 * 
 * Usage:
 *   CopyStream copy(conn, "COPY table FROM STDIN WITH (FORMAT text)");
 *   copy.put("data1\tdata2\n");
 *   copy.put("data3\tdata4\n");
 *   copy.end();  // Explicit end, or auto-ends on destruction
 */
class CopyStream {
public:
    CopyStream(PGconn* conn, const char* copy_cmd)
        : conn_(conn), ended_(false), error_(false) {
        PGresult* res = PQexec(conn_, copy_cmd);
        if (PQresultStatus(res) != PGRES_COPY_IN) {
            error_ = true;
            error_msg_ = PQerrorMessage(conn_);
        }
        PQclear(res);
    }
    
    ~CopyStream() {
        end();
    }
    
    bool ok() const { return !error_; }
    const std::string& error() const { return error_msg_; }
    
    /**
     * Send data chunk. Returns false on error.
     */
    bool put(const char* data, size_t len) {
        if (error_ || ended_) return false;
        if (PQputCopyData(conn_, data, static_cast<int>(len)) != 1) {
            error_ = true;
            error_msg_ = PQerrorMessage(conn_);
            return false;
        }
        return true;
    }
    
    bool put(const std::string& data) {
        return put(data.c_str(), data.size());
    }
    
    /**
     * End COPY stream. Returns false on error.
     * After calling end(), check result() for insert status.
     */
    bool end() {
        if (ended_) return !error_;
        ended_ = true;
        
        if (error_) {
            PQputCopyEnd(conn_, "error");
            PGresult* res = PQgetResult(conn_);
            PQclear(res);
            return false;
        }
        
        if (PQputCopyEnd(conn_, nullptr) != 1) {
            error_ = true;
            error_msg_ = PQerrorMessage(conn_);
            return false;
        }
        
        PGresult* res = PQgetResult(conn_);
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            error_ = true;
            error_msg_ = PQresultErrorMessage(res);
        }
        PQclear(res);
        return !error_;
    }
    
    // Non-copyable
    CopyStream(const CopyStream&) = delete;
    CopyStream& operator=(const CopyStream&) = delete;
    
private:
    PGconn* conn_;
    bool ended_;
    bool error_;
    std::string error_msg_;
};

/**
 * High-level COPY helper with chunking and progress.
 * 
 * Usage:
 *   CopyWriter writer(conn, "tmp_table", {"col1", "col2"});
 *   writer.row("val1", "val2");
 *   writer.row_bytea(hash, "val2");
 *   writer.finish();
 */
class CopyWriter {
public:
    CopyWriter(PGconn* conn, const std::string& table,
               const std::vector<std::string>& columns = {},
               size_t chunk_size = 1 << 20)  // 1MB default chunks
        : conn_(conn), chunk_size_(chunk_size), rows_(0), error_(false) {
        
        // Build COPY command
        std::string cmd = "COPY " + table;
        if (!columns.empty()) {
            cmd += " (";
            for (size_t i = 0; i < columns.size(); ++i) {
                if (i > 0) cmd += ", ";
                cmd += columns[i];
            }
            cmd += ")";
        }
        cmd += " FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";
        
        stream_ = std::make_unique<CopyStream>(conn_, cmd.c_str());
        if (!stream_->ok()) {
            error_ = true;
            error_msg_ = stream_->error();
        }
        
        buffer_.reserve(chunk_size_);
    }
    
    bool ok() const { return !error_; }
    const std::string& error() const { return error_msg_; }
    size_t rows() const { return rows_; }
    
    /**
     * Start a new row. Call column methods, then end_row().
     */
    CopyWriter& col(std::string_view val) {
        if (need_tab_) buffer_ += '\t';
        copy_escape(buffer_, val);
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col(int64_t val) {
        if (need_tab_) buffer_ += '\t';
        buffer_ += std::to_string(val);
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col(double val) {
        if (need_tab_) buffer_ += '\t';
        buffer_ += std::to_string(val);
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col_bytea(const Blake3Hash& hash) {
        if (need_tab_) buffer_ += '\t';
        copy_bytea(buffer_, hash);
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col_bytea(const uint8_t* data, size_t len) {
        if (need_tab_) buffer_ += '\t';
        copy_bytea(buffer_, data, len);
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col_null() {
        if (need_tab_) buffer_ += '\t';
        buffer_ += "\\N";
        need_tab_ = true;
        return *this;
    }
    
    CopyWriter& col_raw(std::string_view raw) {
        if (need_tab_) buffer_ += '\t';
        buffer_ += raw;
        need_tab_ = true;
        return *this;
    }
    
    /**
     * End current row and flush if buffer is large.
     */
    bool end_row() {
        buffer_ += '\n';
        rows_++;
        need_tab_ = false;
        
        if (buffer_.size() >= chunk_size_) {
            return flush();
        }
        return true;
    }
    
    /**
     * Flush buffer to stream.
     */
    bool flush() {
        if (error_ || buffer_.empty()) return !error_;
        if (!stream_->put(buffer_)) {
            error_ = true;
            error_msg_ = stream_->error();
            return false;
        }
        buffer_.clear();
        return true;
    }
    
    /**
     * Finish COPY operation.
     */
    bool finish() {
        if (error_) return false;
        if (!flush()) return false;
        if (!stream_->end()) {
            error_ = true;
            error_msg_ = stream_->error();
            return false;
        }
        return true;
    }
    
private:
    PGconn* conn_;
    std::unique_ptr<CopyStream> stream_;
    std::string buffer_;
    size_t chunk_size_;
    size_t rows_;
    bool need_tab_ = false;
    bool error_;
    std::string error_msg_;
};

// =============================================================================
// Connection Pool
// =============================================================================

/**
 * Thread-safe connection pool for parallel database operations.
 * 
 * Usage:
 *   ConnectionPool pool(config, 4);  // 4 connections
 *   
 *   // Borrow a connection
 *   auto conn = pool.acquire();
 *   exec(conn.get(), "SELECT ...");
 *   // Auto-returns to pool when conn goes out of scope
 */
class ConnectionPool {
public:
    /**
     * RAII handle for borrowed connection.
     */
    class Handle {
    public:
        Handle() : pool_(nullptr), conn_(nullptr) {}
        Handle(ConnectionPool* pool, PGconn* conn) : pool_(pool), conn_(conn) {}
        
        ~Handle() {
            if (pool_ && conn_) {
                pool_->release(conn_);
            }
        }
        
        // Move only
        Handle(Handle&& other) noexcept : pool_(other.pool_), conn_(other.conn_) {
            other.pool_ = nullptr;
            other.conn_ = nullptr;
        }
        
        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                if (pool_ && conn_) pool_->release(conn_);
                pool_ = other.pool_;
                conn_ = other.conn_;
                other.pool_ = nullptr;
                other.conn_ = nullptr;
            }
            return *this;
        }
        
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        
        PGconn* get() const { return conn_; }
        operator PGconn*() const { return conn_; }
        bool ok() const { return conn_ && PQstatus(conn_) == CONNECTION_OK; }
        
    private:
        ConnectionPool* pool_;
        PGconn* conn_;
    };
    
    /**
     * Create pool with specified number of connections.
     */
    ConnectionPool(const ConnectionConfig& config, size_t size)
        : config_(config), shutdown_(false) {
        
        std::string conninfo = config_.to_conninfo();
        for (size_t i = 0; i < size; ++i) {
            PGconn* conn = PQconnectdb(conninfo.c_str());
            if (PQstatus(conn) == CONNECTION_OK) {
                available_.push(conn);
            } else {
                PQfinish(conn);
            }
        }
    }
    
    ~ConnectionPool() {
        shutdown();
    }
    
    /**
     * Acquire a connection (blocks if none available).
     */
    Handle acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (available_.empty() && !shutdown_) {
            cv_.wait(lock);
        }
        
        if (shutdown_ || available_.empty()) {
            return Handle();
        }
        
        PGconn* conn = available_.front();
        available_.pop();
        
        // Reset connection if needed
        if (PQstatus(conn) != CONNECTION_OK) {
            PQreset(conn);
        }
        
        return Handle(this, conn);
    }
    
    /**
     * Try to acquire without blocking.
     */
    Handle try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_.empty()) {
            return Handle();
        }
        
        PGconn* conn = available_.front();
        available_.pop();
        
        if (PQstatus(conn) != CONNECTION_OK) {
            PQreset(conn);
        }
        
        return Handle(this, conn);
    }
    
    /**
     * Get number of available connections.
     */
    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_.size();
    }
    
    /**
     * Shutdown pool and close all connections.
     */
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        while (!available_.empty()) {
            PQfinish(available_.front());
            available_.pop();
        }
        cv_.notify_all();
    }
    
private:
    void release(PGconn* conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            PQfinish(conn);
        } else {
            available_.push(conn);
            cv_.notify_one();
        }
    }
    
    ConnectionConfig config_;
    std::queue<PGconn*> available_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_;
};

// =============================================================================
// Parallel Execution Helpers
// =============================================================================

/**
 * Get optimal thread count for parallel operations.
 */
inline unsigned int optimal_threads(unsigned int hint = 0) {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    if (hint > 0) return std::min(hint, hw);
    // Cap at 16 for DB operations (connection overhead)
    return std::min(hw, 16u);
}

/**
 * Parallel batch builder - accumulates data in thread-local buffers.
 * 
 * Usage:
 *   ParallelBatchBuilder builder(4);  // 4 threads
 *   builder.parallel_for(0, items.size(), [&](size_t i, std::string& batch) {
 *       batch += format_row(items[i]);
 *   });
 *   std::string combined = builder.combine();
 */
class ParallelBatchBuilder {
public:
    explicit ParallelBatchBuilder(size_t num_threads = 0)
        : num_threads_(optimal_threads(static_cast<unsigned>(num_threads))) {
        buffers_.resize(num_threads_);
        for (auto& b : buffers_) {
            b.reserve(1 << 20);  // 1MB each
        }
    }
    
    /**
     * Execute function in parallel, passing thread-local buffer.
     */
    template<typename Func>
    void parallel_for(size_t start, size_t end, Func&& func) {
        std::atomic<size_t> next{start};
        std::vector<std::thread> threads;
        
        for (size_t t = 0; t < num_threads_; ++t) {
            threads.emplace_back([&, t]() {
                while (true) {
                    size_t i = next.fetch_add(1);
                    if (i >= end) break;
                    func(i, buffers_[t]);
                }
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
    }
    
    /**
     * Execute with progress callback.
     */
    template<typename Func, typename Progress>
    void parallel_for_progress(size_t start, size_t end, Func&& func, Progress&& progress) {
        std::atomic<size_t> next{start};
        std::atomic<size_t> done{0};
        std::atomic<bool> finished{false};
        
        // Progress thread
        std::thread progress_thread([&]() {
            while (!finished.load()) {
                progress(done.load(), end - start);
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            progress(done.load(), end - start);  // Final update
        });
        
        // Workers
        std::vector<std::thread> threads;
        for (size_t t = 0; t < num_threads_; ++t) {
            threads.emplace_back([&, t]() {
                while (true) {
                    size_t i = next.fetch_add(1);
                    if (i >= end) break;
                    func(i, buffers_[t]);
                    done.fetch_add(1);
                }
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
        
        finished.store(true);
        progress_thread.join();
    }
    
    /**
     * Combine all buffers into single string.
     */
    std::string combine() const {
        size_t total = 0;
        for (const auto& b : buffers_) total += b.size();
        
        std::string result;
        result.reserve(total);
        for (const auto& b : buffers_) {
            result += b;
        }
        return result;
    }
    
    /**
     * Get total size across all buffers.
     */
    size_t total_size() const {
        size_t total = 0;
        for (const auto& b : buffers_) total += b.size();
        return total;
    }
    
    /**
     * Clear all buffers.
     */
    void clear() {
        for (auto& b : buffers_) b.clear();
    }
    
    /**
     * Access individual buffer.
     */
    std::string& buffer(size_t idx) { return buffers_[idx]; }
    const std::string& buffer(size_t idx) const { return buffers_[idx]; }
    size_t buffer_count() const { return buffers_.size(); }
    
private:
    size_t num_threads_;
    std::vector<std::string> buffers_;
};

// =============================================================================
// Common Temp Table Patterns
// =============================================================================

/**
 * Create a temp table for COPY import.
 * Returns false on error.
 */
inline bool create_temp_table(PGconn* conn, const std::string& name,
                               const std::string& schema) {
    std::string sql = "CREATE TEMP TABLE " + name + " (" + schema + ") ON COMMIT DROP";
    Result res = exec(conn, sql);
    return res.ok();
}

/**
 * Common temp table schemas.
 */
namespace schema {

inline const char* composition() {
    return "id BYTEA, "
           "label TEXT, "
           "depth INTEGER, "
           "child_count INTEGER, "
           "atom_count BIGINT, "
           "geom GEOMETRY(LINESTRINGZM, 0), "
           "centroid GEOMETRY(POINTZM, 0), "
           "hilbert_lo BIGINT, "
           "hilbert_hi BIGINT";
}

inline const char* composition_child() {
    return "composition_id BYTEA, "
           "ordinal SMALLINT, "
           "child_type CHAR(1), "
           "child_id BYTEA";
}

inline const char* relation() {
    return "source_type CHAR(1), "
           "source_id BYTEA, "
           "target_type CHAR(1), "
           "target_id BYTEA, "
           "relation_type CHAR(1), "
           "weight REAL, "
           "source_model TEXT, "
           "layer INTEGER, "
           "component TEXT";
}

inline const char* atom_projection() {
    return "id BYTEA, "
           "geom GEOMETRY(POINTZM, 0), "
           "hilbert_lo BIGINT, "
           "hilbert_hi BIGINT";
}

}  // namespace schema

// =============================================================================
// Prepared Statement Cache
// =============================================================================

/**
 * Cache for prepared statements to avoid repeated parsing.
 * 
 * Usage:
 *   PreparedStatementCache cache(conn);
 *   cache.prepare("lookup_atom", "SELECT * FROM atom WHERE id = $1");
 *   PGresult* res = cache.exec("lookup_atom", {param1});
 */
class PreparedStatementCache {
public:
    explicit PreparedStatementCache(PGconn* conn) : conn_(conn) {}
    
    /**
     * Prepare a statement (no-op if already prepared).
     */
    bool prepare(const std::string& name, const std::string& sql) {
        if (prepared_.count(name)) return true;
        
        PGresult* res = PQprepare(conn_, name.c_str(), sql.c_str(), 0, nullptr);
        bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
        PQclear(res);
        
        if (ok) prepared_.insert(name);
        return ok;
    }
    
    /**
     * Execute prepared statement with string parameters.
     */
    Result exec(const std::string& name, const std::vector<std::string>& params) {
        std::vector<const char*> values;
        values.reserve(params.size());
        for (const auto& p : params) {
            values.push_back(p.c_str());
        }
        
        return Result(PQexecPrepared(
            conn_, name.c_str(),
            static_cast<int>(values.size()),
            values.data(),
            nullptr, nullptr, 0
        ));
    }
    
    /**
     * Execute prepared statement with bytea parameters.
     */
    Result exec_binary(const std::string& name,
                       const std::vector<std::pair<const char*, int>>& params) {
        std::vector<const char*> values;
        std::vector<int> lengths;
        std::vector<int> formats;
        
        for (const auto& [data, len] : params) {
            values.push_back(data);
            lengths.push_back(len);
            formats.push_back(1);  // binary
        }
        
        return Result(PQexecPrepared(
            conn_, name.c_str(),
            static_cast<int>(values.size()),
            values.data(),
            lengths.data(),
            formats.data(),
            0  // text result
        ));
    }
    
private:
    PGconn* conn_;
    std::unordered_set<std::string> prepared_;
};

}  // namespace hypercube::db
