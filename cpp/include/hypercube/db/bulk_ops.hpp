#pragma once

#include <vector>
#include <string>
#include <functional>
#include <optional>
#include <libpq-fe.h>

#include "hypercube/db/connection.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/geometry.hpp"
#include "hypercube/types.hpp"
#include "hypercube/ingest/cpe.hpp"

namespace hypercube::db {

/**
 * @brief Generic bulk inserter with transaction management and error handling
 */
template<typename T>
class BulkInserter {
public:
    /**
     * @brief Constructor with serializer function
     * @param conn PostgreSQL connection
     * @param table_name Target table name
     * @param columns Column names for COPY command
     * @param serializer Function to serialize T to CopyWriter
     * @param chunk_size Buffer size before flush (default 1MB)
     */
    BulkInserter(PGconn* conn, std::string table_name,
                 std::vector<std::string> columns,
                 std::function<bool(CopyWriter&, const T&)> serializer,
                 size_t chunk_size = 1 << 20)
        : conn_(conn), table_name_(std::move(table_name)),
          columns_(std::move(columns)), serializer_(std::move(serializer)),
          chunk_size_(chunk_size), rows_inserted_(0), error_(false) {}

    ~BulkInserter() {
        if (copy_writer_) {
            finish();
        }
    }

    /**
     * @brief Start bulk insertion with transaction
     * @return true on success
     */
    bool begin() {
        if (transaction_ || copy_writer_) return false;

        transaction_ = std::make_unique<Transaction>(conn_);

        std::string copy_cmd = "COPY " + table_name_ + " (";
        for (size_t i = 0; i < columns_.size(); ++i) {
            if (i > 0) copy_cmd += ", ";
            copy_cmd += columns_[i];
        }
        copy_cmd += ") FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";

        copy_writer_ = std::make_unique<CopyWriter>(
            conn_, table_name_, columns_, chunk_size_);

        return copy_writer_->ok();
    }

    /**
     * @brief Insert a single item
     * @param item Item to insert
     * @return true on success
     */
    bool insert(const T& item) {
        if (!copy_writer_ || !serializer_) return false;

        bool success = serializer_(*copy_writer_, item);
        if (success) {
            success = copy_writer_->end_row();
        }
        if (!success) {
            error_ = true;
            error_msg_ = copy_writer_->error();
        }
        return success;
    }

    /**
     * @brief Insert multiple items
     * @param items Items to insert
     * @return true on success
     */
    bool insert_batch(const std::vector<T>& items) {
        for (const auto& item : items) {
            if (!insert(item)) return false;
        }
        return true;
    }

    /**
     * @brief Finish bulk insertion and commit transaction
     * @return true on success
     */
    bool finish() {
        if (!copy_writer_) return true;

        bool success = copy_writer_->finish();
        if (success) {
            transaction_->commit();
        } else {
            error_ = true;
            error_msg_ = copy_writer_->error();
            transaction_->rollback();
        }

        rows_inserted_ = copy_writer_->rows();
        copy_writer_.reset();
        transaction_.reset();

        return success;
    }

    /**
     * @brief Check if inserter is ready
     */
    bool ok() const { return !error_; }

    /**
     * @brief Get error message
     */
    const std::string& error() const { return error_msg_; }

    /**
     * @brief Get number of rows inserted
     */
    size_t rows_inserted() const { return rows_inserted_; }

private:
    PGconn* conn_;
    std::string table_name_;
    std::vector<std::string> columns_;
    std::function<bool(CopyWriter&, const T&)> serializer_;
    size_t chunk_size_;
    size_t rows_inserted_;
    bool error_;
    std::string error_msg_;

    std::unique_ptr<Transaction> transaction_;
    std::unique_ptr<CopyWriter> copy_writer_;
};

// =============================================================================
// Convenience functions for creating bulk inserters
// =============================================================================

/**
 * @brief Create bulk inserter for atoms
 */
inline BulkInserter<UnicodeAtom> create_atom_inserter(PGconn* conn, size_t chunk_size = 1 << 20) {
    auto serializer = [](CopyWriter& writer, const UnicodeAtom& atom) -> bool {
        writer.col_bytea(atom.hash);
        writer.col(static_cast<int64_t>(atom.codepoint));
        writer.col(std::string(1, static_cast<char>(atom.codepoint)));  // UTF-8 value
        writer.col_raw(build_pointzm_ewkb(
            atom.coords.x_raw(), atom.coords.y_raw(),
            atom.coords.z_raw(), atom.coords.m_raw()));
        writer.col(static_cast<int64_t>(atom.hilbert.lo));
        writer.col(static_cast<int64_t>(atom.hilbert.hi));
        return writer.ok();
    };

    return BulkInserter<UnicodeAtom>(conn, "atom",
        {"id", "codepoint", "value", "geom", "hilbert_lo", "hilbert_hi"},
        serializer, chunk_size);
}

/**
 * @brief Create bulk inserter for compositions
 */
inline BulkInserter<ingest::CompositionRecord> create_composition_inserter(PGconn* conn, size_t chunk_size = 1 << 20) {
    auto serializer = [](CopyWriter& writer, const ingest::CompositionRecord& comp) -> bool {
        // Build geometry from children
        std::string geom_str = "\\N";
        if (comp.children.size() >= 2) {
            std::vector<std::array<int32_t, 4>> points;
            points.reserve(comp.children.size());
            for (const auto& child : comp.children) {
                points.push_back({child.x, child.y, child.z, child.m});
            }
            geom_str = build_linestringzm_ewkb(points);
        }

        writer.col_bytea(comp.hash);
        writer.col_null();  // label (NULL)
        writer.col(static_cast<int64_t>(comp.depth));
        writer.col(static_cast<int64_t>(comp.children.size()));
        writer.col(static_cast<int64_t>(comp.atom_count));
        writer.col_raw(geom_str);
        writer.col_raw(build_pointzm_ewkb(
            comp.coord_x, comp.coord_y, comp.coord_z, comp.coord_m));
        writer.col(static_cast<int64_t>(comp.hilbert_lo));
        writer.col(static_cast<int64_t>(comp.hilbert_hi));
        return writer.ok();
    };

    return BulkInserter<ingest::CompositionRecord>(conn, "composition",
        {"id", "label", "depth", "child_count", "atom_count",
         "geom", "centroid", "hilbert_lo", "hilbert_hi"},
        serializer, chunk_size);
}

/**
 * @brief Create bulk inserter for composition children
 */
class CompositionChildInserter {
public:
    CompositionChildInserter(PGconn* conn, const Blake3Hash& composition_id, size_t chunk_size = 1 << 20)
        : inserter_(conn, "composition_child",
                   {"composition_id", "ordinal", "child_type", "child_id"},
                   [this](CopyWriter& writer, const ingest::ChildInfo& child) -> bool {
                       char child_type = child.is_atom ? 'A' : 'C';
                       writer.col_bytea(composition_id_);
                       writer.col(static_cast<int64_t>(ordinal_++));
                       writer.col(std::string(1, child_type));
                       writer.col_bytea(child.hash);
                       return writer.ok();
                   }, chunk_size),
          composition_id_(composition_id), ordinal_(0) {}

    bool begin() { return inserter_.begin(); }
    bool insert(const ingest::ChildInfo& child) { return inserter_.insert(child); }
    bool insert_batch(const std::vector<ingest::ChildInfo>& children) {
        return inserter_.insert_batch(children);
    }
    bool finish() { return inserter_.finish(); }
    bool ok() const { return inserter_.ok(); }
    const std::string& error() const { return inserter_.error(); }
    size_t rows_inserted() const { return inserter_.rows_inserted(); }

private:
    BulkInserter<ingest::ChildInfo> inserter_;
    Blake3Hash composition_id_;
    size_t ordinal_;
};

/**
 * @brief Data structure for relation insertions
 */
struct RelationRecord {
    char source_type;
    Blake3Hash source_id;
    char target_type;
    Blake3Hash target_id;
    char relation_type;
    double weight;
    std::string source_model;
    int source_count = 1;
    int layer = -1;
    std::string component;

    RelationRecord(char src_type, const Blake3Hash& src_id,
                   char tgt_type, const Blake3Hash& tgt_id,
                   char rel_type, double w,
                   std::string model = "", int layer = -1, std::string comp = "")
        : source_type(src_type), source_id(src_id),
          target_type(tgt_type), target_id(tgt_id),
          relation_type(rel_type), weight(w),
          source_model(std::move(model)), layer(layer), component(std::move(comp)) {}
};

/**
 * @brief Create bulk inserter for relations
 */
inline BulkInserter<RelationRecord> create_relation_inserter(PGconn* conn, size_t chunk_size = 1 << 20) {
    auto serializer = [](CopyWriter& writer, const RelationRecord& rel) -> bool {
        writer.col(std::string(1, rel.source_type));
        writer.col_bytea(rel.source_id);
        writer.col(std::string(1, rel.target_type));
        writer.col_bytea(rel.target_id);
        writer.col(std::string(1, rel.relation_type));
        writer.col(rel.weight);
        writer.col(rel.source_model.empty() ? "\\N" : rel.source_model);
        writer.col(static_cast<int64_t>(rel.source_count));
        writer.col(static_cast<int64_t>(rel.layer));
        writer.col(rel.component.empty() ? "\\N" : rel.component);
        return writer.ok();
    };

    return BulkInserter<RelationRecord>(conn, "relation",
        {"source_type", "source_id", "target_type", "target_id",
         "relation_type", "weight", "source_model", "source_count",
         "layer", "component"},
        serializer, chunk_size);
}

/**
 * @brief Data structure for embedding/centroid updates
 */
struct EmbeddingRecord {
    Blake3Hash composition_id;
    double x, y, z, m;
    HilbertIndex hilbert;

    EmbeddingRecord(const Blake3Hash& id, double x_, double y_, double z_, double m_,
                    const HilbertIndex& h)
        : composition_id(id), x(x_), y(y_), z(z_), m(m_), hilbert(h) {}
};

/**
 * @brief Bulk updater for embeddings (centroid updates)
 * Uses UPDATE instead of INSERT since embeddings update existing compositions
 */
class EmbeddingBulkUpdater {
public:
    EmbeddingBulkUpdater(PGconn* conn, size_t batch_size = 1000)
        : conn_(conn), batch_size_(batch_size), pending_(0), error_(false) {}

    ~EmbeddingBulkUpdater() {
        if (pending_ > 0) {
            finish();
        }
    }

    bool begin() {
        if (transaction_) return false;
        transaction_ = std::make_unique<Transaction>(conn_);
        return true;
    }

    bool update(const EmbeddingRecord& emb) {
        if (error_) return false;

        updates_.push_back(emb);
        pending_++;

        if (pending_ >= batch_size_) {
            return flush();
        }
        return true;
    }

    bool update_batch(const std::vector<EmbeddingRecord>& embeddings) {
        for (const auto& emb : embeddings) {
            if (!update(emb)) return false;
        }
        return true;
    }

    bool flush() {
        if (updates_.empty()) return true;

        // Build batch UPDATE query
        std::string query = "UPDATE composition SET centroid = data.centroid, "
                           "hilbert_lo = data.hilbert_lo, hilbert_hi = data.hilbert_hi "
                           "FROM (VALUES ";

        for (size_t i = 0; i < updates_.size(); ++i) {
            if (i > 0) query += ",";
            const auto& emb = updates_[i];
            query += "('\\x" + emb.composition_id.to_hex() + "'::bytea, ";
            query += "'" + build_pointzm_ewkb(emb.x, emb.y, emb.z, emb.m) + "', ";
            query += std::to_string(emb.hilbert.lo) + ", ";
            query += std::to_string(emb.hilbert.hi) + ")";
        }

        query += ") AS data(id, centroid, hilbert_lo, hilbert_hi) "
                "WHERE composition.id = data.id";

        PGresult* res = PQexec(conn_, query.c_str());
        bool success = (PQresultStatus(res) == PGRES_COMMAND_OK);
        PQclear(res);

        if (!success) {
            error_ = true;
            error_msg_ = PQerrorMessage(conn_);
            return false;
        }

        updates_.clear();
        pending_ = 0;
        return true;
    }

    bool finish() {
        if (!flush()) {
            transaction_->rollback();
            return false;
        }

        transaction_->commit();
        return true;
    }

    bool ok() const { return !error_; }
    const std::string& error() const { return error_msg_; }
    size_t pending() const { return pending_; }

private:
    PGconn* conn_;
    size_t batch_size_;
    size_t pending_;
    bool error_;
    std::string error_msg_;
    std::vector<EmbeddingRecord> updates_;
    std::unique_ptr<Transaction> transaction_;
};

// Helper functions (overloads for embedding updates)
inline std::string build_pointzm_ewkb(double x, double y, double z, double m) {
    // Use the int32_t version but cast (this is a temporary implementation)
    // In practice, you might want proper double handling in geometry.hpp
    return build_pointzm_ewkb(static_cast<int32_t>(x), static_cast<int32_t>(y),
                             static_cast<int32_t>(z), static_cast<int32_t>(m));
}

} // namespace hypercube::db