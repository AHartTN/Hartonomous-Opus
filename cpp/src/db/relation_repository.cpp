#include "hypercube/db/relation_repository.hpp"
#include <libpq-fe.h>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>

namespace hypercube {
namespace db {

bool RelationRepository::insert_evidence_batch(const std::vector<RelationEvidence>& batch) {
    if (batch.empty()) return true;

    // 1. Create temp table
    // IDs are BYTEA (native BLAKE3 hash storage)
    const char* create_temp = R"(
        CREATE TEMP TABLE IF NOT EXISTS tmp_evidence_batch (
            source_id BYTEA,
            target_id BYTEA,
            relation_type CHAR(1),
            source_model TEXT,
            layer INT,
            component TEXT,
            weight FLOAT
        ) ON COMMIT DROP;
    )";
    
    PGresult* res = PQexec(conn_, create_temp);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[RelationRepository] Failed to create temp table: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // 2. Start COPY
    res = PQexec(conn_, "COPY tmp_evidence_batch FROM STDIN");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[RelationRepository] COPY start failed: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // 3. Stream data
    // Use text format with hex escape (\x) for BYTEA
    for (const auto& item : batch) {
        std::string line = "\\x" + item.source_id.to_hex() + "\t" + 
                           "\\x" + item.target_id.to_hex() + "\t" + 
                           item.role + "\t" + 
                           item.model_id + "\t" + 
                           std::to_string(item.layer) + "\t" + 
                           item.component + "\t" + 
                           std::to_string(item.similarity) + "\n";
                           
        if (PQputCopyData(conn_, line.c_str(), (int)line.size()) != 1) {
             std::cerr << "[RelationRepository] COPY data failed: " << PQerrorMessage(conn_) << "\n";
             return false;
        }
    }
    
    if (PQputCopyEnd(conn_, nullptr) != 1) {
        std::cerr << "[RelationRepository] COPY end failed: " << PQerrorMessage(conn_) << "\n";
        return false;
    }
    
    // 4. INSERT / MERGE Logic
    // No DECODE needed - temp table has correct BYTEA types
    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        SELECT
            t.source_id,
            t.target_id,
            t.relation_type,
            t.source_model,
            t.layer,
            t.component,
            1500.0,
            1,
            t.weight,
            t.weight
        FROM tmp_evidence_batch t
        WHERE EXISTS (SELECT 1 FROM composition WHERE id = t.source_id)
        AND EXISTS (SELECT 1 FROM composition WHERE id = t.target_id)
        ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
        DO UPDATE SET
            rating = relation_evidence.rating +
                     LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                     (
                         (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                         (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                     ),
            observation_count = relation_evidence.observation_count + 1,
            raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                         (relation_evidence.observation_count + 1),
            normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                               (relation_evidence.observation_count + 1),
            last_updated = NOW();
    )SQL";
    
    res = PQexec(conn_, insert_sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
         std::cerr << "[RelationRepository] INSERT failed: " << PQerrorMessage(conn_) << "\n";
         PQclear(res);
         return false;
    }
    
    int inserted = atoi(PQcmdTuples(res));
    std::cerr << "[RelationRepository] Inserted/Updated " << inserted << " evidence rows\n";

    PQclear(res);
    PQexec(conn_, "DROP TABLE tmp_evidence_batch");
    
    return true;
}

bool RelationRepository::insert_evidence(const RelationEvidence& evidence) {
    return insert_evidence_batch({evidence});
}

} // namespace db
} // namespace hypercube
