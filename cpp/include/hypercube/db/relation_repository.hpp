#pragma once

#include "hypercube/db/operations.hpp"
#include "hypercube/types.hpp"
#include <string>
#include <vector>
#include <optional>

namespace hypercube {
namespace db {

struct RelationEvidence {
    hypercube::Blake3Hash source_id;
    hypercube::Blake3Hash target_id;
    std::string role;       // e.g., "S", "A" (relation_type)
    float similarity;
    std::string model_id;   // source_model is TEXT
    
    // Optional metadata
    int layer = 0;
    std::string component;  // component is TEXT (e.g., "semantic", "attention")
};

class RelationRepository {
public:
    explicit RelationRepository(PGconn* conn) : conn_(conn) {}

    /**
     * @brief Inserts relation evidence with automatic ELO rating updates.
     * 
     * Uses ON CONFLICT DO UPDATE to adjust weights and ELO ratings based on new evidence.
     * 
     * @param evidence The evidence to insert.
     * @return bool True on success.
     */
    bool insert_evidence(const RelationEvidence& evidence);

    /**
     * @brief Batch insert relation evidence.
     * 
     * @param batch Vector of evidence.
     * @return bool True on success.
     */
    bool insert_evidence_batch(const std::vector<RelationEvidence>& batch);

private:
    PGconn* conn_;
};

} // namespace db
} // namespace hypercube
