/**
 * @file elo_relations.hpp
 * @brief ELO-style relation weight tracking for multi-model consensus
 *
 * Core Idea:
 * - Multiple models can observe the same relation (A→B)
 * - Each model's observation updates its ELO rating for that relation
 * - Strong relations get higher ratings, weak relations get lower ratings
 * - Consensus = average rating across models, weighted by confidence
 *
 * Rating System:
 * - Initial rating: 1500 (neutral)
 * - Range: ~0 to ~3000 (theoretically unbounded)
 * - K-factor: 32 (learning rate, decays with more observations)
 * - Normalized weight: tanh((rating - 1500) / 400) ∈ [-1, 1]
 */

#pragma once

#include <cmath>
#include <string>
#include <unordered_map>

namespace hypercube {
namespace ingest {

// =============================================================================
// ELO Rating System for Relations
// =============================================================================

struct ELOConfig {
    double initial_rating = 1500.0;  // Starting rating (neutral)
    double k_factor = 32.0;          // Base learning rate
    double k_decay = 0.95;           // K-factor decay per observation
    double min_k_factor = 4.0;       // Minimum K-factor (never fully rigid)
};

class ELORelationTracker {
public:
    explicit ELORelationTracker(const ELOConfig& config = ELOConfig{})
        : config_(config) {}

    /**
     * Normalize raw weight to [-1, 1]
     *
     * Common input ranges:
     * - Cosine similarity: [0, 1] → map to [0, 1]
     * - Attention scores: [0, 1] → map to [0, 1]
     * - Correlation: [-1, 1] → already normalized
     * - Distance: [0, ∞) → map via exp(-d)
     */
    static double normalize_weight(double raw_weight, double min_val = 0.0, double max_val = 1.0) {
        if (min_val == -1.0 && max_val == 1.0) {
            return raw_weight;  // Already normalized
        }

        // Map [min_val, max_val] → [0, 1]
        double normalized = (raw_weight - min_val) / (max_val - min_val);

        // Map [0, 1] → [-1, 1] with sigmoid-style center
        // Values near 0.5 → 0, strong values → ±1
        return 2.0 * normalized - 1.0;
    }

    /**
     * Update ELO rating based on new observation
     *
     * @param current_rating Current ELO rating
     * @param normalized_weight New observation normalized to [-1, 1]
     * @param observation_count Number of previous observations (for K decay)
     * @return New ELO rating
     *
     * Logic:
     * - Expected score: sigmoid of rating difference
     * - Actual score: (normalized_weight + 1) / 2  (map [-1,1] to [0,1])
     * - Update: rating + K * (actual - expected)
     */
    double update_rating(double current_rating, double normalized_weight, int observation_count) const {
        // Calculate dynamic K-factor (decays with more observations)
        double k = config_.k_factor * std::pow(config_.k_decay, observation_count);
        k = std::max(k, config_.min_k_factor);

        // Expected score: probability that this relation is strong
        // Based on current rating vs. neutral (1500)
        double rating_diff = current_rating - config_.initial_rating;
        double expected = 1.0 / (1.0 + std::exp(-rating_diff / 400.0));

        // Actual score: observed weight mapped to [0, 1]
        // Strong positive → 1.0, neutral → 0.5, strong negative → 0.0
        double actual = (normalized_weight + 1.0) / 2.0;

        // ELO update
        double new_rating = current_rating + k * (actual - expected);

        // Clamp to reasonable range (prevent overflow)
        return std::max(0.0, std::min(3000.0, new_rating));
    }

    /**
     * Convert ELO rating back to normalized weight [-1, 1]
     *
     * Uses tanh to map rating → weight:
     * - 1500 (neutral) → 0.0
     * - 1900+ (strong) → +1.0
     * - 1100- (weak) → -1.0
     */
    static double rating_to_weight(double rating) {
        return std::tanh((rating - 1500.0) / 400.0);
    }

    /**
     * Calculate consensus weight across multiple model ratings
     *
     * @param ratings Array of ELO ratings from different models
     * @param count Number of ratings
     * @return Consensus weight ∈ [-1, 1], confidence ∈ [0, 1]
     */
    static std::pair<double, double> calculate_consensus(const double* ratings, size_t count) {
        if (count == 0) return {0.0, 0.0};

        // Calculate mean and stddev
        double sum = 0.0;
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum += ratings[i];
            sum_sq += ratings[i] * ratings[i];
        }

        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        double stddev = std::sqrt(variance);

        // Consensus weight: average rating mapped to [-1, 1]
        double consensus_weight = rating_to_weight(mean);

        // Confidence: inverse of disagreement
        // High stddev → low confidence, low stddev → high confidence
        double confidence = 1.0 / (1.0 + stddev / 400.0);

        return {consensus_weight, confidence};
    }

private:
    ELOConfig config_;
};

// =============================================================================
// SQL Generation Helpers
// =============================================================================

/**
 * Generate SQL to insert/update relation evidence with ELO
 */
inline std::string generate_elo_upsert_sql(
    const std::string& source_model,
    int layer,
    const std::string& component,
    double raw_weight,
    double normalized_weight
) {
    // Use PostgreSQL's ON CONFLICT to update ELO rating
    return R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
            ($1, $2, $3, $4, $5, $6,
             1500.0, 1, $7, $8)
        ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
        DO UPDATE SET
            rating = (
                -- Calculate new ELO rating
                relation_evidence.rating +
                LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                (
                    (($8 + 1.0) / 2.0) -  -- Actual score [0,1]
                    (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))  -- Expected
                )
            ),
            observation_count = relation_evidence.observation_count + 1,
            raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + $7) /
                         (relation_evidence.observation_count + 1),
            normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + $8) /
                                (relation_evidence.observation_count + 1),
            last_updated = NOW()
    )SQL";
}

/**
 * Generate SQL to refresh consensus materialized view
 */
inline std::string generate_refresh_consensus_sql() {
    return "REFRESH MATERIALIZED VIEW CONCURRENTLY relation_consensus";
}

} // namespace ingest
} // namespace hypercube
