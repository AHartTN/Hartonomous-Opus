/**
 * Generative Walk Engine
 * 
 * LLM-like generation using the hypercube substrate:
 * - 4D centroid similarity (Laplacian eigenmap projected)
 * - PMI / co-occurrence scoring
 * - Attention relation scoring
 * - Hilbert proximity pre-filtering
 */

#ifndef HYPERCUBE_GENERATIVE_HPP
#define HYPERCUBE_GENERATIVE_HPP

#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace hypercube {
namespace generative {

// =============================================================================
// Core Types
// =============================================================================

using Blake3Hash = std::array<uint8_t, 32>;

// Hash function for Blake3Hash (32-byte array) to use in unordered_map
struct Blake3Hasher {
    size_t operator()(const Blake3Hash& h) const noexcept {
        // Use first 8 bytes as hash - Blake3 is already well-distributed
        size_t result;
        std::memcpy(&result, h.data(), sizeof(size_t));
        return result;
    }
};

// Equality is default for std::array, but explicit for clarity
struct Blake3Equal {
    bool operator()(const Blake3Hash& a, const Blake3Hash& b) const noexcept {
        return a == b;
    }
};

// 4D Centroid (Laplacian eigenmap projection)
// Coordinates are stored as uint32 cast to double (range 0 to 4.3 billion)
// For similarity, we normalize to unit scale
struct Centroid4D {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double m = 0.0;
    
    static constexpr double COORD_MAX = 4294967295.0;  // UINT32_MAX
    
    bool valid() const { return x != 0.0 || y != 0.0 || z != 0.0 || m != 0.0; }
    bool has_coordinates() const { return valid(); }
    
    // Normalized coordinates (0 to 1)
    double x_norm() const { return x / COORD_MAX; }
    double y_norm() const { return y / COORD_MAX; }
    double z_norm() const { return z / COORD_MAX; }
    double m_norm() const { return m / COORD_MAX; }
    
    // Euclidean distance in normalized 4D space (range 0 to 2.0)
    double distance(const Centroid4D& other) const {
        double dx = x_norm() - other.x_norm();
        double dy = y_norm() - other.y_norm();
        double dz = z_norm() - other.z_norm();
        double dm = m_norm() - other.m_norm();
        return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
    }
    
    // Similarity (inverse distance, normalized to [0, 1])
    // Max distance in 4D unit hypercube is 2.0 (diagonal)
    double similarity(const Centroid4D& other) const {
        double d = distance(other);
        return 1.0 - (d / 2.0);  // Linear similarity: 1 at d=0, 0 at d=2
    }
};

struct TokenCandidate {
    Blake3Hash id;
    std::string label;
    Centroid4D centroid;          // 4D projected coordinates
    double hilbert_index;
    double frequency;              // Usage frequency (for global prior)
};

struct TokenState {
    Blake3Hash id;
    std::string label;
    Centroid4D centroid;          // Current token's 4D position
    double hilbert_index;
};

struct ScoredCandidate {
    size_t index;           // Index in candidate list
    double score_centroid;  // 4D proximity score (renamed from score_shape)
    double score_pmi;
    double score_attn;
    double score_global;
    double score_total;
};

struct GenerationConfig {
    // Scoring weights
    double w_centroid = 0.4;  // 4D centroid similarity weight
    double w_pmi = 0.3;
    double w_attn = 0.2;
    double w_global = 0.1;
    
    // Selection policy
    bool greedy = true;
    double temperature = 1.0;
    
    // Candidate filtering
    size_t max_candidates = 500;      // After Hilbert pre-filter
    double hilbert_range = 0.1;       // Fraction of Hilbert space to search
    
    // Generation limits
    size_t max_tokens = 50;
    std::vector<std::string> stop_tokens = {".", "!", "?", "\n"};
};

// =============================================================================
// 4D Centroid-Based Vocabulary Cache
// =============================================================================

struct VocabEntry {
    Blake3Hash id;
    std::string label;
    int depth;
    double frequency;
    double hilbert_index;
    
    // 4D centroid (from Laplacian eigenmap projection)
    Centroid4D centroid;
};

class VocabularyCache {
public:
    std::vector<VocabEntry> entries;
    std::unordered_map<std::string, size_t> label_to_index;
    std::unordered_map<Blake3Hash, size_t, Blake3Hasher, Blake3Equal> id_to_index;
    
    void clear() {
        entries.clear();
        label_to_index.clear();
        id_to_index.clear();
    }
    
    void add_entry(const VocabEntry& entry) {
        size_t idx = entries.size();
        entries.push_back(entry);
        label_to_index[entry.label] = idx;
        id_to_index[entry.id] = idx;
    }
    
    void set_centroid(size_t idx, double x, double y, double z, double m) {
        if (idx >= entries.size()) return;
        entries[idx].centroid = Centroid4D{x, y, z, m};
    }
    
    int64_t find_label(const std::string& label) const {
        auto it = label_to_index.find(label);
        return (it != label_to_index.end()) ? static_cast<int64_t>(it->second) : -1;
    }
    
    const VocabEntry* get_entry(size_t idx) const {
        return (idx < entries.size()) ? &entries[idx] : nullptr;
    }
    
    size_t count_with_centroid() const {
        size_t count = 0;
        for (const auto& e : entries) {
            if (e.centroid.valid()) ++count;
        }
        return count;
    }
};

// =============================================================================
// Bigram/PMI Cache
// =============================================================================

struct BigramKey {
    Blake3Hash left;
    Blake3Hash right;
    
    bool operator==(const BigramKey& other) const {
        return left == other.left && right == other.right;
    }
};

struct BigramKeyHash {
    size_t operator()(const BigramKey& k) const {
        size_t h = 0;
        for (int i = 0; i < 16; ++i) {
            h ^= std::hash<uint8_t>()(k.left[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint8_t>()(k.right[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

class BigramCache {
public:
    std::unordered_map<BigramKey, double, BigramKeyHash> pmi_scores;
    double max_pmi = 1.0;
    
    void clear() {
        pmi_scores.clear();
        max_pmi = 1.0;
    }
    
    void add(const Blake3Hash& left, const Blake3Hash& right, double score) {
        BigramKey key{left, right};
        pmi_scores[key] = score;
        if (score > max_pmi) max_pmi = score;
    }
    
    double get(const Blake3Hash& left, const Blake3Hash& right) const {
        BigramKey key{left, right};
        auto it = pmi_scores.find(key);
        return (it != pmi_scores.end()) ? it->second / max_pmi : 0.0;
    }
};

// =============================================================================
// Attention Cache
// =============================================================================

// Inner map type for attention edges
using Blake3DoubleMap = std::unordered_map<Blake3Hash, double, Blake3Hasher, Blake3Equal>;

class AttentionCache {
public:
    // source_id -> (target_id -> weight)
    std::unordered_map<Blake3Hash, Blake3DoubleMap, Blake3Hasher, Blake3Equal> edges;
    
    void clear() {
        edges.clear();
    }
    
    void add(const Blake3Hash& source, const Blake3Hash& target, double weight) {
        edges[source][target] = weight;
    }
    
    double get(const Blake3Hash& source, const Blake3Hash& target) const {
        auto src_it = edges.find(source);
        if (src_it == edges.end()) return 0.0;
        
        auto tgt_it = src_it->second.find(target);
        return (tgt_it != src_it->second.end()) ? tgt_it->second : 0.0;
    }
};

// =============================================================================
// Generative Engine
// =============================================================================

class GenerativeEngine {
public:
    VocabularyCache vocab;
    BigramCache bigrams;
    AttentionCache attention;
    GenerationConfig config;
    
    // Deterministic RNG - default seed 42 for reproducibility
    std::mt19937 rng{42};
    
    // Set seed for reproducible generation
    void seed(uint32_t s) { rng.seed(s); }
    
    // Derive seed from input text for deterministic but varied results
    void seed_from_text(const std::string& text) {
        uint32_t h = 2166136261u;  // FNV-1a seed
        for (char c : text) {
            h ^= static_cast<uint8_t>(c);
            h *= 16777619u;
        }
        rng.seed(h);
    }
    
    // =========================================================================
    // Cache Management
    // =========================================================================
    
    void clear_all() {
        vocab.clear();
        bigrams.clear();
        attention.clear();
    }
    
    // =========================================================================
    // Scoring Functions (4D Centroid-Based)
    // =========================================================================
    
    double score_centroid(const TokenState& current, const VocabEntry& candidate) const {
        if (!current.centroid.valid() || !candidate.centroid.valid()) {
            return 0.0;
        }
        return current.centroid.similarity(candidate.centroid);
    }
    
    double score_pmi(const TokenState& current, const VocabEntry& candidate) const {
        return bigrams.get(current.id, candidate.id);
    }
    
    double score_attn(const TokenState& current, const VocabEntry& candidate) const {
        return attention.get(current.id, candidate.id);
    }
    
    double score_global(const VocabEntry& candidate) const {
        // Frequency-based prior (log scale)
        if (candidate.frequency <= 0) return 0.0;
        return std::log1p(candidate.frequency) / 10.0;  // Normalize roughly
    }
    
    ScoredCandidate score_candidate(
        const TokenState& current, 
        size_t candidate_idx
    ) const {
        const VocabEntry& cand = vocab.entries[candidate_idx];
        
        ScoredCandidate sc;
        sc.index = candidate_idx;
        sc.score_centroid = score_centroid(current, cand);
        sc.score_pmi = score_pmi(current, cand);
        sc.score_attn = score_attn(current, cand);
        sc.score_global = score_global(cand);
        
        sc.score_total = 
            config.w_centroid * sc.score_centroid +
            config.w_pmi * sc.score_pmi +
            config.w_attn * sc.score_attn +
            config.w_global * sc.score_global;
        
        return sc;
    }
    
    // =========================================================================
    // Candidate Filtering (Hilbert proximity)
    // =========================================================================
    
    std::vector<size_t> get_candidates_by_hilbert(const TokenState& current) const {
        std::vector<std::pair<double, size_t>> distances;
        distances.reserve(vocab.entries.size());
        
        for (size_t i = 0; i < vocab.entries.size(); ++i) {
            const auto& e = vocab.entries[i];
            if (e.depth != 1 || e.label.empty() || !e.centroid.valid()) {
                continue;
            }
            
            // Hilbert distance (wrap-around on unit interval)
            double d = std::abs(e.hilbert_index - current.hilbert_index);
            d = std::min(d, 1.0 - d);  // Handle wrap-around
            
            distances.push_back({d, i});
        }
        
        // Sort by Hilbert distance
        std::sort(distances.begin(), distances.end());
        
        // Take top N candidates
        std::vector<size_t> result;
        size_t limit = std::min(config.max_candidates, distances.size());
        for (size_t i = 0; i < limit; ++i) {
            result.push_back(distances[i].second);
        }
        
        return result;
    }
    
    std::vector<size_t> get_all_vocab_candidates() const {
        std::vector<size_t> result;
        for (size_t i = 0; i < vocab.entries.size(); ++i) {
            const auto& e = vocab.entries[i];
            if (e.depth == 1 && !e.label.empty() && e.centroid.has_coordinates()) {
                result.push_back(i);
            }
        }
        return result;
    }
    
    // =========================================================================
    // Selection Policy
    // =========================================================================
    
    size_t select_next_token(const std::vector<ScoredCandidate>& scored) {
        if (scored.empty()) return SIZE_MAX;
        
        if (config.greedy) {
            // Greedy: pick highest score
            size_t best = 0;
            for (size_t i = 1; i < scored.size(); ++i) {
                if (scored[i].score_total > scored[best].score_total) {
                    best = i;
                }
            }
            return scored[best].index;
        } else {
            // Stochastic: softmax sampling with temperature
            std::vector<double> probs(scored.size());
            double max_score = scored[0].score_total;
            for (const auto& s : scored) {
                if (s.score_total > max_score) max_score = s.score_total;
            }
            
            double sum = 0.0;
            for (size_t i = 0; i < scored.size(); ++i) {
                probs[i] = std::exp((scored[i].score_total - max_score) / config.temperature);
                sum += probs[i];
            }
            
            for (auto& p : probs) p /= sum;
            
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t sample_idx = dist(rng);
            return scored[sample_idx].index;
        }
    }
    
    // =========================================================================
    // Generation
    // =========================================================================
    
    TokenState make_token_state(size_t idx) const {
        const VocabEntry& e = vocab.entries[idx];
        TokenState ts;
        ts.id = e.id;
        ts.label = e.label;
        ts.centroid = e.centroid;
        ts.hilbert_index = e.hilbert_index;
        return ts;
    }
    
    std::vector<std::string> generate(const std::string& start_label, size_t max_tokens) {
        // Find starting token
        int64_t start_idx = vocab.find_label(start_label);
        if (start_idx < 0) {
            return {};  // Token not found
        }
        
        TokenState current = make_token_state(start_idx);
        std::vector<std::string> output;
        output.push_back(current.label);
        
        for (size_t step = 0; step < max_tokens; ++step) {
            // Get candidates (optionally pre-filtered by Hilbert)
            auto candidate_indices = config.max_candidates < vocab.entries.size()
                ? get_candidates_by_hilbert(current)
                : get_all_vocab_candidates();
            
            if (candidate_indices.empty()) break;
            
            // Score all candidates
            std::vector<ScoredCandidate> scored;
            scored.reserve(candidate_indices.size());
            
            for (size_t idx : candidate_indices) {
                // Skip self
                if (vocab.entries[idx].id == current.id) continue;
                scored.push_back(score_candidate(current, idx));
            }
            
            if (scored.empty()) break;
            
            // Select next token
            size_t next_idx = select_next_token(scored);
            if (next_idx == SIZE_MAX) break;
            
            const VocabEntry& next_entry = vocab.entries[next_idx];
            
            // Check stop tokens
            bool should_stop = false;
            for (const auto& stop : config.stop_tokens) {
                if (next_entry.label == stop) {
                    should_stop = true;
                    break;
                }
            }
            
            output.push_back(next_entry.label);
            
            if (should_stop) break;
            
            // Update current state
            current = make_token_state(next_idx);
        }
        
        return output;
    }
    
    // =========================================================================
    // Top-K Similar (for testing/debugging)
    // =========================================================================
    
    std::vector<ScoredCandidate> find_similar(const std::string& label, size_t k) {
        int64_t idx = vocab.find_label(label);
        if (idx < 0) return {};
        
        TokenState current = make_token_state(idx);
        
        std::vector<ScoredCandidate> all_scored;
        for (size_t i = 0; i < vocab.entries.size(); ++i) {
            if (i == (size_t)idx) continue;
            if (!vocab.entries[i].centroid.has_coordinates()) continue;
            
            ScoredCandidate sc;
            sc.index = i;
            sc.score_centroid = score_centroid(current, vocab.entries[i]);
            sc.score_total = sc.score_centroid;
            all_scored.push_back(sc);
        }
        
        std::sort(all_scored.begin(), all_scored.end(),
            [](const auto& a, const auto& b) { return a.score_total > b.score_total; });
        
        if (all_scored.size() > k) {
            all_scored.resize(k);
        }
        
        return all_scored;
    }
};

// Global engine instance
inline GenerativeEngine& get_engine() {
    static GenerativeEngine engine;
    return engine;
}

} // namespace generative
} // namespace hypercube

#endif // HYPERCUBE_GENERATIVE_HPP
