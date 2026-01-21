/**
 * PMI-Based Geometric Contraction - OPTIMIZED
 *
 * Performance improvements over naive implementation:
 * - std::list for O(1) sequence modifications (was O(n) vector copy)
 * - Priority queue with lazy invalidation for O(log n) best-pair lookup
 * - Incremental statistics updates (was O(n) full recompute)
 * - Skip-ahead scanning for Phase 2 geometric pairing
 *
 * The question isn't "do A and B appear together often?"
 * The question is "do A and B BELONG together?"
 */

#include "hypercube/ingest/pmi_contraction.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/types.hpp"
#include "hypercube/util/utf8.hpp"

#include <unordered_map>
#include <vector>
#include <list>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <atomic>
#include <thread>
#include <future>
#include <queue>

namespace hypercube::ingest {

// ============================================================================
// 4D Vector operations for hypersphere geometry
// ============================================================================
static constexpr double CENTER = 2147483648.0;
static constexpr double SCALE = 2147483647.0;

struct Vec4D {
    double x, y, z, m;

    Vec4D() : x(0), y(0), z(0), m(0) {}
    Vec4D(double x_, double y_, double z_, double m_) : x(x_), y(y_), z(z_), m(m_) {}

    explicit Vec4D(const Point4D& p) {
        x = (static_cast<double>(p.x) - CENTER) / SCALE;
        y = (static_cast<double>(p.y) - CENTER) / SCALE;
        z = (static_cast<double>(p.z) - CENTER) / SCALE;
        m = (static_cast<double>(p.m) - CENTER) / SCALE;
    }

    Vec4D operator+(const Vec4D& o) const { return {x+o.x, y+o.y, z+o.z, m+o.m}; }
    Vec4D operator-(const Vec4D& o) const { return {x-o.x, y-o.y, z-o.z, m-o.m}; }
    Vec4D operator*(double s) const { return {x*s, y*s, z*s, m*s}; }

    double dot(const Vec4D& o) const { return x*o.x + y*o.y + z*o.z + m*o.m; }
    double norm() const { return std::sqrt(dot(*this)); }

    Vec4D normalized() const {
        double n = norm();
        if (n < 1e-10) return {1, 0, 0, 0};
        return *this * (1.0 / n);
    }

    Point4D to_point() const {
        Point4D p;
        p.x = static_cast<Coord64>(std::round(CENTER + x * SCALE));
        p.y = static_cast<Coord64>(std::round(CENTER + y * SCALE));
        p.z = static_cast<Coord64>(std::round(CENTER + z * SCALE));
        p.m = static_cast<Coord64>(std::round(CENTER + m * SCALE));
        return p;
    }
};

Vec4D geodesic_midpoint(const Vec4D& a, const Vec4D& b) {
    Vec4D an = a.normalized();
    Vec4D bn = b.normalized();
    double dot_val = std::max(-1.0, std::min(1.0, an.dot(bn)));
    double theta = std::acos(dot_val);
    if (theta < 1e-6) return (a + b).normalized();
    double sin_theta = std::sin(theta);
    Vec4D result = an * (std::sin(0.5*theta) / sin_theta) + bn * (std::sin(0.5*theta) / sin_theta);
    return result.normalized();
}

// ============================================================================
// Token structure
// ============================================================================
struct Token {
    Blake3Hash hash;
    Vec4D position;
    uint32_t depth = 0;
    uint64_t atom_count = 1;
    uint32_t codepoint = 0;
    std::string label;
    std::vector<Blake3Hash> children;
    std::vector<Vec4D> child_positions;
    std::vector<uint32_t> child_depths;
    std::vector<uint32_t> child_codepoints;
    std::vector<std::string> child_labels;
};

// ============================================================================
// Pair key for hash map
// ============================================================================
struct PairKey {
    Blake3Hash first;
    Blake3Hash second;
    bool operator==(const PairKey& o) const {
        return first == o.first && second == o.second;
    }
};

struct PairKeyHasher {
    size_t operator()(const PairKey& k) const {
        uint64_t h1, h2;
        std::memcpy(&h1, k.first.bytes.data(), 8);
        std::memcpy(&h2, k.second.bytes.data(), 8);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
    }
};

// ============================================================================
// PMI Priority Queue Entry
// ============================================================================
struct PMIEntry {
    double pmi;
    PairKey pair;
    uint64_t version;  // For lazy invalidation

    bool operator<(const PMIEntry& o) const {
        return pmi < o.pmi;  // Max-heap
    }
};

// ============================================================================
// Optimized PMI Contraction Engine
// ============================================================================
class PMIContractor {
public:
    double pmi_threshold = 0.0;
    size_t min_pair_count = 2;
    size_t max_chunk_size = 2000;
    size_t max_completion_iterations = 10000;
    size_t no_progress_threshold = 10;

    // Use std::list for O(1) splice/removal
    std::list<Token> sequence;

    // Statistics with version tracking for lazy invalidation
    std::unordered_map<Blake3Hash, uint64_t, Blake3HashHasher> token_counts;
    std::unordered_map<PairKey, uint64_t, PairKeyHasher> pair_counts;
    std::unordered_map<PairKey, uint64_t, PairKeyHasher> pair_versions;
    uint64_t total_tokens = 0;
    uint64_t total_pairs = 0;
    uint64_t global_version = 0;

    // Priority queue for best PMI (lazy invalidation)
    std::priority_queue<PMIEntry> pmi_queue;

    // Composites created
    std::unordered_map<Blake3Hash, Token, Blake3HashHasher> composites;

    void initialize(const std::vector<uint32_t>& codepoints) {
        sequence.clear();

        for (uint32_t cp : codepoints) {
            auto atom = AtomCalculator::compute_atom(cp);
            Token t;
            t.hash = atom.hash;
            t.position = Vec4D(atom.coords);
            t.depth = 0;
            t.atom_count = 1;
            t.codepoint = cp;
            t.label = util::encode_utf8(cp);
            sequence.push_back(std::move(t));
        }

        build_initial_statistics();
    }

    void build_initial_statistics() {
        token_counts.clear();
        pair_counts.clear();
        pair_versions.clear();
        global_version = 0;

        // Clear and rebuild priority queue
        pmi_queue = std::priority_queue<PMIEntry>();

        total_tokens = sequence.size();
        total_pairs = (sequence.size() > 1) ? sequence.size() - 1 : 0;

        auto it = sequence.begin();
        auto prev = it;
        if (it != sequence.end()) {
            token_counts[it->hash]++;
            ++it;
        }

        while (it != sequence.end()) {
            token_counts[it->hash]++;
            PairKey pk{prev->hash, it->hash};
            pair_counts[pk]++;
            prev = it;
            ++it;
        }

        // Build initial priority queue
        for (const auto& [pair, count] : pair_counts) {
            double pmi = calculate_pmi(pair, count);
            if (pmi > pmi_threshold) {
                pmi_queue.push({pmi, pair, 0});
                pair_versions[pair] = 0;
            }
        }
    }

    double calculate_pmi(const PairKey& pair, uint64_t pair_count) const {
        if (pair_count < min_pair_count) return -1000.0;
        if (total_tokens < 2 || total_pairs < 1) return -1000.0;

        auto it_a = token_counts.find(pair.first);
        auto it_b = token_counts.find(pair.second);
        if (it_a == token_counts.end() || it_b == token_counts.end()) return -1000.0;

        double p_ab = static_cast<double>(pair_count) / total_pairs;
        double p_a = static_cast<double>(it_a->second) / total_tokens;
        double p_b = static_cast<double>(it_b->second) / total_tokens;

        if (p_a < 1e-10 || p_b < 1e-10) return -1000.0;
        return std::log2(p_ab / (p_a * p_b));
    }

    std::pair<PairKey, double> find_best_pair() {
        // Lazy invalidation: skip entries with outdated versions
        while (!pmi_queue.empty()) {
            PMIEntry top = pmi_queue.top();
            auto ver_it = pair_versions.find(top.pair);
            if (ver_it != pair_versions.end() && ver_it->second == top.version) {
                // Valid entry - but recompute PMI to ensure accuracy
                auto count_it = pair_counts.find(top.pair);
                if (count_it != pair_counts.end() && count_it->second >= min_pair_count) {
                    double current_pmi = calculate_pmi(top.pair, count_it->second);
                    if (current_pmi > pmi_threshold) {
                        return {top.pair, current_pmi};
                    }
                }
            }
            pmi_queue.pop();
        }
        return {PairKey{}, -1000.0};
    }

    Token create_composite(const Token& a, const Token& b) {
        Token composite;
        std::vector<Blake3Hash> child_hashes = {a.hash, b.hash};
        composite.hash = Blake3Hasher::hash_children(child_hashes);
        composite.depth = std::max(a.depth, b.depth) + 1;
        Vec4D direction = geodesic_midpoint(a.position, b.position);
        double factor = 1.0 - 1.0 / static_cast<double>(composite.depth + 2);
        composite.position = direction * factor;
        composite.atom_count = a.atom_count + b.atom_count;
        composite.label = a.label + b.label;
        composite.children = {a.hash, b.hash};
        composite.child_positions = {a.position, b.position};
        composite.child_depths = {a.depth, b.depth};
        composite.child_codepoints = {a.codepoint, b.codepoint};
        composite.child_labels = {a.label, b.label};
        return composite;
    }

    // Incremental update when removing a token
    void decrement_token(const Blake3Hash& hash) {
        auto it = token_counts.find(hash);
        if (it != token_counts.end() && it->second > 0) {
            it->second--;
            if (it->second == 0) token_counts.erase(it);
        }
        total_tokens--;
    }

    // Incremental update when adding a token
    void increment_token(const Blake3Hash& hash) {
        token_counts[hash]++;
        total_tokens++;
    }

    // Incremental update for pair removal
    void decrement_pair(const PairKey& pk) {
        auto it = pair_counts.find(pk);
        if (it != pair_counts.end() && it->second > 0) {
            it->second--;
            if (it->second == 0) {
                pair_counts.erase(it);
                pair_versions.erase(pk);
            } else {
                // Invalidate old queue entries by bumping version
                global_version++;
                pair_versions[pk] = global_version;
                double pmi = calculate_pmi(pk, it->second);
                if (pmi > pmi_threshold) {
                    pmi_queue.push({pmi, pk, global_version});
                }
            }
        }
        total_pairs--;
    }

    // Incremental update for pair addition
    void increment_pair(const PairKey& pk) {
        pair_counts[pk]++;
        total_pairs++;
        global_version++;
        pair_versions[pk] = global_version;
        double pmi = calculate_pmi(pk, pair_counts[pk]);
        if (pmi > pmi_threshold) {
            pmi_queue.push({pmi, pk, global_version});
        }
    }

    size_t contract_pair(const PairKey& pair) {
        // Find and replace all occurrences using list iterators
        Token composite;
        bool composite_created = false;
        size_t replacements = 0;

        auto it = sequence.begin();
        while (it != sequence.end()) {
            auto next = std::next(it);
            if (next == sequence.end()) break;

            if (it->hash == pair.first && next->hash == pair.second) {
                if (!composite_created) {
                    composite = create_composite(*it, *next);
                    composites[composite.hash] = composite;
                    composite_created = true;
                }

                // Get neighbors for incremental updates
                auto prev = (it != sequence.begin()) ? std::prev(it) : sequence.end();
                auto next_next = std::next(next);

                // Update statistics: remove old pairs
                if (prev != sequence.end()) {
                    decrement_pair({prev->hash, it->hash});
                }
                decrement_pair({it->hash, next->hash});
                if (next_next != sequence.end()) {
                    decrement_pair({next->hash, next_next->hash});
                }

                // Remove old tokens from counts
                decrement_token(it->hash);
                decrement_token(next->hash);

                // Replace the pair with composite
                *it = composite;
                it = sequence.erase(next);

                // Add new token to counts
                increment_token(composite.hash);

                // Add new pairs
                if (prev != sequence.end()) {
                    increment_pair({prev->hash, composite.hash});
                }
                auto new_next = std::next(std::prev(it)); // Points to element after composite
                if (it != sequence.end()) {
                    increment_pair({composite.hash, it->hash});
                }

                replacements++;
                // Don't advance - check if new composite forms pair with next
            } else {
                ++it;
            }
        }

        return replacements;
    }

    Token process_recursive(std::list<Token> seq) {
        if (seq.size() <= max_chunk_size) {
            PMIContractor temp;
            temp.pmi_threshold = this->pmi_threshold;
            temp.min_pair_count = this->min_pair_count;
            temp.max_chunk_size = this->max_chunk_size;
            temp.max_completion_iterations = this->max_completion_iterations;
            temp.no_progress_threshold = this->no_progress_threshold;
            temp.sequence = std::move(seq);
            temp.build_initial_statistics();
            temp.run_phases();
            // merge composites
            for (auto& p : temp.composites) {
                this->composites[p.first] = std::move(p.second);
            }
            return temp.get_root();
        } else {
            size_t mid = seq.size() / 2;
            auto mid_it = std::next(seq.begin(), mid);
            std::list<Token> left(seq.begin(), mid_it);
            std::list<Token> right(mid_it, seq.end());
            Token left_root = process_recursive(std::move(left));
            Token right_root = process_recursive(std::move(right));
            std::list<Token> new_seq = {left_root, right_root};
            return process_recursive(std::move(new_seq));
        }
    }

    void run_phases(size_t max_iterations = 1000000) {
        size_t iteration = 0;
        size_t high_pmi_phase = 0;
        size_t completion_phase = 0;
        size_t no_progress_count = 0;
        size_t completion_iters = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t last_report = 0;

        // Phase 1: Contract high-PMI pairs
        while (iteration < max_iterations && sequence.size() > 1) {
            auto [best_pair, best_pmi] = find_best_pair();
            if (best_pmi <= pmi_threshold) break;

            size_t replacements = contract_pair(best_pair);
            if (replacements == 0) {
                no_progress_count++;
            } else {
                no_progress_count = 0;
            }
            if (no_progress_count >= no_progress_threshold) break;

            iteration++;
            high_pmi_phase++;

            // Progress report every 5000 iterations or 2 seconds
            if (iteration - last_report >= 5000) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                double rate = iteration * 1000.0 / (elapsed + 1);
                std::cerr << "\r[PMI Phase 1] " << iteration << " iters, "
                          << sequence.size() << " tokens, PMI=" << std::fixed
                          << std::setprecision(2) << best_pmi
                          << " (" << std::setprecision(0) << rate << " iter/s)   " << std::flush;
                last_report = iteration;
            }
        }

        // Phase 2: Complete using geometric nearest-neighbor
        while (sequence.size() > 1 && iteration < max_iterations) {
            completion_iters++;
            if (completion_iters >= max_completion_iterations) break;

            // Try remaining frequent pairs first
            auto [best_pair, best_pmi] = find_best_pair();
            if (best_pmi > -100.0) {
                auto count_it = pair_counts.find(best_pair);
                if (count_it != pair_counts.end() && count_it->second > 1) {
                    size_t replacements = contract_pair(best_pair);
                    if (replacements > 0) {
                        iteration++;
                        completion_phase++;
                        continue;
                    }
                }
            }

            // Geometric nearest-neighbor: find closest adjacent pair
            if (sequence.size() < 2) break;

            auto best_it = sequence.begin();
            double best_dist = std::numeric_limits<double>::max();

            for (auto it = sequence.begin(); it != sequence.end(); ++it) {
                auto next = std::next(it);
                if (next == sequence.end()) break;
                double dot = it->position.dot(next->position);
                double dist = std::acos(std::clamp(dot, -1.0, 1.0));
                if (dist < best_dist) {
                    best_dist = dist;
                    best_it = it;
                }
            }

            auto next_it = std::next(best_it);
            Token composite = create_composite(*best_it, *next_it);
            composites[composite.hash] = composite;

            // Incremental updates
            auto prev = (best_it != sequence.begin()) ? std::prev(best_it) : sequence.end();
            auto next_next = std::next(next_it);

            if (prev != sequence.end()) {
                decrement_pair({prev->hash, best_it->hash});
            }
            decrement_pair({best_it->hash, next_it->hash});
            if (next_next != sequence.end()) {
                decrement_pair({next_it->hash, next_next->hash});
            }

            decrement_token(best_it->hash);
            decrement_token(next_it->hash);

            *best_it = composite;
            sequence.erase(next_it);

            increment_token(composite.hash);

            if (prev != sequence.end()) {
                increment_pair({prev->hash, composite.hash});
            }
            auto after_composite = std::next(best_it);
            if (after_composite != sequence.end()) {
                increment_pair({composite.hash, after_composite->hash});
            }

            iteration++;
            completion_phase++;

            if (completion_phase % 10000 == 0) {
                std::cerr << "\r[PMI Phase 2] " << sequence.size() << " tokens remaining   " << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cerr << "\n[PMI] Complete: " << high_pmi_phase << " high-PMI + "
                  << completion_phase << " geometric = " << composites.size()
                  << " compositions (" << total_ms << " ms)\n";
    }

    void run(size_t max_iterations = 1000000) {
        if (sequence.size() <= max_chunk_size) {
            run_phases(max_iterations);
        } else {
            Token root = process_recursive(std::move(sequence));
            sequence.clear();
            sequence.push_back(root);
        }
    }

    Token get_root() {
        if (sequence.empty()) return Token{};
        if (sequence.size() != 1) {
            std::cerr << "[WARNING] Sequence not fully reduced: " << sequence.size() << " tokens\n";
        }
        return sequence.front();
    }

    std::vector<CompositionRecord> extract_compositions() {
        std::vector<CompositionRecord> result;
        result.reserve(composites.size());

        for (const auto& [hash, token] : composites) {
            CompositionRecord rec;
            rec.hash = token.hash;

            Point4D p = token.position.to_point();
            rec.coord_x = p.x;
            rec.coord_y = p.y;
            rec.coord_z = p.z;
            rec.coord_m = p.m;

            HilbertIndex h = HilbertCurve::coords_to_index(p);
            rec.hilbert_lo = static_cast<int64_t>(h.lo);
            rec.hilbert_hi = static_cast<int64_t>(h.hi);

            rec.depth = token.depth;
            rec.atom_count = token.atom_count;
            rec.label = token.label;

            for (size_t i = 0; i < token.children.size(); ++i) {
                ChildInfo ci;
                ci.hash = token.children[i];
                Point4D cp = token.child_positions[i].to_point();
                ci.x = cp.x;
                ci.y = cp.y;
                ci.z = cp.z;
                ci.m = cp.m;
                ci.is_atom = (i < token.child_depths.size()) ? (token.child_depths[i] == 0) : true;
                if (i < token.child_codepoints.size()) ci.codepoint = token.child_codepoints[i];
                if (i < token.child_labels.size()) ci.label = token.child_labels[i];
                rec.children.push_back(ci);
            }

            result.push_back(std::move(rec));
        }

        return result;
    }
};

// ============================================================================
// Chunk processing
// ============================================================================
struct PMIChunkResult {
    Blake3Hash root_hash;
    std::vector<CompositionRecord> compositions;
};

static std::atomic<size_t> progress_chars{0};

PMIChunkResult process_chunk_pmi(
    const std::vector<uint32_t>& codepoints,
    double pmi_threshold
) {
    PMIChunkResult result;
    if (codepoints.empty()) return result;

    PMIContractor contractor;
    contractor.pmi_threshold = pmi_threshold;
    contractor.initialize(codepoints);
    contractor.run();

    Token root = contractor.get_root();
    result.root_hash = root.hash;
    result.compositions = contractor.extract_compositions();

    progress_chars += codepoints.size();
    return result;
}

// ============================================================================
// PMI Ingester
// ============================================================================
struct PMIIngester::Impl {
    size_t num_threads;
    double pmi_threshold;
    std::mutex comp_mutex;
    std::unordered_map<Blake3Hash, CompositionRecord, Blake3HashHasher> all_compositions;

    Impl(size_t threads, double threshold)
        : num_threads(threads), pmi_threshold(threshold) {}
};

PMIIngester::PMIIngester(size_t num_threads, double pmi_threshold)
    : impl_(std::make_unique<Impl>(
        num_threads == 0 ? std::thread::hardware_concurrency() : num_threads,
        pmi_threshold
      ))
{}

PMIIngester::~PMIIngester() = default;

std::vector<Blake3Hash> PMIIngester::ingest(
    const std::string& text,
    std::vector<CompositionRecord>& new_compositions
) {
    auto all_codepoints = AtomCalculator::decode_utf8(text);

    if (all_codepoints.empty()) return {};

    size_t total_chars = all_codepoints.size();
    std::cerr << "[PMI] " << total_chars << " codepoints, "
              << impl_->num_threads << " threads, threshold "
              << impl_->pmi_threshold << "\n";

    // Split into chunks at natural boundaries (sentences/paragraphs)
    constexpr size_t MIN_CHUNK_SIZE = 1000;
    constexpr size_t MAX_CHUNK_SIZE = 50000;

    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(MAX_CHUNK_SIZE);

    for (uint32_t cp : all_codepoints) {
        current_chunk.push_back(cp);

        // Split at paragraph boundaries when chunk is large enough
        bool is_paragraph_end = (cp == '\n' && !current_chunk.empty() &&
                                 current_chunk.size() >= 2 &&
                                 current_chunk[current_chunk.size()-2] == '\n');

        if ((is_paragraph_end && current_chunk.size() >= MIN_CHUNK_SIZE) ||
            current_chunk.size() >= MAX_CHUNK_SIZE) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = std::vector<uint32_t>();
            current_chunk.reserve(MAX_CHUNK_SIZE);
        }
    }

    if (!current_chunk.empty()) {
        chunks.push_back(std::move(current_chunk));
    }

    std::cerr << "[PMI] Split into " << chunks.size() << " chunks\n";

    // Process chunks in parallel
    progress_chars = 0;
    std::vector<std::future<PMIChunkResult>> futures;

    for (const auto& chunk : chunks) {
        futures.push_back(std::async(std::launch::async,
            process_chunk_pmi, chunk, impl_->pmi_threshold));
    }

    std::vector<Blake3Hash> root_hashes;

    for (auto& future : futures) {
        PMIChunkResult result = future.get();
        root_hashes.push_back(result.root_hash);

        std::lock_guard<std::mutex> lock(impl_->comp_mutex);
        for (auto& comp : result.compositions) {
            impl_->all_compositions[comp.hash] = std::move(comp);
        }
    }

    // Export unique compositions
    new_compositions.clear();
    new_compositions.reserve(impl_->all_compositions.size());
    for (auto& [hash, comp] : impl_->all_compositions) {
        new_compositions.push_back(std::move(comp));
    }

    std::cerr << "[PMI] Extracted " << new_compositions.size() << " unique compositions\n";

    return root_hashes;
}

} // namespace hypercube::ingest
