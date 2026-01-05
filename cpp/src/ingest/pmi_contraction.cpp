/**
 * PMI-Based Geometric Contraction
 * 
 * NOT lazy frequency counting. This uses:
 * - Pointwise Mutual Information (PMI) to measure cohesion
 * - Branching Entropy for boundary detection  
 * - Geodesic midpoints on the 4D hypersphere for composite positioning
 * 
 * The question isn't "do A and B appear together often?"
 * The question is "do A and B BELONG together?"
 */

#include "hypercube/ingest/pmi_contraction.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/types.hpp"

#include <unordered_map>
#include <vector>
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
// COORDINATE CONVENTION:
//   uint32 coords: [1, 2^32-1] with CENTER at 2^31 = 2147483648
//   unit sphere:   [-1, +1] with center at origin
//   Vec4D operates in UNIT SPHERE space for geometric operations
// ============================================================================

// Constants for coordinate conversion
static constexpr double CENTER = 2147483648.0;  // 2^31
static constexpr double SCALE = 2147483647.0;   // 2^31 - 1

struct Vec4D {
    double x, y, z, m;  // Unit sphere space: [-1, +1] centered at origin
    
    Vec4D() : x(0), y(0), z(0), m(0) {}
    Vec4D(double x_, double y_, double z_, double m_) : x(x_), y(y_), z(z_), m(m_) {}
    
    // Convert FROM uint32 coords TO unit sphere
    // uint32 coords are in [1, 2^32-1] with CENTER at 2^31
    // Unit sphere is [-1, +1] centered at origin
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
    
    // Convert FROM unit sphere BACK TO uint32 coords
    // Scales [-1, +1] back to [1, 2^32-1] with CENTER at 2^31
    Point4D to_point() const {
        Point4D p;
        p.x = static_cast<Coord32>(std::round(CENTER + x * SCALE));
        p.y = static_cast<Coord32>(std::round(CENTER + y * SCALE));
        p.z = static_cast<Coord32>(std::round(CENTER + z * SCALE));
        p.m = static_cast<Coord32>(std::round(CENTER + m * SCALE));
        return p;
    }
};

// Geodesic midpoint on 4D hypersphere (SLERP with t=0.5)
// Returns the midpoint direction on the unit sphere (normalized)
// Caller must apply depth-based scaling for composite positioning
Vec4D geodesic_midpoint(const Vec4D& a, const Vec4D& b) {
    Vec4D an = a.normalized();
    Vec4D bn = b.normalized();
    
    double dot_val = an.dot(bn);
    dot_val = std::max(-1.0, std::min(1.0, dot_val));  // Clamp for numerical stability
    
    double theta = std::acos(dot_val);
    
    if (theta < 1e-6) {
        // Nearly identical points - just return normalized average direction
        return (a + b).normalized();
    }
    
    double sin_theta = std::sin(theta);
    double t = 0.5;  // Midpoint
    
    // SLERP formula - returns unit vector (on surface)
    Vec4D result = an * (std::sin((1-t)*theta) / sin_theta) + 
                   bn * (std::sin(t*theta) / sin_theta);
    
    // Return normalized (on unit sphere surface)
    // Depth-based scaling is applied in create_composite()
    return result.normalized();
}

// ============================================================================
// Token: A position in the sequence with geometric data
// ============================================================================

struct Token {
    Blake3Hash hash;
    Vec4D position;      // Position on 4D hypersphere
    uint32_t depth;      // 0 = atom, >0 = composite
    uint64_t atom_count; // Number of original atoms in this token
    
    // For composites: children info
    std::vector<Blake3Hash> children;
    std::vector<Vec4D> child_positions;
};

// ============================================================================
// Pair statistics for PMI calculation
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
// PMI Contraction Engine
// ============================================================================

class PMIContractor {
public:
    // Configuration
    double pmi_threshold = 0.0;  // Minimum PMI to consider merging
    size_t min_pair_count = 2;   // Pair must occur at least this many times
    
    // The current sequence of tokens
    std::vector<Token> sequence;
    
    // Statistics
    std::unordered_map<Blake3Hash, uint64_t, Blake3HashHasher> token_counts;
    std::unordered_map<PairKey, uint64_t, PairKeyHasher> pair_counts;
    uint64_t total_tokens = 0;
    uint64_t total_pairs = 0;
    
    // All composites created (for database insert)
    std::unordered_map<Blake3Hash, Token, Blake3HashHasher> composites;
    
    // Initialize from codepoints
    void initialize(const std::vector<uint32_t>& codepoints) {
        sequence.clear();
        sequence.reserve(codepoints.size());
        
        for (uint32_t cp : codepoints) {
            auto atom = AtomCalculator::compute_atom(cp);
            
            Token t;
            t.hash = atom.hash;
            t.position = Vec4D(atom.coords);
            t.depth = 0;
            t.atom_count = 1;
            
            sequence.push_back(std::move(t));
        }
        
        recompute_statistics();
    }
    
    // Recompute all statistics from current sequence
    void recompute_statistics() {
        token_counts.clear();
        pair_counts.clear();
        total_tokens = sequence.size();
        total_pairs = (sequence.size() > 1) ? sequence.size() - 1 : 0;
        
        for (size_t i = 0; i < sequence.size(); ++i) {
            token_counts[sequence[i].hash]++;
            
            if (i + 1 < sequence.size()) {
                PairKey pk{sequence[i].hash, sequence[i+1].hash};
                pair_counts[pk]++;
            }
        }
    }
    
    // Calculate PMI for a pair
    double calculate_pmi(const PairKey& pair, uint64_t pair_count) const {
        if (pair_count < min_pair_count) return -1000.0;  // Too rare
        if (total_tokens < 2 || total_pairs < 1) return -1000.0;
        
        auto it_a = token_counts.find(pair.first);
        auto it_b = token_counts.find(pair.second);
        if (it_a == token_counts.end() || it_b == token_counts.end()) {
            return -1000.0;
        }
        
        double p_ab = static_cast<double>(pair_count) / total_pairs;
        double p_a = static_cast<double>(it_a->second) / total_tokens;
        double p_b = static_cast<double>(it_b->second) / total_tokens;
        
        if (p_a < 1e-10 || p_b < 1e-10) return -1000.0;
        
        // PMI = log2(P(AB) / (P(A) * P(B)))
        return std::log2(p_ab / (p_a * p_b));
    }
    
    // Find the pair with highest PMI above threshold
    std::pair<PairKey, double> find_best_pair() const {
        PairKey best_pair{};
        double best_pmi = pmi_threshold;
        
        for (const auto& [pair, count] : pair_counts) {
            double pmi = calculate_pmi(pair, count);
            if (pmi > best_pmi) {
                best_pmi = pmi;
                best_pair = pair;
            }
        }
        
        return {best_pair, best_pmi};
    }
    
    // Create a composite token from two tokens
    // ARCHITECTURE: Atoms (depth=0) are on 3-sphere SURFACE (magnitude=1)
    //               Compositions are INSIDE sphere, scaling toward CENTER
    //               as depth increases: factor = 1 - 1/(depth+2)
    Token create_composite(const Token& a, const Token& b) {
        Token composite;
        
        // Hash: BLAKE3(child_hashes)
        std::vector<Blake3Hash> child_hashes = {a.hash, b.hash};
        composite.hash = Blake3Hasher::hash_children(child_hashes);
        
        // Depth: max(child depths) + 1
        composite.depth = std::max(a.depth, b.depth) + 1;
        
        // Position: Geodesic midpoint direction, scaled by depth
        // geodesic_midpoint returns unit vector (on surface)
        Vec4D direction = geodesic_midpoint(a.position, b.position);
        
        // Apply depth-based scaling: compositions move TOWARD CENTER as depth increases
        // factor = 1 - 1/(depth+2):
        //   depth=1: factor=0.667 (first composite is 66.7% of radius from center)
        //   depth=2: factor=0.75
        //   depth=3: factor=0.8
        //   depth→∞: factor→1.0 (approaches surface but never reaches)
        double factor = 1.0 - 1.0 / static_cast<double>(composite.depth + 2);
        composite.position = direction * factor;
        
        // Atom count: sum
        composite.atom_count = a.atom_count + b.atom_count;
        
        // Children info
        composite.children = {a.hash, b.hash};
        composite.child_positions = {a.position, b.position};
        
        return composite;
    }
    
    // Contract: replace all occurrences of pair with composite
    size_t contract_pair(const PairKey& pair) {
        // Find first occurrence to get token data
        Token first_token, second_token;
        bool found = false;
        
        for (size_t i = 0; i + 1 < sequence.size(); ++i) {
            if (sequence[i].hash == pair.first && sequence[i+1].hash == pair.second) {
                first_token = sequence[i];
                second_token = sequence[i+1];
                found = true;
                break;
            }
        }
        
        if (!found) return 0;
        
        // Create composite
        Token composite = create_composite(first_token, second_token);
        
        // Store for database insert
        composites[composite.hash] = composite;
        
        // Replace all occurrences
        std::vector<Token> new_sequence;
        new_sequence.reserve(sequence.size());
        
        size_t replacements = 0;
        size_t i = 0;
        while (i < sequence.size()) {
            if (i + 1 < sequence.size() && 
                sequence[i].hash == pair.first && 
                sequence[i+1].hash == pair.second) {
                new_sequence.push_back(composite);
                replacements++;
                i += 2;  // Skip both tokens
            } else {
                new_sequence.push_back(sequence[i]);
                i++;
            }
        }
        
        sequence = std::move(new_sequence);
        return replacements;
    }
    
    // Run contraction until sequence is a single token (complete binary reduction)
    // This follows Re-Pair: keep contracting the best available pair until done
    void run(size_t max_iterations = 1000000) {
        size_t iteration = 0;
        size_t high_pmi_phase = 0;
        size_t completion_phase = 0;
        
        // Phase 1: Contract high-PMI pairs (information-theoretic cohesion)
        while (iteration < max_iterations && sequence.size() > 1) {
            auto [best_pair, best_pmi] = find_best_pair();
            
            if (best_pmi <= pmi_threshold) {
                break;  // Transition to completion phase
            }
            
            size_t replacements = contract_pair(best_pair);
            if (replacements == 0) break;
            
            recompute_statistics();
            iteration++;
            high_pmi_phase++;
            
            if (iteration % 1000 == 0) {
                std::cerr << "\r[PMI Phase 1] Iteration " << iteration 
                          << ", length " << sequence.size()
                          << ", PMI " << std::fixed << std::setprecision(2) << best_pmi
                          << "   " << std::flush;
            }
        }
        
        // Phase 2: Complete the binary tree using ANY available pairs
        // Re-Pair algorithm: contract until no pairs occur twice
        // Then: contract remaining pairs geometrically (nearest neighbors)
        while (sequence.size() > 1 && iteration < max_iterations) {
            // First try: any pair that occurs more than once (Re-Pair style)
            auto [best_pair, best_pmi] = find_best_pair();
            
            if (pair_counts[best_pair] > 1) {
                // This pair occurs multiple times - contract it
                size_t replacements = contract_pair(best_pair);
                if (replacements > 0) {
                    recompute_statistics();
                    iteration++;
                    completion_phase++;
                    continue;
                }
            }
            
            // No repeated pairs: use geometric nearest-neighbor pairing
            // Find the two adjacent tokens closest on the hypersphere
            if (sequence.size() < 2) break;
            
            size_t best_idx = 0;
            double best_dist = std::numeric_limits<double>::max();
            
            for (size_t i = 0; i + 1 < sequence.size(); i++) {
                // Geodesic distance on unit sphere: arccos(dot product)
                double dot = sequence[i].position.dot(sequence[i+1].position);
                double dist = std::acos(std::clamp(dot, -1.0, 1.0));
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            
            // Contract this single pair (geometric locality)
            Token composite = create_composite(sequence[best_idx], sequence[best_idx + 1]);
            
            std::vector<Token> new_sequence;
            new_sequence.reserve(sequence.size() - 1);
            
            for (size_t i = 0; i < sequence.size(); i++) {
                if (i == best_idx) {
                    new_sequence.push_back(composite);
                    i++;  // Skip the next token too
                } else {
                    new_sequence.push_back(sequence[i]);
                }
            }
            
            sequence = std::move(new_sequence);
            recompute_statistics();
            iteration++;
            completion_phase++;
            
            if (completion_phase % 1000 == 0) {
                std::cerr << "\r[PMI Phase 2] Iteration " << iteration 
                          << ", length " << sequence.size()
                          << "   " << std::flush;
            }
        }
        
        std::cerr << "\n[PMI] Complete: " << high_pmi_phase << " high-PMI contractions, "
                  << completion_phase << " completion contractions, "
                  << composites.size() << " total compositions\n";
    }
    
    // Get the single root token (sequence should already be reduced to 1)
    Token get_root() {
        if (sequence.empty()) {
            return Token{};
        }
        
        // If run() completed correctly, sequence should be exactly 1 token
        if (sequence.size() != 1) {
            std::cerr << "[WARNING] Sequence not fully reduced: " << sequence.size() 
                      << " tokens remaining\n";
        }
        
        return sequence[0];
    }
    
    // Extract compositions for database
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
            
            for (size_t i = 0; i < token.children.size(); ++i) {
                ChildInfo ci;
                ci.hash = token.children[i];
                Point4D cp = token.child_positions[i].to_point();
                ci.x = cp.x;
                ci.y = cp.y;
                ci.z = cp.z;
                ci.m = cp.m;
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

PMIChunkResult process_chunk_pmi(
    const std::vector<uint32_t>& codepoints,
    double pmi_threshold,
    std::atomic<size_t>& progress_chars
) {
    PMIChunkResult result;
    
    if (codepoints.empty()) {
        return result;
    }
    
    if (codepoints.size() == 1) {
        auto atom = AtomCalculator::compute_atom(codepoints[0]);
        result.root_hash = atom.hash;
        progress_chars += 1;
        return result;
    }
    
    PMIContractor contractor;
    contractor.pmi_threshold = pmi_threshold;
    contractor.min_pair_count = 2;
    
    contractor.initialize(codepoints);
    contractor.run();
    
    Token root = contractor.get_root();
    result.root_hash = root.hash;
    result.compositions = contractor.extract_compositions();
    
    progress_chars += codepoints.size();
    return result;
}

// ============================================================================
// PMI Ingester (replaces SequiturIngester)
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
    
    // Split into chunks at line boundaries
    constexpr size_t MIN_CHUNK_SIZE = 500;
    constexpr size_t MAX_CHUNK_SIZE = 10000;
    
    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(MAX_CHUNK_SIZE);
    
    for (uint32_t cp : all_codepoints) {
        current_chunk.push_back(cp);
        
        if (cp == '\n' && current_chunk.size() >= MIN_CHUNK_SIZE) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = std::vector<uint32_t>();
            current_chunk.reserve(MAX_CHUNK_SIZE);
        } else if (current_chunk.size() >= MAX_CHUNK_SIZE) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = std::vector<uint32_t>();
            current_chunk.reserve(MAX_CHUNK_SIZE);
        }
    }
    if (!current_chunk.empty()) {
        chunks.push_back(std::move(current_chunk));
    }
    
    std::cerr << "[PMI] Split into " << chunks.size() << " chunks\n";
    
    std::atomic<size_t> progress_chars{0};
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process chunks in parallel
    std::vector<std::future<PMIChunkResult>> futures;
    futures.reserve(chunks.size());
    
    double threshold = impl_->pmi_threshold;
    for (auto& chunk : chunks) {
        futures.push_back(std::async(std::launch::async, 
            [&chunk, threshold, &progress_chars]() {
                return process_chunk_pmi(chunk, threshold, progress_chars);
            }
        ));
    }
    
    // Progress display
    std::atomic<bool> done{false};
    std::thread progress_thread([&]() {
        while (!done) {
            size_t chars = progress_chars.load();
            double pct = 100.0 * chars / total_chars;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            double rate = chars / (elapsed + 0.001);
            
            std::cerr << "\r[PMI] " << std::fixed << std::setprecision(1) << pct << "% "
                      << "(" << chars << "/" << total_chars << ") "
                      << std::setprecision(0) << rate << " chars/sec   " << std::flush;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Collect results
    std::vector<Blake3Hash> root_hashes;
    root_hashes.reserve(futures.size());
    
    for (auto& f : futures) {
        auto result = f.get();
        root_hashes.push_back(result.root_hash);
        
        std::lock_guard<std::mutex> lock(impl_->comp_mutex);
        for (auto& comp : result.compositions) {
            impl_->all_compositions[comp.hash] = std::move(comp);
        }
    }
    
    done = true;
    progress_thread.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cerr << "\r[PMI] 100% - " << impl_->all_compositions.size() 
              << " unique compositions in " << std::fixed << std::setprecision(2) 
              << elapsed << "s\n";
    
    new_compositions.reserve(impl_->all_compositions.size());
    for (auto& [hash, comp] : impl_->all_compositions) {
        new_compositions.push_back(std::move(comp));
    }
    
    return root_hashes;
}

size_t PMIIngester::composition_count() const {
    return impl_->all_compositions.size();
}

void PMIIngester::clear() {
    impl_->all_compositions.clear();
}

} // namespace hypercube::ingest
