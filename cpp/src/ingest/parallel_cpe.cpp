/**
 * Parallel CPE (Cascading Pair Encoding) Ingester
 * 
 * ZERO DATABASE ROUNDTRIPS until final insert.
 * Everything is computed client-side using AtomCalculator.
 * 
 * Algorithm:
 * 1. Split text into chunks (paragraphs/lines)
 * 2. Process chunks in parallel using thread pool
 * 3. Each chunk runs iterative pair discovery (CPE)
 * 4. Merge all discovered compositions
 * 5. Single batch insert to database
 * 
 * Performance: O(n) per iteration, O(log n) iterations typically
 * Memory: Proportional to text size
 * CPU: Scales with cores
 */

#include "hypercube/ingest/parallel_cpe.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/blake3.hpp"

#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

namespace hypercube::ingest {

// Convert AtomCalculator::CompositionRecord to ingest::CompositionRecord
static CompositionRecord convert_record(const hypercube::CompositionRecord& src) {
    CompositionRecord dst;
    dst.hash = src.hash;
    dst.coord_x = static_cast<int32_t>(src.centroid.x);
    dst.coord_y = static_cast<int32_t>(src.centroid.y);
    dst.coord_z = static_cast<int32_t>(src.centroid.z);
    dst.coord_m = static_cast<int32_t>(src.centroid.m);
    dst.hilbert_lo = static_cast<int64_t>(src.hilbert.lo);
    dst.hilbert_hi = static_cast<int64_t>(src.hilbert.hi);
    dst.depth = src.depth;
    dst.atom_count = src.atom_count;
    
    // Convert children with correct is_atom based on child_depths
    dst.children.reserve(src.children.size());
    for (size_t i = 0; i < src.children.size(); ++i) {
        ChildInfo ci;
        ci.hash = src.children[i];
        ci.x = static_cast<int32_t>(src.child_coords[i].x);
        ci.y = static_cast<int32_t>(src.child_coords[i].y);
        ci.z = static_cast<int32_t>(src.child_coords[i].z);
        ci.m = static_cast<int32_t>(src.child_coords[i].m);
        // Use child_depths to determine if child is atom (0) or composition (>0)
        ci.is_atom = (i < src.child_depths.size()) ? (src.child_depths[i] == 0) : true;
        dst.children.push_back(ci);
    }
    
    return dst;
}

// ============================================================================
// Hash-based pair key (faster than string concatenation)
// ============================================================================

struct PairKey {
    Blake3Hash first;
    Blake3Hash second;
    
    bool operator==(const PairKey& other) const {
        return first == other.first && second == other.second;
    }
};

struct PairKeyHasher {
    size_t operator()(const PairKey& k) const {
        // XOR first 8 bytes of each hash
        uint64_t h1, h2;
        std::memcpy(&h1, k.first.bytes.data(), 8);
        std::memcpy(&h2, k.second.bytes.data(), 8);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
    }
};

// ============================================================================
// Token with metadata
// ============================================================================

struct Token {
    Blake3Hash hash;
    Point4D coords;
    uint32_t depth;
    uint64_t atom_count;
};

// ============================================================================
// CPE on a single chunk - runs iteratively until no frequent pairs
// ============================================================================

struct ChunkResult {
    Blake3Hash root_hash;
    std::vector<CompositionRecord> compositions;
};

ChunkResult process_chunk(
    const std::vector<uint32_t>& codepoints,
    std::atomic<size_t>& progress_chars,
    size_t max_iterations = 1000
) {
    ChunkResult result;
    
    if (codepoints.empty()) return result;
    if (codepoints.size() == 1) {
        auto atom = AtomCalculator::compute_atom(codepoints[0]);
        result.root_hash = atom.hash;
        return result;
    }
    
    // Initialize tokens from codepoints
    std::vector<Token> current;
    current.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        auto atom = AtomCalculator::compute_atom(cp);
        current.push_back({atom.hash, atom.coords, 0, 1});
    }
    
    // Cache: hash -> Token (for looking up existing compositions)
    std::unordered_map<Blake3Hash, Token, Blake3HashHasher> token_cache;
    for (const auto& t : current) {
        token_cache[t.hash] = t;
    }
    
    size_t iteration = 0;
    
    while (current.size() > 1 && iteration < max_iterations) {
        ++iteration;
        
        // Count all adjacent pairs in O(n)
        std::unordered_map<PairKey, size_t, PairKeyHasher> pair_counts;
        for (size_t i = 0; i + 1 < current.size(); ++i) {
            PairKey key{current[i].hash, current[i+1].hash};
            pair_counts[key]++;
        }
        
        // Find most frequent pair in O(unique pairs)
        PairKey best_pair{};
        size_t best_count = 0;
        for (const auto& [key, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best_pair = key;
            }
        }
        
        // If no pair occurs more than once, we're done
        if (best_count <= 1) break;
        
        // Get metadata for the two tokens
        const Token& t1 = token_cache[best_pair.first];
        const Token& t2 = token_cache[best_pair.second];
        
        // Compute composition using AtomCalculator
        std::vector<AtomCalculator::ChildInfo> children = {
            {t1.hash, t1.coords, t1.depth, t1.atom_count},
            {t2.hash, t2.coords, t2.depth, t2.atom_count}
        };
        auto comp = AtomCalculator::compute_composition(children);
        
        // Store composition (convert to ingest::CompositionRecord)
        result.compositions.push_back(convert_record(comp));
        
        // Create token for new composition
        Token new_token{comp.hash, comp.centroid, comp.depth, comp.atom_count};
        token_cache[comp.hash] = new_token;
        
        // Replace all occurrences of this pair in O(n)
        std::vector<Token> next;
        next.reserve(current.size());
        
        size_t i = 0;
        while (i < current.size()) {
            if (i + 1 < current.size() &&
                current[i].hash == best_pair.first &&
                current[i+1].hash == best_pair.second) {
                next.push_back(new_token);
                i += 2;
            } else {
                next.push_back(current[i]);
                i += 1;
            }
        }
        
        current = std::move(next);
    }
    
    // Continue pairing until only one token remains (proper binary hierarchy)
    // This is the MERGE phase: no more repeated pairs, so pair left-to-right
    while (current.size() > 1) {
        std::vector<Token> next;
        next.reserve((current.size() + 1) / 2);
        
        size_t i = 0;
        while (i < current.size()) {
            if (i + 1 < current.size()) {
                // Pair adjacent tokens
                const Token& t1 = current[i];
                const Token& t2 = current[i + 1];
                
                std::vector<AtomCalculator::ChildInfo> pair_children = {
                    {t1.hash, t1.coords, t1.depth, t1.atom_count},
                    {t2.hash, t2.coords, t2.depth, t2.atom_count}
                };
                auto comp = AtomCalculator::compute_composition(pair_children);
                
                // Store composition
                result.compositions.push_back(convert_record(comp));
                
                // Create token for new composition
                Token new_token{comp.hash, comp.centroid, comp.depth, comp.atom_count};
                token_cache[comp.hash] = new_token;
                
                next.push_back(new_token);
                i += 2;
            } else {
                // Odd token, carry to next level
                next.push_back(current[i]);
                i += 1;
            }
        }
        
        current = std::move(next);
    }
    
    // Now we have exactly one root
    if (current.size() == 1) {
        result.root_hash = current[0].hash;
    }
    
    progress_chars += codepoints.size();
    
    return result;
}

// ============================================================================
// Parallel CPE Ingester Implementation
// ============================================================================

struct ParallelCPEIngester::Impl {
    size_t num_threads;
    std::mutex comp_mutex;
    std::unordered_map<Blake3Hash, CompositionRecord, Blake3HashHasher> all_compositions;
    
    Impl(size_t threads) : num_threads(threads) {}
};

ParallelCPEIngester::ParallelCPEIngester(size_t num_threads)
    : impl_(std::make_unique<Impl>(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads))
{}

ParallelCPEIngester::~ParallelCPEIngester() = default;

std::vector<Blake3Hash> ParallelCPEIngester::ingest(
    const std::string& text,
    std::vector<CompositionRecord>& new_compositions
) {
    // Decode UTF-8 to codepoints
    auto all_codepoints = AtomCalculator::decode_utf8(text);
    
    if (all_codepoints.empty()) return {};
    
    size_t total_chars = all_codepoints.size();
    std::cerr << "[CPE] " << total_chars << " codepoints, " 
              << impl_->num_threads << " threads\n";
    
    // Split into chunks (by newlines for natural boundaries)
    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(10000);
    
    for (uint32_t cp : all_codepoints) {
        current_chunk.push_back(cp);
        if (cp == '\n' && current_chunk.size() >= 100) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = std::vector<uint32_t>();
            current_chunk.reserve(10000);
        }
    }
    if (!current_chunk.empty()) {
        chunks.push_back(std::move(current_chunk));
    }
    
    std::cerr << "[CPE] Split into " << chunks.size() << " chunks\n";
    
    // Progress tracking
    std::atomic<size_t> progress_chars{0};
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process chunks in parallel
    std::vector<std::future<ChunkResult>> futures;
    futures.reserve(chunks.size());

    for (auto& chunk : chunks) {
        futures.push_back(std::async(std::launch::async, [&chunk, &progress_chars]() {
            return process_chunk(chunk, progress_chars);
        }));
    }

    // Progress display thread
    std::atomic<bool> done{false};
    std::thread progress_thread([&]() {
        while (!done) {
            size_t chars = progress_chars.load();
            double pct = 100.0 * chars / total_chars;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            double rate = chars / (elapsed + 0.001);

            std::cerr << "\r[CPE] " << std::fixed << std::setprecision(1) << pct << "% "
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

        // Merge compositions (deduplicate by hash)
        std::lock_guard<std::mutex> lock(impl_->comp_mutex);
        for (auto& comp : result.compositions) {
            impl_->all_compositions[comp.hash] = std::move(comp);
        }
    }

    done = true;
    progress_thread.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cerr << "\r[CPE] 100% - " << impl_->all_compositions.size() 
              << " unique compositions in " << std::fixed << std::setprecision(2) 
              << elapsed << "s\n";
    
    // Output compositions
    new_compositions.reserve(impl_->all_compositions.size());
    for (auto& [hash, comp] : impl_->all_compositions) {
        new_compositions.push_back(std::move(comp));
    }
    
    return root_hashes;
}

size_t ParallelCPEIngester::composition_count() const {
    return impl_->all_compositions.size();
}

void ParallelCPEIngester::clear() {
    impl_->all_compositions.clear();
}

} // namespace hypercube::ingest
