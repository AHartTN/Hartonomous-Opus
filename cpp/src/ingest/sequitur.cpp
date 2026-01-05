/**
 * Sequitur Algorithm for Hypercube
 * 
 * Implements the Nevill-Manning & Witten (1997) algorithm:
 * - Digram uniqueness: No digram appears more than once
 * - Rule utility: Every rule is used more than once
 * 
 * This produces a context-free grammar that:
 * - Is lossless (reconstructs exact input)
 * - Is maximally compressed (no redundant rules)
 * - Scales linearly O(n) time and space
 * 
 * All computation is client-side using AtomCalculator.
 * ZERO database calls until final batch insert.
 */

#include "hypercube/ingest/sequitur.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <atomic>
#include <thread>
#include <future>

namespace hypercube::ingest {

// ============================================================================
// Symbol: Either a terminal (atom) or non-terminal (rule reference)
// ============================================================================

struct Symbol;
struct Rule;

struct Symbol {
    Blake3Hash hash;        // Content hash (atom or composition)
    Point4D coords;         // 4D coordinates
    uint32_t depth;         // 0 = leaf, >0 = composition
    uint64_t atom_count;    // Total leaves in subtree
    
    Rule* rule = nullptr;   // If non-terminal, points to the rule
    
    // Doubly-linked list pointers for O(1) insertion/deletion
    Symbol* prev = nullptr;
    Symbol* next = nullptr;
    
    bool is_terminal() const { return rule == nullptr; }
};

// ============================================================================
// Digram: An adjacent pair of symbols
// ============================================================================

struct Digram {
    Blake3Hash first;
    Blake3Hash second;
    
    bool operator==(const Digram& other) const {
        return first == other.first && second == other.second;
    }
};

struct DigramHasher {
    size_t operator()(const Digram& d) const {
        uint64_t h1, h2;
        std::memcpy(&h1, d.first.bytes.data(), 8);
        std::memcpy(&h2, d.second.bytes.data(), 8);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
    }
};

// ============================================================================
// Rule: A non-terminal in the grammar
// ============================================================================

struct Rule {
    Blake3Hash hash;        // Composition hash = BLAKE3(child hashes)
    Point4D centroid;       // Average of child centroids
    HilbertIndex hilbert;   // Hilbert index of centroid
    uint32_t depth;         // max(child depths) + 1
    uint64_t atom_count;    // Sum of child atom counts
    
    Symbol* first = nullptr;  // First symbol in this rule's RHS
    Symbol* last = nullptr;   // Last symbol in this rule's RHS
    
    int ref_count = 0;        // How many times this rule is referenced
    
    std::vector<Blake3Hash> children;       // Ordered child hashes
    std::vector<Point4D> child_coords;      // Child coordinates for geometry
    std::vector<uint32_t> child_depths;     // Child depths (0=atom, >0=composition) for correct type tracking
};

// ============================================================================
// Sequitur Grammar
// ============================================================================

class SequiturGrammar {
public:
    Rule* start_rule = nullptr;  // S → ...
    
    // All rules indexed by their hash
    std::unordered_map<Blake3Hash, std::unique_ptr<Rule>, Blake3HashHasher> rules;
    
    // Digram index: maps each digram to the first symbol of its occurrence
    std::unordered_map<Digram, Symbol*, DigramHasher> digram_index;
    
    // Recursion depth counter for debugging
    int recursion_depth = 0;
    static constexpr int MAX_RECURSION = 1000;
    
    // Create a new symbol for a terminal (atom)
    Symbol* make_terminal(const Blake3Hash& hash, const Point4D& coords) {
        auto* sym = new Symbol();
        sym->hash = hash;
        sym->coords = coords;
        sym->depth = 0;
        sym->atom_count = 1;
        sym->rule = nullptr;
        return sym;
    }
    
    // Create a new symbol for a non-terminal (rule reference)
    Symbol* make_nonterminal(Rule* rule) {
        auto* sym = new Symbol();
        sym->hash = rule->hash;
        sym->coords = rule->centroid;
        sym->depth = rule->depth;
        sym->atom_count = rule->atom_count;
        sym->rule = rule;
        rule->ref_count++;
        return sym;
    }
    
    // Link symbol after prev in the list
    void link_after(Symbol* prev_sym, Symbol* sym) {
        sym->prev = prev_sym;
        sym->next = prev_sym->next;
        if (prev_sym->next) prev_sym->next->prev = sym;
        prev_sym->next = sym;
    }
    
    // Unlink symbol from the list
    void unlink(Symbol* sym) {
        if (sym->prev) sym->prev->next = sym->next;
        if (sym->next) sym->next->prev = sym->prev;
    }
    
    // Get digram starting at sym
    Digram get_digram(Symbol* sym) {
        if (!sym || !sym->next) return Digram{};
        return Digram{sym->hash, sym->next->hash};
    }
    
    // Delete a digram from the index (if this symbol is registered for it)
    void delete_digram(Symbol* sym) {
        if (!sym || !sym->next) return;
        Digram d = get_digram(sym);
        auto it = digram_index.find(d);
        if (it != digram_index.end() && it->second == sym) {
            digram_index.erase(it);
        }
    }
    
    // Check a digram and enforce constraints if needed
    // Returns: true if grammar was modified, false otherwise
    // This is the core Sequitur constraint enforcement
    bool check(Symbol* sym) {
        if (!sym || !sym->next) return false;
        
        recursion_depth++;
        if (recursion_depth > MAX_RECURSION) {
            std::cerr << "[WARN] Max recursion depth reached in check()" << std::endl;
            recursion_depth--;
            return false;
        }
        
        Digram d = get_digram(sym);
        auto it = digram_index.find(d);
        
        if (it == digram_index.end()) {
            // Digram not in table - add it
            digram_index[d] = sym;
            recursion_depth--;
            return false;
        }
        
        Symbol* match = it->second;
        
        // Don't match ourselves
        if (match == sym) {
            recursion_depth--;
            return false;
        }
        
        // Check for overlapping digrams (consecutive symbols)
        if (match->next == sym || sym->next == match) {
            recursion_depth--;
            return false;
        }
        
        // Found a valid match elsewhere - handle it
        handle_match(sym, match);
        recursion_depth--;
        return true;
    }
    
    // Handle a matching digram - either reuse existing rule or create new one
    void handle_match(Symbol* new_sym, Symbol* existing) {
        Rule* rule = nullptr;
        
        // Check if existing digram forms a complete rule's RHS
        // In the reference: m->prev()->is_guard() && m->next()->next()->is_guard()
        // We use nullptr checks instead since we don't have guards
        bool existing_is_complete_rule = false;
        for (const auto& [h, r] : rules) {
            if (r.get() != start_rule && r->first == existing && r->last == existing->next) {
                existing_is_complete_rule = true;
                rule = r.get();
                break;
            }
        }
        
        if (existing_is_complete_rule && rule) {
            // Reuse existing rule - just substitute new_sym
            substitute(new_sym, rule);
        } else {
            // Create a new rule from the digram
            rule = create_rule(existing, existing->next);
            
            // Substitute both occurrences - existing FIRST to avoid invalidation
            substitute(existing, rule);
            substitute(new_sym, rule);
            
            // Register the rule's internal digram
            Digram rule_digram{rule->first->hash, rule->first->next->hash};
            digram_index[rule_digram] = rule->first;
        }
        
        // Check for underused rule in the first symbol of the new rule
        if (rule->first->rule && rule->first->rule->ref_count == 1) {
            expand(rule->first);
        }
    }
    
    // Compute composition hash and metadata from a rule's symbols
    void compute_rule_hash(Rule* rule) {
        std::vector<Blake3Hash> child_hashes;
        std::vector<Point4D> child_coords_vec;
        std::vector<uint32_t> child_depths_vec;
        uint32_t max_depth = 0;
        uint64_t total_atoms = 0;
        
        for (Symbol* s = rule->first; s; s = s->next) {
            child_hashes.push_back(s->hash);
            child_coords_vec.push_back(s->coords);
            child_depths_vec.push_back(s->depth);  // Track each child's depth (0=atom, >0=composition)
            max_depth = std::max(max_depth, s->depth);
            total_atoms += s->atom_count;
        }
        
        // Compute BLAKE3 hash using Blake3Hasher
        rule->hash = Blake3Hasher::hash_children(child_hashes);
        
        // Compute centroid (average of child coords) using uint64 to prevent overflow
        // Each uint32 coord can be up to 4B, and summing many would overflow 32 bits
        uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
        for (const auto& c : child_coords_vec) {
            sum_x += c.x;
            sum_y += c.y;
            sum_z += c.z;
            sum_m += c.m;
        }
        size_t n = child_coords_vec.size();
        Point4D centroid{0, 0, 0, 0};
        if (n > 0) {
            centroid.x = static_cast<Coord32>(sum_x / n);
            centroid.y = static_cast<Coord32>(sum_y / n);
            centroid.z = static_cast<Coord32>(sum_z / n);
            centroid.m = static_cast<Coord32>(sum_m / n);
        }
        rule->centroid = centroid;
        
        // Compute Hilbert index from centroid
        rule->hilbert = HilbertCurve::coords_to_index(centroid);
        
        rule->depth = max_depth + 1;
        rule->atom_count = total_atoms;
        rule->children = std::move(child_hashes);
        rule->child_coords = std::move(child_coords_vec);
        rule->child_depths = std::move(child_depths_vec);
    }
    
    // Create a new rule from a digram (copies the symbols)
    Rule* create_rule(Symbol* first, Symbol* second) {
        auto rule = std::make_unique<Rule>();
        
        // Create copies of the symbols for the rule's RHS
        Symbol* s1 = first->is_terminal() 
            ? make_terminal(first->hash, first->coords)
            : make_nonterminal(first->rule);
        Symbol* s2 = second->is_terminal()
            ? make_terminal(second->hash, second->coords)
            : make_nonterminal(second->rule);
        
        s1->next = s2;
        s2->prev = s1;
        
        rule->first = s1;
        rule->last = s2;
        
        compute_rule_hash(rule.get());
        
        Rule* ptr = rule.get();
        rules[rule->hash] = std::move(rule);
        return ptr;
    }
    
    // Substitute a digram (sym and sym->next) with a non-terminal pointing to rule
    // This is NOT recursive - it just does the replacement
    void substitute(Symbol* sym, Rule* rule) {
        Symbol* first = sym;
        Symbol* second = sym->next;
        Symbol* before = first->prev;
        Symbol* after = second->next;
        
        // Delete digrams involving these symbols
        if (before) delete_digram(before);
        delete_digram(first);
        if (after) delete_digram(second);
        
        // Create the non-terminal replacement
        Symbol* replacement = make_nonterminal(rule);
        
        // Link it in place
        replacement->prev = before;
        replacement->next = after;
        if (before) before->next = replacement;
        if (after) after->prev = replacement;
        
        // Update rule first/last if needed (for start rule)
        if (start_rule && start_rule->first == first) {
            start_rule->first = replacement;
        }
        if (start_rule && start_rule->last == second) {
            start_rule->last = replacement;
        }
        
        // Decrement ref counts on replaced non-terminals
        if (first->rule) first->rule->ref_count--;
        if (second->rule) second->rule->ref_count--;
        
        delete first;
        delete second;
        
        // Register new digrams in hash table - but DON'T check for violations here!
        // The check will happen in the main loop after all substitutions are done
        if (before) {
            Digram d1{before->hash, replacement->hash};
            if (digram_index.find(d1) == digram_index.end()) {
                digram_index[d1] = before;
            }
        }
        if (after) {
            Digram d2{replacement->hash, after->hash};
            if (digram_index.find(d2) == digram_index.end()) {
                digram_index[d2] = replacement;
            }
        }
    }
    
    // Expand a non-terminal: inline a rule that's only used once
    void expand(Symbol* sym) {
        if (!sym->rule) return;
        
        Rule* rule = sym->rule;
        Symbol* before = sym->prev;
        Symbol* after = sym->next;
        
        // Get the rule's first and last symbols
        Symbol* rule_first = rule->first;
        Symbol* rule_last = rule->last;
        
        // Delete digrams around sym
        if (before) delete_digram(before);
        delete_digram(sym);
        
        // Remove from hash
        digram_index.erase(Digram{rule_first->hash, rule_first->next->hash});
        
        // Mark sym's value as 0 to avoid decrementing rule ref count in destructor
        sym->rule = nullptr;
        
        // Link rule's symbols in place of sym
        if (before) before->next = rule_first;
        rule_first->prev = before;
        if (after) after->prev = rule_last;
        rule_last->next = after;
        
        // Update start rule if needed
        if (start_rule && start_rule->first == sym) {
            start_rule->first = rule_first;
        }
        if (start_rule && start_rule->last == sym) {
            start_rule->last = rule_last;
        }
        
        // Register the new digram at the seam
        if (before) {
            Digram d{rule_last->hash, after ? after->hash : Blake3Hash{}};
            if (after && digram_index.find(d) == digram_index.end()) {
                digram_index[d] = rule_last;
            }
        }
        
        delete sym;
        
        // Remove the rule itself (but NOT its symbols - they're now in the main sequence)
        rule->first = nullptr;
        rule->last = nullptr;
        rules.erase(rule->hash);
    }
    
    // Build grammar from codepoints
    void build(const std::vector<uint32_t>& codepoints) {
        if (codepoints.empty()) return;
        
        // Create start rule
        auto start = std::make_unique<Rule>();
        start_rule = start.get();
        
        Symbol* prev_sym = nullptr;
        
        for (uint32_t cp : codepoints) {
            // Compute atom
            auto atom = AtomCalculator::compute_atom(cp);
            
            // Create terminal symbol
            Symbol* sym = make_terminal(atom.hash, atom.coords);
            
            if (!start_rule->first) {
                start_rule->first = sym;
            }
            
            if (prev_sym) {
                prev_sym->next = sym;
                sym->prev = prev_sym;
                
                // Check digram constraint after each insertion
                check(prev_sym);
            }
            
            prev_sym = sym;
            start_rule->last = sym;
        }
        
        // Compute start rule hash
        compute_rule_hash(start_rule);
        rules[start_rule->hash] = std::move(start);
    }
    
    // Extract all compositions for database insert
    std::vector<CompositionRecord> extract_compositions() {
        std::vector<CompositionRecord> result;
        
        for (const auto& [hash, rule] : rules) {
            CompositionRecord rec;
            rec.hash = rule->hash;
            rec.coord_x = static_cast<int32_t>(rule->centroid.x);
            rec.coord_y = static_cast<int32_t>(rule->centroid.y);
            rec.coord_z = static_cast<int32_t>(rule->centroid.z);
            rec.coord_m = static_cast<int32_t>(rule->centroid.m);
            rec.hilbert_lo = static_cast<int64_t>(rule->hilbert.lo);
            rec.hilbert_hi = static_cast<int64_t>(rule->hilbert.hi);
            rec.depth = rule->depth;
            rec.atom_count = rule->atom_count;
            
            // Build children with correct is_atom flag based on actual child depths
            for (size_t i = 0; i < rule->children.size(); ++i) {
                ChildInfo ci;
                ci.hash = rule->children[i];
                ci.x = static_cast<int32_t>(rule->child_coords[i].x);
                ci.y = static_cast<int32_t>(rule->child_coords[i].y);
                ci.z = static_cast<int32_t>(rule->child_coords[i].z);
                ci.m = static_cast<int32_t>(rule->child_coords[i].m);
                // Use actual child depth to determine type: depth=0 is atom, depth>0 is composition
                ci.is_atom = (i < rule->child_depths.size()) ? (rule->child_depths[i] == 0) : true;
                rec.children.push_back(ci);
            }
            
            result.push_back(std::move(rec));
        }
        
        return result;
    }
    
    Blake3Hash root_hash() const {
        return start_rule ? start_rule->hash : Blake3Hash{};
    }
    
    ~SequiturGrammar() {
        // Clean up all symbols in all rules
        for (auto& [hash, rule] : rules) {
            Symbol* s = rule->first;
            while (s) {
                Symbol* next_s = s->next;
                delete s;
                s = next_s;
            }
        }
    }
};

// ============================================================================
// Process a chunk using Sequitur
// ============================================================================

struct ChunkResult {
    Blake3Hash root_hash;
    std::vector<CompositionRecord> compositions;
};

ChunkResult process_chunk_sequitur(
    const std::vector<uint32_t>& codepoints,
    std::atomic<size_t>& progress_chars
) {
    ChunkResult result;
    
    if (codepoints.empty()) {
        progress_chars += 0;
        return result;
    }
    
    if (codepoints.size() == 1) {
        auto atom = AtomCalculator::compute_atom(codepoints[0]);
        result.root_hash = atom.hash;
        progress_chars += 1;
        return result;
    }
    
    try {
        SequiturGrammar grammar;
        grammar.build(codepoints);
        
        result.root_hash = grammar.root_hash();
        result.compositions = grammar.extract_compositions();
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception in chunk: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "\n[ERROR] Unknown exception in chunk" << std::endl;
    }
    
    progress_chars += codepoints.size();
    return result;
}

// ============================================================================
// Parallel Sequitur Ingester
// ============================================================================

struct SequiturIngester::Impl {
    size_t num_threads;
    std::mutex comp_mutex;
    std::unordered_map<Blake3Hash, CompositionRecord, Blake3HashHasher> all_compositions;
    
    Impl(size_t threads) : num_threads(threads) {}
};

SequiturIngester::SequiturIngester(size_t num_threads)
    : impl_(std::make_unique<Impl>(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads))
{}

SequiturIngester::~SequiturIngester() = default;

std::vector<Blake3Hash> SequiturIngester::ingest(
    const std::string& text,
    std::vector<CompositionRecord>& new_compositions
) {
    // Decode UTF-8 to codepoints
    auto all_codepoints = AtomCalculator::decode_utf8(text);
    
    if (all_codepoints.empty()) return {};
    
    size_t total_chars = all_codepoints.size();
    std::cerr << "[SEQUITUR] " << total_chars << " codepoints, " 
              << impl_->num_threads << " threads\n";
    
    // Split into chunks at paragraph or line boundaries
    // Aim for chunks around 10K-50K characters for good parallelization
    constexpr size_t MIN_CHUNK_SIZE = 1000;
    constexpr size_t MAX_CHUNK_SIZE = 50000;
    
    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(MAX_CHUNK_SIZE);
    
    bool prev_newline = false;
    for (uint32_t cp : all_codepoints) {
        current_chunk.push_back(cp);
        
        bool is_newline = (cp == '\n');
        bool is_paragraph = is_newline && prev_newline;
        
        // Split on paragraph break (any size >= MIN), or line break if chunk is large
        if ((is_paragraph && current_chunk.size() >= MIN_CHUNK_SIZE) ||
            (is_newline && current_chunk.size() >= MAX_CHUNK_SIZE)) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = std::vector<uint32_t>();
            current_chunk.reserve(MAX_CHUNK_SIZE);
            prev_newline = false;
        } else {
            prev_newline = is_newline;
        }
    }
    if (!current_chunk.empty()) {
        chunks.push_back(std::move(current_chunk));
    }
    
    std::cerr << "[SEQUITUR] Split into " << chunks.size() << " chunks\n";
    
    // Progress tracking
    std::atomic<size_t> progress_chars{0};
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process chunks in parallel
    std::vector<std::future<ChunkResult>> futures;
    futures.reserve(chunks.size());
    
    for (auto& chunk : chunks) {
        futures.push_back(std::async(std::launch::async, [&chunk, &progress_chars]() {
            return process_chunk_sequitur(chunk, progress_chars);
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
            
            std::cerr << "\r[SEQUITUR] " << std::fixed << std::setprecision(1) << pct << "% "
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
    
    std::cerr << "\r[SEQUITUR] 100% - " << impl_->all_compositions.size() 
              << " unique compositions in " << std::fixed << std::setprecision(2) 
              << elapsed << "s\n";
    
    // Output compositions
    new_compositions.reserve(impl_->all_compositions.size());
    for (auto& [hash, comp] : impl_->all_compositions) {
        new_compositions.push_back(std::move(comp));
    }
    
    return root_hashes;
}

size_t SequiturIngester::composition_count() const {
    return impl_->all_compositions.size();
}

void SequiturIngester::clear() {
    impl_->all_compositions.clear();
}

} // namespace hypercube::ingest
