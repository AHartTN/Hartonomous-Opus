/**
 * Sequitur Grammar-Based Ingester
 * 
 * Replaces binary CPE with Sequitur algorithm for proper semantic composition.
 * 
 * Key differences from CPE:
 * - Variable-length children (not just binary pairs)
 * - Discovers natural boundaries (words, phrases)
 * - Same input always produces same grammar (deterministic)
 * - Every rule used at least twice (digram uniqueness + rule utility)
 * 
 * Algorithm:
 * 1. Scan input symbols, append to start rule
 * 2. Check each new digram:
 *    - If digram exists elsewhere, create rule for it
 *    - Replace all instances with new non-terminal
 * 3. Rule utility: if rule used only once, inline it
 * 
 * Output: Grammar where each rule = composition in our DAG
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <memory>
#include <filesystem>
#include <chrono>
#include <cstring>

#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;
namespace fs = std::filesystem;

// =============================================================================
// Sequitur Data Structures
// =============================================================================

// Forward declarations
struct Symbol;
struct Rule;
struct Digram;

// Symbol: either terminal (atom codepoint) or non-terminal (rule reference)
struct Symbol {
    bool is_terminal;
    union {
        uint32_t codepoint;     // Terminal: Unicode codepoint
        Rule* rule;             // Non-terminal: pointer to rule
    };
    
    // Doubly-linked list for rule body
    Symbol* prev = nullptr;
    Symbol* next = nullptr;
    
    // Which rule contains this symbol
    Rule* container = nullptr;
    
    Symbol(uint32_t cp) : is_terminal(true), codepoint(cp) {}
    Symbol(Rule* r) : is_terminal(false), rule(r) {}
    
    // Get unique ID for digram hashing
    uint64_t id() const {
        if (is_terminal) {
            return static_cast<uint64_t>(codepoint);
        } else {
            // Rules get IDs above codepoint range
            return 0x200000ULL + reinterpret_cast<uint64_t>(rule);
        }
    }
    
    // Check if this symbol equals another
    bool equals(const Symbol* other) const {
        if (is_terminal != other->is_terminal) return false;
        if (is_terminal) return codepoint == other->codepoint;
        return rule == other->rule;
    }
};

// Digram: pair of adjacent symbols
struct Digram {
    uint64_t first_id;
    uint64_t second_id;
    
    bool operator==(const Digram& other) const {
        return first_id == other.first_id && second_id == other.second_id;
    }
};

struct DigramHash {
    size_t operator()(const Digram& d) const {
        return std::hash<uint64_t>()(d.first_id) ^ 
               (std::hash<uint64_t>()(d.second_id) << 1);
    }
};

// Rule: sequence of symbols
struct Rule {
    uint32_t id;                        // Unique rule ID
    Symbol* first = nullptr;            // Head of symbol list
    Symbol* last = nullptr;             // Tail of symbol list
    int use_count = 0;                  // Number of times this rule is referenced
    int length = 0;                     // Number of symbols in body
    
    // Our composition data (computed after grammar is stable)
    Blake3Hash hash;
    int32_t centroid_x = 0, centroid_y = 0, centroid_z = 0, centroid_m = 0;
    int64_t hilbert_lo = 0, hilbert_hi = 0;
    uint32_t depth = 0;
    uint64_t atom_count = 0;
    bool computed = false;
    
    Rule(uint32_t id_) : id(id_) {}
    
    // Append symbol to end
    void append(Symbol* s) {
        s->container = this;
        s->prev = last;
        s->next = nullptr;
        if (last) {
            last->next = s;
        } else {
            first = s;
        }
        last = s;
        length++;
    }
    
    // Insert symbol after position
    void insert_after(Symbol* pos, Symbol* s) {
        s->container = this;
        s->prev = pos;
        s->next = pos->next;
        if (pos->next) {
            pos->next->prev = s;
        } else {
            last = s;
        }
        pos->next = s;
        length++;
    }
    
    // Remove symbol
    void remove(Symbol* s) {
        if (s->prev) {
            s->prev->next = s->next;
        } else {
            first = s->next;
        }
        if (s->next) {
            s->next->prev = s->prev;
        } else {
            last = s->prev;
        }
        s->container = nullptr;
        length--;
    }
};

// =============================================================================
// Sequitur Algorithm Implementation
// =============================================================================

class Sequitur {
public:
    Sequitur() : next_rule_id_(1) {
        // Rule 0 is the start rule (S)
        start_rule_ = new Rule(0);
        rules_.push_back(start_rule_);
    }
    
    ~Sequitur() {
        // Clean up all symbols and rules
        for (Rule* r : rules_) {
            Symbol* s = r->first;
            while (s) {
                Symbol* next = s->next;
                delete s;
                s = next;
            }
            delete r;
        }
    }
    
    // Process a sequence of codepoints
    void process(const std::vector<uint32_t>& codepoints) {
        for (uint32_t cp : codepoints) {
            // Create terminal symbol
            Symbol* s = new Symbol(cp);
            start_rule_->append(s);
            
            // Check digram ending at this symbol
            if (s->prev) {
                check_digram(s->prev);
            }
        }
    }
    
    // Get all rules (for export to database)
    const std::vector<Rule*>& rules() const { return rules_; }
    
    Rule* start_rule() const { return start_rule_; }
    
    // Statistics
    size_t num_rules() const { return rules_.size(); }
    size_t num_symbols() const {
        size_t count = 0;
        for (Rule* r : rules_) {
            count += r->length;
        }
        return count;
    }
    
private:
    std::vector<Rule*> rules_;
    Rule* start_rule_;
    uint32_t next_rule_id_;
    
    // Map from digram to the first symbol of that digram
    std::unordered_map<Digram, Symbol*, DigramHash> digram_index_;
    
    // Make digram from two adjacent symbols
    Digram make_digram(Symbol* first) {
        return Digram{first->id(), first->next->id()};
    }
    
    // Check if digram at position triggers a rule
    void check_digram(Symbol* first) {
        if (!first->next) return;  // No digram
        
        Digram d = make_digram(first);
        auto it = digram_index_.find(d);
        
        if (it == digram_index_.end()) {
            // First occurrence - just index it
            digram_index_[d] = first;
        } else if (it->second != first && it->second->next != first) {
            // Digram exists elsewhere (and not overlapping)
            Symbol* existing = it->second;
            
            // Check if existing is the entire body of a 2-symbol rule
            if (existing->container != start_rule_ && 
                existing->container->length == 2 &&
                existing == existing->container->first) {
                // Reuse existing rule
                Rule* r = existing->container;
                match_digram(first, r);
            } else {
                // Create new rule for this digram
                Rule* r = new Rule(next_rule_id_++);
                rules_.push_back(r);
                
                // Copy the digram content to new rule
                Symbol* s1;
                Symbol* s2;
                if (existing->is_terminal) {
                    s1 = new Symbol(existing->codepoint);
                } else {
                    s1 = new Symbol(existing->rule);
                    existing->rule->use_count++;
                }
                if (existing->next->is_terminal) {
                    s2 = new Symbol(existing->next->codepoint);
                } else {
                    s2 = new Symbol(existing->next->rule);
                    existing->next->rule->use_count++;
                }
                r->append(s1);
                r->append(s2);
                
                // Replace both occurrences with rule reference
                substitute_digram(existing, r);
                substitute_digram(first, r);
                
                // Index the new rule's digram
                digram_index_[d] = s1;
            }
        }
    }
    
    // Replace digram with rule reference
    void substitute_digram(Symbol* first, Rule* r) {
        // Remove digram from index
        if (first->next) {
            Digram d = make_digram(first);
            auto it = digram_index_.find(d);
            if (it != digram_index_.end() && it->second == first) {
                digram_index_.erase(it);
            }
        }
        
        // Also remove adjacent digrams that will be affected
        if (first->prev && first->prev->next) {
            Digram d_prev = make_digram(first->prev);
            auto it = digram_index_.find(d_prev);
            if (it != digram_index_.end() && it->second == first->prev) {
                digram_index_.erase(it);
            }
        }
        if (first->next && first->next->next) {
            Digram d_next = make_digram(first->next);
            auto it = digram_index_.find(d_next);
            if (it != digram_index_.end() && it->second == first->next) {
                digram_index_.erase(it);
            }
        }
        
        Rule* container = first->container;
        Symbol* prev = first->prev;
        Symbol* second = first->next;  // Save BEFORE any removal modifies pointers
        Symbol* after_second = second->next;

        // Decrement use counts if non-terminals
        if (!first->is_terminal) {
            first->rule->use_count--;
            check_rule_utility(first->rule);
        }
        if (!second->is_terminal) {
            second->rule->use_count--;
            check_rule_utility(second->rule);
        }

        // Remove both symbols from container
        // NOTE: remove() updates neighbor pointers, so we saved 'second' above
        container->remove(second);
        container->remove(first);

        // Now safe to delete - we have independent pointers
        delete second;
        delete first;
        
        // Insert rule reference
        Symbol* ref = new Symbol(r);
        r->use_count++;
        
        if (prev) {
            container->insert_after(prev, ref);
        } else {
            // Insert at beginning
            ref->container = container;
            ref->next = container->first;
            ref->prev = nullptr;
            if (container->first) {
                container->first->prev = ref;
            }
            container->first = ref;
            if (!container->last) {
                container->last = ref;
            }
            container->length++;
        }
        
        // Check new digrams
        if (ref->prev) {
            check_digram(ref->prev);
        }
        if (ref->next) {
            check_digram(ref);
        }
    }
    
    // Match existing digram with a rule (reuse case)
    void match_digram(Symbol* first, Rule* r) {
        substitute_digram(first, r);
    }
    
    // Check if rule should be inlined (used only once)
    void check_rule_utility(Rule* r) {
        if (r == start_rule_) return;  // Never inline start rule
        if (r->use_count != 1) return;  // Still used multiple times
        
        // Find the single reference to this rule
        for (Rule* container : rules_) {
            for (Symbol* s = container->first; s; s = s->next) {
                if (!s->is_terminal && s->rule == r) {
                    // Found it - inline the rule
                    inline_rule(s, r);
                    return;
                }
            }
        }
    }
    
    // Inline a rule at a reference position
    void inline_rule(Symbol* ref, Rule* r) {
        Rule* container = ref->container;
        Symbol* prev = ref->prev;
        
        // Remove digrams around ref
        if (ref->prev) {
            Digram d = make_digram(ref->prev);
            digram_index_.erase(d);
        }
        if (ref->next) {
            Digram d = make_digram(ref);
            digram_index_.erase(d);
        }
        
        // Remove reference
        container->remove(ref);
        delete ref;
        
        // Copy rule body to container
        Symbol* insert_after_pos = prev;
        for (Symbol* s = r->first; s; s = s->next) {
            Symbol* copy;
            if (s->is_terminal) {
                copy = new Symbol(s->codepoint);
            } else {
                copy = new Symbol(s->rule);
                s->rule->use_count++;
            }
            
            if (insert_after_pos) {
                container->insert_after(insert_after_pos, copy);
            } else {
                // Insert at beginning
                copy->container = container;
                copy->next = container->first;
                copy->prev = nullptr;
                if (container->first) {
                    container->first->prev = copy;
                }
                container->first = copy;
                if (!container->last) {
                    container->last = copy;
                }
                container->length++;
            }
            insert_after_pos = copy;
        }
        
        // Decrement use counts in inlined rule
        for (Symbol* s = r->first; s; s = s->next) {
            if (!s->is_terminal) {
                s->rule->use_count--;
            }
        }
        
        // Remove the rule (don't delete yet, might still need it)
        // Actually we should mark it as dead
        r->use_count = 0;
    }
};

// =============================================================================
// Atom Cache (same as CPE)
// =============================================================================

struct AtomInfo {
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
};

static std::unordered_map<uint32_t, AtomInfo> g_atom_cache;

bool load_atom_cache(PGconn* conn) {
    auto start = std::chrono::high_resolution_clock::now();

    PGresult* res = PQexec(conn,
        "SELECT codepoint, id, ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom) "
        "FROM atom WHERE depth = 0 AND codepoint IS NOT NULL");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to load atoms: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = PQntuples(res);
    if (rows == 0) {
        std::cerr << "ERROR: No atoms found. Run setup.sh init first.\n";
        PQclear(res);
        return false;
    }

    g_atom_cache.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        uint32_t cp = static_cast<uint32_t>(std::stoul(PQgetvalue(res, i, 0)));

        AtomInfo info;
        const char* hex = PQgetvalue(res, i, 1);
        if (hex[0] == '\\' && hex[1] == 'x') {
            info.hash = Blake3Hash::from_hex(std::string_view(hex + 2, 64));
        }
        // Raw uint32 values stored as double (no normalization)
        info.coord_x = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 2))));
        info.coord_y = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 3))));
        info.coord_z = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 4))));
        info.coord_m = static_cast<int32_t>(static_cast<uint32_t>(std::stod(PQgetvalue(res, i, 5))));

        g_atom_cache[cp] = info;
    }

    PQclear(res);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[CACHE] Loaded " << g_atom_cache.size() << " atoms in " << ms << " ms\n";

    return true;
}

// =============================================================================
// Compute Composition Data from Sequitur Grammar
// =============================================================================

inline uint32_t int32_to_uint32(int32_t val) {
    return static_cast<uint32_t>(val);
}

inline int32_t uint32_to_int32(uint32_t val) {
    return static_cast<int32_t>(val);
}

inline int64_t uint64_to_int64(uint64_t val) {
    return static_cast<int64_t>(val);
}

// Compute hash, centroid, hilbert for a rule (bottom-up)
void compute_rule_data(Rule* r, std::vector<Rule*>& all_rules) {
    if (r->computed) return;
    
    // First compute all children
    std::vector<Blake3Hash> child_hashes;
    std::vector<int32_t> child_coords_x, child_coords_y, child_coords_z, child_coords_m;
    uint32_t max_depth = 0;
    uint64_t total_atoms = 0;
    
    for (Symbol* s = r->first; s; s = s->next) {
        if (s->is_terminal) {
            // Terminal: use atom data
            auto it = g_atom_cache.find(s->codepoint);
            if (it != g_atom_cache.end()) {
                child_hashes.push_back(it->second.hash);
                child_coords_x.push_back(it->second.coord_x);
                child_coords_y.push_back(it->second.coord_y);
                child_coords_z.push_back(it->second.coord_z);
                child_coords_m.push_back(it->second.coord_m);
                total_atoms++;
            }
        } else {
            // Non-terminal: recursively compute
            compute_rule_data(s->rule, all_rules);
            child_hashes.push_back(s->rule->hash);
            child_coords_x.push_back(s->rule->centroid_x);
            child_coords_y.push_back(s->rule->centroid_y);
            child_coords_z.push_back(s->rule->centroid_z);
            child_coords_m.push_back(s->rule->centroid_m);
            max_depth = std::max(max_depth, s->rule->depth);
            total_atoms += s->rule->atom_count;
        }
    }
    
    if (child_hashes.empty()) {
        r->computed = true;
        return;
    }
    
    // Compute hash: BLAKE3(ordinal||hash||ordinal||hash||...)
    std::vector<uint8_t> hash_input;
    hash_input.reserve(child_hashes.size() * 36);  // 4 + 32 per child
    
    for (size_t i = 0; i < child_hashes.size(); ++i) {
        uint32_t ordinal = static_cast<uint32_t>(i);
        hash_input.insert(hash_input.end(),
            reinterpret_cast<uint8_t*>(&ordinal),
            reinterpret_cast<uint8_t*>(&ordinal) + 4);
        hash_input.insert(hash_input.end(),
            child_hashes[i].bytes.begin(),
            child_hashes[i].bytes.end());
    }
    
    r->hash = Blake3Hasher::hash(std::span<const uint8_t>(hash_input));
    
    // Compute centroid (average of child coordinates)
    uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    for (size_t i = 0; i < child_coords_x.size(); ++i) {
        sum_x += int32_to_uint32(child_coords_x[i]);
        sum_y += int32_to_uint32(child_coords_y[i]);
        sum_z += int32_to_uint32(child_coords_z[i]);
        sum_m += int32_to_uint32(child_coords_m[i]);
    }
    
    size_t n = child_coords_x.size();
    r->centroid_x = uint32_to_int32(static_cast<uint32_t>(sum_x / n));
    r->centroid_y = uint32_to_int32(static_cast<uint32_t>(sum_y / n));
    r->centroid_z = uint32_to_int32(static_cast<uint32_t>(sum_z / n));
    r->centroid_m = uint32_to_int32(static_cast<uint32_t>(sum_m / n));
    
    // Compute Hilbert index
    Point4D coords(
        int32_to_uint32(r->centroid_x),
        int32_to_uint32(r->centroid_y),
        int32_to_uint32(r->centroid_z),
        int32_to_uint32(r->centroid_m)
    );
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    r->hilbert_lo = uint64_to_int64(hilbert.lo);
    r->hilbert_hi = uint64_to_int64(hilbert.hi);
    
    r->depth = max_depth + 1;
    r->atom_count = total_atoms;
    r->computed = true;
}

// =============================================================================
// Database Insertion
// =============================================================================

// Build EWKB for LINESTRINGZM from list of 4D points
std::string build_linestringzm_ewkb(const std::vector<std::array<int32_t, 4>>& points) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string ewkb;
    
    // Byte order (little-endian) + type (LINESTRINGZM with SRID=0)
    ewkb += "01020000c0";
    
    // Number of points (little-endian 32-bit)
    uint32_t npoints = static_cast<uint32_t>(points.size());
    for (int i = 0; i < 4; ++i) {
        uint8_t byte = (npoints >> (i * 8)) & 0xFF;
        ewkb += hex_chars[byte >> 4];
        ewkb += hex_chars[byte & 0x0F];
    }
    
    // Helper to append double as little-endian hex
    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    // Append each point
    for (const auto& pt : points) {
        append_double(static_cast<double>(int32_to_uint32(pt[0])));
        append_double(static_cast<double>(int32_to_uint32(pt[1])));
        append_double(static_cast<double>(int32_to_uint32(pt[2])));
        append_double(static_cast<double>(int32_to_uint32(pt[3])));
    }
    
    return ewkb;
}

bool insert_rules_to_db(PGconn* conn, const std::vector<Rule*>& rules) {
    // Filter to rules that have data and are used
    std::vector<Rule*> to_insert;
    for (Rule* r : rules) {
        if (r->computed && r->atom_count > 0 && (r->id == 0 || r->use_count > 0)) {
            to_insert.push_back(r);
        }
    }
    
    if (to_insert.empty()) return true;
    
    std::cerr << "[DB] Inserting " << to_insert.size() << " rules as compositions...\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_atom ("
        "  id BYTEA PRIMARY KEY,"
        "  geom GEOMETRY(GEOMETRYZM, 0) NOT NULL,"
        "  children BYTEA[],"
        "  value BYTEA,"
        "  hilbert_lo BIGINT NOT NULL,"
        "  hilbert_hi BIGINT NOT NULL,"
        "  depth INTEGER NOT NULL,"
        "  atom_count BIGINT NOT NULL"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY to temp
    res = PQexec(conn,
        "COPY tmp_atom (id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    char num_buf[32];
    
    for (Rule* r : to_insert) {
        // Collect child data for geometry and children array
        std::vector<std::array<int32_t, 4>> child_points;
        std::vector<Blake3Hash> child_hashes;
        
        for (Symbol* s = r->first; s; s = s->next) {
            if (s->is_terminal) {
                auto it = g_atom_cache.find(s->codepoint);
                if (it != g_atom_cache.end()) {
                    child_points.push_back({
                        it->second.coord_x,
                        it->second.coord_y,
                        it->second.coord_z,
                        it->second.coord_m
                    });
                    child_hashes.push_back(it->second.hash);
                }
            } else if (s->rule->computed) {
                child_points.push_back({
                    s->rule->centroid_x,
                    s->rule->centroid_y,
                    s->rule->centroid_z,
                    s->rule->centroid_m
                });
                child_hashes.push_back(s->rule->hash);
            }
        }
        
        if (child_points.empty()) continue;
        
        // id (bytea hex)
        batch += "\\\\x";
        batch += r->hash.to_hex();
        batch += '\t';
        
        // geom: LINESTRINGZM from child centroids
        batch += build_linestringzm_ewkb(child_points);
        batch += '\t';
        
        // children: BYTEA array
        batch += "{";
        for (size_t i = 0; i < child_hashes.size(); ++i) {
            if (i > 0) batch += ",";
            batch += "\"\\\\\\\\x";
            batch += child_hashes[i].to_hex();
            batch += "\"";
        }
        batch += "}";
        batch += '\t';
        
        // value: NULL
        batch += "\\N";
        batch += '\t';
        
        // hilbert_lo, hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(r->hilbert_lo));
        batch += num_buf;
        batch += '\t';

        snprintf(num_buf, sizeof(num_buf), "%lld", static_cast<long long>(r->hilbert_hi));
        batch += num_buf;
        batch += '\t';

        // depth
        snprintf(num_buf, sizeof(num_buf), "%u", r->depth);
        batch += num_buf;
        batch += '\t';

        // atom_count
        snprintf(num_buf, sizeof(num_buf), "%llu", static_cast<unsigned long long>(r->atom_count));
        batch += num_buf;
        batch += '\n';
        
        if (batch.size() > (1 << 19)) {
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "COPY data failed\n";
                PQputCopyEnd(conn, "error");
                return false;
            }
            batch.clear();
        }
    }
    
    if (!batch.empty()) {
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "COPY final failed\n";
            PQputCopyEnd(conn, "error");
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "COPY end failed: " << PQerrorMessage(conn) << std::endl;
        return false;
    }
    
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "COPY result: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // Upsert to atom table
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT id, geom, children, value, hilbert_lo, hilbert_hi, depth, atom_count FROM tmp_atom "
        "ON CONFLICT (id) DO NOTHING");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Upsert failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    return true;
}

// =============================================================================
// UTF-8 Decoding
// =============================================================================

std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());
    
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();
    
    while (p < end) {
        uint32_t cp;
        
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            cp = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            cp = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            cp = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            cp = 0xFFFD;
            ++p;
        }
        
        codepoints.push_back(cp);
    }
    
    return codepoints;
}

// =============================================================================
// File Processing
// =============================================================================

struct IngestResult {
    size_t bytes;
    size_t codepoints;
    size_t rules;
    size_t edges;
    double seconds;
};

// =============================================================================
// Co-occurrence Edge Recording
// =============================================================================

// Accumulate co-occurrence weights between compositions
// Key: pair of hashes (smaller first for symmetry)
// Value: accumulated weight
static std::map<std::pair<std::string, std::string>, double> g_cooccurrence;

void record_cooccurrence(const Blake3Hash& a, const Blake3Hash& b, double weight) {
    std::string a_hex = a.to_hex();
    std::string b_hex = b.to_hex();
    
    // Canonical order (smaller hash first)
    if (a_hex > b_hex) std::swap(a_hex, b_hex);
    
    auto key = std::make_pair(a_hex, b_hex);
    g_cooccurrence[key] += weight;
}

// Record co-occurrence edges from a sequence of composition hashes
// Context window: pairs within N positions accumulate weight = 1/distance
void record_sequence_cooccurrence(
    const std::vector<Blake3Hash>& sequence,
    size_t context_window = 5
) {
    for (size_t i = 0; i < sequence.size(); ++i) {
        for (size_t j = i + 1; j < sequence.size() && j <= i + context_window; ++j) {
            double distance = static_cast<double>(j - i);
            double weight = 1.0 / distance;  // Adjacent = 1.0, 2 apart = 0.5, etc.
            record_cooccurrence(sequence[i], sequence[j], weight);
        }
    }
}

// Insert accumulated co-occurrence edges to database
bool insert_cooccurrence_edges(PGconn* conn, double threshold = 0.5) {
    // Filter edges above threshold
    std::vector<std::tuple<std::string, std::string, double>> edges;
    for (const auto& [key, weight] : g_cooccurrence) {
        if (weight >= threshold) {
            edges.push_back({key.first, key.second, weight});
        }
    }
    
    if (edges.empty()) return true;
    
    std::cerr << "[EDGES] Inserting " << edges.size() << " co-occurrence edges (threshold=" << threshold << ")...\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table for edges
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_edge ("
        "  from_id BYTEA,"
        "  to_id BYTEA,"
        "  weight DOUBLE PRECISION"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_edge failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY edges to temp table
    res = PQexec(conn, "COPY tmp_edge FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto& [from_hex, to_hex, weight] : edges) {
        batch += "\\\\x";
        batch += from_hex;
        batch += "\t\\\\x";
        batch += to_hex;
        batch += "\t";
        batch += std::to_string(weight);
        batch += "\n";
        
        if (batch.size() > (1 << 19)) {
            PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    
    if (!batch.empty()) {
        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
    }
    
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Insert edges as compositions with weight in M coordinate
    // Edge geometry: LINESTRINGZM from source centroid to target centroid
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT "
        "  hypercube_blake3(e.from_id || e.to_id), "
        "  ST_SetSRID(ST_MakeLine("
        "    ST_SetSRID(ST_MakePoint("
        "      ST_X(ST_Centroid(a1.geom)), ST_Y(ST_Centroid(a1.geom)), "
        "      ST_Z(ST_Centroid(a1.geom)), e.weight"
        "    ), 0), "
        "    ST_SetSRID(ST_MakePoint("
        "      ST_X(ST_Centroid(a2.geom)), ST_Y(ST_Centroid(a2.geom)), "
        "      ST_Z(ST_Centroid(a2.geom)), e.weight"
        "    ), 0)"
        "  ), 0), "
        "  ARRAY[e.from_id, e.to_id], "
        "  COALESCE((a1.hilbert_lo + a2.hilbert_lo) / 2, 0), "
        "  COALESCE((a1.hilbert_hi + a2.hilbert_hi) / 2, 0), "
        "  GREATEST(a1.depth, a2.depth) + 1, "
        "  a1.atom_count + a2.atom_count "
        "FROM tmp_edge e "
        "JOIN atom a1 ON a1.id = e.from_id "
        "JOIN atom a2 ON a2.id = e.to_id "
        "ON CONFLICT (id) DO UPDATE SET "
        "  geom = ST_SetSRID(ST_MakeLine("
        "    ST_SetSRID(ST_MakePoint("
        "      ST_X(ST_StartPoint(atom.geom)), ST_Y(ST_StartPoint(atom.geom)), "
        "      ST_Z(ST_StartPoint(atom.geom)), "
        "      ST_M(ST_StartPoint(atom.geom)) + EXCLUDED.geom::geometry->ST_M(ST_StartPoint(EXCLUDED.geom))"
        "    ), 0), "
        "    ST_EndPoint(atom.geom)"
        "  ), 0)");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        // Simpler upsert without complex update
        PQclear(res);
        res = PQexec(conn,
            "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
            "SELECT "
            "  hypercube_blake3(e.from_id || e.to_id), "
            "  ST_SetSRID(ST_MakeLine("
            "    ST_SetSRID(ST_MakePoint("
            "      ST_X(ST_Centroid(a1.geom)), ST_Y(ST_Centroid(a1.geom)), "
            "      ST_Z(ST_Centroid(a1.geom)), e.weight"
            "    ), 0), "
            "    ST_SetSRID(ST_MakePoint("
            "      ST_X(ST_Centroid(a2.geom)), ST_Y(ST_Centroid(a2.geom)), "
            "      ST_Z(ST_Centroid(a2.geom)), e.weight"
            "    ), 0)"
            "  ), 0), "
            "  ARRAY[e.from_id, e.to_id], "
            "  COALESCE((a1.hilbert_lo + a2.hilbert_lo) / 2, 0), "
            "  COALESCE((a1.hilbert_hi + a2.hilbert_hi) / 2, 0), "
            "  GREATEST(a1.depth, a2.depth) + 1, "
            "  a1.atom_count + a2.atom_count "
            "FROM tmp_edge e "
            "JOIN atom a1 ON a1.id = e.from_id "
            "JOIN atom a2 ON a2.id = e.to_id "
            "ON CONFLICT (id) DO NOTHING");
    }
    
    int inserted = 0;
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        inserted = atoi(PQcmdTuples(res));
    } else {
        std::cerr << "Edge insert error: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[EDGES] Inserted " << inserted << " edges\n";
    return true;
}

// Get sequence of composition hashes from start rule
std::vector<Blake3Hash> get_start_rule_sequence(Rule* start_rule, const std::vector<Rule*>& all_rules) {
    std::vector<Blake3Hash> sequence;
    
    for (Symbol* s = start_rule->first; s; s = s->next) {
        if (s->is_terminal) {
            auto it = g_atom_cache.find(s->codepoint);
            if (it != g_atom_cache.end()) {
                sequence.push_back(it->second.hash);
            }
        } else if (s->rule->computed) {
            sequence.push_back(s->rule->hash);
        }
    }
    
    return sequence;
}

IngestResult ingest_file(PGconn* conn, const fs::path& path) {
    IngestResult result{};
    auto start = std::chrono::high_resolution_clock::now();
    
    // Read file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot read: " << path << std::endl;
        return result;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    result.bytes = content.size();
    
    // Decode UTF-8
    std::vector<uint32_t> codepoints = decode_utf8(content);
    result.codepoints = codepoints.size();
    
    if (codepoints.empty()) return result;
    
    // Run Sequitur
    Sequitur seq;
    seq.process(codepoints);
    
    result.rules = seq.num_rules();
    
    // Compute composition data for all rules
    std::vector<Rule*> rules_copy = seq.rules();
    for (Rule* r : rules_copy) {
        compute_rule_data(r, rules_copy);
    }
    
    // Insert to database
    insert_rules_to_db(conn, rules_copy);
    
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    
    return result;
}

void ingest_directory(PGconn* conn, const fs::path& dir) {
    size_t total_bytes = 0;
    size_t total_files = 0;
    size_t total_rules = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
        if (ext == ".txt" || ext == ".md" || ext == ".json" || 
            ext == ".py" || ext == ".cpp" || ext == ".hpp" ||
            ext == ".c" || ext == ".h" || ext == ".js" ||
            ext == ".ts" || ext == ".yaml" || ext == ".yml" ||
            ext == ".xml" || ext == ".html" || ext == ".css" ||
            ext == ".sql" || ext == ".sh" || ext == ".rs" ||
            ext == ".go" || ext == ".java" || ext == ".rb") {
            files.push_back(entry.path());
        }
    }
    
    std::cerr << "[INGEST] Found " << files.size() << " text files\n";
    
    for (const auto& path : files) {
        IngestResult res = ingest_file(conn, path);
        if (res.bytes > 0) {
            total_bytes += res.bytes;
            total_files++;
            total_rules += res.rules;
            
            double rate_kbps = (res.bytes / 1024.0) / res.seconds;
            std::cerr << "  " << path.filename().string()
                      << " (" << res.bytes << " B, "
                      << res.rules << " rules, "
                      << static_cast<int>(rate_kbps) << " KB/s)\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(end - start).count();
    
    std::cerr << "\n[COMPLETE]\n";
    std::cerr << "  Files: " << total_files << "\n";
    std::cerr << "  Bytes: " << total_bytes << " (" << (total_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cerr << "  Rules: " << total_rules << "\n";
    std::cerr << "  Time: " << total_secs << " s\n";
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Sequitur Grammar-Based Ingester\n\n"
              << "Usage: " << prog << " [options] <path>\n\n"
              << "Arguments:\n"
              << "  <path>               File or directory to ingest\n\n"
              << "Options:\n"
              << "  -d, --dbname NAME    Database name (default: hypercube)\n"
              << "  -h, --host HOST      Database host\n"
              << "  -p, --port PORT      Database port\n"
              << "  -U, --user USER      Database user\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    std::string target = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            target = arg;
        }
    }
    
    if (target.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string conninfo = "dbname=" + dbname;
    if (!host.empty()) conninfo += " host=" + host;
    if (!port.empty()) conninfo += " port=" + port;
    if (!user.empty()) conninfo += " user=" + user;
    
    std::cerr << "=== Sequitur Ingester ===\n";
    std::cerr << "Database: " << dbname << "\n";
    std::cerr << "Target: " << target << "\n\n";
    
    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }
    
    if (!load_atom_cache(conn)) {
        PQfinish(conn);
        return 1;
    }
    
    fs::path path(target);
    if (!fs::exists(path)) {
        std::cerr << "Not found: " << target << std::endl;
        PQfinish(conn);
        return 1;
    }
    
    if (fs::is_directory(path)) {
        ingest_directory(conn, path);
    } else {
        IngestResult res = ingest_file(conn, path);
        if (res.bytes > 0) {
            std::cerr << "[OK] " << res.bytes << " bytes → " << res.rules << " rules\n";
        }
    }
    
    // Stats
    PGresult* res = PQexec(conn, "SELECT COUNT(*) FROM atom WHERE depth > 0");
    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
        std::cerr << "\nTotal compositions in DB: " << PQgetvalue(res, 0, 0) << "\n";
    }
    PQclear(res);
    
    PQfinish(conn);
    return 0;
}
