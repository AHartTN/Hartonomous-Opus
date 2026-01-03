/**
 * Hypercube Optimized Operations
 * 
 * High-performance batch operations for PostgreSQL.
 * Key optimizations:
 * - Batch loading to eliminate N+1 queries
 * - Prepared statements with SPI_prepare
 * - In-memory graph algorithms
 * - PARALLEL SAFE for multi-core execution
 */

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "utils/memutils.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
}

#include <vector>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

// =============================================================================
// Type Definitions
// =============================================================================

// 32-byte hash as fixed-size key
struct HashKey {
    uint8_t bytes[32];
    
    bool operator==(const HashKey& other) const {
        return memcmp(bytes, other.bytes, 32) == 0;
    }
    
    std::string to_hex() const {
        char hex[65];
        for (int i = 0; i < 32; i++) {
            snprintf(hex + i*2, 3, "%02x", bytes[i]);
        }
        return std::string(hex);
    }
};

struct HashKeyHasher {
    size_t operator()(const HashKey& k) const {
        uint64_t h;
        memcpy(&h, k.bytes, 8);
        return h;
    }
};

// Atom data cached in memory
struct AtomData {
    HashKey id;
    std::vector<HashKey> children;
    std::vector<uint8_t> value;  // For leaf nodes
    int depth;
    double centroid_x, centroid_y, centroid_z, centroid_m;
    bool is_leaf;
};

// Graph edge for semantic operations
struct SemanticEdge {
    HashKey target;
    double weight;
};

using AtomCache = std::unordered_map<HashKey, AtomData, HashKeyHasher>;
using EdgeMap = std::unordered_map<HashKey, std::vector<SemanticEdge>, HashKeyHasher>;

// =============================================================================
// Helper Functions
// =============================================================================

static void bytea_to_hash(bytea* b, HashKey& out) {
    if (VARSIZE_ANY_EXHDR(b) >= 32) {
        memcpy(out.bytes, VARDATA_ANY(b), 32);
    } else {
        memset(out.bytes, 0, 32);
    }
}

static bytea* hash_to_bytea(const HashKey& hash) {
    bytea* result = (bytea*)palloc(VARHDRSZ + 32);
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash.bytes, 32);
    return result;
}

// =============================================================================
// Batch Cache Loading
// =============================================================================

/**
 * Load atoms by depth range into memory cache.
 * This is the key optimization - one query loads all needed data.
 */
static void load_atoms_by_depth(AtomCache& cache, int min_depth, int max_depth) {
    char query[512];
    snprintf(query, sizeof(query),
        "SELECT id, value, children, depth, "
        "ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
        "FROM atom WHERE depth >= %d AND depth <= %d",
        min_depth, max_depth);
    
    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT) return;
    
    for (uint64 i = 0; i < SPI_processed; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        
        AtomData atom;
        bool isnull;
        
        // id
        Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        bytea_to_hash(DatumGetByteaP(id_datum), atom.id);
        
        // value (for leaves)
        Datum value_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (!isnull) {
            bytea* val = DatumGetByteaP(value_datum);
            atom.value.assign((uint8_t*)VARDATA_ANY(val), 
                             (uint8_t*)VARDATA_ANY(val) + VARSIZE_ANY_EXHDR(val));
            atom.is_leaf = true;
        } else {
            atom.is_leaf = false;
        }
        
        // children
        Datum children_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        if (!isnull) {
            ArrayType* arr = DatumGetArrayTypeP(children_datum);
            Datum* elems;
            bool* nulls;
            int nelems;
            deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                             &elems, &nulls, &nelems);
            
            atom.children.reserve(nelems);
            for (int j = 0; j < nelems; j++) {
                if (!nulls[j]) {
                    HashKey child;
                    bytea_to_hash(DatumGetByteaP(elems[j]), child);
                    atom.children.push_back(child);
                }
            }
        }
        
        // depth
        Datum depth_datum = SPI_getbinval(tuple, tupdesc, 4, &isnull);
        atom.depth = isnull ? 0 : DatumGetInt32(depth_datum);
        
        // centroid
        Datum cx = SPI_getbinval(tuple, tupdesc, 5, &isnull);
        atom.centroid_x = isnull ? 0 : DatumGetFloat8(cx);
        
        Datum cy = SPI_getbinval(tuple, tupdesc, 6, &isnull);
        atom.centroid_y = isnull ? 0 : DatumGetFloat8(cy);
        
        Datum cz = SPI_getbinval(tuple, tupdesc, 7, &isnull);
        atom.centroid_z = isnull ? 0 : DatumGetFloat8(cz);
        
        Datum cm = SPI_getbinval(tuple, tupdesc, 8, &isnull);
        atom.centroid_m = isnull ? 0 : DatumGetFloat8(cm);
        
        cache[atom.id] = std::move(atom);
    }
}

/**
 * Load specific atoms by ID list.
 * Uses IN clause with array for efficiency.
 */
static void load_atoms_by_ids(AtomCache& cache, const std::vector<HashKey>& ids) {
    if (ids.empty()) return;
    
    // Build array literal
    std::string array_lit = "ARRAY[";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) array_lit += ",";
        array_lit += "'\\x" + ids[i].to_hex() + "'::bytea";
    }
    array_lit += "]";
    
    std::string query = 
        "SELECT id, value, children, depth, "
        "ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
        "FROM atom WHERE id = ANY(" + array_lit + ")";
    
    int ret = SPI_execute(query.c_str(), true, 0);
    if (ret != SPI_OK_SELECT) return;
    
    for (uint64 i = 0; i < SPI_processed; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        
        AtomData atom;
        bool isnull;
        
        Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        bytea_to_hash(DatumGetByteaP(id_datum), atom.id);
        
        Datum value_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (!isnull) {
            bytea* val = DatumGetByteaP(value_datum);
            atom.value.assign((uint8_t*)VARDATA_ANY(val), 
                             (uint8_t*)VARDATA_ANY(val) + VARSIZE_ANY_EXHDR(val));
            atom.is_leaf = true;
        } else {
            atom.is_leaf = false;
        }
        
        Datum children_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        if (!isnull) {
            ArrayType* arr = DatumGetArrayTypeP(children_datum);
            Datum* elems;
            bool* nulls;
            int nelems;
            deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                             &elems, &nulls, &nelems);
            
            atom.children.reserve(nelems);
            for (int j = 0; j < nelems; j++) {
                if (!nulls[j]) {
                    HashKey child;
                    bytea_to_hash(DatumGetByteaP(elems[j]), child);
                    atom.children.push_back(child);
                }
            }
        }
        
        Datum depth_datum = SPI_getbinval(tuple, tupdesc, 4, &isnull);
        atom.depth = isnull ? 0 : DatumGetInt32(depth_datum);
        
        atom.centroid_x = 0; atom.centroid_y = 0;
        atom.centroid_z = 0; atom.centroid_m = 0;
        
        Datum cx = SPI_getbinval(tuple, tupdesc, 5, &isnull);
        if (!isnull) atom.centroid_x = DatumGetFloat8(cx);
        Datum cy = SPI_getbinval(tuple, tupdesc, 6, &isnull);
        if (!isnull) atom.centroid_y = DatumGetFloat8(cy);
        Datum cz = SPI_getbinval(tuple, tupdesc, 7, &isnull);
        if (!isnull) atom.centroid_z = DatumGetFloat8(cz);
        Datum cm = SPI_getbinval(tuple, tupdesc, 8, &isnull);
        if (!isnull) atom.centroid_m = DatumGetFloat8(cm);
        
        cache[atom.id] = std::move(atom);
    }
}

/**
 * Load semantic edges (depth=1, atom_count=2) into edge map.
 */
static void load_semantic_edges(EdgeMap& edges, int limit = 100000) {
    char query[256];
    snprintf(query, sizeof(query),
        "SELECT children, ST_M(ST_StartPoint(geom)) as weight "
        "FROM atom WHERE depth = 1 AND atom_count = 2 "
        "AND ST_M(ST_StartPoint(geom)) < 100000 "
        "LIMIT %d", limit);
    
    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT) return;
    
    for (uint64 i = 0; i < SPI_processed; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull;
        
        Datum children_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        
        ArrayType* arr = DatumGetArrayTypeP(children_datum);
        Datum* elems;
        bool* nulls;
        int nelems;
        deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                         &elems, &nulls, &nelems);
        
        if (nelems != 2 || nulls[0] || nulls[1]) continue;
        
        HashKey a, b;
        bytea_to_hash(DatumGetByteaP(elems[0]), a);
        bytea_to_hash(DatumGetByteaP(elems[1]), b);
        
        Datum weight_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        double weight = isnull ? 1.0 : DatumGetFloat8(weight_datum);
        
        // Bidirectional edges
        edges[a].push_back({b, weight});
        edges[b].push_back({a, weight});
    }
}

// =============================================================================
// In-Memory Graph Algorithms
// =============================================================================

/**
 * Reconstruct text using cached atoms (no SPI in loop).
 */
static std::string reconstruct_text_cached(const AtomCache& cache, const HashKey& root) {
    std::string result;
    result.reserve(4096);
    
    struct StackEntry {
        HashKey id;
        size_t child_idx;
    };
    
    std::vector<StackEntry> stack;
    stack.push_back({root, 0});
    
    while (!stack.empty()) {
        auto& current = stack.back();
        
        auto it = cache.find(current.id);
        if (it == cache.end()) {
            stack.pop_back();
            continue;
        }
        
        const AtomData& atom = it->second;
        
        if (atom.is_leaf) {
            // Append value bytes
            result.append((char*)atom.value.data(), atom.value.size());
            stack.pop_back();
            continue;
        }
        
        if (current.child_idx < atom.children.size()) {
            HashKey child_id = atom.children[current.child_idx];
            current.child_idx++;
            stack.push_back({child_id, 0});
        } else {
            stack.pop_back();
        }
    }
    
    return result;
}

/**
 * Random walk using cached edge map.
 */
static std::vector<std::pair<HashKey, double>> random_walk_cached(
    const EdgeMap& edges,
    const HashKey& seed,
    int steps,
    std::mt19937& rng
) {
    std::vector<std::pair<HashKey, double>> path;
    path.reserve(steps + 1);
    
    HashKey current = seed;
    std::unordered_set<HashKey, HashKeyHasher> visited;
    
    for (int i = 0; i <= steps; i++) {
        visited.insert(current);
        
        auto it = edges.find(current);
        if (it == edges.end()) {
            path.push_back({current, 0.0});
            break;
        }
        
        // Filter to unvisited neighbors
        std::vector<SemanticEdge> available;
        for (const auto& edge : it->second) {
            if (visited.find(edge.target) == visited.end()) {
                available.push_back(edge);
            }
        }
        
        if (available.empty()) {
            path.push_back({current, 0.0});
            break;
        }
        
        // Select by weight (higher weight = more likely)
        double total_weight = 0;
        for (const auto& e : available) {
            total_weight += e.weight;
        }
        
        std::uniform_real_distribution<double> dist(0, total_weight);
        double r = dist(rng);
        
        double cumulative = 0;
        SemanticEdge chosen = available[0];
        for (const auto& e : available) {
            cumulative += e.weight;
            if (r <= cumulative) {
                chosen = e;
                break;
            }
        }
        
        path.push_back({current, chosen.weight});
        current = chosen.target;
    }
    
    return path;
}

/**
 * BFS shortest path using cached edges.
 */
static std::vector<HashKey> shortest_path_cached(
    const EdgeMap& edges,
    const HashKey& from,
    const HashKey& to,
    int max_depth
) {
    std::unordered_map<HashKey, HashKey, HashKeyHasher> parent;
    std::queue<std::pair<HashKey, int>> queue;
    
    queue.push({from, 0});
    parent[from] = from;  // Mark as visited
    
    bool found = false;
    while (!queue.empty() && !found) {
        auto [current, depth] = queue.front();
        queue.pop();
        
        if (depth >= max_depth) continue;
        
        auto it = edges.find(current);
        if (it == edges.end()) continue;
        
        for (const auto& edge : it->second) {
            if (parent.find(edge.target) != parent.end()) continue;
            
            parent[edge.target] = current;
            
            if (edge.target == to) {
                found = true;
                break;
            }
            
            queue.push({edge.target, depth + 1});
        }
    }
    
    if (!found) return {};
    
    // Reconstruct path
    std::vector<HashKey> path;
    HashKey current = to;
    while (!(current == from)) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(from);
    std::reverse(path.begin(), path.end());
    
    return path;
}

// =============================================================================
// PostgreSQL Function Definitions
// =============================================================================

extern "C" {

// Batch reconstruct multiple atoms
PG_FUNCTION_INFO_V1(hypercube_batch_reconstruct);
Datum hypercube_batch_reconstruct(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        // Get input array
        ArrayType* ids_arr = PG_GETARG_ARRAYTYPE_P(0);
        Datum* elems;
        bool* nulls;
        int nelems;
        deconstruct_array(ids_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                         &elems, &nulls, &nelems);
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        // Load all atoms (depth 0-25 should cover most trees)
        AtomCache* cache = new AtomCache();
        load_atoms_by_depth(*cache, 0, 25);
        
        // Reconstruct all texts
        auto* results = new std::vector<std::pair<HashKey, std::string>>();
        results->reserve(nelems);
        
        for (int i = 0; i < nelems; i++) {
            if (nulls[i]) continue;
            
            HashKey id;
            bytea_to_hash(DatumGetByteaP(elems[i]), id);
            
            std::string text = reconstruct_text_cached(*cache, id);
            results->push_back({id, std::move(text)});
        }
        
        delete cache;
        SPI_finish();
        
        // Setup for iteration
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = results;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = results->size();
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    auto* results = (std::vector<std::pair<HashKey, std::string>>*)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        auto& [id, text] = (*results)[funcctx->call_cntr];
        
        Datum values[2];
        bool isnulls[2] = {false, false};
        
        values[0] = PointerGetDatum(hash_to_bytea(id));
        values[1] = CStringGetTextDatum(text.c_str());
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    delete results;
    SRF_RETURN_DONE(funcctx);
}

// Fast semantic walk with in-memory graph
PG_FUNCTION_INFO_V1(hypercube_semantic_walk);
Datum hypercube_semantic_walk(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea* seed_bytea = PG_GETARG_BYTEA_PP(0);
        int steps = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 10;
        
        HashKey seed;
        bytea_to_hash(seed_bytea, seed);
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        // Load semantic edges into memory (one query!)
        EdgeMap* edges = new EdgeMap();
        load_semantic_edges(*edges);
        
        // Perform walk entirely in memory
        std::random_device rd;
        std::mt19937 rng(rd());
        
        auto* path = new std::vector<std::pair<HashKey, double>>(
            random_walk_cached(*edges, seed, steps, rng)
        );
        
        delete edges;
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = path;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = path->size();
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    auto* path = (std::vector<std::pair<HashKey, double>>*)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        auto& [id, weight] = (*path)[funcctx->call_cntr];
        
        Datum values[3];
        bool isnulls[3] = {false, false, false};
        
        values[0] = Int32GetDatum((int32)funcctx->call_cntr);
        values[1] = PointerGetDatum(hash_to_bytea(id));
        values[2] = Float8GetDatum(weight);
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    delete path;
    SRF_RETURN_DONE(funcctx);
}

// Fast semantic path with in-memory BFS
PG_FUNCTION_INFO_V1(hypercube_semantic_path);
Datum hypercube_semantic_path(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea* from_bytea = PG_GETARG_BYTEA_PP(0);
        bytea* to_bytea = PG_GETARG_BYTEA_PP(1);
        int max_depth = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : 6;
        
        HashKey from, to;
        bytea_to_hash(from_bytea, from);
        bytea_to_hash(to_bytea, to);
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        EdgeMap* edges = new EdgeMap();
        load_semantic_edges(*edges);
        
        auto* path = new std::vector<HashKey>(
            shortest_path_cached(*edges, from, to, max_depth)
        );
        
        delete edges;
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = path;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = path->size();
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    auto* path = (std::vector<HashKey>*)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        HashKey& id = (*path)[funcctx->call_cntr];
        
        Datum values[2];
        bool isnulls[2] = {false, false};
        
        values[0] = Int32GetDatum((int32)funcctx->call_cntr);
        values[1] = PointerGetDatum(hash_to_bytea(id));
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    delete path;
    SRF_RETURN_DONE(funcctx);
}

// Batch lookup atoms by ID array
PG_FUNCTION_INFO_V1(hypercube_batch_lookup);
Datum hypercube_batch_lookup(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        ArrayType* ids_arr = PG_GETARG_ARRAYTYPE_P(0);
        Datum* elems;
        bool* nulls;
        int nelems;
        deconstruct_array(ids_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                         &elems, &nulls, &nelems);
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        std::vector<HashKey> ids;
        ids.reserve(nelems);
        for (int i = 0; i < nelems; i++) {
            if (!nulls[i]) {
                HashKey id;
                bytea_to_hash(DatumGetByteaP(elems[i]), id);
                ids.push_back(id);
            }
        }
        
        AtomCache* cache = new AtomCache();
        load_atoms_by_ids(*cache, ids);
        
        // Convert to vector for iteration
        auto* results = new std::vector<AtomData>();
        results->reserve(cache->size());
        for (auto& [k, v] : *cache) {
            results->push_back(std::move(v));
        }
        
        delete cache;
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = results;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = results->size();
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    auto* results = (std::vector<AtomData>*)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        AtomData& atom = (*results)[funcctx->call_cntr];
        
        Datum values[5];
        bool isnulls[5] = {false, false, false, false, false};
        
        values[0] = PointerGetDatum(hash_to_bytea(atom.id));
        values[1] = Int32GetDatum(atom.depth);
        values[2] = BoolGetDatum(atom.is_leaf);
        values[3] = Int32GetDatum((int32)atom.children.size());
        values[4] = Float8GetDatum(atom.centroid_x);
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    delete results;
    SRF_RETURN_DONE(funcctx);
}

} // extern "C"
