/**
 * PostgreSQL Extension: semantic_ops
 * 
 * High-performance C++ implementations for semantic query operations.
 * Offloads recursive traversals, distance calculations, and graph walks from SQL.
 * 
 * Functions:
 * - semantic_traverse(root_id) -> SETOF (id, depth, path, content)
 * - semantic_find_path(from_id, to_id) -> path
 * - semantic_nearest_k(target_id, k) -> SETOF (id, distance)
 * - semantic_hilbert_range(center_id, range) -> SETOF id
 * - semantic_reconstruct(root_id) -> text
 * - semantic_centroid_4d(x[], y[], z[], m[]) -> (x, y, z, m)
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

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
}

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <cmath>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"

using namespace hypercube;

// =============================================================================
// Helper Functions
// =============================================================================

// Parse bytea to 32-byte hash
static void bytea_to_hash(bytea* b, uint8_t* out) {
    if (VARSIZE_ANY_EXHDR(b) >= 32) {
        memcpy(out, VARDATA_ANY(b), 32);
    } else {
        memset(out, 0, 32);
    }
}

// Create bytea from 32-byte hash
static bytea* hash_to_bytea(const uint8_t* hash) {
    bytea* result = (bytea*)palloc(VARHDRSZ + 32);
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash, 32);
    return result;
}

// Hash key for unordered_map
struct HashKey {
    uint8_t bytes[32];
    
    bool operator==(const HashKey& other) const {
        return memcmp(bytes, other.bytes, 32) == 0;
    }
};

struct HashKeyHasher {
    size_t operator()(const HashKey& k) const {
        // Use first 8 bytes as hash
        uint64_t h;
        memcpy(&h, k.bytes, 8);
        return h;
    }
};

extern "C" {

// =============================================================================
// semantic_traverse: DFS traversal of composition DAG
// Returns all descendants with depth and path info
// =============================================================================

typedef struct {
    uint8_t id[32];
    int depth;
    int ordinal;
    uint8_t* path;  // Array of ordinals
    int path_len;
} TraverseNode;

typedef struct {
    std::vector<TraverseNode>* nodes;
    size_t current;
    TupleDesc tupdesc;
} TraverseState;

PG_FUNCTION_INFO_V1(semantic_traverse);
Datum semantic_traverse(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    TraverseState* state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        // Get parameters
        bytea* root_id = PG_GETARG_BYTEA_PP(0);
        int max_depth = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 100;
        
        // Get tuple descriptor
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
        }
        
        state = (TraverseState*)palloc(sizeof(TraverseState));
        state->nodes = new std::vector<TraverseNode>();
        state->current = 0;
        state->tupdesc = BlessTupleDesc(tupdesc);
        
        // Connect to SPI for database access
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        // BFS/DFS traversal using SPI
        std::queue<TraverseNode> to_visit;
        std::unordered_set<std::string> visited;
        
        TraverseNode root;
        bytea_to_hash(root_id, root.id);
        root.depth = 0;
        root.ordinal = 0;
        root.path = NULL;
        root.path_len = 0;
        to_visit.push(root);
        
        while (!to_visit.empty()) {
            TraverseNode node = to_visit.front();
            to_visit.pop();
            
            // Check if visited
            std::string key((char*)node.id, 32);
            if (visited.count(key)) continue;
            visited.insert(key);
            
            // Add to results
            state->nodes->push_back(node);
            
            if (node.depth >= max_depth) continue;
            
            // Query for children
            char query[256];
            char id_hex[65];
            for (int i = 0; i < 32; i++) {
                snprintf(id_hex + i*2, 3, "%02x", node.id[i]);
            }
            
            snprintf(query, sizeof(query),
                "SELECT unnest(children), generate_subscripts(children, 1) - 1 "
                "FROM atom WHERE id = '\\x%s' AND children IS NOT NULL",
                id_hex);
            
            int ret = SPI_execute(query, true, 0);
            if (ret == SPI_OK_SELECT && SPI_processed > 0) {
                for (uint64 i = 0; i < SPI_processed; i++) {
                    HeapTuple tuple = SPI_tuptable->vals[i];
                    
                    bool isnull;
                    Datum child_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &isnull);
                    if (isnull) continue;
                    
                    Datum ord_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 2, &isnull);
                    int ordinal = isnull ? 0 : DatumGetInt32(ord_datum);
                    
                    TraverseNode child;
                    bytea* child_bytea = DatumGetByteaP(child_datum);
                    bytea_to_hash(child_bytea, child.id);
                    child.depth = node.depth + 1;
                    child.ordinal = ordinal;
                    
                    // Build path
                    child.path_len = node.path_len + 1;
                    child.path = (uint8_t*)MemoryContextAlloc(
                        funcctx->multi_call_memory_ctx, child.path_len);
                    if (node.path_len > 0) {
                        memcpy(child.path, node.path, node.path_len);
                    }
                    child.path[node.path_len] = (uint8_t)ordinal;
                    
                    to_visit.push(child);
                }
            }
        }
        
        SPI_finish();
        
        funcctx->user_fctx = state;
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (TraverseState*)funcctx->user_fctx;
    
    if (state->current < state->nodes->size()) {
        const TraverseNode& node = (*state->nodes)[state->current++];
        
        Datum values[4];
        bool nulls[4] = {false, false, false, false};
        
        // id
        values[0] = PointerGetDatum(hash_to_bytea(node.id));
        
        // depth
        values[1] = Int32GetDatum(node.depth);
        
        // ordinal
        values[2] = Int32GetDatum(node.ordinal);
        
        // path_len (simplified - could return full path array)
        values[3] = Int32GetDatum(node.path_len);
        
        HeapTuple tuple = heap_form_tuple(state->tupdesc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    // Cleanup
    delete state->nodes;
    SRF_RETURN_DONE(funcctx);
}

// =============================================================================
// semantic_reconstruct: Fast text reconstruction from composition
// Uses iterative DFS instead of recursive SQL CTE
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_reconstruct);
Datum semantic_reconstruct(PG_FUNCTION_ARGS) {
    bytea* root_id = PG_GETARG_BYTEA_PP(0);
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    // Stack for DFS: (id, depth, visited_children)
    struct StackEntry {
        uint8_t id[32];
        int child_idx;
        std::vector<uint8_t> child_ids;  // Each child is 32 bytes
    };
    
    std::vector<StackEntry> stack;
    std::string result;
    result.reserve(4096);
    
    // Start with root
    StackEntry root;
    bytea_to_hash(root_id, root.id);
    root.child_idx = -1;  // Not yet queried
    stack.push_back(root);
    
    while (!stack.empty()) {
        StackEntry& current = stack.back();
        
        // First visit - query for children or value
        if (current.child_idx == -1) {
            char id_hex[65];
            for (int i = 0; i < 32; i++) {
                snprintf(id_hex + i*2, 3, "%02x", current.id[i]);
            }
            
            // Check if leaf (has value)
            char query[256];
            snprintf(query, sizeof(query),
                "SELECT value, children FROM atom WHERE id = '\\x%s'", id_hex);
            
            int ret = SPI_execute(query, true, 1);
            if (ret == SPI_OK_SELECT && SPI_processed > 0) {
                HeapTuple tuple = SPI_tuptable->vals[0];
                
                bool value_null, children_null;
                Datum value_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &value_null);
                Datum children_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 2, &children_null);
                
                if (!value_null) {
                    // Leaf node - append value to result
                    bytea* value = DatumGetByteaP(value_datum);
                    result.append((char*)VARDATA_ANY(value), VARSIZE_ANY_EXHDR(value));
                    stack.pop_back();
                    continue;
                }
                
                if (!children_null) {
                    // Composition - get children array
                    ArrayType* arr = DatumGetArrayTypeP(children_datum);
                    int n = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
                    
                    current.child_idx = 0;
                    current.child_ids.reserve(n * 32);
                    
                    // Extract children bytea values
                    Datum* elems;
                    bool* nulls;
                    int nelems;
                    deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                                     &elems, &nulls, &nelems);
                    
                    for (int i = 0; i < nelems; i++) {
                        if (!nulls[i]) {
                            bytea* child = DatumGetByteaP(elems[i]);
                            uint8_t hash[32];
                            bytea_to_hash(child, hash);
                            current.child_ids.insert(current.child_ids.end(), hash, hash + 32);
                        }
                    }
                } else {
                    // No value, no children - skip
                    stack.pop_back();
                    continue;
                }
            } else {
                // Not found
                stack.pop_back();
                continue;
            }
        }
        
        // Process next child
        int num_children = current.child_ids.size() / 32;
        if (current.child_idx < num_children) {
            StackEntry child;
            memcpy(child.id, &current.child_ids[current.child_idx * 32], 32);
            child.child_idx = -1;
            current.child_idx++;
            stack.push_back(child);
        } else {
            // All children processed
            stack.pop_back();
        }
    }
    
    SPI_finish();
    
    // Return as text
    text* result_text = cstring_to_text_with_len(result.c_str(), result.size());
    PG_RETURN_TEXT_P(result_text);
}

// =============================================================================
// semantic_hilbert_distance: 128-bit Hilbert distance calculation
// Returns (distance_lo, distance_hi) as proper 128-bit value
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_hilbert_distance_128);
Datum semantic_hilbert_distance_128(PG_FUNCTION_ARGS) {
    int64 lo1 = PG_GETARG_INT64(0);
    int64 hi1 = PG_GETARG_INT64(1);
    int64 lo2 = PG_GETARG_INT64(2);
    int64 hi2 = PG_GETARG_INT64(3);
    
    // Convert to unsigned for proper arithmetic
    uint64_t ulo1 = static_cast<uint64_t>(lo1);
    uint64_t uhi1 = static_cast<uint64_t>(hi1);
    uint64_t ulo2 = static_cast<uint64_t>(lo2);
    uint64_t uhi2 = static_cast<uint64_t>(hi2);
    
    // Compute absolute difference of 128-bit values
    // a = (hi1, lo1), b = (hi2, lo2)
    // |a - b| = distance
    
    uint64_t dist_lo, dist_hi;
    
    if (uhi1 > uhi2 || (uhi1 == uhi2 && ulo1 >= ulo2)) {
        // a >= b, compute a - b
        dist_lo = ulo1 - ulo2;
        dist_hi = uhi1 - uhi2;
        if (ulo1 < ulo2) {
            dist_hi--;  // Borrow
        }
    } else {
        // b > a, compute b - a
        dist_lo = ulo2 - ulo1;
        dist_hi = uhi2 - uhi1;
        if (ulo2 < ulo1) {
            dist_hi--;  // Borrow
        }
    }
    
    // Return as tuple
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[2];
    bool nulls[2] = {false, false};
    values[0] = Int64GetDatum(static_cast<int64>(dist_lo));
    values[1] = Int64GetDatum(static_cast<int64>(dist_hi));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

// =============================================================================
// semantic_4d_distance: True 4D Euclidean distance
// Uses raw uint32 coordinates stored as doubles
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_4d_distance);
Datum semantic_4d_distance(PG_FUNCTION_ARGS) {
    double x1 = PG_GETARG_FLOAT8(0);
    double y1 = PG_GETARG_FLOAT8(1);
    double z1 = PG_GETARG_FLOAT8(2);
    double m1 = PG_GETARG_FLOAT8(3);
    double x2 = PG_GETARG_FLOAT8(4);
    double y2 = PG_GETARG_FLOAT8(5);
    double z2 = PG_GETARG_FLOAT8(6);
    double m2 = PG_GETARG_FLOAT8(7);
    
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    double dm = m1 - m2;
    
    double dist = sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
    
    PG_RETURN_FLOAT8(dist);
}

// =============================================================================
// semantic_centroid_4d: Compute centroid of multiple 4D points
// Handles large coordinate values without overflow
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_centroid_4d);
Datum semantic_centroid_4d(PG_FUNCTION_ARGS) {
    ArrayType* x_arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* y_arr = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType* z_arr = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType* m_arr = PG_GETARG_ARRAYTYPE_P(3);
    
    int n = ArrayGetNItems(ARR_NDIM(x_arr), ARR_DIMS(x_arr));
    if (n == 0) {
        PG_RETURN_NULL();
    }
    
    // Extract arrays as float8
    Datum* x_elems;
    Datum* y_elems;
    Datum* z_elems;
    Datum* m_elems;
    bool* x_nulls;
    bool* y_nulls;
    bool* z_nulls;
    bool* m_nulls;
    int nx, ny, nz, nm;
    
    deconstruct_array(x_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &x_elems, &x_nulls, &nx);
    deconstruct_array(y_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &y_elems, &y_nulls, &ny);
    deconstruct_array(z_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &z_elems, &z_nulls, &nz);
    deconstruct_array(m_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &m_elems, &m_nulls, &nm);
    
    // Compute sums
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    int count = 0;
    
    for (int i = 0; i < n && i < nx && i < ny && i < nz && i < nm; i++) {
        if (x_nulls && x_nulls[i]) continue;
        if (y_nulls && y_nulls[i]) continue;
        if (z_nulls && z_nulls[i]) continue;
        if (m_nulls && m_nulls[i]) continue;
        
        sum_x += DatumGetFloat8(x_elems[i]);
        sum_y += DatumGetFloat8(y_elems[i]);
        sum_z += DatumGetFloat8(z_elems[i]);
        sum_m += DatumGetFloat8(m_elems[i]);
        count++;
    }
    
    if (count == 0) {
        PG_RETURN_NULL();
    }
    
    // Return tuple
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    values[0] = Float8GetDatum(sum_x / count);
    values[1] = Float8GetDatum(sum_y / count);
    values[2] = Float8GetDatum(sum_z / count);
    values[3] = Float8GetDatum(sum_m / count);
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

// =============================================================================
// semantic_coords_from_hilbert: Inverse Hilbert mapping
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_coords_from_hilbert);
Datum semantic_coords_from_hilbert(PG_FUNCTION_ARGS) {
    int64 lo = PG_GETARG_INT64(0);
    int64 hi = PG_GETARG_INT64(1);
    
    HilbertIndex idx(static_cast<uint64_t>(lo), static_cast<uint64_t>(hi));
    Point4D point = HilbertCurve::index_to_coords(idx);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    values[0] = Float8GetDatum(static_cast<double>(point.x));
    values[1] = Float8GetDatum(static_cast<double>(point.y));
    values[2] = Float8GetDatum(static_cast<double>(point.z));
    values[3] = Float8GetDatum(static_cast<double>(point.m));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

// =============================================================================
// semantic_hilbert_from_coords: Forward Hilbert mapping
// =============================================================================

PG_FUNCTION_INFO_V1(semantic_hilbert_from_coords);
Datum semantic_hilbert_from_coords(PG_FUNCTION_ARGS) {
    double x = PG_GETARG_FLOAT8(0);
    double y = PG_GETARG_FLOAT8(1);
    double z = PG_GETARG_FLOAT8(2);
    double m = PG_GETARG_FLOAT8(3);
    
    Point4D point(
        static_cast<uint32_t>(x),
        static_cast<uint32_t>(y),
        static_cast<uint32_t>(z),
        static_cast<uint32_t>(m)
    );
    
    HilbertIndex idx = HilbertCurve::coords_to_index(point);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[2];
    bool nulls[2] = {false, false};
    values[0] = Int64GetDatum(static_cast<int64>(idx.lo));
    values[1] = Int64GetDatum(static_cast<int64>(idx.hi));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

} // extern "C"
