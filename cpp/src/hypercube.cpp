/**
 * PostgreSQL Extension: hypercube
 * 
 * Provides 4D Hilbert curve indexing for semantic coordinate system.
 */

// PostgreSQL headers must come first on Windows
// Include order per PostgreSQL wiki: port/win32_msvc, port/win32, server headers
extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
}

// Now include C++ headers
#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;

extern "C" {

/**
 * Convert 4D coordinates to Hilbert index
 * hypercube_coords_to_hilbert(x int8, y int8, z int8, m int8) 
 *   RETURNS TABLE(hilbert_lo int8, hilbert_hi int8)
 */
PG_FUNCTION_INFO_V1(hypercube_coords_to_hilbert);
Datum hypercube_coords_to_hilbert(PG_FUNCTION_ARGS) {
    // Get coordinates as int64 (will be cast from int8 in SQL)
    int64 x = PG_GETARG_INT64(0);
    int64 y = PG_GETARG_INT64(1);
    int64 z = PG_GETARG_INT64(2);
    int64 m = PG_GETARG_INT64(3);
    
    // Convert to Point4D (unsigned)
    Point4D point(
        static_cast<uint32_t>(x),
        static_cast<uint32_t>(y),
        static_cast<uint32_t>(z),
        static_cast<uint32_t>(m)
    );
    
    // Compute Hilbert index
    HilbertIndex hilbert = HilbertCurve::coords_to_index(point);
    
    // Build result tuple
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[2];
    bool nulls[2] = {false, false};
    
    // Store as signed int64 (PostgreSQL doesn't have unsigned types)
    values[0] = Int64GetDatum(static_cast<int64>(hilbert.lo));
    values[1] = Int64GetDatum(static_cast<int64>(hilbert.hi));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/**
 * Convert Hilbert index to 4D coordinates
 * hypercube_hilbert_to_coords(hilbert_lo int8, hilbert_hi int8)
 *   RETURNS TABLE(x int8, y int8, z int8, m int8)
 */
PG_FUNCTION_INFO_V1(hypercube_hilbert_to_coords);
Datum hypercube_hilbert_to_coords(PG_FUNCTION_ARGS) {
    int64 lo = PG_GETARG_INT64(0);
    int64 hi = PG_GETARG_INT64(1);
    
    HilbertIndex hilbert(static_cast<uint64_t>(lo), static_cast<uint64_t>(hi));
    Point4D point = HilbertCurve::index_to_coords(hilbert);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    
    values[0] = Int64GetDatum(static_cast<int64>(point.x));
    values[1] = Int64GetDatum(static_cast<int64>(point.y));
    values[2] = Int64GetDatum(static_cast<int64>(point.z));
    values[3] = Int64GetDatum(static_cast<int64>(point.m));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/**
 * BLAKE3 hash of data
 * hypercube_blake3(data bytea) RETURNS bytea
 */
PG_FUNCTION_INFO_V1(hypercube_blake3);
Datum hypercube_blake3(PG_FUNCTION_ARGS) {
    bytea* input = PG_GETARG_BYTEA_PP(0);
    
    size_t len = VARSIZE_ANY_EXHDR(input);
    const uint8_t* data = reinterpret_cast<const uint8_t*>(VARDATA_ANY(input));
    
    Blake3Hash hash = Blake3Hasher::hash(std::span<const uint8_t>(data, len));
    
    // Allocate result bytea
    bytea* result = static_cast<bytea*>(palloc(VARHDRSZ + 32));
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash.bytes.data(), 32);
    
    PG_RETURN_BYTEA_P(result);
}

/**
 * BLAKE3 hash of a codepoint (UTF-8 encoded)
 * hypercube_blake3_codepoint(codepoint int4) RETURNS bytea
 */
PG_FUNCTION_INFO_V1(hypercube_blake3_codepoint);
Datum hypercube_blake3_codepoint(PG_FUNCTION_ARGS) {
    int32 codepoint = PG_GETARG_INT32(0);
    
    Blake3Hash hash = Blake3Hasher::hash_codepoint(static_cast<uint32_t>(codepoint));
    
    bytea* result = static_cast<bytea*>(palloc(VARHDRSZ + 32));
    SET_VARSIZE(result, VARHDRSZ + 32);
    memcpy(VARDATA(result), hash.bytes.data(), 32);
    
    PG_RETURN_BYTEA_P(result);
}

/**
 * Map a Unicode codepoint to its 4D coordinates and Hilbert index
 * hypercube_map_codepoint(codepoint int4)
 *   RETURNS TABLE(x int8, y int8, z int8, m int8, hilbert_lo int8, hilbert_hi int8, hash bytea, category text)
 */
PG_FUNCTION_INFO_V1(hypercube_map_codepoint);
Datum hypercube_map_codepoint(PG_FUNCTION_ARGS) {
    int32 codepoint = PG_GETARG_INT32(0);
    uint32_t cp = static_cast<uint32_t>(codepoint);
    
    // Get coordinates, category, and hash
    Point4D coords = CoordinateMapper::map_codepoint(cp);
    AtomCategory cat = CoordinateMapper::categorize(cp);
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
    Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[8];
    bool nulls[8] = {false, false, false, false, false, false, false, false};
    
    values[0] = Int64GetDatum(static_cast<int64>(coords.x));
    values[1] = Int64GetDatum(static_cast<int64>(coords.y));
    values[2] = Int64GetDatum(static_cast<int64>(coords.z));
    values[3] = Int64GetDatum(static_cast<int64>(coords.m));
    values[4] = Int64GetDatum(static_cast<int64>(hilbert.lo));
    values[5] = Int64GetDatum(static_cast<int64>(hilbert.hi));
    
    // Hash as bytea
    bytea* hash_bytea = static_cast<bytea*>(palloc(VARHDRSZ + 32));
    SET_VARSIZE(hash_bytea, VARHDRSZ + 32);
    memcpy(VARDATA(hash_bytea), hash.bytes.data(), 32);
    values[6] = PointerGetDatum(hash_bytea);
    
    // Category as text
    const char* cat_str = category_to_string(cat);
    values[7] = CStringGetTextDatum(cat_str);
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/**
 * Get category for a codepoint
 * hypercube_categorize(codepoint int4) RETURNS text
 */
PG_FUNCTION_INFO_V1(hypercube_categorize);
Datum hypercube_categorize(PG_FUNCTION_ARGS) {
    int32 codepoint = PG_GETARG_INT32(0);
    AtomCategory cat = CoordinateMapper::categorize(static_cast<uint32_t>(codepoint));
    PG_RETURN_TEXT_P(cstring_to_text(category_to_string(cat)));
}

/**
 * Seed all Unicode atoms
 * Returns a set of all valid Unicode codepoints with their coordinates, Hilbert indices, and hashes
 * 
 * hypercube_seed_atoms() RETURNS SETOF RECORD
 */

// State for set-returning function
typedef struct {
    uint32_t current_codepoint;
    TupleDesc tupdesc;
} SeedAtomsState;

PG_FUNCTION_INFO_V1(hypercube_seed_atoms);
Datum hypercube_seed_atoms(PG_FUNCTION_ARGS) {
    FuncCallContext* funcctx;
    SeedAtomsState* state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        // Get result tuple descriptor
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("function returning record called in context that cannot accept type record")));
        }
        
        state = static_cast<SeedAtomsState*>(palloc(sizeof(SeedAtomsState)));
        state->current_codepoint = 0;
        state->tupdesc = BlessTupleDesc(tupdesc);
        
        funcctx->user_fctx = state;
        funcctx->max_calls = constants::MAX_CODEPOINT + 1;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = static_cast<SeedAtomsState*>(funcctx->user_fctx);
    
    // Skip surrogates
    while (state->current_codepoint >= constants::SURROGATE_START && 
           state->current_codepoint <= constants::SURROGATE_END) {
        state->current_codepoint++;
    }
    
    if (state->current_codepoint <= constants::MAX_CODEPOINT) {
        uint32_t cp = state->current_codepoint++;
        
        // Compute all values
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        AtomCategory cat = CoordinateMapper::categorize(cp);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
        
        // Build tuple: (codepoint, x, y, z, m, hilbert_lo, hilbert_hi, hash, category)
        Datum values[9];
        bool nulls[9] = {false, false, false, false, false, false, false, false, false};
        
        values[0] = Int32GetDatum(static_cast<int32>(cp));
        values[1] = Int64GetDatum(static_cast<int64>(coords.x));
        values[2] = Int64GetDatum(static_cast<int64>(coords.y));
        values[3] = Int64GetDatum(static_cast<int64>(coords.z));
        values[4] = Int64GetDatum(static_cast<int64>(coords.m));
        values[5] = Int64GetDatum(static_cast<int64>(hilbert.lo));
        values[6] = Int64GetDatum(static_cast<int64>(hilbert.hi));
        
        bytea* hash_bytea = static_cast<bytea*>(palloc(VARHDRSZ + 32));
        SET_VARSIZE(hash_bytea, VARHDRSZ + 32);
        memcpy(VARDATA(hash_bytea), hash.bytes.data(), 32);
        values[7] = PointerGetDatum(hash_bytea);
        
        values[8] = CStringGetTextDatum(category_to_string(cat));
        
        HeapTuple tuple = heap_form_tuple(state->tupdesc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/**
 * Compute centroid of multiple 4D points
 * hypercube_centroid(x int8[], y int8[], z int8[], m int8[])
 *   RETURNS TABLE(x int8, y int8, z int8, m int8)
 */
PG_FUNCTION_INFO_V1(hypercube_centroid);
Datum hypercube_centroid(PG_FUNCTION_ARGS) {
    ArrayType* x_arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* y_arr = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType* z_arr = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType* m_arr = PG_GETARG_ARRAYTYPE_P(3);
    
    int n = ArrayGetNItems(ARR_NDIM(x_arr), ARR_DIMS(x_arr));
    
    if (n == 0) {
        PG_RETURN_NULL();
    }
    
    int64* x_data = reinterpret_cast<int64*>(ARR_DATA_PTR(x_arr));
    int64* y_data = reinterpret_cast<int64*>(ARR_DATA_PTR(y_arr));
    int64* z_data = reinterpret_cast<int64*>(ARR_DATA_PTR(z_arr));
    int64* m_data = reinterpret_cast<int64*>(ARR_DATA_PTR(m_arr));
    
    std::vector<Point4D> points;
    points.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        points.emplace_back(
            static_cast<uint32_t>(x_data[i]),
            static_cast<uint32_t>(y_data[i]),
            static_cast<uint32_t>(z_data[i]),
            static_cast<uint32_t>(m_data[i])
        );
    }
    
    Point4D centroid = CoordinateMapper::centroid(points);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    
    values[0] = Int64GetDatum(static_cast<int64>(centroid.x));
    values[1] = Int64GetDatum(static_cast<int64>(centroid.y));
    values[2] = Int64GetDatum(static_cast<int64>(centroid.z));
    values[3] = Int64GetDatum(static_cast<int64>(centroid.m));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/**
 * Check if a point is on the hypercube surface
 * hypercube_is_on_surface(x int8, y int8, z int8, m int8) RETURNS bool
 */
PG_FUNCTION_INFO_V1(hypercube_is_on_surface);
Datum hypercube_is_on_surface(PG_FUNCTION_ARGS) {
    int64 x = PG_GETARG_INT64(0);
    int64 y = PG_GETARG_INT64(1);
    int64 z = PG_GETARG_INT64(2);
    int64 m = PG_GETARG_INT64(3);
    
    Point4D point(
        static_cast<uint32_t>(x),
        static_cast<uint32_t>(y),
        static_cast<uint32_t>(z),
        static_cast<uint32_t>(m)
    );
    
    PG_RETURN_BOOL(point.is_on_surface());
}

/**
 * Hilbert distance between two indices
 * hypercube_hilbert_distance(lo1 int8, hi1 int8, lo2 int8, hi2 int8)
 *   RETURNS TABLE(lo int8, hi int8)
 */
PG_FUNCTION_INFO_V1(hypercube_hilbert_distance);
Datum hypercube_hilbert_distance(PG_FUNCTION_ARGS) {
    int64 lo1 = PG_GETARG_INT64(0);
    int64 hi1 = PG_GETARG_INT64(1);
    int64 lo2 = PG_GETARG_INT64(2);
    int64 hi2 = PG_GETARG_INT64(3);
    
    HilbertIndex a(static_cast<uint64_t>(lo1), static_cast<uint64_t>(hi1));
    HilbertIndex b(static_cast<uint64_t>(lo2), static_cast<uint64_t>(hi2));
    HilbertIndex dist = HilbertCurve::distance(a, b);
    
    TupleDesc tupdesc;
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    }
    tupdesc = BlessTupleDesc(tupdesc);
    
    Datum values[2];
    bool nulls[2] = {false, false};
    
    values[0] = Int64GetDatum(static_cast<int64>(dist.lo));
    values[1] = Int64GetDatum(static_cast<int64>(dist.hi));
    
    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

} // extern "C"
