/**
 * hypercube_c.h - C API for Hypercube Core Library
 * 
 * This header provides a pure C interface to the C++ hypercube library.
 * It is designed to be included by PostgreSQL extensions (which are compiled as C)
 * and any other C code that needs to use hypercube functionality.
 * 
 * Architecture:
 *   - C++ core library: Heavy computation, optimized, no PostgreSQL dependencies
 *   - C API (this file): extern "C" bridge between C++ and C
 *   - PostgreSQL extension: Pure C, includes PG headers, calls this C API
 * 
 * This design avoids PostgreSQL header incompatibilities with modern C++ compilers.
 */

#ifndef HYPERCUBE_C_H
#define HYPERCUBE_C_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* DLL export/import macros for Windows */
#ifdef _WIN32
    #ifdef HYPERCUBE_C_EXPORTS
        #define HC_API __declspec(dllexport)
    #else
        #define HC_API __declspec(dllimport)
    #endif
#else
    #define HC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

/** 4D point with 32-bit coordinates per dimension */
typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t m;
} hc_point4d_t;

/** 128-bit Hilbert curve index (two 64-bit parts) */
typedef struct {
    uint64_t lo;  /**< Lower 64 bits */
    uint64_t hi;  /**< Upper 64 bits */
} hc_hilbert_t;

/** BLAKE3 hash (32 bytes) */
typedef struct {
    uint8_t bytes[32];
} hc_hash_t;

/** Atom category enumeration */
typedef enum {
    HC_CAT_CONTROL = 0,
    HC_CAT_FORMAT,
    HC_CAT_PRIVATE_USE,
    HC_CAT_SURROGATE,
    HC_CAT_NONCHARACTER,
    HC_CAT_SPACE,
    HC_CAT_PUNCTUATION_OPEN,
    HC_CAT_PUNCTUATION_CLOSE,
    HC_CAT_PUNCTUATION_OTHER,
    HC_CAT_DIGIT,
    HC_CAT_NUMBER_LETTER,
    HC_CAT_MATH_SYMBOL,
    HC_CAT_CURRENCY,
    HC_CAT_MODIFIER,
    HC_CAT_LETTER_UPPER,
    HC_CAT_LETTER_LOWER,
    HC_CAT_LETTER_TITLECASE,
    HC_CAT_LETTER_MODIFIER,
    HC_CAT_LETTER_OTHER,
    HC_CAT_MARK_NONSPACING,
    HC_CAT_MARK_SPACING,
    HC_CAT_MARK_ENCLOSING,
    HC_CAT_SYMBOL_OTHER,
    HC_CAT_SEPARATOR,
    HC_CAT_COUNT
} hc_category_t;

/* ============================================================================
 * Constants
 * ============================================================================ */

#define HC_MAX_CODEPOINT    0x10FFFF
#define HC_SURROGATE_START  0xD800
#define HC_SURROGATE_END    0xDFFF
#define HC_HASH_SIZE        32

/* ============================================================================
 * Hilbert Curve Functions
 * ============================================================================ */

/**
 * Convert 4D coordinates to 128-bit Hilbert index
 * @param point 4D point with 32-bit coordinates per dimension
 * @return 128-bit Hilbert index
 */
HC_API hc_hilbert_t hc_coords_to_hilbert(hc_point4d_t point);

/**
 * Convert 128-bit Hilbert index to 4D coordinates
 * @param index 128-bit Hilbert index
 * @return 4D point with 32-bit coordinates per dimension
 */
HC_API hc_point4d_t hc_hilbert_to_coords(hc_hilbert_t index);

/**
 * Compute absolute distance between two Hilbert indices
 * @param a First index
 * @param b Second index
 * @return Absolute distance as 128-bit value
 */
HC_API hc_hilbert_t hc_hilbert_distance(hc_hilbert_t a, hc_hilbert_t b);

/**
 * Compare two Hilbert indices
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
HC_API int hc_hilbert_compare(hc_hilbert_t a, hc_hilbert_t b);

/* ============================================================================
 * Coordinate Mapping Functions
 * ============================================================================ */

/**
 * Map a Unicode codepoint to its 4D coordinates on the 3-sphere surface
 * @param codepoint Unicode codepoint (0 to 0x10FFFF)
 * @return 4D point on 3-sphere surface
 */
HC_API hc_point4d_t hc_map_codepoint(uint32_t codepoint);

/**
 * Determine category of a Unicode codepoint
 * @param codepoint Unicode codepoint
 * @return Category enum value
 */
HC_API hc_category_t hc_categorize(uint32_t codepoint);

/**
 * Get string name for a category
 * @param cat Category enum value
 * @return Null-terminated string (static, do not free)
 */
HC_API const char* hc_category_name(hc_category_t cat);

/**
 * Calculate centroid of multiple points
 * @param points Array of 4D points
 * @param count Number of points
 * @return Centroid point
 */
HC_API hc_point4d_t hc_centroid(const hc_point4d_t* points, size_t count);

/**
 * Calculate weighted centroid
 * @param points Array of 4D points
 * @param weights Array of weights (same length as points)
 * @param count Number of points
 * @return Weighted centroid point
 */
HC_API hc_point4d_t hc_weighted_centroid(const hc_point4d_t* points, 
                                   const double* weights, 
                                   size_t count);

/**
 * Check if a point is on the 3-sphere surface
 * @param point 4D point to check
 * @return true if on surface (r² ≈ 1 within tolerance)
 */
HC_API bool hc_is_on_surface(hc_point4d_t point);


/**
 * Calculate Euclidean distance in 4D space
 */
HC_API double hc_euclidean_distance(hc_point4d_t a, hc_point4d_t b);

/* ============================================================================
 * BLAKE3 Hashing Functions
 * ============================================================================ */

/**
 * Hash arbitrary data with BLAKE3
 * @param data Input bytes
 * @param len Length of input
 * @return 32-byte BLAKE3 hash
 */
HC_API hc_hash_t hc_blake3(const uint8_t* data, size_t len);

/**
 * Hash a string with BLAKE3
 * @param str Null-terminated string
 * @return 32-byte BLAKE3 hash
 */
HC_API hc_hash_t hc_blake3_str(const char* str);

/**
 * Hash a Unicode codepoint (as UTF-8)
 * @param codepoint Unicode codepoint
 * @return BLAKE3 hash of UTF-8 encoding
 */
HC_API hc_hash_t hc_blake3_codepoint(uint32_t codepoint);

/**
 * Hash a sequence of child hashes (for Merkle DAG)
 * @param children Array of child hashes
 * @param count Number of children
 * @return BLAKE3 hash of concatenated children
 */
HC_API hc_hash_t hc_blake3_children(const hc_hash_t* children, size_t count);

/**
 * Hash children with ordinals (position-sensitive)
 * Format: [ordinal_0][hash_0][ordinal_1][hash_1]...
 */
HC_API hc_hash_t hc_blake3_children_ordered(const hc_hash_t* children, size_t count);

/**
 * Compute content hash for UTF-8 text using CPE cascade
 * This is the deterministic Merkle DAG root hash for any text content.
 * Uses atom hashes from the database via callback.
 * 
 * @param text UTF-8 encoded text
 * @param len Length of text in bytes
 * @param atom_hashes Array of atom hashes (one per codepoint)
 * @param atom_count Number of atoms
 * @return Root hash of CPE Merkle DAG
 */
HC_API hc_hash_t hc_content_hash(const uint8_t* text, size_t len,
                                  const hc_hash_t* atom_hashes, size_t atom_count);

/**
 * Compute content hash from codepoint array
 * @param codepoints Array of Unicode codepoints
 * @param count Number of codepoints
 * @param atom_hashes Array of atom hashes (must match codepoints)
 * @return Root hash of CPE Merkle DAG
 */
HC_API hc_hash_t hc_content_hash_codepoints(const uint32_t* codepoints, size_t count,
                                             const hc_hash_t* atom_hashes);

/**
 * Convert hash to hex string
 * @param hash Hash to convert
 * @param out Output buffer (must be at least 65 bytes for null terminator)
 */
HC_API void hc_hash_to_hex(hc_hash_t hash, char* out);

/**
 * Parse hash from hex string
 * @param hex 64-character hex string
 * @return Parsed hash (zero if invalid)
 */
HC_API hc_hash_t hc_hash_from_hex(const char* hex);

/**
 * Compare two hashes
 * @return 0 if equal, non-zero if different
 */
HC_API int hc_hash_compare(hc_hash_t a, hc_hash_t b);

/**
 * Check if hash is all zeros
 */
HC_API bool hc_hash_is_zero(hc_hash_t hash);

/* ============================================================================
 * Batch Processing Functions (High Performance)
 * ============================================================================ */

/**
 * Map multiple codepoints to coordinates (batch, parallelized)
 * @param codepoints Array of codepoints
 * @param count Number of codepoints
 * @param out_points Output array (must be pre-allocated with 'count' elements)
 */
HC_API void hc_map_codepoints_batch(const uint32_t* codepoints, 
                              size_t count, 
                              hc_point4d_t* out_points);

/**
 * Hash multiple codepoints (batch, parallelized)
 * @param codepoints Array of codepoints
 * @param count Number of codepoints
 * @param out_hashes Output array (must be pre-allocated with 'count' elements)
 */
HC_API void hc_hash_codepoints_batch(const uint32_t* codepoints, 
                               size_t count, 
                               hc_hash_t* out_hashes);

/**
 * Convert multiple coordinates to Hilbert indices (batch, parallelized)
 */
HC_API void hc_coords_to_hilbert_batch(const hc_point4d_t* points,
                                 size_t count,
                                 hc_hilbert_t* out_indices);

/**
 * Full atom mapping: codepoint -> (coords, hilbert, hash, category)
 * Used for seeding all Unicode atoms efficiently
 */
typedef struct {
    uint32_t codepoint;
    hc_point4d_t coords;
    hc_hilbert_t hilbert;
    hc_hash_t hash;
    hc_category_t category;
} hc_atom_t;

/**
 * Map a single codepoint to full atom data
 */
HC_API hc_atom_t hc_map_atom(uint32_t codepoint);

/**
 * Map multiple codepoints to full atom data (batch, parallelized)
 * @param codepoints Array of codepoints
 * @param count Number of codepoints
 * @param out_atoms Output array (must be pre-allocated with 'count' elements)
 */
HC_API void hc_map_atoms_batch(const uint32_t* codepoints,
                         size_t count,
                         hc_atom_t* out_atoms);

/**
 * Get the number of valid Unicode codepoints (excluding surrogates)
 */
HC_API uint32_t hc_valid_codepoint_count(void);

/**
 * Iterator for all valid codepoints
 * Returns the next valid codepoint after 'current', or (uint32_t)-1 if done
 * Start with current = (uint32_t)-1 to get first codepoint (0)
 */
HC_API uint32_t hc_next_codepoint(uint32_t current);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Allocate memory that can be freed by hc_free
 * Use for buffers that will be passed to hypercube functions
 */
HC_API void* hc_alloc(size_t size);

/**
 * Free memory allocated by hc_alloc or returned by hypercube functions
 */
HC_API void hc_free(void* ptr);

/* ============================================================================
 * Optimized Batch Operations (SIMD + Threading)
 * ============================================================================ */

/**
 * Batch 4D Euclidean distance calculation with SIMD
 * @param target Target centroid (4 doubles: x, y, z, m)
 * @param points Array of point centroids (N*4 doubles, interleaved XYZM)
 * @param count Number of points
 * @param distances_out Output distances (N doubles)
 */
HC_API void hc_batch_distances(const double* target, const double* points,
                               size_t count, double* distances_out);

/**
 * Find k-nearest neighbors from distance array
 * @param distances Array of distances
 * @param count Number of distances
 * @param k Number of neighbors to find
 * @param out_indices Output indices (k elements)
 * @param out_distances Output distances (k elements)
 * @return Actual number of neighbors found (may be less than k)
 */
HC_API size_t hc_find_knn(const double* distances, size_t count, size_t k,
                          size_t* out_indices, double* out_distances);

/**
 * Discrete Fréchet distance between two trajectories
 * @param traj1 First trajectory (n1*4 doubles, interleaved XYZM)
 * @param n1 Number of points in first trajectory
 * @param traj2 Second trajectory (n2*4 doubles, interleaved XYZM)
 * @param n2 Number of points in second trajectory
 * @return Fréchet distance
 */
HC_API double hc_frechet_distance(const double* traj1, size_t n1,
                                  const double* traj2, size_t n2);

/**
 * Jaccard similarity between two sets of neighbors
 * @param set1 First set of hash IDs
 * @param count1 Number of elements in first set
 * @param set2 Second set of hash IDs
 * @param count2 Number of elements in second set
 * @return Jaccard similarity (0.0 to 1.0)
 */
HC_API double hc_jaccard_similarity(const hc_hash_t* set1, size_t count1,
                                    const hc_hash_t* set2, size_t count2);

/**
 * Analogy vector arithmetic: D = C + B - A
 * @param a Centroid of A (4 doubles)
 * @param b Centroid of B (4 doubles)
 * @param c Centroid of C (4 doubles)
 * @param out_d Output centroid D (4 doubles)
 */
HC_API void hc_analogy_vector(const double* a, const double* b, 
                              const double* c, double* out_d);

/**
 * Get number of hardware threads available
 */
HC_API size_t hc_thread_count(void);

/**
 * Seed atoms table with all Unicode codepoints (parallel seeding)
 * @param conninfo PostgreSQL connection string
 * @return 0 on success, non-zero on error
 */
HC_API int hc_seed_atoms_parallel(const char* conninfo);

#ifdef __cplusplus
}
#endif

#endif /* HYPERCUBE_C_H */
