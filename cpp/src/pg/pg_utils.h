/**
 * pg_utils.h - Shared PostgreSQL Extension Utilities
 * 
 * Common helper functions for all Hypercube PostgreSQL extensions.
 * Include this header to avoid duplicating bytea/hash conversion code.
 */

#ifndef PG_UTILS_H
#define PG_UTILS_H

#include "postgres.h"
#include "fmgr.h"
#include <string.h>

#define HASH_SIZE 32

/* Convert PostgreSQL bytea to raw hash bytes - handles TOAST safely 
 * Note: Caller is responsible for memory management of b.
 * This function may detoast which allocates in CurrentMemoryContext.
 */
static inline void bytea_to_hash(bytea *b, uint8 *out)
{
    /* Detoast if necessary - allocates in CurrentMemoryContext */
    bytea *detoasted = (bytea *)PG_DETOAST_DATUM(PointerGetDatum(b));
    int len = VARSIZE(detoasted) - VARHDRSZ;
    if (len >= HASH_SIZE) {
        memcpy(out, VARDATA(detoasted), HASH_SIZE);
    } else {
        memset(out, 0, HASH_SIZE);
        if (len > 0)
            memcpy(out, VARDATA(detoasted), len);
    }
    /* Only free if we actually created a copy (different pointer) */
    if ((Pointer)detoasted != (Pointer)b)
        pfree(detoasted);
}

/* Convert raw hash bytes to PostgreSQL bytea */
static inline bytea *hash_to_bytea(const uint8 *hash)
{
    bytea *result = (bytea *)palloc(VARHDRSZ + HASH_SIZE);
    SET_VARSIZE(result, VARHDRSZ + HASH_SIZE);
    memcpy(VARDATA(result), hash, HASH_SIZE);
    return result;
}

/* Compare two hash byte arrays */
static inline bool hash_equals(const uint8 *a, const uint8 *b)
{
    return memcmp(a, b, HASH_SIZE) == 0;
}

/* Check if hash is all zeros */
static inline bool hash_is_zero(const uint8 *hash)
{
    for (int i = 0; i < HASH_SIZE; i++) {
        if (hash[i] != 0) return false;
    }
    return true;
}

/* Convert hash to hex string (caller must provide 65-byte buffer) */
static inline void hash_to_hex(const uint8 *hash, char *out)
{
    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < HASH_SIZE; i++) {
        out[i * 2] = hex_chars[hash[i] >> 4];
        out[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out[64] = '\0';
}

#endif /* PG_UTILS_H */
