/*-------------------------------------------------------------------------
 *
 * generic-msvc.h
 *        Atomic operations support when using MSVC
 *
 * Portions Copyright (c) 1996-2025, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * PATCHED VERSION: This file has been patched to work with VS2026+ which
 * has stricter type checking for interlocked intrinsics. The Windows SDK
 * expects LONG/LONG64* types for interlocked operations, but PostgreSQL
 * uses uint32/uint64. We add explicit casts to fix the type mismatch.
 *
 * NOTES:
 *
 * Documentation:
 * * Interlocked Variable Access
 *   http://msdn.microsoft.com/en-us/library/ms684122%28VS.85%29.aspx
 *
 * src/include/port/atomics/generic-msvc.h
 *
 *-------------------------------------------------------------------------
 */
#include <intrin.h>

/* intentionally no include guards, should only be included by atomics.h */
#ifndef INSIDE_ATOMICS_H
#error "should be included via atomics.h"
#endif

#pragma intrinsic(_ReadWriteBarrier)
#define pg_compiler_barrier_impl()      _ReadWriteBarrier()

#ifndef pg_memory_barrier_impl
#define pg_memory_barrier_impl()        MemoryBarrier()
#endif

#define PG_HAVE_ATOMIC_U32_SUPPORT
typedef struct pg_atomic_uint32
{
        volatile uint32 value;
} pg_atomic_uint32;

#define PG_HAVE_ATOMIC_U64_SUPPORT
typedef struct pg_attribute_aligned(8) pg_atomic_uint64
{
        volatile uint64 value;
} pg_atomic_uint64;


#define PG_HAVE_ATOMIC_COMPARE_EXCHANGE_U32
static inline bool
pg_atomic_compare_exchange_u32_impl(volatile pg_atomic_uint32 *ptr,
                                                                        uint32 *expected, uint32 newval)
{
        bool    ret;
        uint32  current;
        /* Cast to LONG* for VS2026+ compatibility */
        current = InterlockedCompareExchange((volatile LONG *)&ptr->value, (LONG)newval, (LONG)*expected);
        ret = current == *expected;
        *expected = current;
        return ret;
}

#define PG_HAVE_ATOMIC_EXCHANGE_U32
static inline uint32
pg_atomic_exchange_u32_impl(volatile pg_atomic_uint32 *ptr, uint32 newval)
{
        /* Cast to LONG* for VS2026+ compatibility */
        return InterlockedExchange((volatile LONG *)&ptr->value, (LONG)newval);
}

#define PG_HAVE_ATOMIC_FETCH_ADD_U32
static inline uint32
pg_atomic_fetch_add_u32_impl(volatile pg_atomic_uint32 *ptr, int32 add_)
{
        /* Cast to LONG* for VS2026+ compatibility */
        return InterlockedExchangeAdd((volatile LONG *)&ptr->value, (LONG)add_);
}

/*
 * The non-intrinsics versions are only available in vista upwards, so use the
 * intrinsic version. Only supported on >486, but we require XP as a minimum
 * baseline, which doesn't support the 486, so we don't need to add checks for
 * that case.
 */
#pragma intrinsic(_InterlockedCompareExchange64)

#define PG_HAVE_ATOMIC_COMPARE_EXCHANGE_U64
static inline bool
pg_atomic_compare_exchange_u64_impl(volatile pg_atomic_uint64 *ptr,
                                                                        uint64 *expected, uint64 newval)
{
        bool    ret;
        uint64  current;
        /* Cast to LONG64* for VS2026+ compatibility - LONG64 is __int64 which is same size as uint64 */
        current = (uint64)_InterlockedCompareExchange64((volatile LONG64 *)&ptr->value, (LONG64)newval, (LONG64)*expected);
        ret = current == *expected;
        *expected = current;
        return ret;
}

/* Only implemented on 64bit builds */
#ifdef _WIN64

#pragma intrinsic(_InterlockedExchange64)

#define PG_HAVE_ATOMIC_EXCHANGE_U64
static inline uint64
pg_atomic_exchange_u64_impl(volatile pg_atomic_uint64 *ptr, uint64 newval)
{
        /* Cast to LONG64* for VS2026+ compatibility */
        return (uint64)_InterlockedExchange64((volatile LONG64 *)&ptr->value, (LONG64)newval);
}

#pragma intrinsic(_InterlockedExchangeAdd64)

#define PG_HAVE_ATOMIC_FETCH_ADD_U64
static inline uint64
pg_atomic_fetch_add_u64_impl(volatile pg_atomic_uint64 *ptr, int64 add_)
{
        /* Cast to LONG64* for VS2026+ compatibility */
        return (uint64)_InterlockedExchangeAdd64((volatile LONG64 *)&ptr->value, (LONG64)add_);
}

#endif /* _WIN64 */
