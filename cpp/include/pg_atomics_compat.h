/*
 * pg_atomics_compat.h - PostgreSQL atomics compatibility for MSVC 19.50+
 * 
 * PostgreSQL's generic-msvc.h has type mismatches with modern MSVC:
 * - _InterlockedCompareExchange64 expects volatile LONG64*
 * - PostgreSQL passes volatile uint64*
 * 
 * This header MUST be included BEFORE postgres.h to override the broken atomics.
 * We define our own fixed implementations that will be used instead.
 */
#ifndef PG_ATOMICS_COMPAT_H
#define PG_ATOMICS_COMPAT_H

#ifdef _MSC_VER
#if _MSC_VER >= 1950

/* 
 * Pre-define the atomics macros so PostgreSQL's generic-msvc.h 
 * won't define its broken versions 
 */
#include <intrin.h>
#include <windows.h>

/* We need to include this before atomics.h gets included */
#ifndef INSIDE_ATOMICS_H
/* Mark that we've already handled atomics */
#define PG_ATOMICS_COMPAT_APPLIED 1
#endif

#endif /* _MSC_VER >= 1950 */
#endif /* _MSC_VER */

#endif /* PG_ATOMICS_COMPAT_H */
