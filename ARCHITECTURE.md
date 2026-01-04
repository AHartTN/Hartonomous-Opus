# Hartonomous Hypercube - Architecture Document

**Last Updated**: 2026-01-03
**Status**: Canonical - N-ary Merkle DAG (v4)

---

## Executive Summary

The Hypercube is a **deterministic, lossless, content-addressable geometric semantic substrate**. All digital content is decomposed into a Merkle DAG where:

1. **Atoms** (Unicode codepoints) form the perimeter landmarks at fixed 4D coordinates (POINTZM)
2. **Compositions** are N-ary ordered sequences (LINESTRINGZM) - NOT binary trees
3. **Children** array stores references to component atoms/compositions in order
4. **Deduplication** is global - identical patterns share the same hash regardless of source
5. **Reconstruction** is bit-perfect via DFS traversal of children arrays

### The Single Table Model (IMPLEMENTED)

The system uses a **single `atom` table** where:
- **POINTZM** = Unicode codepoints (the seeded perimeter - depth 0)
- **LINESTRINGZM** = Compositions (ordered path through N child centroids - depth > 0)

The `geom GEOMETRY(GEOMETRYZM, 0)` column stores both geometry types.
The `children BYTEA[]` column stores N child hash references (any number, not just 2).

---

## Core Principles

### 1. Determinism
- Same bytes → same composition ID, always
- No randomness, no floating-point conversion, no approximations
- Hash = BLAKE3 of canonical ordered child sequence

### 2. Losslessness
- Bit-perfect reconstruction from root composition
- DFS traversal of children → original byte sequence
- All coordinates stored as 32-bit signed integers (bit pattern same as uint32)

### 3. Global Deduplication
- "the" from Moby Dick = "the" from a children's book = same composition ID
- First ingest creates the pattern; subsequent ingests only add references
- The more you ingest, the more deduplication occurs

### 4. N-ary Compositions (NOT Binary Trees)

**"the" = ONE composition with children [t, h, e]**

This is fundamentally different from binary tree nonsense:
- Binary tree (WRONG): `th` = `[t, h]`, then `the` = `[th, e]` - artificial nesting
- N-ary (CORRECT): `the` = `[t, h, e]` - one composition, three children

**Tokenization determines composition boundaries:**
- Words: sequences of codepoints between whitespace
- Sentences: sequences of words ending in sentence punctuation
- Paragraphs: sequences of sentences
- Documents: hierarchical composition of all content

**LINESTRINGZM geometry** represents the ordered path through child centroids:
- `ST_MakeLine(child1.centroid, child2.centroid, ..., childN.centroid)`
- The trajectory shape IS semantic information (Fréchet distance for similarity)

### 5. Hypersphere Geometry

**Leaves on perimeter, compositions move inward:**
- Depth 0 (atoms): On the outer surface of the 4D hypersphere
- Depth 1 (words): Centroid of component atoms, radius decreases
- Depth N: Further inward, approaching origin as complexity increases

**Origin = most abstract/complex, Perimeter = most atomic**

### 6. Emergent Topology as Semantics

**The structure IS the meaning.** This is fundamentally different from:
- Vector embeddings (opaque dimensions)
- Probability distributions (training artifacts)
- Learned projections (black box)

Semantic signal emerges from:
- **Connectivity**: How many compositions include this atom?
- **Trajectory shape**: What path through 4D space? (Fréchet distance)
- **Neighborhood density**: How clustered are connections?
- **Path multiplicity**: How many ways to reach X from Y?

---

## Data Model

### Unified Atom Table
```sql
atom (
    -- Content-addressed identifier
    id              BYTEA PRIMARY KEY,      -- BLAKE3 hash (32 bytes)

    -- Geometry: POINTZM for leaves, LINESTRINGZM for compositions
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL,
    centroid        GEOMETRY(POINTZM, 0),   -- Pre-computed 4D centroid

    -- Child references for compositions (NULL for leaves)
    children        BYTEA[],                -- Array of N child hashes in order

    -- Canonical value for leaves only (UTF-8 bytes)
    value           BYTEA,                  -- NULL for compositions

    -- Unicode codepoint for leaf atoms (NULL for compositions)
    codepoint       INTEGER UNIQUE,

    -- 128-bit Hilbert index (from centroid -> 4D coords)
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,

    -- Depth in DAG (0 = leaf, 1 = word, 2 = sentence, etc.)
    depth           INTEGER NOT NULL DEFAULT 0,

    -- Total leaf atoms in subtree (1 for leaves)
    atom_count      BIGINT NOT NULL DEFAULT 1
)
```

**Key Points**:
- Single table for ALL content - atoms AND compositions
- `depth = 0` means leaf (POINTZM geometry, has value and codepoint)
- `depth > 0` means composition (LINESTRINGZM geometry, has N children)
- LINESTRINGZM vertices are child centroids in sequence order
- ST_Centroid(geom) gives the 4D centroid for Hilbert encoding
- DFS traversal of children arrays = original byte sequence

---

## Ingestion Pipeline

### Universal Substrate Ingester (C++ Implementation)

```
Input → Integer Sequence → Sliding Window → Vocabulary Match → Hierarchical Tiers → Batch Insert
```

**The substrate is COMPLETELY AGNOSTIC.** It doesn't know or care what it's ingesting:

| Input Type | Tier 0 Alphabet | Example |
|------------|-----------------|---------|
| Text | Unicode codepoints (1.1M) | "Hello" = [H,e,l,l,o] |
| Binary | Byte values (256) | [0x48,0x65,0x6c,0x6c,0x6f] |
| Audio | Sample amplitudes | [-32768...32767] or [0...65535] |
| Image | Pixel values | [0...255] per channel |
| Numbers | Codepoints | "0.987" = [0,.,9,8,7] |

**All are just sequences of integers.** Same algorithm, same storage, same semantics.

### Greedy Pattern Matching (Vocabulary-driven)

```
Tier 0: Atoms (seeded alphabet - Unicode codepoints, bytes, etc.)
        "The cat sat" → [T,h,e, ,c,a,t, ,s,a,t] (11 atoms, including spaces)
        
Tier 1+: N-ary compositions discovered through greedy matching
         First pass (empty vocab): composition([T,h,e, ,c,a,t, ,s,a,t])
         After learning "the", "cat", "sat" patterns:
           composition([The, ,cat, ,sat]) where The=composition([T,h,e])
```

**No linguistic rules. No language detection. No special cases.**

- "Hello world" = H,e,l,l,o, ,w,o,r,l,d (11 atoms) - space is codepoint 32
- "public class" = p,u,b,l,i,c, ,c,l,a,s,s (12 atoms)  
- "{ get; set; }" = {, ,g,e,t,;, ,s,e,t,;, ,} (13 atoms)
- "0.987" = 0,.,9,8,7 (5 atoms)
- Attention weight 0.987 in a model? Same thing - text representation.

**The vocabulary trie builds organically through usage:**
1. Ingest content → greedy match against known vocabulary
2. Unmatched sequences become new compositions
3. New compositions added to vocabulary trie
4. Next ingest → trie matches known patterns, creates new for unknown
5. Over time → vocabulary grows, compression improves, patterns emerge

**Depth and atom_count computation:**
```
depth = max(child depths) + 1      // Atoms are depth 0
atom_count = sum(child atom_counts) // Atoms have atom_count 1
```

**Hash Computation (N-ary, position-sensitive):**
```
hash = BLAKE3(ord_0 || hash_0 || ord_1 || hash_1 || ... || ord_N-1 || hash_N-1)
```
Each ordinal is 4 bytes (uint32), position in the sequence matters.

**Centroid Computation**:
```
centroid.x = average(child[i].x for all i)
centroid.y = average(child[i].y for all i)
centroid.z = average(child[i].z for all i)
centroid.m = average(child[i].m for all i)
```

### What C++ Does (Heavy Lifting)
1. Load atom cache once (token ID → hash, coords)
2. Decode input to integer sequence (codepoints, bytes, samples, etc.)
3. Greedy longest-match against vocabulary trie
4. Create N-ary compositions for unmatched sequences
5. Hash computation (BLAKE3), centroid calculation, Hilbert indexing
6. Batch COPY to PostgreSQL (parallel connections)

### What SQL Does (Orchestration ONLY)
1. Store compositions (INSERT...ON CONFLICT DO NOTHING)
2. Spatial queries (PostGIS GIST index)
3. Hilbert range queries for neighborhood search
4. Simple lookups and joins

---

## Type System - CRITICAL

### Coordinate Storage

| Field | PostgreSQL Type | Interpretation | Range |
|-------|----------------|----------------|-------|
| coord_x/y/z/m | INTEGER | Signed 32-bit | -2,147,483,648 to 2,147,483,647 |
| (As uint32) | N/A | Unsigned 32-bit | 0 to 4,294,967,295 |

**Conversion** (lossless bit reinterpretation):
```c++
// C++
uint32_t as_unsigned = static_cast<uint32_t>(signed_val);
int32_t as_signed = static_cast<int32_t>(unsigned_val);

// SQL
int32_to_uint32(INTEGER) → BIGINT  -- Add 2^32 if negative
uint32_to_int32(BIGINT) → INTEGER  -- Subtract 2^32 if >= 2^31
```

### Hash Storage
- `BYTEA(32)` - 256-bit BLAKE3 hash
- Domain: `blake3_hash` with length check

### Hilbert Index
- 128-bit split into two `BIGINT` columns
- `hilbert_hi` (upper 64 bits), `hilbert_lo` (lower 64 bits)
- Stored as signed but interpreted as unsigned for ordering

---

## FORBIDDEN Patterns

### ❌ Never Do This:

1. **Binary tree compositions**:
   ```c++
   // WRONG: Creates artificial nesting
   composition("th", [t, h]);
   composition("the", [th, e]);
   ```

2. **Lossy double conversion**:
   ```sql
   -- BAD: Loses precision!
   ST_X(coords) * 4294967295  -- Double precision loses bits
   ```

3. **Linguistic tokenization**:
   ```c++
   // BAD: Language-specific, loses semantic structure
   split_on_whitespace("Hello world");  // Assumes English-like words
   sentence_boundary_detection();        // Language-specific rules
   ```

4. **Line-by-line ingestion**:
   ```bash
   # BAD: Creates disconnected compositions per line
   while read line; do psql -c "SELECT ingest('$line')"; done
   ```

5. **Recursion/loops in SQL**:
   ```sql
   -- BAD: C++ should do heavy computation
   WITH RECURSIVE tree AS (...)  -- Only for simple traversal
   ```

### ✅ Always Do This:

1. **N-ary compositions at natural boundaries**:
   ```c++
   // "the" = single composition with 3 children
   composition = {hash, children: [t_id, h_id, e_id]};
   ```

2. **Integer arithmetic for centroids**:
   ```c++
   for (child : children) sum_x += child.x;
   centroid.x = sum_x / children.size();
   ```

3. **Whole-file ingestion as codepoint sequence**:
   ```c++
   auto codepoints = decode_utf8(read_file(path));  // No tokenization!
   auto root = ingester.ingest(codepoints, atom_cache, new_comps);
   // Greedy vocabulary matching handles composition boundaries
   ```

4. **COPY for batch insert**:
   ```c++
   PQexec(conn, "COPY atom FROM STDIN...");
   for (comp : compositions) send_row(comp);
   ```

---

## Reconstruction

To reconstruct original content from a composition ID:

```sql
-- Using the built-in function
SELECT semantic_reconstruct('\x...'::BYTEA);

-- Manual recursive traversal
WITH RECURSIVE tree AS (
    SELECT id, children, value, 1 as ord, ARRAY[1] as path
    FROM atom WHERE id = $root_id

    UNION ALL

    SELECT a.id, a.children, a.value, c.ordinal, t.path || c.ordinal
    FROM tree t
    CROSS JOIN LATERAL unnest(t.children) WITH ORDINALITY AS c(child_id, ordinal)
    JOIN atom a ON a.id = c.child_id
    WHERE t.children IS NOT NULL
)
SELECT convert_from(string_agg(value, ''::BYTEA ORDER BY path), 'UTF8')
FROM tree WHERE value IS NOT NULL;
```

---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Init (seed atoms) | ~10s | 1.1M Unicode codepoints (parallel COPY) |
| Ingest text | ~10 MB/s | N-ary tokenization in C++ |
| Query similar | <100ms | Hilbert + GIST indexes |
| Reconstruct | O(n) | DFS traversal of children |

---

## Change Log

### 2026-01-03 - N-ary Merkle DAG (v4)
- **REMOVED**: Binary tree/CPE cascade - fundamentally wrong approach
- **ADDED**: N-ary compositions with arbitrary children count
- **ADDED**: Token-aware ingestion (words, sentences, paragraphs)
- **FIXED**: "the" is now ONE composition with 3 children, not nested binary tree
- **UPDATED**: CompositionRecord uses vector<ChildInfo> instead of left/right
- **UPDATED**: LINESTRINGZM built from all N child centroids

### 2026-01-02 - Unified Schema (v3)
- Single `atom` table replaces atom + relation + relation_edge
- GEOMETRY(GEOMETRYZM, 0) for both POINTZM and LINESTRINGZM

### 2026-01-02 - Course Correction (v2)
- Removed PostGIS coords from relation table
- Added lossless integer coordinates
