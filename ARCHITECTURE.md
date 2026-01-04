# Hartonomous Hypercube - Architecture Document

**Last Updated**: 2025-01-16
**Status**: Canonical - Binary PMI Merkle DAG (v5)

---

## Executive Summary

The Hypercube is a **deterministic, lossless, content-addressable geometric semantic substrate**. All digital content is decomposed into a Merkle DAG where:

1. **Atoms** (Unicode codepoints) form the perimeter landmarks at fixed 4D coordinates (POINTZM)
2. **Compositions** are binary pairs forming a deduplicated dictionary (not content directly)
3. **Relations** explicitly store parent-child edges with ordinals for ordering
4. **Deduplication** is global - identical patterns share the same hash regardless of source
5. **Reconstruction** is bit-perfect via ordered traversal of the relation table

### The Two Table Model (IMPLEMENTED)

The system uses:
- **`atom` table**: Nodes (codepoints and compositions)
- **`relation` table**: Edges (parent→child with ordinal and type)

This is the **dictionary model**:
- We build a dictionary of binary compositions via PMI contraction
- Content references the dictionary through sequences of edges
- The dictionary grows logarithmically while content grows linearly

---

## Core Principles

### 1. Determinism
- Same bytes → same composition ID, always
- No randomness, no floating-point conversion, no approximations
- Hash = BLAKE3 of canonical ordered child pair

### 2. Losslessness
- Bit-perfect reconstruction from composition
- DFS traversal of relations (ordinal 1 then 2) → original byte sequence
- All coordinates stored as 32-bit signed integers (bit pattern same as uint32)

### 3. Global Deduplication
- "the" from Moby Dick = "the" from a children's book = same composition ID
- First ingest creates the pattern; subsequent ingests only add references
- The more you ingest, the more deduplication occurs

### 4. Binary Compositions (PMI Contraction)

**Binary pairs form the dictionary:**
- PMI (Pointwise Mutual Information) identifies significant co-occurrences
- Highest PMI pairs are contracted into new compositions
- Process repeats until single root composition remains
- Result: Logarithmic growth of dictionary, linear growth of content

**Example:** "the" might decompose as:
- `th` = composition of `[t, h]` (ordinal 1, ordinal 2)
- `the` = composition of `[th, e]` (ordinal 1, ordinal 2)

**LINESTRINGZM geometry** represents the path through 2 child centroids:
- `ST_MakeLine(child1.centroid, child2.centroid)`
- The trajectory shape IS semantic information (Fréchet distance for similarity)

### 5. Relation Table for Edges

**Explicit edge storage:**
- `relation(parent_id, child_id, ordinal, relation_type)`
- `ordinal` = 1 or 2 for binary compositions
- `relation_type` = 'C' (composition), 'S' (sequence), 'M' (metadata), 'R' (reference)
- Enables rich graph traversal and pattern matching

### 6. Hypersphere Geometry

**Leaves on perimeter, compositions move inward:**
- Depth 0 (atoms): On the outer surface of the 4D hypersphere
- Depth 1 (pairs): Centroid of component atoms, radius decreases
- Depth N: Further inward, approaching origin as complexity increases

**Origin = most abstract/complex, Perimeter = most atomic**

### 7. Emergent Topology as Semantics

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

### Atom Table (Nodes)
```sql
atom (
    -- Content-addressed identifier
    id              BYTEA PRIMARY KEY,      -- BLAKE3 hash (32 bytes)

    -- Geometry: POINTZM for leaves, LINESTRINGZM for compositions
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL,
    centroid        GEOMETRY(POINTZM, 0),   -- Pre-computed 4D centroid

    -- Canonical content for leaves only (NULL for compositions)
    content         BYTEA,                  -- Raw bytes for depth 0

    -- Unicode codepoint for leaf atoms (NULL for compositions)
    codepoint       INTEGER UNIQUE,

    -- 128-bit Hilbert index (from centroid -> 4D coords)
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,

    -- Depth in DAG (0 = leaf, 1+ = composition)
    depth           INTEGER NOT NULL DEFAULT 0,

    -- Total leaf atoms in subtree (1 for leaves)
    atom_count      BIGINT NOT NULL DEFAULT 1
)
```

### Relation Table (Edges)
```sql
relation (
    -- References
    parent_id       BYTEA NOT NULL REFERENCES atom(id),
    child_id        BYTEA NOT NULL REFERENCES atom(id),

    -- Order within parent (1 or 2 for binary compositions)
    ordinal         INTEGER NOT NULL,

    -- Relationship type
    relation_type   CHAR(1) NOT NULL DEFAULT 'C',
    -- 'C' = Composition (binary tree structure)
    -- 'S' = Sequence (document content referencing dictionary)
    -- 'M' = Metadata (annotations, provenance)
    -- 'R' = Reference (cross-links)

    PRIMARY KEY (parent_id, ordinal, relation_type)
)
```

**Key Points**:
- Two tables: `atom` for nodes, `relation` for edges
- `depth = 0` means leaf (POINTZM geometry, has content and codepoint)
- `depth > 0` means composition (LINESTRINGZM geometry, 2 children via relation table)
- Ordinals 1 and 2 preserve binary order for reconstruction
- DFS traversal of relations (ordinal 1 first, then 2) = original byte sequence

---

## Ingestion Pipeline

### PMI-Based Binary Contraction (C++ Implementation)

```
Input → Codepoint Sequence → PMI Calculation → Highest-Pair Contraction → Binary Merkle DAG
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
