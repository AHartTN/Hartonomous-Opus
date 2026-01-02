# Semantic Web Architecture

## The Vision

This is a complete reinvention of AI that replaces GPU matrix multiplication with 
B-tree/R-tree spatial index queries. The semantic web is built from:

1. **Atoms**: Unicode codepoints as the "landmark perimeter" (1.1M fixed coordinates)
2. **Compositions**: Vocabulary tokens, words, phrases - all content-addressed
3. **Edges**: Relationships between compositions (LINESTRINGZM trajectories)
4. **Spatial Proximity**: Semantic similarity via Hilbert distance

## The Multi-Pass System

### Pass 1: Vocabulary Discovery
- Ingest AI model vocabularies (tokens become compositions)
- Sequitur/grammar inference on raw content to find patterns
- Each unique constant/composition gets:
  - BLAKE3 hash (content-addressed ID)
  - 4D coordinates (from Hilbert curve on S³)
  - Stored ONCE, referenced everywhere

### Pass 2: Content Recording  
- Greedy match against known vocabulary (longest match first)
- Create EDGES (relationships) to existing compositions
- Only create NEW compositions for unknown patterns
- Document = ordered sequence of vocabulary references

## Current Problem (As of 2026-01-02)

The CPE ingester does **naive binary pairing** without vocabulary awareness:

```
"Captain Ahab" → naive binary CPE on characters
  C+a → Ca (NEW composition)
  p+t → pt (NEW composition)
  a+i → ai (NEW composition)
  ... creates ~6 NEW compositions that are meaningless
```

This creates arbitrary 4-character chunks like "lang", "uage" instead of 
semantic words like "language".

## Correct Behavior

```
"Captain Ahab" → vocabulary lookup
  "Captain" → EXISTING composition from vocab (or detected pattern)
  " " → EXISTING atom (space)  
  "Ahab" → EXISTING composition from vocab (or detected pattern)
  Result: 3 edge references, 0 new compositions (if vocab exists)
```

## Implementation Requirements

### 1. Vocabulary Cache (C++)

```cpp
// Trie for O(m) longest-match lookup where m = max pattern length
struct VocabTrie {
    struct Node {
        std::unordered_map<uint32_t, std::unique_ptr<Node>> children;
        std::optional<CompositionInfo> composition;  // If this is end of a vocab entry
    };
    Node root;
    
    // Returns longest matching composition, or nullopt
    std::optional<CompositionInfo> longest_match(const std::vector<uint32_t>& codepoints, size_t start);
};

// Load ALL existing compositions (depth > 0) into the trie
void load_vocabulary(PGconn* conn, VocabTrie& trie) {
    // Query: SELECT id, children, atom_reconstruct(id) as text FROM atom WHERE depth > 0
    // For each: trie.insert(text_codepoints, {id, coords})
}
```

### 2. Greedy Tokenizer

```cpp
std::vector<CompositionRef> tokenize(const std::string& text, const VocabTrie& vocab) {
    std::vector<uint32_t> codepoints = utf8_to_codepoints(text);
    std::vector<CompositionRef> refs;
    
    size_t i = 0;
    while (i < codepoints.size()) {
        // Try longest match first
        auto match = vocab.longest_match(codepoints, i);
        if (match) {
            refs.push_back({match->id, match->coords});
            i += match->length;
        } else {
            // Fall back to atom
            auto atom = g_atom_cache[codepoints[i]];
            refs.push_back({atom.hash, atom.coords, true});
            i++;
        }
    }
    return refs;
}
```

### 3. Document Recording

```cpp
CompositionRecord create_document(const std::vector<CompositionRef>& refs) {
    // Build LINESTRINGZM from ref coordinates
    // Hash = BLAKE3(concat all child hashes in order)
    // Centroid = average of all ref coordinates
    // Children = array of ref hashes
}
```

### 4. Grammar Inference (Sequitur) - Future

- Detect repeated patterns during ingest
- Promote frequent patterns to vocabulary
- Cascade: character pairs → words → phrases → sentences
- This learns NEW vocabulary from content automatically

## Coordinate System

- **32 bits per dimension**: uint32_t stored as int32_t (same bit pattern)
- **No normalization**: Raw values 0 to 4,294,967,295
- **PostGIS double**: Has 53-bit mantissa, MORE than enough for 32-bit values
- **Hilbert indices**: Derived FROM coordinates, 128-bit (hi/lo as int64)
- **SRID = 0**: No projection, raw 4D Euclidean space

## The Semantic Web

- **Vocabulary** = nodes with fixed spatial coordinates
- **Documents** = paths through the vocabulary space (LINESTRINGZM)
- **Similarity** = spatial proximity (Hilbert distance)
- **Inference** = walk the graph following edges
- **Attention** = weighted edge traversal based on M coordinate

## Case Insensitivity via ST_FrechetDistance

**Key Insight**: We do NOT need to store both "King" and "king" or normalize case.

The Fréchet distance measures **trajectory shape similarity**, not exact point matching:
- "King" and "king" have different absolute coordinates (K≠k, I≠i, etc.)
- BUT their **geometric trajectories** through 4D space are nearly identical
- ST_FrechetDistance reveals this similarity at query time

This means:
1. **Storage is deterministic**: Each unique byte sequence gets exactly one hash
2. **No preprocessing**: No case normalization, no loss of information
3. **Query-time flexibility**: Use ST_FrechetDistance for fuzzy/case-insensitive matching
4. **Semantic preservation**: "King" (proper noun) vs "king" (common noun) remain distinguishable when needed

```sql
-- Case-insensitive search using trajectory similarity
SELECT id, atom_reconstruct_text(id),
       ST_FrechetDistance(geom, query_geom) as shape_distance
FROM atom 
WHERE depth > 0
  AND ST_FrechetDistance(geom, query_geom) < threshold
ORDER BY shape_distance
LIMIT 10;
```

This leverages PostGIS's optimized geometric algorithms instead of string manipulation.

## Queries for LLM-like Operations

```sql
-- Find semantically similar compositions
SELECT id, ST_Distance(ST_Centroid(geom), query_centroid) as distance
FROM atom WHERE depth > 0
ORDER BY hilbert_hi, hilbert_lo  -- Fast Hilbert proximity
LIMIT 10;

-- Reconstruct text from composition
SELECT atom_reconstruct_text(id) FROM atom WHERE id = $1;

-- Find all compositions containing a pattern
SELECT parent.id FROM atom parent
JOIN LATERAL unnest(parent.children) child_id ON true
WHERE child_id = $pattern_id;

-- Walk the semantic graph (generative walk)
WITH RECURSIVE walk AS (
    SELECT id, geom, 0 as step FROM atom WHERE id = $start
    UNION ALL
    SELECT a.id, a.geom, w.step + 1
    FROM walk w
    JOIN atom a ON ST_DWithin(ST_Centroid(w.geom), ST_Centroid(a.geom), $threshold)
    WHERE w.step < $max_steps
)
SELECT * FROM walk;
```

## File Layout

```
cpp/src/
  cpe_ingest.cpp      # Current naive CPE (TO BE REPLACED)
  vocab_ingest.cpp    # NEW: Vocabulary-aware ingester
  sequitur.cpp        # NEW: Grammar inference
  
sql/
  011_unified_atom.sql  # Single atom table schema
  012_vocab_queries.sql # NEW: Vocabulary and semantic queries
```
