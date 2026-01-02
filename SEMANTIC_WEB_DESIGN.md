# Semantic Web Architecture

## The Multi-Pass System

### Pass 1: Vocabulary Discovery
- Ingest AI model vocabularies (tokens become compositions)
- Sequitur/grammar inference on raw content to find patterns
- Each unique constant/composition gets:
  - BLAKE3 hash (content-addressed ID)
  - 4D coordinates (from Hilbert curve)
  - Stored ONCE, referenced everywhere

### Pass 2: Content Recording  
- Greedy match against known vocabulary (longest match first)
- Create EDGES (relationships) to existing compositions
- Only create NEW compositions for unknown patterns
- Document = ordered sequence of vocabulary references

## Current Problem
```
"Captain Ahab" → naive binary CPE on characters
  C+a → Ca (NEW composition)
  p+t → pt (NEW composition)
  ... creates ~6 NEW compositions
```

## Correct Behavior
```
"Captain Ahab" → vocabulary lookup
  "Captain" → EXISTING composition from vocab
  " " → EXISTING atom (space)  
  "Ahab" → EXISTING composition from vocab
  Result: 3 edge references, 0 new compositions
```

## Implementation Requirements

1. **Vocabulary Index**: Trie or hash map of known compositions
   - Key: byte sequence
   - Value: composition ID + coordinates

2. **Greedy Tokenizer**: 
   - Scan input left-to-right
   - Match longest known composition
   - Fall back to character atoms for unknowns

3. **Edge Recording**:
   - Document = root composition
   - Children = ordered vocabulary references
   - LINESTRINGZM = trajectory through vocabulary space

4. **Grammar Inference** (Sequitur):
   - Detect repeated patterns during ingest
   - Promote frequent patterns to vocabulary
   - Cascade: character pairs → words → phrases → sentences

## The Semantic Web
- Vocabulary = nodes with fixed spatial coordinates
- Documents = paths through the vocabulary space
- Similarity = spatial proximity (Hilbert distance)
- Inference = walk the graph following edges
