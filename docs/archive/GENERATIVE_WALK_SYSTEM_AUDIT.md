# Generative Walk System: Current State Audit

**Date**: 2026-01-17
**Purpose**: Document existing infrastructure and identify gaps for implementing energy-based generative walks

---

## Executive Summary

The Hartonomous-Opus codebase has **substantial infrastructure** for generation, but the current system uses a **weighted-sum scoring model** rather than the **energy-based walk system** described in your architecture notes. The core pieces exist; they need to be restructured around an explicit energy function with goal-conditioning, novelty terms, and context accumulation.

---

## 1. C++ Core: What EXISTS

### 1.1 Generation Engine (`cpp/include/hypercube/generative.hpp`)

**Classes**:
- `VocabularyCache` - Stores tokens with 4D centroids, Hilbert indices, frequencies
- `BigramCache` - PMI scores between token pairs
- `AttentionCache` - Attention weights from ingested models
- `GenerativeEngine` - Main engine with scoring and generation

**Current Scoring (lines 307-349)**:
```cpp
// Individual score components
double score_centroid(current, candidate)  // 4D proximity
double score_pmi(current, candidate)       // Co-occurrence
double score_attn(current, candidate)      // Attention relations
double score_global(candidate)             // Frequency prior

// Combined score (weighted sum, NOT energy)
score_total = w_centroid * score_centroid +
              w_pmi * score_pmi +
              w_attn * score_attn +
              w_global * score_global
```

**Candidate Filtering (lines 356-395)**:
- `get_candidates_by_hilbert()` - Filters by Hilbert proximity
- `get_all_vocab_candidates()` - All depth-1 tokens with centroids

**Selection Policy (lines 401-433)**:
- Greedy: Pick highest score
- Stochastic: Softmax sampling with temperature

**Generation Loop (lines 449-504)**:
```cpp
std::vector<std::string> generate(start_label, max_tokens) {
    current = make_token_state(start_idx);
    for (step = 0; step < max_tokens; ++step) {
        candidates = get_candidates_by_hilbert(current);
        scored = score_all_candidates(current, candidates);
        next_idx = select_next_token(scored);  // greedy or softmax
        // Check stop tokens
        current = make_token_state(next_idx);
    }
}
```

### 1.2 C API Bridge (`cpp/include/hypercube/generative_c.h`)

Full C interface for interop:
- `gen_vocab_add()`, `gen_vocab_set_centroid()` - Vocabulary management
- `gen_bigram_add()`, `gen_bigram_get()` - PMI cache
- `gen_attention_add()` - Attention cache
- `gen_config_set_weights()`, `gen_config_set_policy()` - Configuration
- `gen_generate()` - Main generation function
- `gen_find_similar()` - Similarity search
- `gen_score_candidates()` - Debug scoring

### 1.3 Geometric Operations

- `geom_map_codepoint()` - Unicode → 4D coordinates
- `geom_euclidean_distance()` - 4D distance
- `geom_centroid()`, `geom_weighted_centroid()` - Centroid calculation

---

## 2. Database Schema: What EXISTS

### 2.1 Core Tables (`sql/schema/01_tables.sql`)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `atom` | Unicode codepoints with 4D geometry | `id` (BLAKE3), `codepoint`, `geom` (POINTZM), `hilbert_lo/hi` |
| `composition` | Aggregations (BPE, words, phrases) | `id`, `label`, `centroid` (POINTZM), `geom` (LINESTRINGZM), `depth` |
| `composition_child` | Ordered parent-child relationships | `composition_id`, `ordinal`, `child_type`, `child_id` |
| `relation` | Semantic edges | `source_id`, `target_id`, `relation_type` (S/A/P), `weight`, `source_model` |
| `bigram_stats` | Token co-occurrence | `left_id`, `right_id`, `count`, `pmi` |
| `relation_evidence` | ELO-rated evidence | `rating`, `observation_count` |

### 2.2 Key SQL Functions

**Generation** (`sql/functions/queries/`):
- `generate_tokens(prompt, max_tokens, temperature, top_k)` - SQL-based generation
- `score_candidates(context_ids, k)` - Scores by centroid (40%), PMI (30%), attention (30%)
- `encode_prompt(text)` - Tokenizes prompt to composition IDs

**Neighbors** (`sql/functions/relations/`):
- `semantic_neighbors(id, limit)` - Get relation-based neighbors
- `attention(id, limit)` - Get attention-weighted neighbors

**Geometry** (`sql/functions/geometry/`):
- `centroid_similarity(geom1, geom2)` - 4D similarity
- `atom_knn(point, k)` - k-nearest atoms

---

## 3. C# API Layer: What EXISTS

### 3.1 OpenAI-Compatible Endpoints

| Endpoint | Status | Location |
|----------|--------|----------|
| `POST /v1/completions` | Implemented | `Controllers/CompletionsController.cs` |
| `GET /v1/models` | Implemented | `Controllers/ModelsController.cs` |
| `GET /health` | Implemented | Built-in |

### 3.2 Services

**GenerativeService** (`Services/GenerativeService.cs`):
- Loads vocabulary, bigrams, attention via C++ interop
- Token-by-token generation loop
- Stop sequence handling
- OpenAI response formatting

**TokenizationService** (`Services/TokenizationService.cs`):
- Prompt tokenization
- Token validation against database

**PostgresService** (`Services/PostgresService.cs`):
- Database connectivity
- Token existence checking

### 3.3 Critical Issues (per SUBSTRATE_CONNECTION_AUDIT.md)

**Interop Layer BROKEN**:
- P/Invoke signatures assume simplified types
- Reality: IDs are 32-byte BLAKE3 hashes (`uint8_t*`)
- Needs complete rewrite to handle BYTEA properly

**PostgresService PLACEHOLDER**:
- Uses fake hash-based lookups
- Should use real BYTEA queries with PostGIS geometry

---

## 4. What's MISSING for Energy-Based Walk System

### 4.1 Energy Function Framework

**Current**: Weighted sum of scores (higher = better)
```
score_total = w1*s1 + w2*s2 + w3*s3 + w4*s4
```

**Needed**: Energy function (lower = better, exponential sampling)
```
E(x'; s_t) = E_dist + E_sem + E_novel + E_goal
P(x_{t+1} | s_t) ∝ exp(-E(x_{t+1}; s_t))
```

### 4.2 Missing Energy Terms

| Term | Purpose | Current Status |
|------|---------|----------------|
| `E_dist` | Distance/locality preference | Exists as `score_centroid` (inverted) |
| `E_sem` | Semantic compatibility with goal | **MISSING** - No goal parameter |
| `E_novel` | Anti-repetition/novelty | **MISSING** - No history tracking |
| `E_goal` | Goal attraction | **MISSING** - No target point |

### 4.3 Missing Context Accumulation

**Current**: Only uses last token for scoring
```cpp
current = make_token_state(next_idx);  // Single state
```

**Needed**: Full trajectory/context
```cpp
struct WalkState {
    Point4D current_position;
    std::vector<Point4D> trajectory;     // History
    Point4D goal;                         // Target
    std::unordered_set<Blake3Hash> visited;  // For novelty
    // ... additional context
};
```

### 4.4 Missing Goal-Conditioning

**Current**: No goal parameter in generation
```cpp
std::vector<std::string> generate(start_label, max_tokens);
```

**Needed**: Goal-conditioned generation
```cpp
std::vector<std::string> generate(
    start_label,
    goal_embedding,      // Target 4D point
    max_tokens,
    energy_config        // Energy function parameters
);
```

### 4.5 Missing Local Programs

**Current**: All tokens are passive data
**Needed**: Tokens that modify walk behavior:
- **Control tokens**: Modify energy function weights
- **Operator tokens**: Transform context
- **Structural tokens**: Encode boundaries/nesting

### 4.6 Missing k-NN Neighbor Graph

**Current**: Hilbert-based filtering (approximates proximity)
**Needed**: Precomputed k-NN neighbors per point for O(1) lookup

**Schema addition**:
```sql
CREATE TABLE composition_neighbors (
    composition_id  BYTEA NOT NULL,
    neighbor_id     BYTEA NOT NULL,
    distance        REAL NOT NULL,
    neighbor_rank   SMALLINT NOT NULL,  -- 1 = nearest, 2 = second nearest, etc.
    PRIMARY KEY (composition_id, neighbor_rank)
);
```

---

## 5. Architecture Mapping: Your Vision → Current Code

| Your Concept | Current Implementation | Gap |
|--------------|------------------------|-----|
| **Field F(x)** | `VocabEntry.centroid`, `composition.centroid` | Exists |
| **Policy π(x)** | `GenerativeEngine::select_next_token()` | Needs energy reformulation |
| **State s_t = (x_t, C_t)** | `TokenState` (position only) | Missing context C_t |
| **Transition kernel P(x' \| s_t)** | Softmax over weighted scores | Needs energy-based kernel |
| **E_dist** | `score_centroid()` (inverted) | Reformulate as energy |
| **E_sem** | Partially via `score_attn()` | Missing explicit goal matching |
| **E_novel** | Not implemented | Add visited set tracking |
| **E_goal** | Not implemented | Add goal attraction term |
| **Local programs P_x** | Not implemented | Add program table + interpreter |

---

## 6. Implementation Roadmap

### Phase 1: Energy Function Refactor (C++)

1. Create `WalkState` struct with full context
2. Create `EnergyFunction` class with composable terms
3. Refactor `generate()` to use energy-based selection
4. Add goal parameter to generation API

### Phase 2: Context & Novelty

1. Track visited tokens in walk state
2. Implement `E_novel` term
3. Implement trajectory-aware scoring

### Phase 3: Database Extensions

1. Add `composition_neighbors` table
2. Add precomputed k-NN for all compositions
3. Add local program storage

### Phase 4: C# Interop Fix

1. Rewrite P/Invoke for BYTEA handling
2. Fix PostgresService with real queries
3. Add goal parameter to API endpoints

### Phase 5: Local Programs

1. Define program types (control, operator, structural)
2. Create program table schema
3. Implement program interpreter in walk loop

---

## 7. Key Files Reference

### C++ Core
- `cpp/include/hypercube/generative.hpp` - Engine classes
- `cpp/include/hypercube/generative_c.h` - C API
- `cpp/src/bridge/generative_c.cpp` - C API implementation

### Database
- `sql/schema/01_tables.sql` - Core tables
- `sql/functions/queries/score_candidates.sql` - Scoring function
- `sql/functions/queries/generate_tokens.sql` - SQL generation

### C# API
- `csharp/HypercubeGenerativeApi/Services/GenerativeService.cs`
- `csharp/HypercubeGenerativeApi/docs/GENERATIVE_SERVICE.md`

### Documentation
- `docs/archive/4D_SUBSTRATE_THEORY.md` - Mathematical foundations
- `csharp/HypercubeGenerativeApi/SUBSTRATE_CONNECTION_AUDIT.md` - Interop issues

---

## 8. Immediate Next Steps

1. **Design `EnergyFunction` class** - Define interface and composable terms
2. **Design `WalkState` struct** - Full context with history, goal, visited set
3. **Add goal parameter to C API** - `gen_generate_goal_conditioned()`
4. **Fix C# interop** - Critical for app layer to work

---

**Document Status**: Research Complete
**Next Action**: Design energy function architecture
