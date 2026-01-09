# ELO-Style Multi-Model Relation Consensus

## Core Concept

**The database IS the context window. The database IS the model.**

Semantic structure emerges through **relations** between atoms and compositions. Multiple models observe the same content (e.g., "neural" → "network") and vote on relation strength. We use **ELO ratings** to track per-model confidence and compute consensus.

## Architecture

### Content-Addressed Merkle DAG

```
Atoms (UTF-8 codepoints)
    ↓ hash
Compositions (ordered children)
    ↓ hash
Relations (typed edges)
    ↓ ELO consensus
Semantic Graph
```

**Key Properties:**
- **Deterministic**: Same content = same hash (client-side calculation)
- **Lossless**: Full provenance tracking (which model, when, how strong)
- **Compositional**: BPE/CPE merges, AST hierarchies, attention patterns
- **Temporal**: Relation strength evolves as more models observe

### Relation Types

| Type | Name | Description | Example |
|------|------|-------------|---------|
| `S` | Semantic | Cosine similarity from embeddings | "king" ↔ "queen" (0.85) |
| `H` | Hierarchy | AST parent-child | `model.config` → `model.config.hidden_size` |
| `V` | Value | Key-value pair | `hidden_size` → `4096` |
| `M` | Merge | BPE/CPE composition | `"un"` + `"able"` → `"unable"` |
| `A` | Attention | Transformer attention weights | Token 42 → Token 17 (layer 12) |
| `T` | Transformation | FFN geometric relations | Embedding space warping |

## ELO Rating System

### Why ELO?

Traditional averaging loses information:
```
Model A: weight = 0.95 (very confident)
Model B: weight = 0.52 (barely above threshold)
Average: 0.735 (looks moderately confident)
```

**Problem**: Can't tell if models agree or if we're averaging noise.

ELO solution:
```
Model A: rating = 1850 (strong relation)
Model B: rating = 1520 (weak relation)
Consensus: weight = 0.45, confidence = 0.65
```

**Benefit**: Separate consensus strength from confidence.

### Rating Update Formula

```python
def update_elo(current_rating, normalized_weight, observation_count):
    # K-factor decays with more observations (converges to stable estimate)
    k = max(4.0, 32.0 * (0.95 ** observation_count))

    # Expected: How strong should this relation be based on current rating?
    rating_diff = current_rating - 1500.0
    expected = 1.0 / (1.0 + exp(-rating_diff / 400.0))

    # Actual: Observed weight normalized to [0, 1]
    actual = (normalized_weight + 1.0) / 2.0

    # Update rating based on surprise
    new_rating = current_rating + k * (actual - expected)

    return clamp(new_rating, 0.0, 3000.0)
```

### Normalization

Different models use different weight ranges:

| Source | Raw Range | Normalization |
|--------|-----------|---------------|
| Cosine similarity | [0, 1] | `2*w - 1` → [-1, 1] |
| Attention scores | [0, 1] | `2*w - 1` → [-1, 1] |
| Correlation | [-1, 1] | Already normalized |
| L2 distance | [0, ∞) | `exp(-d)` then normalize |
| Logits | (-∞, ∞) | `tanh(w)` → [-1, 1] |

### Consensus Calculation

```sql
SELECT
    AVG(rating) as avg_rating,              -- Mean ELO
    STDDEV(rating) as rating_stddev,        -- Disagreement measure

    -- Consensus weight: map rating to [-1, 1]
    TANH((AVG(rating) - 1500.0) / 400.0) as consensus_weight,

    -- Confidence: inverse of disagreement
    1.0 / (1.0 + STDDEV(rating) / 400.0) as confidence,

    COUNT(*) as num_models,
    SUM(observation_count) as total_observations
FROM relation_evidence
GROUP BY source_id, target_id, relation_type;
```

**Example Results:**

| Scenario | Avg Rating | StdDev | Consensus Weight | Confidence |
|----------|------------|--------|------------------|------------|
| Strong agreement (1850, 1830, 1870) | 1850 | 20 | +0.68 | 0.95 |
| Moderate agreement (1600, 1550, 1650) | 1600 | 50 | +0.25 | 0.89 |
| Weak relation (1510, 1490, 1505) | 1502 | 10 | +0.005 | 0.98 |
| Disagreement (1900, 1200) | 1550 | 495 | +0.12 | 0.45 |

## Database Schema

### relation_evidence (raw observations)

```sql
CREATE TABLE relation_evidence (
    source_id BYTEA,
    target_id BYTEA,
    relation_type CHAR(1),
    source_model TEXT,      -- e.g., "llama-3.3-70b", "qwen2.5-72b"
    layer INT,              -- -1 for embeddings, 0-47 for layers
    component TEXT,         -- "attn_qk", "ffn_up", "embeddings"

    -- ELO system
    rating REAL DEFAULT 1500.0,
    observation_count INT DEFAULT 1,

    -- Evidence
    raw_weight REAL,
    normalized_weight REAL,

    -- Metadata
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (source_id, target_id, relation_type, source_model, layer, component)
);
```

### relation_consensus (materialized view)

```sql
CREATE MATERIALIZED VIEW relation_consensus AS
SELECT
    source_id,
    target_id,
    relation_type,

    AVG(rating) as avg_rating,
    STDDEV(rating) as rating_stddev,
    TANH((AVG(rating) - 1500.0) / 400.0) as consensus_weight,
    1.0 / (1.0 + STDDEV(rating) / 400.0) as confidence,

    COUNT(*) as num_models,
    SUM(observation_count) as total_observations

FROM relation_evidence
GROUP BY source_id, target_id, relation_type;
```

### relation (backward-compatible view)

```sql
CREATE VIEW relation AS
SELECT
    source_id,
    target_id,
    relation_type,
    consensus_weight as weight,
    'consensus' as source_model,
    total_observations as source_count
FROM relation_consensus;
```

## Ingestion Pipeline

### Phase 1: Seed Atoms (One-Time)

```bash
./seed_atoms_parallel
```

Seeds all UTF-8 codepoints (1.1M atoms) with Hilbert coordinates.

### Phase 2: Model Metadata

```cpp
insert_model_metadata(conn, meta);
```

1. Parse `config.json` → config atoms + AST compositions
2. Parse `tokenizer.json` → vocab tokens + BPE merges
3. Build compositions with **deterministic hashes**
4. Insert into `relation_evidence` with initial rating 1500

### Phase 3: Tensor Hierarchy

```cpp
insert_tensor_hierarchy(conn, ctx);
```

1. Extract tensor names from safetensors
2. Build hierarchy: `model.layers.0.attn.q_proj.weight`
3. Insert H-relations (hierarchy edges)

### Phase 4: Embeddings + k-NN

```cpp
extract_all_semantic_relations(conn, ctx);
```

1. Load token embeddings (BF16/F16/F32)
2. **Validate & fix corrupt values** (NaN, extreme outliers)
3. Build HNSW index:
   - Vocab <100K: Sequential with progress
   - Vocab >100K: **Parallel partitioned** (4x speedup)
4. Query k-NN, insert S-relations with cosine similarity

### Phase 5: Attention Relations

```cpp
insert_attention_relations(conn, ctx);
```

1. Load attention weight tensors
2. Extract attention patterns per layer
3. Insert A-relations with attention scores

### Phase 6: Update Consensus

```sql
SELECT refresh_relation_consensus();
```

Rebuilds materialized view with updated ELO consensus.

## Query Examples

### Find strongly related tokens (high consensus)

```sql
SELECT
    c1.label as source,
    c2.label as target,
    r.consensus_weight,
    r.confidence,
    r.num_models
FROM relation_consensus r
JOIN composition c1 ON r.source_id = c1.id
JOIN composition c2 ON r.target_id = c2.id
WHERE r.relation_type = 'S'
  AND r.consensus_weight > 0.7
  AND r.confidence > 0.8
ORDER BY r.consensus_weight DESC, r.confidence DESC
LIMIT 100;
```

### Compare how different models see a relation

```sql
SELECT
    source_model,
    rating,
    normalized_weight,
    observation_count
FROM relation_evidence
WHERE source_id = (SELECT id FROM composition WHERE label = 'neural')
  AND target_id = (SELECT id FROM composition WHERE label = 'network')
  AND relation_type = 'S'
ORDER BY rating DESC;
```

### Find controversial relations (high disagreement)

```sql
SELECT
    c1.label as source,
    c2.label as target,
    r.avg_rating,
    r.rating_stddev,
    r.confidence,
    r.num_models
FROM relation_consensus r
JOIN composition c1 ON r.source_id = c1.id
JOIN composition c2 ON r.target_id = c2.id
WHERE r.rating_stddev > 200  -- High disagreement
  AND r.num_models >= 3        -- At least 3 models voted
ORDER BY r.rating_stddev DESC;
```

### Temporal analysis: How relation strength evolves

```sql
WITH evidence_history AS (
    SELECT
        source_id,
        target_id,
        source_model,
        rating,
        observation_count,
        last_updated,
        LAG(rating) OVER (
            PARTITION BY source_id, target_id, source_model
            ORDER BY last_updated
        ) as prev_rating
    FROM relation_evidence
)
SELECT
    source_model,
    rating - prev_rating as rating_change,
    last_updated
FROM evidence_history
WHERE source_id = $1 AND target_id = $2
  AND prev_rating IS NOT NULL
ORDER BY last_updated DESC;
```

## Prompt Ingestion

Prompts flow into the SAME graph:

```
User prompt: "Explain transformers"
    ↓ tokenize
Tokens: [Explain, transformers]
    ↓ hash
Compositions: [existing atoms]
    ↓ observe
Relations: Update ELO for "explain" → "transformers"
```

Each prompt **updates** existing relations with new evidence. Frequently co-occurring tokens gain stronger relations over time.

## Benefits

1. **Multi-Model Consensus**: Aggregate knowledge from multiple models
2. **Confidence Tracking**: Know when models agree vs. disagree
3. **Temporal Evolution**: Relations strengthen/weaken over time
4. **Provenance**: Always know which model contributed what
5. **Controversy Detection**: Find relations where models disagree
6. **Prompt Integration**: User interactions update the graph

## Implementation Status

- [x] ELO algorithm design
- [x] C++ header with update formulas
- [x] SQL migration with tables/views
- [ ] Integrate into ingestion pipeline
- [ ] Consensus refresh triggers
- [ ] Query API with confidence filtering
- [ ] Prompt ingestion flow
- [ ] Web UI for relation visualization

## References

- ELO Rating: https://en.wikipedia.org/wiki/Elo_rating_system
- Merkle DAG: https://docs.ipfs.tech/concepts/merkle-dag/
- Content Addressing: https://en.wikipedia.org/wiki/Content-addressable_storage
