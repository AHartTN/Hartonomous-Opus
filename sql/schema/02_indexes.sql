-- =============================================================================
-- DATABASE INDEXES - Performance optimization for hypercube queries
-- =============================================================================
-- Indexes are critical for the hypercube's performance characteristics:
-- - Hilbert indexes enable O(log n) locality queries
-- - GIST indexes enable efficient geometric operations
-- - Foreign key indexes optimize joins
-- =============================================================================

-- =============================================================================
-- ATOM INDEXES
-- =============================================================================

-- Codepoint lookup (frequent for Unicode operations)
CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint);

-- Hilbert curve ordering for locality-sensitive queries
CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);

-- Spatial operations (KNN, distance, containment)
CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);

-- =============================================================================
-- COMPOSITION INDEXES
-- =============================================================================

-- Centroid similarity search (most important query pattern)
CREATE INDEX IF NOT EXISTS idx_comp_centroid ON composition USING GIST(centroid);

-- Hilbert locality for composition queries
CREATE INDEX IF NOT EXISTS idx_comp_hilbert ON composition(hilbert_hi, hilbert_lo);

-- Label lookups for vocabulary operations
CREATE INDEX IF NOT EXISTS idx_comp_label ON composition(label);

-- Depth-based queries (tree traversal)
CREATE INDEX IF NOT EXISTS idx_comp_depth ON composition(depth);

-- Child count queries (statistics and filtering)
CREATE INDEX IF NOT EXISTS idx_comp_child_count ON composition(child_count);

-- =============================================================================
-- COMPOSITION_CHILD INDEXES
-- =============================================================================

-- Child lookup (finding parents of a given entity)
CREATE INDEX IF NOT EXISTS idx_comp_child_child ON composition_child(child_id);

-- Ordinal ordering for sequence operations
CREATE INDEX IF NOT EXISTS idx_comp_child_ordinal ON composition_child(composition_id, ordinal);

-- =============================================================================
-- RELATION INDEXES
-- =============================================================================

-- Source entity lookups (outgoing edges)
CREATE INDEX IF NOT EXISTS idx_relation_source ON relation(source_id);

-- Target entity lookups (incoming edges)
CREATE INDEX IF NOT EXISTS idx_relation_target ON relation(target_id);

-- Relation type filtering
CREATE INDEX IF NOT EXISTS idx_relation_type ON relation(relation_type);

-- Model attribution (filtering by model)
CREATE INDEX IF NOT EXISTS idx_relation_model ON relation(source_model) WHERE source_model != '';

-- Weight ordering (top-K queries)
CREATE INDEX IF NOT EXISTS idx_relation_weight ON relation(weight DESC);

-- Layer filtering (neural network layer attribution)
CREATE INDEX IF NOT EXISTS idx_relation_layer ON relation(layer) WHERE layer != -1;

-- =============================================================================
-- MODEL REGISTRY INDEXES
-- =============================================================================

-- Model name lookups
CREATE INDEX IF NOT EXISTS idx_model_name ON model(name);

-- Source filtering
CREATE INDEX IF NOT EXISTS idx_model_source ON model(source);

-- =============================================================================
-- BIGRAM STATISTICS INDEXES
-- =============================================================================

-- Left token lookups (continuation queries)
CREATE INDEX IF NOT EXISTS idx_bigram_left ON bigram_stats(left_id);

-- Right token lookups (predecessor queries)
CREATE INDEX IF NOT EXISTS idx_bigram_right ON bigram_stats(right_id);

-- PMI ordering (top continuations)
CREATE INDEX IF NOT EXISTS idx_bigram_pmi ON bigram_stats(pmi DESC) WHERE pmi IS NOT NULL;

-- =============================================================================
-- PROJECTION METADATA INDEXES
-- =============================================================================

-- Model lookups (which models have been projected)
CREATE INDEX IF NOT EXISTS idx_proj_meta_model ON projection_metadata(model_id);

-- Role filtering (embeddings vs attention vs ffn)
CREATE INDEX IF NOT EXISTS idx_proj_meta_role ON projection_metadata(role);

-- Quality-based queries (champion model selection)
CREATE INDEX IF NOT EXISTS idx_proj_meta_quality ON projection_metadata(quality_score DESC);

-- Geometry write status (track which projections define coordinates)
CREATE INDEX IF NOT EXISTS idx_proj_meta_written ON projection_metadata(geom_written);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON INDEX idx_atom_hilbert IS 'Hilbert curve ordering for O(log n) locality queries';
COMMENT ON INDEX idx_atom_geom IS 'GIST index for 4D geometric operations (KNN, distance)';
COMMENT ON INDEX idx_comp_centroid IS 'Critical index for 4D similarity search';
COMMENT ON INDEX idx_comp_hilbert IS 'Hilbert ordering for composition locality queries';
COMMENT ON INDEX idx_relation_source IS 'Outgoing edge lookups from source entities';
COMMENT ON INDEX idx_relation_target IS 'Incoming edge lookups to target entities';
COMMENT ON INDEX idx_relation_weight IS 'Weight-ordered queries for top-K semantic neighbors';