-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Score candidate next tokens given context (using 4D centroids)
CREATE OR REPLACE FUNCTION score_candidates(
    p_context_ids BYTEA[],
    p_k INTEGER DEFAULT 50
)
RETURNS TABLE(
    candidate_id BYTEA,
    candidate_label TEXT,
    centroid_score DOUBLE PRECISION,   -- 4D proximity to context
    attention_score DOUBLE PRECISION,  -- Attention relation weights
    pmi_score DOUBLE PRECISION,        -- Bigram PMI score
    combined_score DOUBLE PRECISION
) AS $$
    WITH context_centroid AS (
        -- Get centroid of last context token
        SELECT c.centroid, c.id
        FROM composition c
        WHERE c.id = p_context_ids[array_length(p_context_ids, 1)]
          AND c.centroid IS NOT NULL
        LIMIT 1
    ),
    attention_from_context AS (
        -- Sum attention weights from context
        SELECT r.target_id, SUM(r.weight) as total_attention
        FROM relation r
        WHERE r.source_id = ANY(p_context_ids)
          AND r.relation_type = 'A'
        GROUP BY r.target_id
    ),
    pmi_from_last AS (
        -- PMI (sequence relation) from last token
        SELECT r.target_id, r.weight as pmi_weight
        FROM relation r
        CROSS JOIN context_centroid ctx
        WHERE r.source_id = ctx.id
          AND r.relation_type = 'S'
    ),
    candidates AS (
        -- Vocabulary tokens with centroids, not in context
        SELECT c.id, c.label, c.centroid
        FROM composition c
        WHERE c.centroid IS NOT NULL
          AND c.label IS NOT NULL
          AND c.label NOT LIKE '[%'
          AND NOT (c.id = ANY(p_context_ids))
        LIMIT 5000
    )
    SELECT
        cand.id,
        cand.label,
        centroid_similarity(cand.centroid, ctx.centroid) as cent_score,
        COALESCE(attn.total_attention, 0) as attn_score,
        COALESCE(pmi.pmi_weight, 0) as pmi_score,
        -- Combined: 40% centroid, 30% PMI, 30% attention
        centroid_similarity(cand.centroid, ctx.centroid) * 0.4 +
        COALESCE(pmi.pmi_weight, 0) * 0.3 +
        COALESCE(attn.total_attention, 0) * 0.3 as combined
    FROM candidates cand
    CROSS JOIN context_centroid ctx
    LEFT JOIN attention_from_context attn ON attn.target_id = cand.id
    LEFT JOIN pmi_from_last pmi ON pmi.target_id = cand.id
    ORDER BY combined DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;