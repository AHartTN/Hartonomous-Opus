-- Semantic Edge Schema for Hartonomous Hypercube
-- Stores sparse semantic relationships extracted from AI model embeddings
--
-- These edges represent the "learned" semantics from the model:
-- - Token i relates to Token j with weight W
-- - Only edges above threshold are stored (sparse)
-- - Links to compositions via vocab lookup

BEGIN;

-- Drop and recreate for clean schema
DROP TABLE IF EXISTS semantic_edge CASCADE;

-- Semantic edges extracted from model embeddings
CREATE TABLE semantic_edge (
    id              BIGSERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,

    -- Source and destination (token indices in original vocab)
    src_token_idx   INTEGER NOT NULL,
    dst_token_idx   INTEGER NOT NULL,

    -- Relationship weight (cosine similarity)
    weight          REAL NOT NULL,

    -- Optional: link to compositions (populated by linking step)
    src_composition_id  bytea REFERENCES relation(id),
    dst_composition_id  bytea REFERENCES relation(id),

    -- Trajectory between source and destination centroids
    trajectory      GEOMETRY(LINESTRINGZM, 0),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for efficient queries
CREATE INDEX idx_semantic_edge_model ON semantic_edge(model_name);
CREATE INDEX idx_semantic_edge_src ON semantic_edge(model_name, src_token_idx);
CREATE INDEX idx_semantic_edge_dst ON semantic_edge(model_name, dst_token_idx);
CREATE INDEX idx_semantic_edge_weight ON semantic_edge(weight DESC);
CREATE INDEX idx_semantic_edge_src_comp ON semantic_edge(src_composition_id) WHERE src_composition_id IS NOT NULL;
CREATE INDEX idx_semantic_edge_dst_comp ON semantic_edge(dst_composition_id) WHERE dst_composition_id IS NOT NULL;
CREATE INDEX idx_semantic_edge_trajectory ON semantic_edge USING GIST(trajectory) WHERE trajectory IS NOT NULL;

-- Model ingestion tracking
CREATE TABLE IF NOT EXISTS model_ingestion (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL UNIQUE,
    model_path      TEXT,
    vocab_size      INTEGER,
    hidden_dim      INTEGER,
    threshold       REAL,
    edges_extracted BIGINT,
    sparsity        REAL,
    config          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Link semantic edges to compositions (run after vocab ingestion)
CREATE OR REPLACE FUNCTION hypercube_link_semantic_edges(
    p_model_name TEXT,
    p_vocab_file TEXT
)
RETURNS INTEGER
AS $$
DECLARE
    v_count INTEGER := 0;
    v_line TEXT;
    v_idx INTEGER := 0;
    v_vocab TEXT[];
    v_src_id bytea;
    v_dst_id bytea;
    v_src_coords geometry;
    v_dst_coords geometry;
BEGIN
    -- This would read vocab and link edges
    -- For now, placeholder that shows the pattern
    RAISE NOTICE 'Linking semantic edges for model: %', p_model_name;

    -- Update edges with composition IDs
    UPDATE semantic_edge se
    SET
        src_composition_id = (
            SELECT r.id FROM relation r
            WHERE r.id = hypercube_ingest_text(
                -- Would need vocab lookup here
                'token_' || se.src_token_idx::text
            )
        ),
        dst_composition_id = (
            SELECT r.id FROM relation r
            WHERE r.id = hypercube_ingest_text(
                'token_' || se.dst_token_idx::text
            )
        )
    WHERE se.model_name = p_model_name
      AND se.src_composition_id IS NULL;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Create trajectory geometries for edges
CREATE OR REPLACE FUNCTION hypercube_create_edge_trajectories(p_model_name TEXT)
RETURNS INTEGER
AS $$
DECLARE
    v_count INTEGER := 0;
BEGIN
    UPDATE semantic_edge se
    SET trajectory = ST_MakeLine(
        src_r.coords,
        dst_r.coords
    )
    FROM relation src_r, relation dst_r
    WHERE se.model_name = p_model_name
      AND se.src_composition_id = src_r.id
      AND se.dst_composition_id = dst_r.id
      AND se.trajectory IS NULL;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Query functions for semantic edges
CREATE OR REPLACE FUNCTION hypercube_semantic_neighbors(
    p_text TEXT,
    p_model_name TEXT DEFAULT 'model',
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE(
    neighbor_text TEXT,
    weight REAL,
    distance DOUBLE PRECISION
)
AS $$
DECLARE
    v_id bytea;
BEGIN
    v_id := hypercube_ingest_text(p_text);

    RETURN QUERY
    SELECT
        hypercube_retrieve_text(se.dst_composition_id) as neighbor_text,
        se.weight,
        ST_Length(se.trajectory) as distance
    FROM semantic_edge se
    WHERE se.model_name = p_model_name
      AND se.src_composition_id = v_id
    ORDER BY se.weight DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Statistics view
CREATE OR REPLACE VIEW semantic_edge_stats AS
SELECT
    model_name,
    count(*) as total_edges,
    count(DISTINCT src_token_idx) as unique_sources,
    count(DISTINCT dst_token_idx) as unique_targets,
    avg(weight)::numeric(10,4) as avg_weight,
    min(weight)::numeric(10,4) as min_weight,
    max(weight)::numeric(10,4) as max_weight,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY weight)::numeric(10,4) as median_weight,
    count(*) FILTER (WHERE src_composition_id IS NOT NULL)::numeric /
        NULLIF(count(*), 0)::numeric * 100 as pct_linked
FROM semantic_edge
GROUP BY model_name;

COMMIT;
