-- Metadata table: External references to compositions
-- This is the ONLY additional table needed beyond atom/relation/relation_edge
--
-- Everything else lives in the Merkle DAG:
-- - Tokens → compositions of atoms
-- - Semantic edges → compositions of [src, dst] with weight in M coordinate
-- - Model config → compositions of key-value pairs
-- - Documents → compositions of sentences/words

BEGIN;

-- Drop all the unnecessary tables
DROP TABLE IF EXISTS semantic_edge CASCADE;
DROP TABLE IF EXISTS token_edge CASCADE;
DROP TABLE IF EXISTS vocab_token CASCADE;
DROP TABLE IF EXISTS model_ingestion CASCADE;
DROP TABLE IF EXISTS model CASCADE;

-- Simple metadata: external reference → composition
CREATE TABLE IF NOT EXISTS metadata (
    ref         TEXT PRIMARY KEY,           -- "model:all-MiniLM-L6-v2", "file:/path/to/doc"
    comp_id     bytea NOT NULL REFERENCES relation(id),
    ref_type    TEXT,                       -- "model", "document", "edge", etc.
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metadata_type ON metadata(ref_type);
CREATE INDEX IF NOT EXISTS idx_metadata_comp ON metadata(comp_id);

-- Helper: Store a semantic edge as a relation
-- Weight is encoded in the M coordinate of the centroid
CREATE OR REPLACE FUNCTION hypercube_store_edge(
    p_src_id bytea,
    p_dst_id bytea,
    p_weight real
) RETURNS bytea
AS $$
DECLARE
    v_id bytea;
    v_src_coords geometry;
    v_dst_coords geometry;
    v_centroid geometry;
    v_hilbert_lo bigint;
    v_hilbert_hi bigint;
BEGIN
    -- Get source and destination coordinates
    SELECT coords INTO v_src_coords FROM relation WHERE id = p_src_id;
    SELECT coords INTO v_dst_coords FROM relation WHERE id = p_dst_id;

    IF v_src_coords IS NULL OR v_dst_coords IS NULL THEN
        RETURN NULL;
    END IF;

    -- Compute centroid with weight encoded in M dimension
    v_centroid := ST_SetSRID(ST_MakePoint(
        (ST_X(v_src_coords) + ST_X(v_dst_coords)) / 2,
        (ST_Y(v_src_coords) + ST_Y(v_dst_coords)) / 2,
        (ST_Z(v_src_coords) + ST_Z(v_dst_coords)) / 2,
        p_weight  -- Weight encoded in M coordinate
    ), 0);

    -- Compute Hilbert index
    SELECT h.hilbert_lo, h.hilbert_hi INTO v_hilbert_lo, v_hilbert_hi
    FROM hypercube_coords_to_hilbert(
        (ST_X(v_centroid) * 4294967295)::bigint,
        (ST_Y(v_centroid) * 4294967295)::bigint,
        (ST_Z(v_centroid) * 4294967295)::bigint,
        (ST_M(v_centroid) * 4294967295)::bigint
    ) h;

    -- Create composition hash from ordered children
    v_id := hypercube_blake3(int4send(0) || p_src_id || int4send(1) || p_dst_id);

    -- Insert relation (deduplication via content-addressing)
    INSERT INTO relation (id, coords, hilbert_lo, hilbert_hi, depth, child_count, atom_count)
    VALUES (v_id, v_centroid, v_hilbert_lo, v_hilbert_hi,
            (SELECT max(depth) + 1 FROM relation WHERE id IN (p_src_id, p_dst_id)),
            2,
            (SELECT sum(atom_count) FROM relation WHERE id IN (p_src_id, p_dst_id)))
    ON CONFLICT (id) DO NOTHING;

    -- Insert edges
    INSERT INTO relation_edge (parent_id, child_id, ordinal, is_atom)
    VALUES (v_id, p_src_id, 0, false), (v_id, p_dst_id, 1, false)
    ON CONFLICT DO NOTHING;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Get weight from an edge composition (stored in M coordinate)
CREATE OR REPLACE FUNCTION hypercube_edge_weight(p_edge_id bytea)
RETURNS real
AS $$
    SELECT ST_M(coords)::real FROM relation WHERE id = p_edge_id;
$$ LANGUAGE sql STABLE;

-- Find edges by weight threshold
CREATE OR REPLACE FUNCTION hypercube_edges_above_threshold(
    p_threshold real,
    p_limit integer DEFAULT 100
)
RETURNS TABLE(
    edge_id bytea,
    weight real,
    src_id bytea,
    dst_id bytea
)
AS $$
    SELECT
        r.id as edge_id,
        ST_M(r.coords)::real as weight,
        (SELECT child_id FROM relation_edge WHERE parent_id = r.id AND ordinal = 0) as src_id,
        (SELECT child_id FROM relation_edge WHERE parent_id = r.id AND ordinal = 1) as dst_id
    FROM relation r
    WHERE r.child_count = 2
      AND ST_M(r.coords) >= p_threshold
    ORDER BY ST_M(r.coords) DESC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

COMMIT;
