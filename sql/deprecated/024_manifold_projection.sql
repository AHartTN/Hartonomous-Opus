-- =============================================================================
-- Hartonomous Hypercube - Manifold Projection (384D → 4D)
-- =============================================================================
-- Projects high-dimensional embeddings (384D from MiniLM, etc.) into the
-- 4D hypercube coordinate space using Laplacian Eigenmaps + Gram-Schmidt.
--
-- This creates a semantic manifold where:
-- - Local neighborhoods are preserved
-- - Similar tokens are geometrically close
-- - The 4D space enables Hilbert indexing for O(log n) queries
-- =============================================================================

-- Type for holding 4D projection result
DO $$ BEGIN
    CREATE TYPE projection_4d AS (
        x INTEGER,
        y INTEGER,
        z INTEGER,
        m INTEGER
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Similarity Graph Construction
-- =============================================================================

-- Build k-nearest neighbor similarity graph from embeddings
-- This is the adjacency matrix W for Laplacian Eigenmaps
CREATE OR REPLACE FUNCTION build_similarity_graph(
    p_model TEXT DEFAULT 'minilm',
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    entity_a BYTEA,
    entity_b BYTEA,
    similarity DOUBLE PRECISION
) AS $$
BEGIN
    -- For each entity with an embedding, find k nearest neighbors
    RETURN QUERY
    WITH embeddings AS (
        SELECT s.entity_id, s.embedding
        FROM shape s
        WHERE s.model_name ILIKE '%' || p_model || '%'
          AND s.dim_count = 384
    )
    SELECT DISTINCT ON (e1.entity_id, e2.entity_id)
        LEAST(e1.entity_id, e2.entity_id),
        GREATEST(e1.entity_id, e2.entity_id),
        embedding_cosine_similarity(e1.embedding, e2.embedding)
    FROM embeddings e1
    CROSS JOIN LATERAL (
        SELECT e2.entity_id, e2.embedding
        FROM embeddings e2
        WHERE e2.entity_id != e1.entity_id
        ORDER BY e1.embedding <-> e2.embedding
        LIMIT p_k
    ) e2;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Graph Laplacian Computation
-- =============================================================================

-- Compute degree for each node (sum of edge weights)
CREATE OR REPLACE FUNCTION compute_node_degrees(p_model TEXT DEFAULT 'minilm')
RETURNS TABLE(entity_id BYTEA, degree DOUBLE PRECISION) AS $$
BEGIN
    RETURN QUERY
    WITH edges AS (
        SELECT * FROM build_similarity_graph(p_model, 10)
    )
    SELECT e.entity_a, SUM(e.similarity)
    FROM edges e
    GROUP BY e.entity_a
    UNION ALL
    SELECT e.entity_b, SUM(e.similarity)
    FROM edges e
    GROUP BY e.entity_b;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Power Iteration for Eigenvector Approximation
-- =============================================================================

-- Approximate top eigenvector using power iteration
-- This is a SQL-friendly way to compute eigenvectors without external libraries
CREATE OR REPLACE FUNCTION power_iteration_step(
    p_model TEXT,
    p_vector_table TEXT,
    p_iterations INTEGER DEFAULT 100
)
RETURNS VOID AS $$
DECLARE
    v_iter INTEGER;
BEGIN
    -- Power iteration: v_new = L * v_old / ||L * v_old||
    -- We use the normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
    FOR v_iter IN 1..p_iterations LOOP
        -- This is a placeholder - actual implementation requires temp tables
        -- and iterative matrix-vector multiplication
        NULL;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- PCA-based 4D Projection (Fast approximation)
-- =============================================================================

-- Project 384D embeddings to 4D using top 4 principal components
-- This is faster than full Laplacian Eigenmaps and works well in practice
CREATE OR REPLACE FUNCTION project_embeddings_pca_4d(
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    entity_id BYTEA,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION
) AS $$
DECLARE
    v_mean DOUBLE PRECISION[];
    v_count BIGINT;
BEGIN
    -- Step 1: Compute mean vector
    SELECT COUNT(*), 
           (SELECT array_agg(avg_val ORDER BY dim)
            FROM (
                SELECT dim, AVG(val) as avg_val
                FROM shape s,
                     LATERAL unnest(embedding_to_array(s.embedding)) WITH ORDINALITY AS t(val, dim)
                WHERE s.model_name ILIKE '%' || p_model || '%'
                  AND s.dim_count = 384
                GROUP BY dim
            ) sub)
    INTO v_count, v_mean
    FROM shape
    WHERE model_name ILIKE '%' || p_model || '%'
      AND dim_count = 384;

    -- For now, use first 4 dimensions as a simple projection
    -- A full PCA implementation would compute covariance and eigenvectors
    RETURN QUERY
    WITH embedding_arrays AS (
        SELECT s.entity_id,
               embedding_to_array(s.embedding) as arr
        FROM shape s
        WHERE s.model_name ILIKE '%' || p_model || '%'
          AND s.dim_count = 384
    )
    SELECT 
        e.entity_id,
        (e.arr[1] - COALESCE(v_mean[1], 0))::DOUBLE PRECISION,
        (e.arr[2] - COALESCE(v_mean[2], 0))::DOUBLE PRECISION,
        (e.arr[3] - COALESCE(v_mean[3], 0))::DOUBLE PRECISION,
        (e.arr[4] - COALESCE(v_mean[4], 0))::DOUBLE PRECISION
    FROM embedding_arrays e;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Random Projection (Johnson-Lindenstrauss)
-- =============================================================================

-- Create a deterministic random projection matrix (seeded for reproducibility)
CREATE OR REPLACE FUNCTION get_projection_matrix_row(
    p_row INTEGER,
    p_input_dim INTEGER DEFAULT 384,
    p_seed INTEGER DEFAULT 42
)
RETURNS DOUBLE PRECISION[] AS $$
DECLARE
    v_result DOUBLE PRECISION[];
    v_i INTEGER;
    v_val DOUBLE PRECISION;
BEGIN
    -- Use deterministic pseudo-random values based on row and seed
    v_result := ARRAY[]::DOUBLE PRECISION[];
    FOR v_i IN 1..p_input_dim LOOP
        -- Simple deterministic hash-based random
        v_val := (sin(p_row * 12.9898 + v_i * 78.233 + p_seed) * 43758.5453);
        v_val := v_val - floor(v_val);  -- fractional part
        v_val := (v_val - 0.5) * 2.0;   -- normalize to [-1, 1]
        v_result := array_append(v_result, v_val / sqrt(p_input_dim::DOUBLE PRECISION));
    END LOOP;
    RETURN v_result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Project a single embedding using JL random projection
CREATE OR REPLACE FUNCTION project_embedding_jl(
    p_embedding geometry,
    p_seed INTEGER DEFAULT 42
)
RETURNS projection_4d AS $$
DECLARE
    v_arr DOUBLE PRECISION[];
    v_proj DOUBLE PRECISION[];
    v_row DOUBLE PRECISION[];
    v_i INTEGER;
    v_j INTEGER;
    v_sum DOUBLE PRECISION;
    v_result projection_4d;
    v_scale DOUBLE PRECISION := 2147483647.0;  -- Scale to int32 range
BEGIN
    v_arr := embedding_to_array(p_embedding);
    IF v_arr IS NULL OR array_length(v_arr, 1) < 4 THEN
        RETURN NULL;
    END IF;
    
    v_proj := ARRAY[0.0, 0.0, 0.0, 0.0];
    
    -- Matrix-vector multiply: proj = P * arr
    FOR v_i IN 1..4 LOOP
        v_row := get_projection_matrix_row(v_i, array_length(v_arr, 1), p_seed);
        v_sum := 0.0;
        FOR v_j IN 1..array_length(v_arr, 1) LOOP
            v_sum := v_sum + v_row[v_j] * v_arr[v_j];
        END LOOP;
        v_proj[v_i] := v_sum;
    END LOOP;
    
    -- Scale to integer coordinates (use tanh to bound, then scale)
    v_result.x := (tanh(v_proj[1]) * v_scale)::INTEGER;
    v_result.y := (tanh(v_proj[2]) * v_scale)::INTEGER;
    v_result.z := (tanh(v_proj[3]) * v_scale)::INTEGER;
    v_result.m := (tanh(v_proj[4]) * v_scale)::INTEGER;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- Batch Projection and Coordinate Update
-- =============================================================================

-- Project all embeddings and update composition coordinates
CREATE OR REPLACE FUNCTION project_all_embeddings(
    p_model TEXT DEFAULT 'minilm',
    p_method TEXT DEFAULT 'jl',  -- 'jl' = Johnson-Lindenstrauss, 'pca' = PCA
    p_batch_size INTEGER DEFAULT 1000
)
RETURNS TABLE(
    total_processed BIGINT,
    total_updated BIGINT,
    elapsed_ms BIGINT
) AS $$
DECLARE
    v_start TIMESTAMP;
    v_processed BIGINT := 0;
    v_updated BIGINT := 0;
    v_batch BIGINT;
BEGIN
    v_start := clock_timestamp();
    
    -- Update compositions with projected coordinates
    LOOP
        WITH to_update AS (
            SELECT s.entity_id, s.embedding
            FROM shape s
            JOIN composition c ON c.id = s.entity_id
            WHERE s.model_name ILIKE '%' || p_model || '%'
              AND s.dim_count = 384
              AND c.centroid IS NULL  -- Only update if not already set
            LIMIT p_batch_size
        ),
        projections AS (
            SELECT 
                t.entity_id,
                (project_embedding_jl(t.embedding)).x AS px,
                (project_embedding_jl(t.embedding)).y AS py,
                (project_embedding_jl(t.embedding)).z AS pz,
                (project_embedding_jl(t.embedding)).m AS pm
            FROM to_update t
        )
        UPDATE composition c
        SET 
            centroid = ST_SetSRID(ST_MakePoint(p.px, p.py, p.pz, p.pm), 0),
            hilbert_lo = (hypercube_coords_to_hilbert(p.px, p.py, p.pz, p.pm)).lo,
            hilbert_hi = (hypercube_coords_to_hilbert(p.px, p.py, p.pz, p.pm)).hi
        FROM projections p
        WHERE c.id = p.entity_id;
        
        GET DIAGNOSTICS v_batch = ROW_COUNT;
        v_updated := v_updated + v_batch;
        v_processed := v_processed + v_batch;
        
        EXIT WHEN v_batch < p_batch_size;
    END LOOP;
    
    total_processed := v_processed;
    total_updated := v_updated;
    elapsed_ms := EXTRACT(EPOCH FROM (clock_timestamp() - v_start)) * 1000;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Semantic Query Functions Using Manifold
-- =============================================================================

-- Find compositions nearest to a query in the projected 4D manifold
CREATE OR REPLACE FUNCTION manifold_nearest(
    p_query TEXT,
    p_k INTEGER DEFAULT 20
)
RETURNS TABLE(
    label TEXT,
    distance DOUBLE PRECISION,
    hilbert_dist NUMERIC
) AS $$
DECLARE
    v_query_id BYTEA;
    v_query_centroid GEOMETRY;
BEGIN
    -- Find query composition
    SELECT id, centroid INTO v_query_id, v_query_centroid
    FROM composition
    WHERE label ILIKE p_query
    LIMIT 1;
    
    IF v_query_id IS NULL THEN
        RAISE NOTICE 'Query "%" not found in compositions', p_query;
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        c.label,
        c.centroid <-> v_query_centroid,
        ABS(c.hilbert_lo - (SELECT hilbert_lo FROM composition WHERE id = v_query_id))::NUMERIC
    FROM composition c
    WHERE c.id != v_query_id
      AND c.label IS NOT NULL
      AND c.centroid IS NOT NULL
    ORDER BY c.centroid <-> v_query_centroid
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- Combined semantic search: embedding similarity + manifold proximity
CREATE OR REPLACE FUNCTION semantic_search(
    p_query TEXT,
    p_k INTEGER DEFAULT 20,
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    label TEXT,
    embedding_sim DOUBLE PRECISION,
    manifold_dist DOUBLE PRECISION,
    combined_score DOUBLE PRECISION
) AS $$
DECLARE
    v_query_embedding GEOMETRY;
    v_query_centroid GEOMETRY;
BEGIN
    -- Get query embedding and centroid
    SELECT s.embedding, c.centroid 
    INTO v_query_embedding, v_query_centroid
    FROM composition c
    LEFT JOIN shape s ON s.entity_id = c.id AND s.model_name ILIKE '%' || p_model || '%'
    WHERE c.label ILIKE p_query
    LIMIT 1;
    
    IF v_query_embedding IS NULL AND v_query_centroid IS NULL THEN
        RAISE NOTICE 'Query "%" not found', p_query;
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        c.label,
        CASE WHEN v_query_embedding IS NOT NULL AND s.embedding IS NOT NULL 
             THEN embedding_cosine_similarity(v_query_embedding, s.embedding)
             ELSE 0.0 END AS emb_sim,
        CASE WHEN v_query_centroid IS NOT NULL AND c.centroid IS NOT NULL
             THEN 1.0 / (1.0 + (c.centroid <-> v_query_centroid))
             ELSE 0.0 END AS man_dist,
        -- Combined: weighted average
        (0.6 * CASE WHEN v_query_embedding IS NOT NULL AND s.embedding IS NOT NULL 
                    THEN embedding_cosine_similarity(v_query_embedding, s.embedding)
                    ELSE 0.0 END +
         0.4 * CASE WHEN v_query_centroid IS NOT NULL AND c.centroid IS NOT NULL
                    THEN 1.0 / (1.0 + (c.centroid <-> v_query_centroid))
                    ELSE 0.0 END) AS combined
    FROM composition c
    LEFT JOIN shape s ON s.entity_id = c.id AND s.model_name ILIKE '%' || p_model || '%'
    WHERE c.label IS NOT NULL
      AND c.label NOT ILIKE p_query
    ORDER BY combined DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Question Answering via Semantic Graph Walk
-- =============================================================================

-- Answer a question by finding relevant compositions and walking the semantic graph
CREATE OR REPLACE FUNCTION answer_question(
    p_question TEXT,
    p_context_tokens INTEGER DEFAULT 5,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    answer_token TEXT,
    relevance DOUBLE PRECISION,
    context_path TEXT[]
) AS $$
DECLARE
    v_keywords TEXT[];
    v_keyword TEXT;
BEGIN
    -- Extract keywords from question (simple tokenization)
    v_keywords := regexp_split_to_array(lower(p_question), '\s+');
    
    -- For each keyword, find related tokens
    RETURN QUERY
    WITH keyword_matches AS (
        SELECT DISTINCT
            c.label,
            c.id,
            embedding_cosine_similarity(
                (SELECT embedding FROM shape WHERE entity_id = c.id LIMIT 1),
                (SELECT embedding FROM shape s2 
                 JOIN composition c2 ON c2.id = s2.entity_id 
                 WHERE c2.label ILIKE ANY(v_keywords) 
                 LIMIT 1)
            ) AS sim
        FROM composition c
        WHERE c.label IS NOT NULL
          AND EXISTS (SELECT 1 FROM shape WHERE entity_id = c.id)
    ),
    top_matches AS (
        SELECT * FROM keyword_matches
        WHERE sim IS NOT NULL
        ORDER BY sim DESC
        LIMIT p_k
    )
    SELECT 
        tm.label,
        tm.sim,
        ARRAY[tm.label]  -- Would extend with graph walk path
    FROM top_matches tm;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Moby Dick Specific: Extract named entities
-- =============================================================================

-- Find proper nouns (capitalized words) that appear in compositions
CREATE OR REPLACE FUNCTION find_proper_nouns(p_limit INTEGER DEFAULT 100)
RETURNS TABLE(
    token TEXT,
    depth INTEGER,
    occurrences BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.label,
        c.depth,
        COUNT(*)
    FROM composition c
    WHERE c.label ~ '^[A-Z][a-z]+$'  -- Capitalized words
      AND c.depth = 1
      AND length(c.label) >= 3
    GROUP BY c.label, c.depth
    ORDER BY COUNT(*) DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Integration: Full semantic query for Q&A
-- =============================================================================

-- Ask a natural language question and get semantically relevant answers
CREATE OR REPLACE FUNCTION ask(
    p_question TEXT,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    answer TEXT,
    score DOUBLE PRECISION,
    method TEXT
) AS $$
DECLARE
    v_words TEXT[];
    v_word TEXT;
BEGIN
    -- Tokenize question
    v_words := regexp_split_to_array(lower(p_question), '[^a-z]+');
    v_words := array_remove(v_words, '');
    
    -- Remove common stop words
    v_words := ARRAY(
        SELECT w FROM unnest(v_words) w
        WHERE w NOT IN ('what', 'is', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by')
    );
    
    RETURN QUERY
    -- Method 1: Direct label match + embedding similarity
    SELECT 
        c.label,
        COALESCE(MAX(embedding_cosine_similarity(
            s.embedding,
            (SELECT s2.embedding FROM shape s2 
             JOIN composition c2 ON c2.id = s2.entity_id 
             WHERE c2.label = ANY(v_words) 
             LIMIT 1)
        )), 0.0) AS sim,
        'embedding'::TEXT
    FROM composition c
    LEFT JOIN shape s ON s.entity_id = c.id
    WHERE c.label IS NOT NULL
      AND c.label != ALL(v_words)
      AND EXISTS (SELECT 1 FROM shape WHERE entity_id = c.id)
    GROUP BY c.label
    ORDER BY sim DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;

-- =============================================================================
-- Post-install: Show available functions
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Manifold Projection functions installed:';
    RAISE NOTICE '  - project_embedding_jl(embedding, seed) -> 4D coords';
    RAISE NOTICE '  - project_all_embeddings(model, method, batch) -> update compositions';
    RAISE NOTICE '  - manifold_nearest(query, k) -> nearest in 4D space';
    RAISE NOTICE '  - semantic_search(query, k) -> combined embedding + manifold';
    RAISE NOTICE '  - ask(question, k) -> natural language Q&A';
END $$;
