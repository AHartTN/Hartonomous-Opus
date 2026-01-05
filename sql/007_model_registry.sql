-- =============================================================================
-- MODEL REGISTRY
-- =============================================================================
-- Tracks discovered models and their ingestion status
-- =============================================================================

CREATE TABLE IF NOT EXISTS model (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,           -- e.g., "sentence-transformers/all-MiniLM-L6-v2"
    path            TEXT NOT NULL,                  -- Absolute path to snapshot dir
    model_type      TEXT NOT NULL,                  -- e.g., "bert", "llama4", "florence2"
    tokenizer_type  TEXT,                           -- e.g., "BPE", "WordPiece", "SentencePiece"
    vocab_size      INTEGER NOT NULL DEFAULT 0,
    hidden_size     INTEGER NOT NULL DEFAULT 0,
    num_layers      INTEGER NOT NULL DEFAULT 0,
    num_experts     INTEGER NOT NULL DEFAULT 0,     -- For MoE models
    is_multimodal   BOOLEAN NOT NULL DEFAULT FALSE,
    vocab_ingested  BOOLEAN NOT NULL DEFAULT FALSE, -- Has vocab been ingested as compositions?
    edges_extracted BOOLEAN NOT NULL DEFAULT FALSE, -- Have weight matrices been processed?
    last_scanned    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_type ON model(model_type);
CREATE INDEX IF NOT EXISTS idx_model_vocab ON model(vocab_ingested);

-- Function to register or update a model
CREATE OR REPLACE FUNCTION upsert_model(
    p_name TEXT,
    p_path TEXT,
    p_model_type TEXT,
    p_tokenizer_type TEXT,
    p_vocab_size INTEGER,
    p_hidden_size INTEGER,
    p_num_layers INTEGER,
    p_num_experts INTEGER DEFAULT 0,
    p_is_multimodal BOOLEAN DEFAULT FALSE
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO model (name, path, model_type, tokenizer_type, vocab_size, hidden_size, num_layers, num_experts, is_multimodal, last_scanned)
    VALUES (p_name, p_path, p_model_type, p_tokenizer_type, p_vocab_size, p_hidden_size, p_num_layers, p_num_experts, p_is_multimodal, now())
    ON CONFLICT (name) DO UPDATE SET
        path = EXCLUDED.path,
        model_type = EXCLUDED.model_type,
        tokenizer_type = EXCLUDED.tokenizer_type,
        vocab_size = EXCLUDED.vocab_size,
        hidden_size = EXCLUDED.hidden_size,
        num_layers = EXCLUDED.num_layers,
        num_experts = EXCLUDED.num_experts,
        is_multimodal = EXCLUDED.is_multimodal,
        last_scanned = now()
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- View: Model stats
CREATE OR REPLACE VIEW model_stats AS
SELECT
    m.id,
    m.name,
    m.model_type,
    m.tokenizer_type,
    m.vocab_size,
    m.hidden_size,
    m.num_layers,
    CASE WHEN m.num_experts > 0 THEN format('%s experts (MoE)', m.num_experts) ELSE NULL END AS moe_info,
    m.is_multimodal,
    m.vocab_ingested,
    m.edges_extracted,
    (SELECT COUNT(*) FROM composition c WHERE c.label LIKE m.name || ':%') AS compositions_created,
    (SELECT COUNT(*) FROM relation r WHERE r.source_model = m.name) AS edges_created,
    m.last_scanned
FROM model m
ORDER BY m.name;
