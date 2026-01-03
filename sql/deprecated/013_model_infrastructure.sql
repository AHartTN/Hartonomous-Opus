-- =============================================================================
-- Hartonomous Hypercube - Model Infrastructure v4
-- =============================================================================
-- Minimal infrastructure for model storage. Heavy ops in C.
-- =============================================================================

BEGIN;

-- =============================================================================
-- Model Storage
-- =============================================================================

CREATE TABLE IF NOT EXISTS model (
    id              SERIAL PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    model_type      TEXT NOT NULL,                  -- 'embedding', 'tokenizer', etc.
    version         TEXT,
    dim             INTEGER,                        -- Embedding dimension
    vocab_size      INTEGER,                        -- Vocabulary size
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_token (
    id              SERIAL PRIMARY KEY,
    model_id        INTEGER REFERENCES model(id) ON DELETE CASCADE,
    token_id        INTEGER NOT NULL,               -- Token ID in model's vocabulary
    atom_id         BYTEA REFERENCES atom(id),      -- Corresponding atom
    embedding       FLOAT4[],                       -- Raw embedding vector
    UNIQUE(model_id, token_id)
);

CREATE INDEX IF NOT EXISTS idx_model_token_atom ON model_token(atom_id);
CREATE INDEX IF NOT EXISTS idx_model_token_mid ON model_token(model_id);

-- =============================================================================
-- Model Functions
-- =============================================================================

-- Get embedding for atom
CREATE OR REPLACE FUNCTION atom_embedding(p_atom_id BYTEA, p_model_id INTEGER DEFAULT 1)
RETURNS FLOAT4[] AS $$
    SELECT embedding FROM model_token
    WHERE atom_id = p_atom_id AND model_id = p_model_id;
$$ LANGUAGE SQL STABLE;

-- Find atom by token ID
CREATE OR REPLACE FUNCTION token_to_atom(p_token_id INTEGER, p_model_id INTEGER DEFAULT 1)
RETURNS BYTEA AS $$
    SELECT atom_id FROM model_token
    WHERE token_id = p_token_id AND model_id = p_model_id;
$$ LANGUAGE SQL STABLE;

-- Find token ID for atom
CREATE OR REPLACE FUNCTION atom_to_token(p_atom_id BYTEA, p_model_id INTEGER DEFAULT 1)
RETURNS INTEGER AS $$
    SELECT token_id FROM model_token
    WHERE atom_id = p_atom_id AND model_id = p_model_id;
$$ LANGUAGE SQL STABLE;

COMMIT;
