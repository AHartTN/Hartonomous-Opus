-- Hartonomous Hypercube Schema
-- 4D Hilbert-indexed content-addressable storage system
-- Requires: PostgreSQL 15+, PostGIS 3.3+, pgcrypto

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Custom domain for BLAKE3 hashes (32 bytes = 256 bits)
CREATE DOMAIN blake3_hash AS BYTEA
    CONSTRAINT blake3_hash_length CHECK (octet_length(VALUE) = 32);

-- Enum for atom categories (semantic clustering on hypercube surface)
CREATE TYPE atom_category AS ENUM (
    'control',           -- C0, C1 control characters
    'format',            -- Format characters (ZWJ, etc.)
    'private_use',       -- Private use areas
    'surrogate',         -- Surrogate pairs (not standalone)
    'noncharacter',      -- Noncharacters
    'space',             -- Whitespace
    'punctuation_open',  -- Opening brackets, quotes
    'punctuation_close', -- Closing brackets, quotes
    'punctuation_other', -- Other punctuation
    'digit',             -- 0-9 and other numeric digits
    'number_letter',     -- Letter-like numbers (Roman numerals)
    'math_symbol',       -- Mathematical operators
    'currency',          -- Currency symbols
    'modifier',          -- Modifier symbols
    'letter_upper',      -- Uppercase letters
    'letter_lower',      -- Lowercase letters
    'letter_titlecase',  -- Titlecase letters
    'letter_modifier',   -- Modifier letters
    'letter_other',      -- Other letters (CJK, etc.)
    'mark_nonspacing',   -- Combining marks (non-spacing)
    'mark_spacing',      -- Combining marks (spacing)
    'mark_enclosing',    -- Enclosing marks
    'symbol_other',      -- Other symbols (emoji, etc.)
    'separator'          -- Line/paragraph separators
);

-- Atoms: Unicode codepoints as fundamental constants
-- Distributed on 4D hypercube SURFACE with semantic clustering
CREATE TABLE atom (
    id              blake3_hash PRIMARY KEY,
    codepoint       INTEGER NOT NULL UNIQUE,
    category        atom_category NOT NULL,
    
    -- 4D coordinates (32-bit per dimension, stored as POINTZM)
    -- X, Y, Z = spatial dimensions, M = 4th dimension
    coords          GEOMETRY(POINTZM, 0) NOT NULL,
    
    -- Raw 32-bit coordinates (lossless storage)
    -- PostgreSQL INTEGER is 32-bit signed, but bit pattern is same as uint32
    coord_x         INTEGER,
    coord_y         INTEGER,
    coord_z         INTEGER,
    coord_m         INTEGER,
    
    -- Hilbert curve index (128-bit position as two 64-bit unsigned integers)
    -- Stored as BIGINT but interpreted as unsigned
    hilbert_lo      BIGINT NOT NULL,  -- Lower 64 bits
    hilbert_hi      BIGINT NOT NULL,  -- Upper 64 bits
    
    -- Metadata
    name            TEXT,  -- Unicode character name
    block           TEXT,  -- Unicode block name
    script          TEXT,  -- Unicode script
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Relations: Edges in the Merkle DAG
-- Parent compositions reference child compositions/atoms
CREATE TABLE relation (
    id              blake3_hash PRIMARY KEY,
    
    -- Raw 32-bit centroid coordinates (lossless, source of truth)
    -- PostgreSQL INTEGER is 32-bit signed, same bit pattern as uint32
    coord_x         INTEGER NOT NULL,
    coord_y         INTEGER NOT NULL,
    coord_z         INTEGER NOT NULL,
    coord_m         INTEGER NOT NULL,
    
    -- Hilbert curve index of centroid (for spatial range queries)
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    
    -- Composition metadata
    depth           INTEGER NOT NULL DEFAULT 1,  -- DAG depth from atoms
    child_count     INTEGER NOT NULL,            -- Number of direct children
    atom_count      BIGINT NOT NULL,             -- Total atoms in subtree
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Relation edges: The actual parent-child relationships
CREATE TABLE relation_edge (
    parent_id       blake3_hash NOT NULL REFERENCES relation(id) ON DELETE CASCADE,
    child_id        blake3_hash NOT NULL,  -- References either atom(id) or relation(id)
    ordinal         INTEGER NOT NULL,      -- Position in sequence
    
    -- Is this child an atom or another relation?
    is_atom         BOOLEAN NOT NULL,
    
    PRIMARY KEY (parent_id, ordinal),
    UNIQUE (parent_id, child_id, ordinal)
);

-- Index for reverse lookups (find all parents of a child)
CREATE INDEX idx_relation_edge_child ON relation_edge(child_id);

-- Spatial indices for geometric queries
-- Spatial index on atom coords (PostGIS for geometric queries)
CREATE INDEX idx_atom_coords ON atom USING GIST(coords);

-- Hilbert curve indices for range queries (primary spatial index for relations)
CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
CREATE INDEX idx_relation_hilbert ON relation(hilbert_hi, hilbert_lo);

-- Composite coordinate index for relations (no PostGIS needed)
CREATE INDEX idx_relation_coords ON relation(coord_x, coord_y, coord_z, coord_m);

-- Category index for semantic filtering
CREATE INDEX idx_atom_category ON atom(category);

-- Depth index for level-based queries
CREATE INDEX idx_relation_depth ON relation(depth);

-- Partial indices for common queries
CREATE INDEX idx_atom_letters ON atom(codepoint) 
    WHERE category IN ('letter_upper', 'letter_lower', 'letter_titlecase', 'letter_other');
CREATE INDEX idx_atom_digits ON atom(codepoint)
    WHERE category = 'digit';

-- Function to verify blake3 hash of content matches stored id
CREATE OR REPLACE FUNCTION verify_relation_hash(rel_id blake3_hash)
RETURNS BOOLEAN AS $$
DECLARE
    computed_hash blake3_hash;
    child_hashes BYTEA := ''::BYTEA;
BEGIN
    -- Concatenate all child hashes in ordinal order
    SELECT string_agg(child_id::BYTEA, ''::BYTEA ORDER BY ordinal)
    INTO child_hashes
    FROM relation_edge
    WHERE parent_id = rel_id;
    
    -- Hash computation would be done in C++ extension
    -- This is a placeholder - actual verification uses hypercube_blake3()
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql STABLE;

-- View for atom statistics
CREATE OR REPLACE VIEW atom_stats AS
SELECT 
    category,
    COUNT(*) as count,
    ST_Extent(coords::GEOMETRY) as bounding_box,
    AVG(ST_X(coords)) as avg_x,
    AVG(ST_Y(coords)) as avg_y,
    AVG(ST_Z(coords)) as avg_z,
    AVG(ST_M(coords)) as avg_m
FROM atom
GROUP BY category;

-- View for relation depth distribution
CREATE OR REPLACE VIEW relation_depth_stats AS
SELECT 
    depth,
    COUNT(*) as count,
    AVG(child_count) as avg_children,
    AVG(atom_count) as avg_atoms,
    SUM(atom_count) as total_atoms
FROM relation
GROUP BY depth
ORDER BY depth;

COMMIT;
