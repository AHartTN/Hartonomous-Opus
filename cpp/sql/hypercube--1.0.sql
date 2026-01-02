-- Hypercube extension SQL definitions
-- Provides 4D Hilbert curve indexing for semantic coordinate system

-- Convert 4D coordinates to Hilbert index
CREATE OR REPLACE FUNCTION hypercube_coords_to_hilbert(
    x bigint, y bigint, z bigint, m bigint
) RETURNS TABLE(hilbert_lo bigint, hilbert_hi bigint)
AS 'hypercube', 'hypercube_coords_to_hilbert'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Convert Hilbert index to 4D coordinates
CREATE OR REPLACE FUNCTION hypercube_hilbert_to_coords(
    hilbert_lo bigint, hilbert_hi bigint
) RETURNS TABLE(x bigint, y bigint, z bigint, m bigint)
AS 'hypercube', 'hypercube_hilbert_to_coords'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- BLAKE3 hash of arbitrary data
CREATE OR REPLACE FUNCTION hypercube_blake3(data bytea)
RETURNS bytea
AS 'hypercube', 'hypercube_blake3'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- BLAKE3 hash of a Unicode codepoint (UTF-8 encoded)
CREATE OR REPLACE FUNCTION hypercube_blake3_codepoint(codepoint integer)
RETURNS bytea
AS 'hypercube', 'hypercube_blake3_codepoint'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Map a codepoint to its full atom representation
CREATE OR REPLACE FUNCTION hypercube_map_codepoint(codepoint integer)
RETURNS TABLE(
    x bigint, 
    y bigint, 
    z bigint, 
    m bigint, 
    hilbert_lo bigint, 
    hilbert_hi bigint, 
    hash bytea, 
    category text
)
AS 'hypercube', 'hypercube_map_codepoint'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get category for a codepoint
CREATE OR REPLACE FUNCTION hypercube_categorize(codepoint integer)
RETURNS text
AS 'hypercube', 'hypercube_categorize'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Seed all Unicode atoms (set-returning function)
CREATE OR REPLACE FUNCTION hypercube_seed_atoms()
RETURNS TABLE(
    codepoint integer,
    x bigint, 
    y bigint, 
    z bigint, 
    m bigint, 
    hilbert_lo bigint, 
    hilbert_hi bigint, 
    hash bytea, 
    category text
)
AS 'hypercube', 'hypercube_seed_atoms'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Compute centroid of multiple 4D points
CREATE OR REPLACE FUNCTION hypercube_centroid(
    x bigint[], y bigint[], z bigint[], m bigint[]
) RETURNS TABLE(x bigint, y bigint, z bigint, m bigint)
AS 'hypercube', 'hypercube_centroid'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Check if point is on hypercube surface
CREATE OR REPLACE FUNCTION hypercube_is_on_surface(
    x bigint, y bigint, z bigint, m bigint
) RETURNS boolean
AS 'hypercube', 'hypercube_is_on_surface'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Hilbert distance between two indices
CREATE OR REPLACE FUNCTION hypercube_hilbert_distance(
    lo1 bigint, hi1 bigint, lo2 bigint, hi2 bigint
) RETURNS TABLE(lo bigint, hi bigint)
AS 'hypercube', 'hypercube_hilbert_distance'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Convenience function: create POINTZM from hypercube coordinates
-- Stores raw uint32 values as doubles - NO normalization
-- PostGIS double has 53-bit mantissa, more than enough for 32-bit values
CREATE OR REPLACE FUNCTION hypercube_to_pointzm(
    x bigint, y bigint, z bigint, m bigint
) RETURNS geometry(POINTZM, 0)
AS $$
    SELECT ST_MakePoint(
        x::float8,  -- Raw uint32 value (no normalization)
        y::float8,
        z::float8,
        m::float8
    )::geometry(POINTZM, 0);
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

-- Convenience function: seed atom table directly
CREATE OR REPLACE FUNCTION hypercube_populate_atoms()
RETURNS bigint
AS $$
DECLARE
    inserted_count bigint;
BEGIN
    INSERT INTO atom (id, codepoint, category, coords, hilbert_lo, hilbert_hi)
    SELECT 
        s.hash,
        s.codepoint,
        s.category::atom_category,
        hypercube_to_pointzm(s.x, s.y, s.z, s.m),
        s.hilbert_lo,
        s.hilbert_hi
    FROM hypercube_seed_atoms() s
    ON CONFLICT (codepoint) DO NOTHING;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Create operator class for Hilbert index ordering
-- This allows ORDER BY on (hilbert_hi, hilbert_lo) to be efficient

-- Composite type for Hilbert index
CREATE TYPE hilbert_index AS (
    hi bigint,
    lo bigint
);

-- Comparison function for hilbert_index
CREATE OR REPLACE FUNCTION hilbert_index_cmp(a hilbert_index, b hilbert_index)
RETURNS integer
AS $$
BEGIN
    IF a.hi < b.hi THEN RETURN -1;
    ELSIF a.hi > b.hi THEN RETURN 1;
    ELSIF a.lo < b.lo THEN RETURN -1;
    ELSIF a.lo > b.lo THEN RETURN 1;
    ELSE RETURN 0;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION hilbert_index_lt(a hilbert_index, b hilbert_index)
RETURNS boolean AS $$
    SELECT hilbert_index_cmp(a, b) < 0;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION hilbert_index_le(a hilbert_index, b hilbert_index)
RETURNS boolean AS $$
    SELECT hilbert_index_cmp(a, b) <= 0;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION hilbert_index_eq(a hilbert_index, b hilbert_index)
RETURNS boolean AS $$
    SELECT hilbert_index_cmp(a, b) = 0;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION hilbert_index_ge(a hilbert_index, b hilbert_index)
RETURNS boolean AS $$
    SELECT hilbert_index_cmp(a, b) >= 0;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION hilbert_index_gt(a hilbert_index, b hilbert_index)
RETURNS boolean AS $$
    SELECT hilbert_index_cmp(a, b) > 0;
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR < (
    LEFTARG = hilbert_index,
    RIGHTARG = hilbert_index,
    FUNCTION = hilbert_index_lt,
    COMMUTATOR = >,
    NEGATOR = >=,
    RESTRICT = scalarltsel,
    JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = hilbert_index,
    RIGHTARG = hilbert_index,
    FUNCTION = hilbert_index_le,
    COMMUTATOR = >=,
    NEGATOR = >,
    RESTRICT = scalarlesel,
    JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = hilbert_index,
    RIGHTARG = hilbert_index,
    FUNCTION = hilbert_index_eq,
    COMMUTATOR = =,
    NEGATOR = <>,
    RESTRICT = eqsel,
    JOIN = eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR >= (
    LEFTARG = hilbert_index,
    RIGHTARG = hilbert_index,
    FUNCTION = hilbert_index_ge,
    COMMUTATOR = <=,
    NEGATOR = <,
    RESTRICT = scalargesel,
    JOIN = scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = hilbert_index,
    RIGHTARG = hilbert_index,
    FUNCTION = hilbert_index_gt,
    COMMUTATOR = <,
    NEGATOR = <=,
    RESTRICT = scalargtsel,
    JOIN = scalargtjoinsel
);

CREATE OPERATOR CLASS hilbert_index_ops
    DEFAULT FOR TYPE hilbert_index USING btree AS
        OPERATOR 1 <,
        OPERATOR 2 <=,
        OPERATOR 3 =,
        OPERATOR 4 >=,
        OPERATOR 5 >,
        FUNCTION 1 hilbert_index_cmp(hilbert_index, hilbert_index);
