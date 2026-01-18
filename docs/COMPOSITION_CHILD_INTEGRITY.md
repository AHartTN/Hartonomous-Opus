# Composition Child Insertion Integrity

This document explains the composition child insertion logic, data integrity guarantees, and maintenance procedures for the Hypercube database system.

## Overview

The composition child insertion system manages the relationship between compositions and their child atoms/compositions. This system was redesigned to fix the "510 bad compositions" issue by implementing proper deduplication and integrity checks.

## Insertion Pipeline

### High-Level Flow

1. **Temp Table Creation**: Temporary tables are created for bulk data staging
2. **Data Staging**: Composition and child records are COPY'd into temporary tables
3. **Deduplication**: Compositions are inserted with `ON CONFLICT DO NOTHING`
4. **Child Filtering**: Children are inserted only for newly created compositions
5. **Validation**: Child counts are verified against expected values

### Detailed Implementation

The insertion process uses `hypercube::db::insert_compositions()` in [`cpp/src/db/insert.cpp`](cpp/src/db/insert.cpp:38):

```cpp
bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps)
```

#### Step 1: Temp Table Setup

```sql
CREATE TEMP TABLE temp_composition (...) ON COMMIT DROP;
CREATE TEMP TABLE temp_composition_child (...) ON COMMIT DROP;
```

#### Step 2: Bulk Data Loading

- Compositions are COPY'd with labels computed in C++
- Children are COPY'd with ordinals and types ('A' for atoms, 'C' for compositions)

#### Step 3: Deduplication Insertion

```sql
INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi)
SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi
FROM temp_composition
ON CONFLICT (id) DO NOTHING;
```

#### Step 4: Child Insertion (Filtered)

Children are inserted only for compositions that were actually inserted:

```sql
INSERT INTO composition_child (composition_id, ordinal, child_type, child_id)
SELECT tc.composition_id, tc.ordinal, tc.child_type, tc.child_id
FROM temp_composition_child tc
WHERE EXISTS (SELECT 1 FROM composition c WHERE c.id = tc.composition_id)
ON CONFLICT DO NOTHING;
```

This prevents duplicate child links for existing compositions.

## Data Integrity Guarantees

### Automatic Child Count Maintenance

A database trigger automatically maintains the `child_count` field in the composition table:

**Trigger Function**: `maintain_composition_child_count()` in [`sql/functions/compositions/maintain_child_count_integrity.sql`](sql/functions/compositions/maintain_child_count_integrity.sql:14)

```sql
CREATE TRIGGER trigger_maintain_child_count
    AFTER INSERT OR DELETE OR UPDATE OF composition_id ON composition_child
    FOR EACH ROW EXECUTE FUNCTION maintain_composition_child_count();
```

The trigger:
- Increments `child_count` on child insertion
- Decrements `child_count` on child deletion
- Adjusts counts when `composition_id` changes

**Constraint**: Non-negative child counts are enforced:
```sql
ALTER TABLE composition ADD CONSTRAINT check_child_count_non_negative CHECK (child_count >= 0);
```

### Foreign Key Constraints

The `composition_child` table has a foreign key to `composition(id)` with `ON DELETE CASCADE`, ensuring orphaned child records are automatically removed when compositions are deleted.

### Child Reference Validation (Defined but Not Enforced)

A validation function `validate_composition_child()` exists in [`sql/schema/03_constraints.sql`](sql/schema/03_constraints.sql:2) but is **not applied as a trigger**. This allows insertion of child links to non-existent atoms/compositions during bulk loading, which is intentional for performance reasons during ingestion.

## Validation vs Enforcement

### Validation (Reporting)

Validation checks data integrity and reports violations without preventing them:

- **Purpose**: Detect inconsistencies for monitoring and alerting
- **Implementation**: Query functions like `atom_child_count()` and `atom_children()`
- **Action**: Log warnings, send alerts, but allow operation to continue

### Enforcement (Automatic Fixing)

Enforcement automatically maintains consistency:

- **Purpose**: Prevent or fix inconsistencies automatically
- **Implementation**: Database triggers and constraints
- **Action**: Modify data to maintain consistency, or reject operations

**Current System**: Uses enforcement for child counts, validation for child existence.

## Troubleshooting Child Count Mismatches

### Detection

Count mismatches can be detected by comparing the `child_count` field with actual child records:

```sql
SELECT c.id, c.child_count, COUNT(cc.*) as actual_count
FROM composition c
LEFT JOIN composition_child cc ON c.id = cc.composition_id
GROUP BY c.id, c.child_count
HAVING c.child_count != COUNT(cc.*);
```

### Common Causes

1. **Trigger Failure**: Database trigger not firing during manual operations
2. **Concurrent Modifications**: Race conditions during parallel insertions
3. **Manual Data Changes**: Direct SQL operations bypassing triggers
4. **Import Errors**: Bulk imports that don't update counts properly

### Resolution

#### Option 1: Recalculate All Counts

Use the maintenance function to fix all compositions:

```sql
SELECT recalculate_all_child_counts();
```

This function updates all `child_count` values to match actual child records.

#### Option 2: Selective Repair

For specific compositions:

```sql
UPDATE composition
SET child_count = (
    SELECT COUNT(*)
    FROM composition_child
    WHERE composition_id = composition.id
)
WHERE id = 'your-composition-id';
```

### Prevention

- Always use the `insert_compositions()` C++ function instead of raw SQL
- Avoid direct modifications to `composition_child` table
- Monitor for count mismatches in production logs

## Examples: Normal vs Problematic Scenarios

### Normal Insertion (All Dependencies Exist)

```sql
-- Atoms exist
INSERT INTO atom (id, codepoint, value, geom, hilbert_lo, hilbert_hi) VALUES
('\\x1111111111111111111111111111111111111111111111111111111111111111', 65, '\\x41', 'POINTZM(0 0 0 0)', 0, 0);

-- Composition insertion succeeds
-- Result: composition inserted, child_count = 1, child link created
```

### Missing Dependencies (Allowed for Performance)

```sql
-- Atom does not exist
-- Composition insertion still succeeds (child link created)
-- Result: composition inserted, child_count = 1, dangling child reference
```

### Partial Batch Insertion

```sql
-- Batch: [existing_comp, new_comp]
-- existing_comp: already in database
-- new_comp: new composition

-- Result:
-- existing_comp: skipped (ON CONFLICT DO NOTHING)
-- new_comp: inserted
-- Children: only inserted for new_comp
-- child_count: maintained correctly for both
```

### Problematic: Manual Child Insertion

```sql
-- Direct insertion bypassing trigger
INSERT INTO composition_child (composition_id, ordinal, child_type, child_id)
VALUES ('\\xcomp_id', 0, 'A', '\\xatom_id');

-- Result: child_count not updated, mismatch occurs
```

## Maintenance Procedures

### Daily Integrity Checks

Run automated checks to detect inconsistencies:

```sql
-- Count mismatches
SELECT COUNT(*) as mismatches
FROM (
    SELECT c.id
    FROM composition c
    LEFT JOIN composition_child cc ON c.id = cc.composition_id
    GROUP BY c.id, c.child_count
    HAVING c.child_count != COUNT(cc.*)
) t;

-- Orphaned child records
SELECT COUNT(*) as orphans
FROM composition_child cc
LEFT JOIN composition c ON cc.composition_id = c.id
WHERE c.id IS NULL;
```

### Weekly Recalculation

For systems with high write loads, perform weekly full recalculations:

```sql
-- During maintenance window
BEGIN;
SELECT recalculate_all_child_counts();
COMMIT;
```

### Emergency Repair

When mismatches are detected in production:

1. **Assess Impact**: Determine affected compositions and downstream effects
2. **Backup**: Create database backup before repair
3. **Repair**: Run recalculate_all_child_counts()
4. **Verify**: Re-run integrity checks
5. **Monitor**: Watch for recurrence

## Performance Considerations

### Bulk Insertion Strategy

The temp table + COPY + INSERT pattern provides:
- **No round-trips**: Single batch operations
- **Deduplication**: ON CONFLICT DO NOTHING prevents duplicates
- **Atomicity**: All-or-nothing transaction semantics
- **Performance**: COPY is faster than individual INSERTs

### Trigger Overhead

The child count maintenance trigger adds overhead:
- Each child insertion/deletion causes an UPDATE to composition
- Acceptable for normal operations, but monitor during bulk imports
- Can be temporarily disabled for maintenance (with manual recalculation)

## Future Improvements

### Planned Enhancements

1. **Apply Validation Trigger**: Enable `validate_composition_child()` trigger for stricter integrity
2. **Asynchronous Repair**: Background jobs to fix inconsistencies
3. **Integrity Monitoring**: Automated alerting for data inconsistencies
4. **Child Existence Indexing**: Indexes to speed up existence checks during validation

### Migration Notes

When enabling stricter validation:
- Existing dangling references must be resolved first
- May require multiple passes for complex dependency chains
- Consider impact on ingestion performance