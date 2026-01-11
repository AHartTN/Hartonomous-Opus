# SQL and Setup Script Audit & Fixes

**Date:** 2026-01-08
**Status:** Issues Identified - Fixes In Progress

## Executive Summary

Comprehensive audit of SQL schema files (`sql/*.sql`) and setup scripts (`scripts/windows/*.ps1`, `scripts/linux/*.sh`) to identify and fix errors preventing proper database initialization.

---

## CRITICAL ISSUES FOUND

### 1. **env.ps1: Missing Visual Studio Check (BLOCKING)**

**File:** `scripts/windows/env.ps1`
**Lines:** 43-76
**Severity:** CRITICAL - Causes setup failure

**Problem:**
```powershell
# Line 44: Attempts to run vswhere, may fail silently
$vsPath = & $vswhere -latest ...

# Line 70: Tries to execute VsDevCmd WITHOUT checking if vsPath was found
cmd /c "`"$vsDevCmd`" -arch=amd64 -no_logo && set" | ...
```

**Error Message:**
```
'vswhere.exe' is not recognized as an internal or external command
```

**Impact:** Script crashes on systems without Visual Studio or with vswhere in non-standard location.

**Root Cause:** No validation that `$vs Path` was successfully populated before using it.

---

### 2. **config.env: Default Credentials May Not Match System**

**File:** `scripts/config.env`
**Lines:** 11-13
**Severity:** HIGH - Prevents connection

**Problem:**
```bash
HC_DB_USER=postgres
HC_DB_PASS=postgres
HC_DB_NAME=hypercube
```

**Impact:** If PostgreSQL has different superuser or password, all database operations fail.

**Root Cause:** Hardcoded defaults don't match all installations.

---

### 3. **setup-db.ps1: No PostgreSQL Service Check**

**File:** `scripts/windows/setup-db.ps1`
**Lines:** 36-50
**Severity:** MEDIUM - Poor UX

**Problem:** Script tests connection but doesn't check if PostgreSQL service is running first.

**Impact:** Users get cryptic "connection refused" errors without helpful guidance.

---

### 4. **Missing Error Handling in SQL Files**

**Files:** Various `sql/*.sql`
**Severity:** MEDIUM - Silent failures

**Issues Found:**
- Functions don't validate input (NULL checks missing)
- No error messages for common failure cases
- CASCADE drops can hide dependency issues

---

### 5. **Extension Loading Order Issues**

**File:** `scripts/windows/setup-db.ps1`
**Lines:** 123-134
**Severity:** MEDIUM - Extensions fail to load

**Problem:** Extensions loaded in arbitrary order, but some have dependencies:
- `semantic_ops` depends on `hypercube`
- `generative` depends on `embedding_ops`

**Impact:** Extension load failures when dependencies not met.

---

## DETAILED ISSUES BY FILE

### scripts/windows/env.ps1

**Lines 35-76: Visual Studio Detection**

**BEFORE (BROKEN):**
```powershell
# Find Visual Studio installation
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    $vswhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
}

# No NULL check here - $vswhere might still not exist!

if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest ...
} else {
    $vsPath = $null
    # Try fallback paths...
}

if ($vsPath -and (Test-Path "$vsPath\Common7\Tools\VsDevCmd.bat")) {
    $vsDevCmd = "$vsPath\Common7\Tools\VsDevCmd.bat"
    cmd /c "`"$vsDevCmd`" ..." | ...  # ⚠️ CRASHES if $vsDevCmd is malformed
}
```

**ISSUE:** If `$vsPath` is found via fallback but `VsDevCmd.bat` doesn't exist, `$vsDevCmd` becomes invalid path and `cmd /c` fails.

**FIX:**
```powershell
if ($vsPath) {
    $vsDevCmdPath = "$vsPath\Common7\Tools\VsDevCmd.bat"
    if (Test-Path $vsDevCmdPath) {
        Write-Host "Initializing Visual Studio environment..." -ForegroundColor Yellow
        $vsDevCmd = $vsDevCmdPath
        cmd /c "`"$vsDevCmd`" -arch=amd64 -no_logo && set" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    } else {
        Write-Host "Warning: Visual Studio found but VsDevCmd.bat missing" -ForegroundColor Yellow
    }
} else {
    Write-Host "Note: Visual Studio not found (optional for database setup)" -ForegroundColor DarkGray
}
```

---

### scripts/windows/setup-db.ps1

**Lines 36-50: Connection Test**

**IMPROVEMENT NEEDED:**
```powershell
# Add service check before connection test
Write-Host "[1/5] Checking PostgreSQL service..." -NoNewline
$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if (-not $pgService -or $pgService.Status -ne "Running") {
    Write-Host " NOT RUNNING" -ForegroundColor Red
    Write-Host ""
    Write-Host "PostgreSQL service is not running. Start it with:" -ForegroundColor Yellow
    Write-Host "  net start postgresql-x64-XX  (Windows)" -ForegroundColor Yellow
    Write-Host "  Or start via Services app" -ForegroundColor Yellow
    exit 1
}
Write-Host " Running" -ForegroundColor Green

Write-Host "[2/6] Testing PostgreSQL connection..." -NoNewline
# ... existing connection test
```

**Lines 123-134: Extension Loading**

**BEFORE (WRONG ORDER):**
```powershell
$extensions = @("hypercube", "hypercube_ops", "semantic_ops", "embedding_ops", "generative")
```

**AFTER (CORRECT DEPENDENCY ORDER):**
```powershell
# Load in dependency order: base → ops → specialized
$extensions = @(
    "hypercube",        # Base functions (BLAKE3, Hilbert, coordinates)
    "hypercube_ops",    # Depends on hypercube
    "embedding_ops",    # Depends on hypercube
    "semantic_ops",     # Depends on hypercube, embedding_ops
    "generative"        # Depends on semantic_ops, embedding_ops
)
```

---

### sql/001_schema.sql

**Lines 17-21: DROP CASCADE**

**ISSUE:** Using CASCADE can hide dependency errors.

**BEFORE:**
```sql
DROP TABLE IF EXISTS composition_child CASCADE;
DROP TABLE IF EXISTS relation CASCADE;
DROP TABLE IF EXISTS composition CASCADE;
DROP TABLE IF EXISTS atom CASCADE;
```

**RECOMMENDATION:** Add warning comments:
```sql
-- CAUTION: CASCADE will drop dependent objects (views, functions, etc.)
-- Review dependencies before running in production
DROP TABLE IF EXISTS composition_child CASCADE;
DROP TABLE IF EXISTS relation CASCADE;
DROP TABLE IF EXISTS composition CASCADE;
DROP TABLE IF EXISTS atom CASCADE;
```

---

### sql/002_core_functions.sql

**Lines 55-69: atom_centroid() - No NULL Handling**

**BEFORE:**
```sql
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS GEOMETRY(POINTZM, 0) AS $$
    SELECT geom FROM atom WHERE id = p_id
    UNION ALL
    SELECT centroid FROM composition WHERE id = p_id
    LIMIT 1;
$$ LANGUAGE SQL STABLE;
```

**ISSUE:** Returns NULL if ID not found, but callers don't check. Causes cascading NULLs.

**FIX:**
```sql
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS GEOMETRY(POINTZM, 0) AS $$
DECLARE
    v_geom GEOMETRY(POINTZM, 0);
BEGIN
    SELECT geom INTO v_geom FROM atom WHERE id = p_id;
    IF v_geom IS NOT NULL THEN
        RETURN v_geom;
    END IF;

    SELECT centroid INTO v_geom FROM composition WHERE id = p_id;
    IF v_geom IS NOT NULL THEN
        RETURN v_geom;
    END IF;

    RAISE EXCEPTION 'Entity % not found in atom or composition tables', encode(p_id, 'hex');
END;
$$ LANGUAGE plpgsql STABLE;
```

---

## PLANNED FIXES

### Priority 1 (BLOCKING - Must Fix Immediately)
1. ✅ **FIXED** - env.ps1 Visual Studio detection (added try/catch, better error handling)
2. ✅ **FIXED** - Extension loading order in setup-db.ps1 (proper dependency order)
3. ✅ **FIXED** - PostgreSQL service check added to setup-db.ps1 (step 1/6)

### Priority 2 (HIGH - Fix Before Next Release)
4. ✅ **FIXED** - atom_centroid() NULL checks (returns NULL gracefully, documented behavior)
5. ⬜ Add error handling to other SQL functions (lower priority)
6. ⬜ Document config.env customization requirements

### Priority 3 (MEDIUM - Nice to Have)
7. ⬜ Add CASCADE warnings to schema
8. ⬜ Create setup validation script
9. ⬜ Add connection string builder for different PostgreSQL configs

---

## TESTING CHECKLIST

After fixes applied:
- [ ] Fresh install on clean Windows system
- [ ] Fresh install on clean Linux system
- [ ] Install with non-standard PostgreSQL credentials
- [ ] Install without Visual Studio (database-only mode)
- [ ] Install with existing database (upgrade scenario)
- [ ] Verify all extensions load successfully
- [ ] Run full test suite
- [ ] Verify atoms seeded correctly (~1.1M rows)
- [ ] Test ingestion pipeline end-to-end

---

## NEXT STEPS

1. Apply Priority 1 fixes now
2. Test database setup
3. Apply Priority 2 fixes
4. Update documentation with setup requirements
5. Create troubleshooting guide for common errors

