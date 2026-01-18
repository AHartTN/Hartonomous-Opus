param(
    [string]$Hostname = "hart-server",
    [int]$Port = 5432,
    [string]$User = "postgres",
    [string]$Database = "hypercube",
    [string]$Password = $env:PGPASSWORD
)

if (-not $Password) {
    $Password = "postgres"  # Fallback, but prefer env var
}

$env:PGPASSWORD = $Password

function Run-SQL {
    param([string]$Query, [string]$Label)

    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Cyan
    Write-Host " $Label" -ForegroundColor Cyan
    Write-Host "====================================================" -ForegroundColor Cyan

    $escaped = $Query.Replace("`n"," ")
    psql -h $Hostname -p $Port -U $User -d $Database -c "$escaped" 2>&1
}

function Run-SQL-File {
    param([string]$FilePath, [string]$Label)

    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host " $Label" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green

    psql -h $Hostname -p $Port -U $User -d $Database -f $FilePath 2>&1
}

# ============================================================
# COMPOSITION FIX VALIDATION SCRIPT
# ============================================================

Write-Host "Composition Fix Validation Test" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Yellow
Write-Host "Database: $Database on ${Hostname}:$Port as $User"
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

# 1. Apply Integrity Triggers (if not already applied)
Run-SQL-File "sql/functions/compositions/maintain_child_count_integrity.sql" "Applying Integrity Triggers"

# 2. Query Bad Compositions Before Fix
$badBefore = Run-SQL @"
SELECT COUNT(*) AS bad_compositions
FROM composition c
LEFT JOIN composition_child cc ON cc.composition_id = c.id
GROUP BY c.id, c.child_count
HAVING COUNT(cc.*) <> c.child_count AND c.child_count > 0;
"@ "Bad Compositions Before Fix (child_count > 0 but mismatched children)"

# 3. Recalculate Child Counts to Repair Existing Data
Run-SQL @"
SELECT recalculate_all_child_counts();
"@ "Recalculating Child Counts"

# 4. Query Bad Compositions After Fix
$badAfter = Run-SQL @"
SELECT COUNT(*) AS bad_compositions
FROM composition c
LEFT JOIN composition_child cc ON cc.composition_id = c.id
GROUP BY c.id, c.child_count
HAVING COUNT(cc.*) <> c.child_count AND c.child_count > 0;
"@ "Bad Compositions After Fix (child_count > 0 but mismatched children)"

# 5. Detailed Report
Run-SQL @"
SELECT c.id, c.label, c.child_count, COUNT(cc.*) AS actual_children,
       CASE WHEN c.child_count = 1 AND COUNT(cc.*) = 0 THEN 'PREVIOUSLY BROKEN: child_count=1 but no children'
            WHEN COUNT(cc.*) <> c.child_count THEN 'MISMATCHED'
            ELSE 'OK' END AS status
FROM composition c
LEFT JOIN composition_child cc ON cc.composition_id = c.id
GROUP BY c.id, c.label, c.child_count
HAVING COUNT(cc.*) <> c.child_count
ORDER BY c.id
LIMIT 20;
"@ "Detailed Fix Results (Sample of 20)"

# 6. Final Integrity Check
Run-SQL @"
SELECT COUNT(*) AS total_compositions,
       COUNT(CASE WHEN child_count > 0 THEN 1 END) AS compositions_with_children,
       COUNT(CASE WHEN child_count = 0 THEN 1 END) AS leaf_compositions,
       SUM(child_count) AS total_child_references,
       (SELECT COUNT(*) FROM composition_child) AS actual_child_records
FROM composition;
"@ "Composition Integrity Summary"

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host " VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Script completed safely. Triggers are now active to prevent future integrity issues."
Write-Host "Run this script again to verify ongoing integrity." -ForegroundColor Gray