-- 1. Combine all program tables into one view
CREATE OR REPLACE VIEW gold_bom_completion_all AS
SELECT * FROM london_abc_bom_completion_snapshot
UNION ALL
SELECT * FROM london_def_bom_completion_snapshot
-- Add more UNIONs as needed for other programs
;

-- 2. Split into Make and Buy views
CREATE OR REPLACE VIEW gold_bom_completion_make AS
SELECT *
FROM gold_bom_completion_all
WHERE LOWER(make_or_buy) = 'make';

CREATE OR REPLACE VIEW gold_bom_completion_buy AS
SELECT *
FROM gold_bom_completion_all
WHERE LOWER(make_or_buy) = 'buy';

-- 3a. Percent Matched Over Time View
CREATE OR REPLACE VIEW gold_percent_matched_trend AS
SELECT 
    snapshot_date,
    variant_id,
    make_or_buy,
    program,
    ROUND(AVG(percent_matched), 2) AS avg_percent_matched
FROM gold_bom_completion_all
GROUP BY snapshot_date, variant_id, make_or_buy, program
ORDER BY snapshot_date;

-- 3b. Missing Parts Summary View
CREATE OR REPLACE VIEW gold_missing_parts_summary AS
SELECT 
    snapshot_date,
    variant_id,
    make_or_buy,
    program,
    SUM(missing_parts) AS total_missing_parts
FROM gold_bom_completion_all
GROUP BY snapshot_date, variant_id, make_or_buy, program
ORDER BY snapshot_date;

-- 3c. Quantity Mismatches Summary View
CREATE OR REPLACE VIEW gold_qty_mismatch_summary AS
SELECT 
    snapshot_date,
    variant_id,
    make_or_buy,
    program,
    SUM(quantity_mismatches) AS total_quantity_mismatches
FROM gold_bom_completion_all
GROUP BY snapshot_date, variant_id, make_or_buy, program
ORDER BY snapshot_date;

-- 4. Latest Snapshot per Program and Variant
CREATE OR REPLACE VIEW gold_latest_snapshot AS
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY program, variant_id 
               ORDER BY snapshot_date DESC
           ) AS rn
    FROM gold_bom_completion_all
)
WHERE rn = 1;