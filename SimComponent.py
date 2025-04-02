-- Create per-program views
CREATE OR REPLACE VIEW bom_completion_snapshot_xm30 AS
SELECT 
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM delta.`/Volumes/poc/default/gold_bom_snapshot/xm30`;

CREATE OR REPLACE VIEW bom_completion_snapshot_xy22 AS
SELECT 
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM delta.`/Volumes/poc/default/gold_bom_snapshot/xy22`;

CREATE OR REPLACE VIEW bom_completion_snapshot_zq19 AS
SELECT 
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM delta.`/Volumes/poc/default/gold_bom_snapshot/zq19`;

-- Unified all-program view for dashboarding
CREATE OR REPLACE VIEW bom_completion_all_variants AS
SELECT 
  'xm30' AS program,
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM bom_completion_snapshot_xm30

UNION ALL

SELECT 
  'xy22' AS program,
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM bom_completion_snapshot_xy22

UNION ALL

SELECT 
  'zq19' AS program,
  snapshot_date,
  variant_id,
  source,
  make_or_buy,
  percent_matched,
  total_parts,
  matched_parts,
  quantity_mismatches,
  missing_parts
FROM bom_completion_snapshot_zq19;