SELECT 
  snapshot_date,
  source,
  make_or_buy,
  percent_matched
FROM xm30_bom_completion_snapshot
ORDER BY snapshot_date, source, make_or_buy
