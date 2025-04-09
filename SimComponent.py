SELECT 
  snapshot_date,
  make_or_buy,
  source,
  'Percent Matched' AS metric_type,
  percent_matched AS metric_value
FROM poc.default.xm30_bom_completion_snapshot

UNION ALL

SELECT 
  snapshot_date,
  make_or_buy,
  source,
  'Percent Matched Qty' AS metric_type,
  percent_matched_qty AS metric_value
FROM poc.default.xm30_bom_completion_snapshot