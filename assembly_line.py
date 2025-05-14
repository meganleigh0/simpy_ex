import pandas as pd

# Load your original wide-format CSV
df = pd.read_csv("snapshot_04-22-2025.csv")

# Reshape percent columns into long format
df_long = df.melt(
    id_vars=[
        'snapshot_date', 'variant_id', 'source', 'make_or_buy',
        'total_parts', 'matched_parts', 'quantity_mismatches', 'missing_parts'
    ],
    value_vars=['percent_matched', 'percent_matched_qty'],
    var_name='match_type',
    value_name='percent_value'
)

# Optional: Clean match_type values
df_long['match_type'] = df_long['match_type'].map({
    'percent_matched': 'Percent Matched',
    'percent_matched_qty': 'Percent Matched Qty'
})

# Save to CSV (ready for Power BI)
df_long.to_csv("mbom_snapshot_long_format.csv", index=False)