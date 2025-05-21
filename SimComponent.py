import pandas as pd

# Reshape percent columns into a long format for stacked bar chart
percent_melted = SNAPSHOT_DATA.melt(
    id_vars=["snapshot_date", "source", "make_or_buy", "total_parts"],
    value_vars=["percent_matched", "percent_matched_qty"],
    var_name="percent_type",
    value_name="percent_value"
)

# Save this to CSV for Power BI
percent_melted.to_csv("reshaped_snapshot_percent_data.csv", index=False)
