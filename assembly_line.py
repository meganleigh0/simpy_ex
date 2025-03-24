# EBOM Aggregation
ebom_agg = (
    ebom_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Template': 'first',  # or another aggregator if needed
        'Quantity': 'sum'
    })
)

# Teamcenter MBOM Aggregation
mbom_tc_agg = (
    mbom_tc_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Template': 'first',  # or 'max', 'min', 'nunique', etc.
        'Quantity': 'sum'
    })
)

# Oracle MBOM Aggregation
mbom_oracle_agg = (
    mbom_oracle_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Type': 'first',
        'Quantity': 'sum'
    })
)

# Rename 'Item Type' -> 'Item Template' for consistent comparisons
mbom_oracle_agg.rename(columns={'Item Type': 'Item Template'}, inplace=True)

# Inspect aggregated shapes
print("EBOM_agg:\n", ebom_agg)
print("MBOM_TC_agg:\n", mbom_tc_agg)
print("MBOM_Oracle_agg:\n", mbom_oracle_agg)


# Merge EBOM & Teamcenter MBOM
compare_tc = ebom_agg.merge(
    mbom_tc_agg,
    on='PART_NUMBER',
    how='outer',  # or 'left', depending on your needs
    suffixes=('_EBOM', '_TC')
)

# Then merge that result with Oracle MBOM
compare_all = compare_tc.merge(
    mbom_oracle_agg,
    on='PART_NUMBER',
    how='outer',
    suffixes=('_TC', '_ORACLE')
)

print(compare_all)


compare_all['Template_Match_EBOM_TC'] = (
    compare_all['Item Template_EBOM'] == compare_all['Item Template_TC']
)

compare_all['Template_Match_EBOM_Oracle'] = (
    compare_all['Item Template_EBOM'] == compare_all['Item Template_ORACLE']
)

compare_all['Qty_Diff_EBOM_vs_TC'] = compare_all['Quantity_EBOM'].fillna(0) - compare_all['Quantity_TC'].fillna(0)
compare_all['Qty_Diff_EBOM_vs_Oracle'] = compare_all['Quantity_EBOM'].fillna(0) - compare_all['Quantity_ORACLE'].fillna(0)

print(compare_all)
