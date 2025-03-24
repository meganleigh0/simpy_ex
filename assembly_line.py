import pandas as pd

# -----------------------------------------------------------------------------
# 1. Mock Data Creation
# -----------------------------------------------------------------------------
ebom_df = pd.DataFrame({
    'PART_NUMBER':   ['PN-001', 'PN-001', 'PN-002', 'PN-003', 'PN-003'],
    'Item Template': ['Purchase Item', 'Purchase Item', 'Finished Good', 'Reference Item', 'Reference Item'],
    'Quantity':      [10, 2, 1,  5, 10]
})

mbom_tc_df = pd.DataFrame({
    'PART_NUMBER':   ['PN-001', 'PN-001', 'PN-004'],
    'Item Template': ['Purchase Item', 'Purchase Item', 'GFM'],
    'Quantity':      [8, 4, 3]
})

mbom_oracle_df = pd.DataFrame({
    'PART_NUMBER': ['PN-001', 'PN-002', 'PN-003', 'PN-005'],
    'Item Type':   ['Purchase Item', 'Finished Good', 'Reference Item', 'Purchase Item'],
    'Quantity':    [12, 2, 15, 1]
})

# -----------------------------------------------------------------------------
# 2. Aggregate Each DataFrame by PART_NUMBER
#    - Summation of "Quantity"
#    - "first" of "Item Template" or "Item Type"
# -----------------------------------------------------------------------------
ebom_agg = (
    ebom_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Template': 'first',  # or another approach if needed
        'Quantity': 'sum'
    })
)

mbom_tc_agg = (
    mbom_tc_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Template': 'first',
        'Quantity': 'sum'
    })
)

mbom_oracle_agg = (
    mbom_oracle_df
    .groupby('PART_NUMBER', as_index=False)
    .agg({
        'Item Type': 'first',
        'Quantity': 'sum'
    })
)

# -----------------------------------------------------------------------------
# 3. Rename "Item Type" -> "Item Template" in the Oracle MBOM for consistency
# -----------------------------------------------------------------------------
mbom_oracle_agg.rename(columns={'Item Type': 'Item Template'}, inplace=True)

# -----------------------------------------------------------------------------
# 4. Merge the Aggregated DataFrames
#    - Merge EBOM & Teamcenter
#    - Then merge that result with Oracle
# -----------------------------------------------------------------------------
compare_tc = pd.merge(
    ebom_agg,
    mbom_tc_agg,
    on='PART_NUMBER',
    how='outer',
    suffixes=('_EBOM', '_TC')
)

compare_all = pd.merge(
    compare_tc,
    mbom_oracle_agg,
    on='PART_NUMBER',
    how='outer',
    suffixes=('_TC', '_ORACLE')
)

# Now we have columns:
# PART_NUMBER
# Item Template_EBOM, Quantity_EBOM
# Item Template_TC,   Quantity_TC
# Item Template_ORACLE, Quantity_ORACLE

# -----------------------------------------------------------------------------
# 5. Create Comparison Columns
# -----------------------------------------------------------------------------
# (A) Compare Item Templates
compare_all['Template_Match_EBOM_TC'] = (
    compare_all['Item Template_EBOM'] == compare_all['Item Template_TC']
)
compare_all['Template_Match_EBOM_Oracle'] = (
    compare_all['Item Template_EBOM'] == compare_all['Item Template_ORACLE']
)

# (B) Compare Quantities (EBOM minus MBOM)
compare_all['Qty_Diff_EBOM_vs_TC'] = (
    compare_all['Quantity_EBOM'].fillna(0) - compare_all['Quantity_TC'].fillna(0)
)
compare_all['Qty_Diff_EBOM_vs_Oracle'] = (
    compare_all['Quantity_EBOM'].fillna(0) - compare_all['Quantity_ORACLE'].fillna(0)
)

print(compare_all)
