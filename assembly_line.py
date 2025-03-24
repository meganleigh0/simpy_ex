import pandas as pd

# -----------------------------------------------------------------------------
# 1. Example DataFrames (mock data)
# -----------------------------------------------------------------------------
ebom_df = pd.DataFrame({
    'PART_NUMBER':  ['PN-001', 'PN-002', 'PN-003', 'PN-004'],
    'Item Template': ['Purchase Item', 'Finished Good', 'Reference Item', 'GFM'],
    'Quantity':      [10, 2, 5, 8]
})

mbom_tc_df = pd.DataFrame({
    'PART_NUMBER':   ['PN-001', 'PN-002', 'PN-003', 'PN-005'],
    'Item Template': ['Purchase Item', 'Finished Good', 'Reference Item', 'Purchase Item'],
    'Quantity':      [10, 2, 7, 1]
})

mbom_oracle_df = pd.DataFrame({
    'PART_NUMBER': ['PN-001', 'PN-002', 'PN-004'],
    'Item Type':   ['Purchase Item', 'Finished Good', 'GFM'],
    'Quantity':    [10, 3, 8]
})

# -----------------------------------------------------------------------------
# 2. Rename columns to have a consistent key for comparison (optional)
#    - 'Item Type' in Oracle MBOM becomes 'Item Template'
# -----------------------------------------------------------------------------
mbom_oracle_renamed = mbom_oracle_df.rename(columns={'Item Type': 'Item Template'})

# -----------------------------------------------------------------------------
# 3. Merge all DataFrames
#    - Use how='outer' to capture parts that appear in any DF but not the others
#    - Suffixes to distinguish columns from each source
# -----------------------------------------------------------------------------
compare_df = ebom_df.merge(
    mbom_tc_df,
    on='PART_NUMBER',
    how='outer',
    suffixes=('_EBOM', '_TC')
).merge(
    mbom_oracle_renamed,
    on='PART_NUMBER',
    how='outer',
    suffixes=('_TC', '_ORACLE')
)

# Now compare_df columns are:
# [
#   'PART_NUMBER',
#   'Item Template_EBOM', 'Quantity_EBOM',
#   'Item Template_TC',   'Quantity_TC',
#   'Item Template',      'Quantity'   <-- from Oracle rename
# ]

# For clarity, rename the last two columns:
compare_df.rename(
    columns={
        'Item Template': 'Item Template_ORACLE',
        'Quantity': 'Quantity_ORACLE'
    },
    inplace=True
)

# -----------------------------------------------------------------------------
# 4. Add comparison columns
# -----------------------------------------------------------------------------

# (a) Check if Item Template matches EBOM vs Teamcenter
compare_df['Template_Match_EBOM_TC'] = (
    compare_df['Item Template_EBOM'] == compare_df['Item Template_TC']
)

# (b) Check if Item Template matches EBOM vs Oracle
compare_df['Template_Match_EBOM_Oracle'] = (
    compare_df['Item Template_EBOM'] == compare_df['Item Template_ORACLE']
)

# (c) Compare quantities (simple difference: EBOM minus MBOM)
#     If you just want a boolean match vs mismatch, compare equality or within tolerance.
compare_df['Qty_Diff_EBOM_vs_TC'] = compare_df['Quantity_EBOM'].fillna(0) - compare_df['Quantity_TC'].fillna(0)
compare_df['Qty_Diff_EBOM_vs_Oracle'] = compare_df['Quantity_EBOM'].fillna(0) - compare_df['Quantity_ORACLE'].fillna(0)

# (d) Optionally, create a boolean for quantity match
compare_df['Qty_Match_EBOM_TC'] = compare_df['Qty_Diff_EBOM_vs_TC'] == 0
compare_df['Qty_Match_EBOM_Oracle'] = compare_df['Qty_Diff_EBOM_vs_Oracle'] == 0

# -----------------------------------------------------------------------------
# 5. Inspect/Use the merged comparison DataFrame
# -----------------------------------------------------------------------------
print(compare_df)
