import pandas as pd
import numpy as np

# --- assumes you already have merged_hours from your merges ---
mh = merged_hours.copy()

# Normalize column names so they're easy to reference
mh.columns = (
    mh.columns
      .str.strip()
      .str.replace('#', '', regex=False)
      .str.replace(' ', '_', regex=False)
      .str.replace('/', '_', regex=False)
      .str.replace('-', '_', regex=False)
)

# Make sure these columns exist (create as 0 if not present)
expected_cols = [
    'Usr_Org', 'Make_Buy',
    'SEP_CWS_Hours', 'SEP_Part_Count',
    'XM30_CWS_Hours', 'XM30_Part_Count',
    'M10_CWS_Hours',  'M10_Part_Count'
]
for c in expected_cols:
    if c not in mh.columns:
        mh[c] = 0.0 if 'Hours' in c or 'Count' in c else mh.get(c, np.nan)

mh = mh.fillna(0)

# Per-row averages (guard against divide-by-zero)
mh['SEP_Avg_Hours_per_Part']   = np.where(mh['SEP_Part_Count']>0,   mh['SEP_CWS_Hours']/mh['SEP_Part_Count'],   0.0)
mh['XM30_Avg_Hours_per_Part']  = np.where(mh['XM30_Part_Count']>0,  mh['XM30_CWS_Hours']/mh['XM30_Part_Count'], 0.0)
mh['M10_Avg_Hours_per_Part']   = np.where(mh['M10_Part_Count']>0,   mh['M10_CWS_Hours']/mh['M10_Part_Count'],   0.0)

# Org x Make/Buy summary (sums)
org_mb = (
    mh.groupby(['Usr_Org','Make_Buy'], as_index=False)
      .agg(
          SEP_Hours=('SEP_CWS_Hours','sum'),
          SEP_Parts=('SEP_Part_Count','sum'),
          XM30_Hours=('XM30_CWS_Hours','sum'),
          XM30_Parts=('XM30_Part_Count','sum'),
          M10_Hours=('M10_CWS_Hours','sum'),
          M10_Parts=('M10_Part_Count','sum')
      )
)

# Total parts per org (for BUY % later)
org_totals = org_mb.groupby('Usr_Org', as_index=False)['SEP_Parts'].sum().rename(columns={'SEP_Parts':'Org_SEP_Parts_Total'})
org_mb = org_mb.merge(org_totals, on='Usr_Org', how='left')

# BUY percentage on SEP parts within each org
org_mb['SEP_Buyer_Pct'] = np.where(
    (org_mb['Make_Buy'].str.upper()=='BUY') & (org_mb['Org_SEP_Parts_Total']>0),
    org_mb['SEP_Parts'] / org_mb['Org_SEP_Parts_Total'],
    0.0
)

# Wide view (handy for dashboards)
wide = (
    org_mb
      .pivot(index='Usr_Org', columns='Make_Buy', values=['SEP_Hours','SEP_Parts','XM30_Hours','M10_Hours'])
      .fillna(0)
)
wide.columns = ['{}_{}'.format(a,b) for a,b in wide.columns]
wide = wide.reset_index()

# Helpful outputs:
print("Long (org x Make/Buy):")
display(org_mb.sort_values(['Usr_Org','Make_Buy']).reset_index(drop=True))
print("\nWide (one row per org):")
display(wide)