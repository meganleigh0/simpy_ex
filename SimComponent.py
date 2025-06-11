# ------------------------------------------------------------
# ONE‑CELL MAPPING DISCOVERY
# ------------------------------------------------------------
import re, numpy as np, pandas as pd

# ────────────────────────────────────────────────────────────
BASE_YEAR  = 2022      # the “anchor” year you trust is already synced
TOLERANCE  = 1e-10     # float tolerance for fuzzy equality
# ────────────────────────────────────────────────────────────

# ── 1. ensure Burden Pool is the index in other_rates ─────────────────────────
if other_rates.index.name != 'Burden Pool':
    other_rates = other_rates.set_index('Burden Pool')

# grab first column that contains the BASE_YEAR string (handles CY2022, CY2022.1 …)
base_col = next(col for col in other_rates.columns if re.search(str(BASE_YEAR), col))
other_base = (
    other_rates[[base_col]]
       .rename(columns={base_col: 'Rate'})
       .reset_index()                               # columns: Burden Pool | Rate
)

# ── 2. slice BURDEN_RATE down to that same year & melt numeric columns ───────
br = BURDEN_RATE.copy()
br['Year'] = pd.to_numeric(br['# Date'], errors='coerce')
br_base = br.loc[br['Year'] == BASE_YEAR].copy()

num_cols  = br_base.select_dtypes(include='number').columns.tolist()

br_long = (
    br_base[['Burden Pool', 'Description', 'Year'] + num_cols]
      .melt(id_vars=['Burden Pool', 'Description', 'Year'],
            var_name='BR_Column',
            value_name='Rate')
      .dropna(subset=['Rate'])
)

# ── 3. vectorised “nearest join” on the rate value ────────────────────────────
idx = (np.abs(other_base['Rate'].values[:, None] - br_long['Rate'].values) <= TOLERANCE)

# where there is *no* match, idx row is all False
row_match  = idx.argmax(axis=1)             # index of first True in each row
has_match  = idx.any(axis=1)

matched    = br_long.iloc[row_match].reset_index(drop=True)
matched.loc[~has_match, :] = np.nan         # keep NaNs for pools we couldn’t map

mapping_df = pd.concat([other_base, matched.add_suffix('_BR')], axis=1)
mapping_df = (
    mapping_df
      .loc[mapping_df['Burden Pool_BR'].notna()]   # keep successful hits
      .rename(columns={
          'Burden Pool'      : 'Other_BurdenPool',
          'Burden Pool_BR'   : 'BR_BurdenPool',
          'Description_BR'   : 'BR_Description',
          'BR_Column_BR'     : 'BR_NumericColumn'
      })
      [['Other_BurdenPool', 'BR_BurdenPool',
        'BR_Description',  'BR_NumericColumn', 'Rate']]
      .sort_values('Other_BurdenPool')
      .reset_index(drop=True)
)

print(">>> AUTOMATIC MAPPING <<<")
display(mapping_df)

# (optional) save:
# mapping_df.to_csv('burden_mapping.csv', index=False)

# You can now feed `mapping_df` into your real pipeline:
#    for each row in mapping_df:
#        merge on (BR_BurdenPool, # Date) and update the column stored in BR_NumericColumn