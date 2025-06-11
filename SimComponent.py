# ----------------------------------------------------------
# ONE‑CELL SOLUTION – update BURDEN_RATE with other_rates
# ----------------------------------------------------------
import re
import numpy as np
import pandas as pd

# --- 1) make sure Burden Pool is an index in other_rates ---
if other_rates.index.name != 'Burden Pool':
    other_rates = other_rates.set_index('Burden Pool')

# --- 2) reshape to long and pull the year out of column names ---
other_long = (
    other_rates
      .reset_index()                              # Burden Pool back to a column
      .melt(id_vars='Burden Pool',
            var_name='year_col',
            value_name='rate')
      .dropna(subset=['rate'])                    # keep rows that actually have rates
)

# keep only the first 4‑digit sequence we find in the column name
other_long['Year'] = other_long['year_col'].str.extract(r'(\d{4})').astype(int)

# eliminate duplicates that can arise from CY2022 / CY2022.1, etc.
other_long = other_long.drop_duplicates(['Burden Pool', 'Year'])

# --- 3) prep BURDEN_RATE for the merge ---
rate_col = 'Rate'                     # <- change if your file's numeric column is named differently
if rate_col not in BURDEN_RATE.columns:
    BURDEN_RATE[rate_col] = np.nan

# make sure # Date is numeric so it matches the Year column
BURDEN_RATE['Year'] = pd.to_numeric(BURDEN_RATE['# Date'], errors='coerce')

# --- 4) merge and update ---
merged = BURDEN_RATE.merge(
    other_long[['Burden Pool', 'Year', 'rate']],
    how='left',
    on=['Burden Pool', 'Year'],
    suffixes=('', '_new')
)

# if a matching rate exists, overwrite; otherwise keep original
merged[rate_col] = merged['rate'].combine_first(merged[rate_col])

# clean‑up columns we no longer need
BURDEN_RATE_UPDATED = (
    merged
      .drop(columns=['rate', 'Year'])             # drop helper cols
      .copy()
)

# --- 5) (optional) formatting & export ---
BURDEN_RATE_UPDATED[rate_col] = BURDEN_RATE_UPDATED[rate_col].round(5)  # or whatever precision
# BURDEN_RATE_UPDATED.to_excel('updated_BurdenRateImport.xlsx', index=False)

# If you want the result back in the original variable:
BURDEN_RATE = BURDEN_RATE_UPDATED