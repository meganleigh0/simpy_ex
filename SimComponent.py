# ------------------------------------------------------------
#  FY‑2022 VALUE‑BASED  M A P P E R
#  (uses existing `other_rates`  &  `BURDEN_RATE` DataFrames)
# ------------------------------------------------------------
import re, numpy as np, pandas as pd

ANCHOR_YEAR = 2022     # year to compare
TOL         = 1e-6     # allowed abs diff

# ──────────────────────────────────────────────────────────────
# 1. normalise headers in other_rates and drop duplicate years
# ──────────────────────────────────────────────────────────────
year_re = re.compile(r'(\d{4})')
rename  = {c: int(year_re.search(str(c)).group(1))
           for c in other_rates.columns if year_re.search(str(c))}

orates  = (other_rates
           .rename(columns=rename)
           .loc[:, ~other_rates.rename(columns=rename).columns.duplicated()])

if ANCHOR_YEAR not in orates.columns:
    raise ValueError(f"{ANCHOR_YEAR} not found among year columns after renaming")

# ──────────────────────────────────────────
# 2. 2022 values from each source row
# ──────────────────────────────────────────
src_vals = (orates[['Burden Pool', ANCHOR_YEAR]]
            .dropna(subset=[ANCHOR_YEAR])
            .astype({ANCHOR_YEAR: float})
            .groupby('Burden Pool', as_index=False)    # if duplicates keep first
            .first()
            .rename(columns={ANCHOR_YEAR: 'Value'}))

# ──────────────────────────────────────────
# 3. 2022 values from each BURDEN_RATE column
# ──────────────────────────────────────────
BURDEN_RATE['Date'] = BURDEN_RATE['Date'].astype(int)
row_2022 = BURDEN_RATE.loc[BURDEN_RATE['Date'] == ANCHOR_YEAR]

if row_2022.empty:
    raise ValueError(f"No row with Date == {ANCHOR_YEAR} in BURDEN_RATE")

numeric_cols = [c for c in BURDEN_RATE.columns
                if c not in {'Date', 'Burden Pool', 'Description', 'Effective Date'}
                and np.issubdtype(BURDEN_RATE[c].dtype, np.number)]

tgt_vals = (row_2022[numeric_cols].iloc[0]
            .dropna()
            .astype(float)
            .to_dict())

# ──────────────────────────────────────────
# 4. build 1‑to‑1 mapping on 2022 value
# ──────────────────────────────────────────
pairs, used_cols = [], set()

for _, r in src_vals.iterrows():
    pool, v = r['Burden Pool'], r['Value']
    match = next((c for c, tv in tgt_vals.items()
                  if abs(tv - v) < TOL and c not in used_cols), None)
    note  = '' if match else '⚠ no match'
    if match:
        used_cols.add(match)
    pairs.append((pool, match, note))

mapping_df = (pd.DataFrame(pairs,
                           columns=['Burden Pool (other_rates)',
                                    'BURDEN_RATE column',
                                    'Note'])
              .sort_values('Burden Pool (other_rates)')
              .reset_index(drop=True))

display(mapping_df)
print(f"✅  matched {mapping_df['BURDEN_RATE column'].notna().sum()}"
      f" of {len(mapping_df)} rows on FY‑{ANCHOR_YEAR}")

# ──────────────────────────────────────────
# 5. dict ready for update_burden_rate()
# ──────────────────────────────────────────
AUTO_MAPPING = {(row['Burden Pool (other_rates)'], None): row['BURDEN_RATE column']
                for _, row in mapping_df[mapping_df['BURDEN_RATE column'].notna()].iterrows()}