# ---------------------------------------------------------------
#  ONE‑SHOT 2022‑VALUE‑MAPPER   other_rates  →  BURDEN_RATE
# ---------------------------------------------------------------
import re, numpy as np, pandas as pd

YEAR_TO_MATCH = 2022      # change if you need a different anchor year
TOL           = 1e-6      # allowed absolute difference for a "match"

# ── 1. normalise year headers in other_rates ────────────────────
year_re  = re.compile(r'(\d{4})')
rename   = {c:int(year_re.search(str(c)).group(1))
            for c in other_rates.columns if year_re.search(str(c))}
orates   = other_rates.rename(columns=rename)
if YEAR_TO_MATCH not in orates.columns:
    raise ValueError(f"{YEAR_TO_MATCH} not found among year columns in other_rates")

# ── 2. pull 2022 values from source rows ────────────────────────
src_vals = (orates[['Burden Pool', YEAR_TO_MATCH]]
            .dropna(subset=[YEAR_TO_MATCH])
            .assign(Value=lambda d: d[YEAR_TO_MATCH].astype(float))
            .groupby('Burden Pool', as_index=False)      # if duplicates, keep first
            .first())

# ── 3. pull 2022 values from BURDEN_RATE columns ───────────────
BURDEN_RATE['Date'] = BURDEN_RATE['Date'].astype(int)
tgt_2022 = BURDEN_RATE[BURDEN_RATE['Date'] == YEAR_TO_MATCH]

numeric_cols = [c for c in BURDEN_RATE.columns
                if c not in {'Date', 'Burden Pool', 'Description', 'Effective Date'}
                and np.issubdtype(BURDEN_RATE[c].dtype, np.number)]

tgt_vals = (tgt_2022[numeric_cols].iloc[0]   # one row, because Date == 2022
            .dropna()
            .astype(float)
            .to_dict())

# ── 4. build mapping by exact (±TOL) value match ───────────────
mapping_rows = []
used_cols    = set()

for _, row in src_vals.iterrows():
    pool, v = row['Burden Pool'], row['Value']
    # find candidate columns whose 2022 value equals v within tolerance
    cands = [c for c, tv in tgt_vals.items() if abs(tv - v) < TOL and c not in used_cols]
    
    if not cands:
        mapping_rows.append((pool, None, '⚠ no match'))
    else:
        chosen = cands[0]
        mapping_rows.append((pool, chosen, '' if len(cands) == 1 else 'dup‑value'))
        used_cols.add(chosen)

mapping_df = (pd.DataFrame(mapping_rows,
                           columns=['Burden Pool (other_rates)',
                                    'BURDEN_RATE column',
                                    'Note'])
              .sort_values('Burden Pool (other_rates)')
              .reset_index(drop=True))

display(mapping_df)
print(f"✅  Matched {mapping_df['BURDEN_RATE column'].notna().sum()} "
      f"of {len(src_vals)} rows on {YEAR_TO_MATCH} values")

# ── 5. dict you can feed into update_burden_rate() ──────────────
AUTO_MAPPING = {(r['Burden Pool (other_rates)'], None): r['BURDEN_RATE column']
                for _, r in mapping_df[mapping_df['BURDEN_RATE column'].notna()].iterrows()}