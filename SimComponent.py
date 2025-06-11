# ------------------------------------------------------------------
#  AUTO‑DETECT MAPPING  :  other_rates  →  BURDEN_RATE
#  (assumes both DataFrames already exist in memory)
# ------------------------------------------------------------------
import re, numpy as np, pandas as pd

# ── 1. normalise year columns in other_rates ──────────────────────
year_re = re.compile(r'(\d{4})')
year_cols = {col: int(year_re.search(str(col)).group(1))
             for col in other_rates.columns if year_re.search(str(col))}
orates = other_rates.rename(columns=year_cols)              # CY2024.1 → 2024
orates = orates.loc[:, ~orates.columns.duplicated()]        # drop dupes if renaming clashed
years = sorted(set(year_cols.values()))

# ── 2. isolate numeric burden‑rate columns in BURDEN_RATE ─────────
BURDEN_RATE['Date'] = BURDEN_RATE['Date'].astype(int)
num_cols = [c for c in BURDEN_RATE.columns
            if np.issubdtype(BURDEN_RATE[c].dtype, np.number) and c != 'Date']

tgt_sig = {c: BURDEN_RATE.set_index('Date')[c].dropna().round(7).to_dict()
           for c in num_cols}

# ── 3. build row‑wise signatures in other_rates ───────────────────
src_sig = {}
for _, row in orates.iterrows():
    pool = str(row.get('Burden Pool', '')).strip()
    if not pool:
        continue
    desc = str(row.get('Description', '')).strip()
    key  = f"{pool} || {desc}" if desc else pool          # helps uniqueness
    vals = {yr: round(float(row[yr]), 7)
            for yr in years if yr in orates.columns and pd.notna(row[yr])}
    if vals:
        src_sig[key] = vals

# ── 4. greedy 1‑to‑1 matching (all shared years must match) ───────
MAPPING = {}
used_tgt = set()
TOL      = 1e-6

for skey, svals in src_sig.items():
    for tcol, tvals in tgt_sig.items():
        if tcol in used_tgt:
            continue
        shared = set(svals) & set(tvals)
        if shared and all(abs(svals[y] - tvals[y]) < TOL for y in shared):
            MAPPING[skey] = tcol
            used_tgt.add(tcol)
            break   # each source row maps to at most one column

# ── 5. display + give you a plain dict for the update function ────
map_df = (pd.DataFrame(MAPPING.items(),
                       columns=['Source row (Burden Pool | Desc)',
                                'BURDEN_RATE column'])
          .sort_values('Source row (Burden Pool | Desc)')
          .reset_index(drop=True))
display(map_df)
print(f"✅  Matched {len(MAPPING)} of {len(src_sig)} source rows")

# Dict keyed just by the Burden‑Pool text (pattern) → target column
AUTO_MAPPING = {(k.split(' || ')[0], None): v for k, v in MAPPING.items()}