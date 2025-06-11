# ------------------------------------------------------------------
#  SMART‑AUTO‑MAP   other_rates  →  BURDEN_RATE
#  (assumes both DataFrames exist in memory)
# ------------------------------------------------------------------
import re, numpy as np, pandas as pd

#####  🔧  KNOBS you can adjust  ####################################
TOL        = 1e-4       # max allowed per‑year absolute difference
MIN_YEARS  = 3          # need ≥ this many overlapping fiscal years
####################################################################

# ── 1. normalise year columns in other_rates ──────────────────────
year_re  = re.compile(r'(\d{4})')
rename   = {c:int(year_re.search(str(c)).group(1))
            for c in other_rates.columns if year_re.search(str(c))}
orates   = other_rates.rename(columns=rename)
orates   = orates.loc[:, ~orates.columns.duplicated()]          # safety
YEAR_COLS = sorted([c for c in orates.columns if isinstance(c, int)])

# ── 2. prep BURDEN_RATE (pivot by 'Date') ─────────────────────────
BURDEN_RATE['Date'] = BURDEN_RATE['Date'].astype(int)
numeric_cols = [c for c in BURDEN_RATE.columns
                if c not in {'Date', 'Burden Pool', 'Description', 'Effective Date'}
                and np.issubdtype(BURDEN_RATE[c].dtype, np.number)]

tgt_wide = BURDEN_RATE.set_index('Date')[numeric_cols]           # years × columns

# ── 3. build MAE matrix  (rows = other_rates, cols = BURDEN_RATE) ─
mae_records = []
for r_idx, r in orates.iterrows():
    r_name  = str(r.get('Burden Pool', '')).strip() or f"row{r_idx}"
    r_vals  = r[YEAR_COLS].astype(float)
    r_mask  = r_vals.notna()
    if r_mask.sum() == 0:
        continue
    r_years = np.array(YEAR_COLS)[r_mask.values]
    
    for c in numeric_cols:
        c_series   = tgt_wide[c].dropna()
        overlap    = np.intersect1d(r_years, c_series.index.values)
        if len(overlap) < MIN_YEARS:
            continue
        
        diffs = np.abs(r.loc[overlap].astype(float).values -
                       c_series.loc[overlap].values)
        mae   = diffs.mean()
        mae_records.append((r_name, c, len(overlap), mae))

mae_df = (pd.DataFrame(mae_records, columns=['SourceRow','TargetCol',
                                            'YearsOverlap','MAE'])
          .sort_values('MAE'))

# ── 4. greedy pick: each source row & target col used at most once ─
picked_src, picked_tgt, pairs = set(), set(), []
for src, tgt, yrs, err in mae_df.itertuples(index=False):
    if (src in picked_src) or (tgt in picked_tgt) or (err > TOL):
        continue
    picked_src.add(src); picked_tgt.add(tgt)
    pairs.append((src, tgt, yrs, err))

mapping_df = (pd.DataFrame(pairs, columns=['Burden Pool (other_rates)',
                                           'BURDEN_RATE column',
                                           '#yrs','MAE'])
              .sort_values('Burden Pool (other_rates)')
              .reset_index(drop=True))
display(mapping_df.style.format({'MAE':'{:.2e}'}))
print(f"✅  Matched {len(mapping_df)} of {orates.shape[0]} source rows")

# ── 5. produce dict compatible with update_burden_rate() ──────────
AUTO_MAPPING = {(row['Burden Pool (other_rates)'], None)
                : row['BURDEN_RATE column'] for _, row in mapping_df.iterrows()}