import re
import pandas as pd
import numpy as np

# ── 0. helper to extract 4‑digit years from column names ───────────────
def get_year_cols(df):
    years, renamed = {}, {}
    for col in df.columns:
        m = re.search(r'(\d{4})', str(col))
        if m:
            yr = int(m.group(1))
            years[col] = yr
            renamed[col] = yr         # rename to plain int for easier lookup
    return years, renamed

# ── 1. normalise both tables ───────────────────────────────────────────
year_map, rename_src = get_year_cols(other_rates)
other_src = other_rates.rename(columns=rename_src).copy()

year_col = "Date"  # column in BURDEN_RATE that holds fiscal year
years_tgt = BURDEN_RATE[year_col].astype(int).tolist()

# pick only numeric burden columns (skip meta / flags)
meta_cols = {year_col, "Burden Pool", "Description", "Effective Date"}
numeric_cols_tgt = [c for c in BURDEN_RATE.columns if c not in meta_cols]

# ── 2. build column‑wise signatures in BURDEN_RATE ────────────────────
sig_tgt = {}
for col in numeric_cols_tgt:
    series = BURDEN_RATE.set_index(year_col)[col]
    sig_tgt[col] = series.dropna().round(6).to_dict()   # round for fuzzy compare

# ── 3. build row‑wise signatures in other_rates  ───────────────────────
sig_src = {}
for i, row in other_src.iterrows():
    desc = f"{row['Burden Pool']}".strip()   # use Burden Pool text as key
    values = {yr: round(row[yr], 6) for yr in year_map.values() if not pd.isna(row[yr])}
    if values:
        sig_src[desc] = values

# ── 4. match: ≥80 % year/value overlap & ≤1e‑6 difference ─────────────
mapping = {}
for desc, s_vals in sig_src.items():
    best_match, best_score = None, 0
    for col, t_vals in sig_tgt.items():
        # intersect years
        yrs = set(s_vals).intersection(t_vals)
        if not yrs:
            continue
        equal = [abs(s_vals[y] - t_vals[y]) < 1e-6 for y in yrs]  # tolerance
        score = sum(equal) / len(yrs)
        if score > best_score:
            best_match, best_score = col, score
    if best_score >= 0.80:          # tweak threshold if needed
        mapping[desc] = best_match

# ── 5. show results  ───────────────────────────────────────────────────
auto_map_df = (
    pd.DataFrame(mapping.items(), columns=["Burden Pool (row text)", "BURDEN_RATE column"])
    .sort_values("Burden Pool (row text)")
    .reset_index(drop=True)
)
display(auto_map_df)
print(f"✅  Found {len(mapping)}/{len(sig_src)} burden‑pool → column matches")

# If you want a plain dict for the earlier update function:
# AUTO_MAPPING = {(k, None): v for k, v in mapping.items()}