import pandas as pd
import numpy as np

# --- INPUTS (change if your path/sheet are different) ---
file_path = "data/Cobra-XM30.xlsx"
sheet     = "tbl_Weekly Extract"
ROUND = 4
# --------------------------------------------------------

# helper to find a column by several possible names
def pick(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols: 
            return cols[n.lower()]
    # substring fallback
    for c in df.columns:
        cu = c.lower()
        if any(n.lower() in cu for n in names):
            return c
    raise KeyError(f"Couldn't find any of {names} in {list(df.columns)}")

df = pd.read_excel(file_path, sheet_name=sheet)
df.columns = df.columns.str.strip()

# --- identify the main columns (robust to name variants) ---
col_chg   = pick(df, ["CHG#", "CHG", "WORK PACKAGE", "ROW LABELS"])
col_cost  = pick(df, ["COST-SET", "COST SET", "COSTSET"])
col_hours = pick(df, ["HOURS", "QTY", "AMOUNT"])

# optional filter columns if present
col_cum   = None
for cand in ["CUM/PER", "CUM PER", "CUMPER", "CUM"]:
    try: col_cum = pick(df, [cand]); break
    except: pass

col_date  = None
for cand in ["DATE", "STATUS DATE", "AS OF", "ASOF", "REPORT DATE"]:
    try: col_date = pick(df, [cand]); break
    except: pass

col_plug  = None
try: col_plug = pick(df, ["PLUG"])
except: pass

# --- normalize key fields ---
df[col_cost]  = df[col_cost].astype(str).str.strip().str.upper()
df[col_hours] = pd.to_numeric(df[col_hours], errors="coerce").fillna(0.0)

# --- build the Excel-equivalent FILTER (very important for BCWS/ETC) ---
mask = pd.Series(True, index=df.index)

# 1) keep CUM rows (if CUM/PER exists)
if col_cum is not None:
    mask &= df[col_cum].astype(str).str.upper().str.contains("CUM", na=False)

# 2) keep DATE = (Missing value)   => blank/NaT
if col_date is not None:
    mask &= (df[col_date].isna() | (df[col_date].astype(str).str.strip() == ""))

# 3) keep PLUG = (Missing value)   => blank/NaN/0 — only apply if such blanks exist
if col_plug is not None:
    plug_blank = (df[col_plug].isna() | (df[col_plug].astype(str).str.strip() == "") | (df[col_plug] == 0))
    # only enforce if there are actually some blank PLUG rows (matches how the Excel page filter was set)
    if plug_blank.any():
        mask &= plug_blank

df_f = df[mask].copy()

# --- pivot exactly like Excel: Sum of HOURS by CHG# x COST-SET ---
grouped = (
    df_f
      .groupby([col_chg, col_cost], dropna=False)[col_hours]
      .sum()
      .unstack(fill_value=0.0)
)

# Normalize columns to the four buckets Excel shows
rename = {}
for c in grouped.columns:
    u = str(c).upper()
    if u in ["ACWP", "ACMP"]: rename[c] = "ACWP"
    elif u == "BCWP":         rename[c] = "BCWP"
    elif u == "BCWS":         rename[c] = "BCWS"
    elif u == "ETC":          rename[c] = "ETC"
grouped = grouped.rename(columns=rename)

for need in ["ACWP", "BCWP", "BCWS", "ETC"]:
    if need not in grouped.columns:
        grouped[need] = 0.0

grouped = grouped[["ACWP", "BCWP", "BCWS", "ETC"]].sort_index()

# --- optional: Excel-style Grand Total row + rounding like your sheet ---
grouped.loc["Grand Total"] = grouped.sum(numeric_only=True)
grouped = grouped.round(ROUND)

# show result
print(grouped.head(20))