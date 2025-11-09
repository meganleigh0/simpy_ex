import pandas as pd, numpy as np
from pathlib import Path

# ---------- paths (edit if needed) ----------
weekly_candidates = ["data/Cobra-XM30.xlsx", "data/cobra-XM30.xlsx"]
weekly_sheet = "tbl_Weekly Extract"
excel_pivot_csv = "data/cum_table.csv"     # your CSV export from Excel Pivot (Row Labels, ACWP, BCWP, BCWS, ETC)
ROUND = 4
TOL = 1e-3
# -------------------------------------------

def pick_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(paths)

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in cols: return cols[x.lower()]
    for c in df.columns:
        if any(x.lower() in c.lower() for x in candidates): return c
    raise KeyError(f"Missing any of: {candidates}")

def to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# --- load weekly extract ---
weekly_path = pick_existing(weekly_candidates)
df = pd.read_excel(weekly_path, sheet_name=weekly_sheet)
df.columns = df.columns.str.strip()

col_chg   = find_col(df, ["CHG#", "CHG", "WORK PACKAGE"])
col_cost  = find_col(df, ["COST-SET", "COST SET", "COSTSET"])
col_hours = find_col(df, ["HOURS", "QTY", "AMOUNT"])
df[col_hours] = to_num(df[col_hours])
df[col_cost]  = df[col_cost].astype(str).str.strip().str.upper()

# optional columns
col_cum = None
for c in ["CUM/PER", "CUM PER", "CUMPER", "CUM"]:
    try: col_cum = find_col(df, [c]); break
    except: pass

col_date = None
for c in ["DATE", "STATUS DATE", "AS OF", "ASOF", "REPORT DATE"]:
    try: col_date = find_col(df, [c]); break
    except: pass

col_plug = None
try: col_plug = find_col(df, ["PLUG"])
except: pass

# show baseline counts
print(f"Rows in weekly extract: {len(df):,}")

def build_filtered(mode):
    """mode in {'none','date_blank','latest'}"""
    d = df.copy()

    # CUM filter if column exists and would keep something
    if col_cum is not None:
        keep_cum = d[col_cum].astype(str).str.upper().str.contains("CUM", na=False)
        if keep_cum.any():
            d = d[keep_cum]
    # PLUG = blank/NaN/0 only if such rows exist (otherwise skip)
    if col_plug is not None:
        plug_blank = (d[col_plug].isna() |
                      (d[col_plug].astype(str).str.strip().eq("")) |
                      (d[col_plug] == 0))
        if plug_blank.any():
            d = d[plug_blank]

    if mode == "date_blank" and col_date is not None:
        blank = (d[col_date].isna() | (d[col_date].astype(str).str.strip()==""))
        if blank.any():
            d = d[blank]
        else:
            # nothing would remain; fall back to 'none'
            mode = "none"

    if mode == "latest" and col_date is not None:
        d[col_date] = pd.to_datetime(d[col_date], errors="coerce")
        grpmax = d.groupby([col_chg, col_cost], dropna=False)[col_date].transform("max")
        d = d[(d[col_date].isna() & grpmax.isna()) | (d[col_date] == grpmax)]

    print(f"   Mode={mode:11s} -> kept {len(d):,} rows")
    # pivot
    pv = (d.groupby([col_chg, col_cost], dropna=False)[col_hours]
            .sum()
            .unstack(fill_value=0.0))
    # normalize cost-set names
    rename = {}
    for c in pv.columns:
        u = str(c).upper()
        if u in ("ACWP","ACMP"): rename[c]="ACWP"
        elif u=="BCWP": rename[c]="BCWP"
        elif u=="BCWS": rename[c]="BCWS"
        elif u=="ETC":  rename[c]="ETC"
    pv = pv.rename(columns=rename)
    for need in ["ACWP","BCWP","BCWS","ETC"]:
        if need not in pv.columns:
            pv[need]=0.0
    pv = pv[["ACWP","BCWP","BCWS","ETC"]]
    pv.index = pv.index.astype(str).str.strip()
    return pv.sort_index()

# build three candidates
cand = {
    "none":        build_filtered("none"),
    "date_blank":  build_filtered("date_blank"),
    "latest":      build_filtered("latest"),
}

# --- load Excel pivot CSV (ground truth) ---
cum = pd.read_csv(excel_pivot_csv, dtype=str).fillna("")
name_map = {c:c.strip() for c in cum.columns}
cum = cum.rename(columns=name_map)
chg_col = "Row Labels" if "Row Labels" in cum.columns else "CHG#"
cum.rename(columns={chg_col:"CHG#"}, inplace=True)
for k in ["ACWP","BCWP","BCWS","ETC"]:
    if k not in cum.columns: cum[k]=0
    cum[k] = to_num(cum[k].replace({"Missing value":0,"-":0,"":0}))
cum["CHG#"]=cum["CHG#"].astype(str).str.strip()
cum = cum.set_index("CHG#")[["ACWP","BCWP","BCWS","ETC"]].sort_index()

# --- choose the best matching mode ---
def score(pv):
    keys = sorted(set(pv.index).union(cum.index))
    a = pv.reindex(keys).fillna(0.0)
    b = cum.reindex(keys).fillna(0.0)
    return (a-b).abs().sum().sum()  # L1 error over all cells

scores = {m: score(pv) for m, pv in cand.items()}
best_mode = min(scores, key=scores.get)
grouped = cand[best_mode].copy()

print("\nCandidate errors (lower is better):")
for m, s in scores.items():
    print(f"  {m:11s}: {s:,.4f}")
print(f"\n>>> Selected mode: {best_mode}\n")

# totals comparison + save mismatches
keys = sorted(set(grouped.index).union(cum.index))
g = grouped.reindex(keys).fillna(0.0)
e = cum.reindex(keys).fillna(0.0)
tot = pd.DataFrame({
    "Python_total": g.sum(),
    "Excel_total": e.sum(),
    "Delta": g.sum()-e.sum()
}).round(ROUND)
print("Totals comparison:")
print(tot, "\n")

delta = (g-e).round(ROUND)
top = (delta.abs().sum(axis=1).sort_values(ascending=False).head(25).index)
print("Top 25 mismatches:")
print(pd.concat({"python": g.loc[top], "excel": e.loc[top], "delta": delta.loc[top]}, axis=1))

out = pd.concat([g.add_prefix("py_"), e.add_prefix("xl_"), delta.add_prefix("delta_")], axis=1)
out.to_csv("mismatch_best_mode.csv")
print("\nSaved side-by-side to mismatch_best_mode.csv")

# Add a Grand Total row and round like Excel
grouped.loc["Grand Total"] = grouped.sum(numeric_only=True)
grouped = grouped.round(ROUND)
print("\nPreview of final grouped:")
print(grouped.head(15))