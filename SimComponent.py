import pandas as pd, numpy as np
from pathlib import Path

# -------------------- CONFIG --------------------
weekly_xlsx_candidates = ["data/cobra-XM30.xlsx", "data/Cobra-XM30.xlsx"]
weekly_sheet = "tbl_Weekly Extract"
excel_pivot_csv = "data/cum_table.csv"   # your cleaned export from Excel "PIVOT"
TOL = 1e-3                                # match tolerance (≈ Excel display)
ROUND = 4                                 # rounding to display like Excel
# ------------------------------------------------

def pick_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")

def find_col(df, candidates):
    """Return the first matching column by exact case-insensitive name or substring."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for want in candidates:
        if want.lower() in lower:
            return lower[want.lower()]
    for c in cols:
        cu = c.lower()
        if any(w.lower() in cu for w in candidates):
            return c
    raise KeyError(f"Could not find any of {candidates} in: {list(df.columns)}")

def to_num(s, zero_missing=True):
    out = pd.to_numeric(s, errors="coerce")
    if zero_missing:
        out = out.fillna(0.0)
    return out

def build_pivot(df, latest_date_per_key=False):
    """Make a pivot like Excel: SUM(HOURS) by CHG# x COST-SET, with common page-filters."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    chg = find_col(df, ["CHG#", "CHG", "WORK PACKAGE"])
    cost = find_col(df, ["COST-SET", "COST SET", "COSTSET"])
    hours = find_col(df, ["HOURS", "QTY", "AMOUNT"])
    df[hours] = to_num(df[hours])

    # Excel-style 'page filters' commonly used in your screenshots:
    # 1) Keep CUM (if a "CUM/PER"ish column exists)
    cumcol = None
    for cand in ["CUM/PER", "CUM PER", "CUMPER", "CUM"]:
        try:
            cumcol = find_col(df, [cand])
            break
        except Exception:
            pass
    if cumcol is not None:
        df = df[df[cumcol].astype(str).str.upper().str.contains("CUM", na=False)]

    # 2) PLUG = Missing value (i.e., blank/NaN/0) IF a PLUG column exists
    try:
        plug = find_col(df, ["PLUG"])
        df = df[df[plug].isna() | (df[plug].astype(str).str.strip().eq("")) | (df[plug]==0)]
    except Exception:
        pass

    # 3) DATE = Missing value (if a date-ish column exists)
    datecol = None
    for cand in ["DATE", "STATUS DATE", "AS OF", "ASOF", "REPORT DATE"]:
        try:
            datecol = find_col(df, [cand])
            break
        except Exception:
            pass
    if datecol is not None:
        # If we're NOT doing latest-date logic, match the Excel screenshot filter:
        # keep rows where date is missing.
        if not latest_date_per_key:
            df = df[df[datecol].isna() | (df[datecol].astype(str).str.strip().eq(""))]
        else:
            # Latest-date per (CHG#, COST-SET). If some are blank, treat blank as "oldest".
            tmp = df.copy()
            # Convert to datetime (coerce errors -> NaT)
            tmp[datecol] = pd.to_datetime(tmp[datecol], errors="coerce")
            # pick max date per key; if all NaT for a key, keep the NaT rows
            max_dates = tmp.groupby([chg, cost], dropna=False)[datecol].transform("max")
            df = tmp[(tmp[datecol].isna() & max_dates.isna()) | (tmp[datecol] == max_dates)]

    # Build pivot
    pv = df.pivot_table(index=chg, columns=cost, values=hours, aggfunc="sum", fill_value=0.0)

    # Normalize columns to the four of interest
    rename = {}
    for c in pv.columns:
        u = str(c).strip().upper()
        if u in ["ACWP", "ACMP"]:
            rename[c] = "ACWP"
        elif u == "BCWP":
            rename[c] = "BCWP"
        elif u == "BCWS":
            rename[c] = "BCWS"
        elif u == "ETC":
            rename[c] = "ETC"
    pv = pv.rename(columns=rename)
    for need in ["ACWP", "BCWP", "BCWS", "ETC"]:
        if need not in pv.columns:
            pv[need] = 0.0
    pv = pv[["ACWP", "BCWP", "BCWS", "ETC"]]
    pv.index = pv.index.astype(str).str.strip()
    return pv.sort_index()

# 1) Load weekly extract + build two versions of the pivot
weekly_path = pick_existing(weekly_xlsx_candidates)
weekly = pd.read_excel(weekly_path, sheet_name=weekly_sheet)

grouped_all = build_pivot(weekly, latest_date_per_key=False)          # Excel page-filter: DATE missing
grouped_latest = build_pivot(weekly, latest_date_per_key=True)        # try latest DATE per (CHG#, COST-SET)

# 2) Load your Excel pivot export (CSV) -> tidy shape
cum = pd.read_csv(excel_pivot_csv, dtype=str).fillna("")
# Expect columns: 'Row Labels', 'ACWP','BCWP','BCWS','ETC' (strings like 'Missing value' present)
rename_map = {c: c.strip() for c in cum.columns}
cum = cum.rename(columns=rename_map)
chg_csv_col = "Row Labels" if "Row Labels" in cum.columns else "CHG#"
cum.rename(columns={chg_csv_col: "CHG#"}, inplace=True)
for col in ["ACWP", "BCWP", "BCWS", "ETC"]:
    if col not in cum.columns:
        cum[col] = 0
    # Turn 'Missing value' or blanks into 0
    vals = cum[col].replace({"Missing value": 0, "": 0, "-": 0})
    cum[col] = to_num(vals, zero_missing=True)
cum["CHG#"] = cum["CHG#"].astype(str).str.strip()
cum = cum.set_index("CHG#").sort_index()
cum = cum[["ACWP", "BCWP", "BCWS", "ETC"]]

# 3) Compare (totals + per-row deltas)
def compare_and_report(py_pivot, name):
    # align
    keys = sorted(set(py_pivot.index).union(set(cum.index)))
    g = py_pivot.reindex(keys).fillna(0.0)
    e = cum.reindex(keys).fillna(0.0)
    delta = (g - e)
    absmax = delta.abs().max(axis=1)

    print(f"\n================= COMPARISON: {name} =================")
    print("Totals (Python vs Excel pivot):")
    totals = pd.DataFrame({
        "Python_total": g.sum(),
        "Excel_total": e.sum(),
        "Delta": (g.sum() - e.sum())
    })[["Python_total","Excel_total","Delta"]].round(ROUND)
    print(totals, "\n")

    mism = delta[absmax > TOL].copy()
    print(f"Rows compared: {len(keys)}   mismatches (>|{TOL}|): {len(mism)}")
    # show top 20 by largest absolute sum of deltas
    top = (delta.abs().sum(axis=1).sort_values(ascending=False).head(20).index)
    view = pd.concat(
        {
            "python": g.loc[top, ["ACWP","BCWP","BCWS","ETC"]],
            "excel":  e.loc[top, ["ACWP","BCWP","BCWS","ETC"]],
            "delta":  delta.loc[top, ["ACWP","BCWP","BCWS","ETC"]],
        }, axis=1
    ).round(ROUND)
    print("\nTop 20 row mismatches:")
    print(view)

    # Save full mismatch file
    report = pd.concat(
        [g.add_prefix("py_"), e.add_prefix("xl_"), delta.add_prefix("delta_")],
        axis=1
    ).round(ROUND)
    outpath = f"mismatch_{name.replace(' ','_').lower()}.csv"
    report.to_csv(outpath)
    print(f"\nSaved full comparison to: {outpath}")

    # Diagnostics for BCWS/ETC mismatches: show potential duplicate “keys” in raw data
    # (only for CHG# present in top mismatches)
    top_keys = list(top)
    chg_col = find_col(weekly, ["CHG#", "CHG", "WORK PACKAGE"])
    cost_col = find_col(weekly, ["COST-SET", "COST SET", "COSTSET"])
    diag_cols = [c for c in ["DATE","STATUS DATE","AS OF","ASOF","CUM/PER","CUM PER","CUMPER","PLUG","RESP_DEPT","BE_DEPT","Control_Acct"] if any(c.lower() in x.lower() for x in weekly.columns)]
    diag_cols = [find_col(weekly, [c]) for c in diag_cols if c] if diag_cols else []
    keep_cols = list(dict.fromkeys([chg_col, cost_col] + diag_cols))  # unique order
    w = weekly.copy()
    # normalize cost set labels to four buckets
    w[cost_col] = w[cost_col].astype(str).str.upper().str.strip()
    w = w[w[chg_col].astype(str).isin(top_keys) & w[cost_col].isin(["ACWP","BCWP","BCWS","ETC"])]
    print("\nSample raw rows for top mismatches (to spot duplicates by DATE / CUM / PLUG etc.):")
    try:
        display_cols = [c for c in keep_cols if c in w.columns]
        print(w[display_cols].head(50))
    except Exception:
        print("(Could not display diagnostics subset.)")

# Run both comparisons
compare_and_report(grouped_all, "page_filters_DATE_missing")
if any(x.lower() in c.lower() for x in ["date","status date","as of","asof","report date"] for c in weekly.columns):
    compare_and_report(grouped_latest, "latest_DATE_per_CHG_COST")

# Also expose the 'grouped' you asked for (matching the page-filter version)
grouped = grouped_all.round(ROUND)
print("\nPreview of grouped (first 15 rows):")
print(grouped.head(15))