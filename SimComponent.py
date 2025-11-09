import pandas as pd
import numpy as np
from pathlib import Path

# ==== CONFIG (change paths if yours differ) ====
weekly_path    = Path("data/cobra-XM30.xlsx")
weekly_sheet   = "tbl_Weekly Extract"
dashboard_path = Path("data/Dashboard-XM30_10.15.25.xlsx")
dash_pivot_sh  = "PIVOT"
TOL = 1e-3  # tolerance to consider a value "matching" (0.001 ~ Excel display)

# ---------- helpers ----------
def _find_col(df, candidates):
    """Return the first existing column name matching any of the case-insensitive candidates or substring hits."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    # substring fallback
    for c in df.columns:
        cu = c.lower()
        if any(x.lower() in cu for x in candidates):
            return c
    raise KeyError(f"Could not find any of {candidates} in {list(df.columns)}")

def _coerce_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def read_dashboard_pivot_table(xlsx_path, sheet_name="PIVOT"):
    """
    The 'PIVOT' sheet is an Excel Pivot export with extra header rows/columns like:
      [ 'PLUG', 'DATE', 'CUM', ... ]
      then a row containing 'Row Labels', and one nearby row containing 'ACWP','BCWP','BCWS','ETC'.
    This parses it into a tidy df with columns: CHG#, ACWP, BCWP, BCWS, ETC. Excludes 'Grand Total' row.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=object)
    # locate the column that contains 'Row Labels'
    loc = np.argwhere(raw.values == "Row Labels")
    if len(loc) == 0:
        # Some Pivot exports say 'Row Labels' in another locale/case or put it one col right; try a contains search
        got = False
        for r in range(min(50, len(raw))):  # search near the top
            for c in range(min(50, raw.shape[1])):
                val = str(raw.iat[r, c]) if pd.notna(raw.iat[r, c]) else ""
                if "row labels" in val.lower():
                    rowlabels_row, chg_col = r, c
                    got = True
                    break
            if got: break
        if not got:
            raise RuntimeError("Couldn't find 'Row Labels' in the PIVOT sheet.")
    else:
        rowlabels_row, chg_col = loc[0]  # first occurrence

    # find the header row that contains the cost set names
    targets = ["ACWP", "BCWP", "BCWS", "ETC"]
    header_row = None
    for r in range(rowlabels_row, min(rowlabels_row + 6, len(raw))):
        row_vals = [str(x).strip().upper() if pd.notna(x) else "" for x in raw.iloc[r].tolist()]
        if sum(t in row_vals for t in targets) >= 2:
            header_row = r
            break
    if header_row is None:
        # last resort: scan first 50 rows
        for r in range(0, min(50, len(raw))):
            row_vals = [str(x).strip().upper() if pd.notna(x) else "" for x in raw.iloc[r].tolist()]
            if sum(t in row_vals for t in targets) >= 2:
                header_row = r
                break
    if header_row is None:
        raise RuntimeError("Couldn't find the ACWP/BCWP/BCWS/ETC header row in PIVOT.")

    # figure out which columns those targets live in
    header_vals = [str(x).strip().upper() if pd.notna(x) else "" for x in raw.iloc[header_row].tolist()]
    target_cols = {}
    for t in targets:
        try:
            target_cols[t] = header_vals.index(t)
        except ValueError:
            target_cols[t] = None  # missing in sheet (we'll treat as zeros)

    # data start is next row after header; end is row where CHG# col == 'Grand Total'
    start = header_row + 1
    # find first 'Grand Total'
    end = None
    for r in range(start, len(raw)):
        v = str(raw.iat[r, chg_col]) if pd.notna(raw.iat[r, chg_col]) else ""
        if v.strip().lower().startswith("grand total"):
            end = r
            break
    if end is None:
        end = len(raw)

    # build the clean dataframe
    recs = []
    for r in range(start, end):
        chg = raw.iat[r, chg_col]
        if pd.isna(chg) or str(chg).strip() == "":
            continue
        row = {"CHG#": str(chg).strip()}
        for t in targets:
            cidx = target_cols.get(t)
            val = 0.0 if cidx is None else _coerce_num(raw.iat[r, cidx])
            row[t] = float(val)
        recs.append(row)
    df = pd.DataFrame(recs).set_index("CHG#")

    # totals row (optional diagnostics)
    totals = {"ACWP":0.0,"BCWP":0.0,"BCWS":0.0,"ETC":0.0}
    if end < len(raw):
        for t in targets:
            cidx = target_cols.get(t)
            if cidx is not None:
                totals[t] = float(_coerce_num(raw.iat[end, cidx]))
    return df.sort_index(), totals

def make_grouped_from_weekly(xlsx_path, sheet_name):
    """Re-create the Excel-like pivot from the weekly extract with the same filtering Excel used."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # canonicalize columns and pick the important ones
    df.columns = df.columns.str.strip()
    chg_col   = _find_col(df, ["CHG#", "CHG", "WORK PACKAGE", "ROW LABELS"])
    cost_col  = _find_col(df, ["COST-SET", "COST SET", "COSTSET"])
    hours_col = _find_col(df, ["HOURS","QTY","AMOUNT"])

    # Excel pivot you showed had the page filters: PLUG = Missing value, DATE = Missing value, CUM = (likely 'CUM')
    # Apply those if columns exist.
    if any(x in df.columns.str.upper() for x in ["CUM/PER","CUM PER","CUMPER","CUM"]):
        cumcol = _find_col(df, ["CUM/PER","CUM PER","CUMPER","CUM"])
        df = df[df[cumcol].astype(str).str.upper().str.contains("CUM", na=False)]

    if "PLUG" in df.columns.str.upper().tolist():
        plugcol = _find_col(df, ["PLUG"])
        df = df[df[plugcol].isna() | (df[plugcol].astype(str).str.strip().eq("")) | (df[plugcol]==0)]

    # The PIVOT screenshot shows DATE "Missing value". When there is a DATE column, exclude rows that have a date.
    # (This avoids double-counting periodized rows when the Pivot is cumulative.)
    for cand in ["DATE", "STATUS DATE", "AS OF", "ASOF", "REPORT DATE"]:
        if any(cand == c.upper() for c in df.columns):
            dcol = _find_col(df, [cand])
            df = df[df[dcol].isna() | (df[dcol].astype(str).str.strip().eq(""))]
            break

    df[hours_col] = _coerce_num(df[hours_col])

    # Create pivot (sum of HOURS)
    pt = df.pivot_table(index=chg_col, columns=cost_col, values=hours_col, aggfunc="sum", fill_value=0.0)

    # normalize column names (Excel shows ACWP/BCWP/BCWS/ETC)
    rename_map = {}
    for c in pt.columns:
        cu = str(c).strip().upper()
        if cu in ["ACWP","ACMP"]: rename_map[c] = "ACWP"
        elif cu == "BCWP":        rename_map[c] = "BCWP"
        elif cu == "BCWS":        rename_map[c] = "BCWS"
        elif cu == "ETC":         rename_map[c] = "ETC"
    pt = pt.rename(columns=rename_map)
    # ensure all expected columns exist
    for col in ["ACWP","BCWP","BCWS","ETC"]:
        if col not in pt.columns:
            pt[col] = 0.0

    # keep only the four columns, sorted by CHG#
    pt = pt[["ACWP","BCWP","BCWS","ETC"]].copy()
    pt.index = pt.index.astype(str).str.strip()
    pt = pt.sort_index()

    totals = pt.sum(numeric_only=True).to_dict()
    return pt, totals

# ---------- run ----------
# 1) Build grouped from weekly extract
grouped, grouped_totals = make_grouped_from_weekly(weekly_path, weekly_sheet)

# 2) Parse the Excel pivot sheet to get the expected numbers
piv_expected, excel_totals = read_dashboard_pivot_table(dashboard_path, dash_pivot_sh)

# 3) Align and compare
#    (some CHG# might exist in one but not the other)
all_keys = sorted(set(grouped.index).union(set(piv_expected.index)))
g = grouped.reindex(all_keys).fillna(0.0)
e = piv_expected.reindex(all_keys).fillna(0.0)

delta = (g - e).assign(
    abs_max_diff = (g[["ACWP","BCWP","BCWS","ETC"]] - e[["ACWP","BCWP","BCWS","ETC"]]).abs().max(axis=1)
)
mismatches = delta[delta["abs_max_diff"] > TOL].drop(columns=["abs_max_diff"])

# totals side-by-side
totals_df = pd.DataFrame({
    "grouped_total": g.sum(),
    "excel_pivot_total": e.sum(),
    "delta": (g.sum() - e.sum())
}).T[["ACWP","BCWP","BCWS","ETC"]]

# 4) Print concise diagnostics
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print("=== Totals (Python grouped vs Excel pivot) ===")
print(totals_df, "\n")

print(f"Rows compared: {len(all_keys)}")
print(f"Rows with any mismatch > {TOL}: {len(mismatches)}")
print("\n--- Top 25 largest absolute deltas ---")
top25 = (g - e).abs().sum(axis=1).sort_values(ascending=False).head(25).index
out = pd.concat(
    {
        "python_grouped": g.loc[top25, ["ACWP","BCWP","BCWS","ETC"]],
        "excel_pivot":    e.loc[top25, ["ACWP","BCWP","BCWS","ETC"]],
        "delta":          (g.loc[top25, ["ACWP","BCWP","BCWS","ETC"]] - e.loc[top25, ["ACWP","BCWP","BCWS","ETC"]])
    },
    axis=1
)
print(out)

# 5) Optional: save full mismatch report to inspect in Excel
mismatch_report = pd.concat(
    [g.add_prefix("py_"), e.add_prefix("xl_"), (g - e).add_prefix("delta_")],
    axis=1
)
mismatch_report.to_csv("bcws_etc_mismatch_report.csv")
print("\nSaved: bcws_etc_mismatch_report.csv (full side-by-side + deltas).")
