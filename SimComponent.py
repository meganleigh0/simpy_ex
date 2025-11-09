import pandas as pd

# --- load weekly extract ---
file_path = "data/Cobra-XM30.xlsx"
sheet     = "tbl_Weekly Extract"
df = pd.read_excel(file_path, sheet_name=sheet)
df.columns = df.columns.str.strip()

# helpers to grab columns even if names vary a bit
def pick(df, names):
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m: 
            return m[n.lower()]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in names):
            return c
    raise KeyError(f"Need one of {names}")

col_chg   = pick(df, ["CHG#", "CHG", "WORK PACKAGE"])
col_cost  = pick(df, ["COST-SET", "COST SET", "COSTSET"])
col_hours = pick(df, ["HOURS", "QTY", "AMOUNT"])

# normalize
df[col_cost]  = df[col_cost].astype(str).str.strip().str.upper()
df[col_hours] = pd.to_numeric(df[col_hours], errors="coerce").fillna(0.0)

# === THE GROUPBY (Excel-equivalent): sum HOURS by CHG# x COST-SET ===
grouped = (
    df[df[col_cost].isin(["ACWP", "BCWP", "BCWS", "ETC"])]
      .groupby([col_chg, col_cost], dropna=False, observed=True)[col_hours]
      .sum()
      .unstack(col_cost, fill_value=0.0)
      .reindex(columns=["ACWP", "BCWP", "BCWS", "ETC"], fill_value=0.0)
      .sort_index()
)

# optional: add Grand Total row like Excel + round for display
grouped.loc["Grand Total"] = grouped.sum(numeric_only=True)
grouped = grouped.round(4)
print(grouped.head(20))