import pandas as pd
from datetime import datetime

# --- inputs you may change ---
excel_path = "data/cobra-XM30.xlsx"
sheet_name = "tbl_Weekly Extract"
round_to = 4  # decimals to match Excel display
# -----------------------------

# 1) Load
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# 2) Light cleanup / column detection
df.columns = df.columns.str.strip()
def pick(col_candidates):
    up = {c.upper(): c for c in df.columns}
    for want in col_candidates:
        if want in up: return up[want]
    # fuzzy contains
    for c in df.columns:
        cu = c.upper()
        if any(w in cu for w in col_candidates): return c
    raise KeyError(f"Could not find any of {col_candidates}")

chg_col   = pick(["CHG#", "CHG", "CHANGE", "WORK PACKAGE"])
cost_col  = pick(["COST-SET", "COST SET", "COSTSET"])
hours