# ==== Labor Hours Performance table (CTD) =====================================
import pandas as pd
import numpy as np
from datetime import datetime

# --- SETTINGS -----------------------------------------------------------------
DATA_PATH  = "data/Cobra-XM30.xlsx"       # your workbook
SHEET_NAME = "tbl_Weekly Extract"         # your sheet
GROUP_COL  = "CHG#"                       # team key in your data (e.g., "CHG#" or "RESP_DEPT")
ANCHOR     = datetime.now()               # cut off future-dated rows if any

# --- LOAD ---------------------------------------------------------------------
xl = pd.ExcelFile(DATA_PATH)
df = xl.parse(SHEET_NAME)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

# use CTD (cumulative to date) and ignore any future-dated rows
df = df[df["DATE"] <= ANCHOR]

# --- CTD totals by team -------------------------------------------------------
# Sum HOURS by team and COST-SET
g = (df.groupby([GROUP_COL, "COST-SET"], dropna=False)["HOURS"]
       .sum()
       .unstack(fill_value=0.0))

# guarantee required cost-sets exist
for k in ["ACWP", "BCWP", "BCWS", "ETC"]:
    if k not in g.columns:
        g[k] = 0.0
g = g[["ACWP", "BCWP", "BCWS", "ETC"]].astype(float)

# --- Build Labor Hours Performance columns -----------------------------------
# BAC = BCWS; EAC = ACWP + ETC; VAC = BAC - EAC; %COMP = BCWP / BAC * 100
BAC = g["BCWS"]
EAC = g["ACWP"] + g["ETC"]
VAC = BAC - EAC
PCOMP = np.where(BAC.eq(0), np.nan, (g["BCWP"] / BAC) * 100)

labor_tbl = pd.DataFrame({
    "%COMP":  np.round(PCOMP, 0),
    "BAC (K)": np.round(BAC / 1000.0, 1),
    "EAC (K)": np.round(EAC / 1000.0, 1),
    "VAC (K)": np.round(VAC / 1000.0, 1),
})
labor_tbl.index.name = None

# --- TOTAL row (ratio-of-sums, not average of team ratios) -------------------
tot = g.sum()
tot_bac  = tot["BCWS"]
tot_eac  = tot["ACWP"] + tot["ETC"]
tot_vac  = tot_bac - tot_eac
tot_pcmp = np.nan if np.isclose(tot_bac, 0) else round((tot["BCWP"] / tot_bac) * 100, 0)

labor_tbl.loc["TOTAL"] = [
    tot_pcmp,
    round(tot_bac / 1000.0, 1),
    round(tot_eac / 1000.0, 1),
    round(tot_vac / 1000.0, 1),
]

print(f"Anchor: {ANCHOR:%Y-%m-%d}")
print("\n=== Labor Hours Performance (paste to slide) ===")
display(labor_tbl)
# ==============================================================================