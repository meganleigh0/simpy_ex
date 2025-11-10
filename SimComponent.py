# ===== EVMS Tables for PPT (single cell) =====================================
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------ SETTINGS --------------------------------------------------
DATA_PATH  = "data/Cobra-XM30.xlsx"         # your file
SHEET_NAME = "tbl_Weekly Extract"           # your sheet
GROUP_COL  = "CHG#"                          # team column in your data (e.g., "CHG#" or "RESP_DEPT")
PROGRAM    = "XM30"                          # label only
ANCHOR     = datetime.now()                  # or set datetime(2025,10,15)

# ------------------ HELPERS ---------------------------------------------------
def to_dt(s): return pd.to_datetime(s, errors="coerce")

def safe_ratio(num, den):
    num = num.astype(float)
    den = den.astype(float)
    out = num / den.replace({0: np.nan})
    return out

def totals_by_group(df, start=None, end=None):
    """Sum HOURS by GROUP_COL and COST-SET for a date window."""
    m = pd.Series(True, index=df.index)
    if start is not None: m &= df["DATE"] >= pd.Timestamp(start)
    if end   is not None: m &= df["DATE"] <= pd.Timestamp(end)

    g = (df.loc[m]
           .groupby([GROUP_COL, "COST-SET"], dropna=False)["HOURS"]
           .sum()
           .unstack(fill_value=0.0))

    # guarantee missing COST-SET columns exist as 0.0
    for k in ["ACWP", "BCWP", "BCWS", "ETC"]:
        if k not in g.columns: g[k] = 0.0
    return g[["ACWP","BCWP","BCWS","ETC"]].astype(float)

def totals_all(df, start=None, end=None):
    """Totals across ALL teams for a window (returns a 1-row DataFrame)."""
    t = totals_by_group(df, start, end).sum(axis=0).to_frame().T
    t.index = ["TOTAL"]
    return t

def add_total_row(table, compute_pct=None):
    """Append a TOTAL row. If compute_pct is provided, recompute that column as ratio of sums."""
    tbl = table.copy()
    sums = tbl.drop(index=["TOTAL"], errors="ignore").sum(numeric_only=True)
    total = sums.to_frame().T
    total.index = ["TOTAL"]
    tbl = pd.concat([tbl.drop(index=["TOTAL"], errors="ignore"), total], axis=0)
    if compute_pct:
        num, den, col = compute_pct
        n = tbl.at["TOTAL", num]
        d = tbl.at["TOTAL", den]
        tbl.at["TOTAL", col] = np.nan if pd.isna(d) or d == 0 else round((n / d) * 100, 0)
    return tbl

# ------------------ LOAD ------------------------------------------------------
xl = pd.ExcelFile(DATA_PATH)
df = xl.parse(SHEET_NAME)
df["DATE"] = to_dt(df["DATE"])

# Windows
year_start = datetime(ANCHOR.year, 1, 1)
# CTD = all data up to ANCHOR; YTD = from Jan 1 of ANCHOR.year to ANCHOR
ctd = totals_by_group(df, end=ANCHOR)
ytd = totals_by_group(df, start=year_start, end=ANCHOR)
ctd_all = totals_all(df, end=ANCHOR).iloc[0]
ytd_all = totals_all(df, start=year_start, end=ANCHOR).iloc[0]

# ------------------ TABLE 1: LABOR HOURS PERFORMANCE -------------------------
# %COMP = BCWP / BAC ;  BAC = BCWS ;  EAC = ACWP + ETC ;  VAC = BAC - EAC  (display in K hours)
BAC  = ctd["BCWS"]
EAC  = ctd["ACWP"] + ctd["ETC"]
VAC  = BAC - EAC
PCMP = safe_ratio(ctd["BCWP"], BAC) * 100

labor_tbl = pd.DataFrame({
    "%COMP":  PCMP.round(0),
    "BAC (K)": (BAC/1000.0).round(1),
    "EAC (K)": (EAC/1000.0).round(1),
    "VAC (K)": (VAC/1000.0).round(1),
})
labor_tbl.index.name = None

# Proper TOTAL row (recompute %COMP from totals, not average of rows)
labor_tot_bac  = ctd_all["BCWS"]
labor_tot_eac  = ctd_all["ACWP"] + ctd_all["ETC"]
labor_tot_vac  = labor_tot_bac - labor_tot_eac
labor_tot_pcmp = np.nan if labor_tot_bac == 0 else round((ctd_all["BCWP"] / labor_tot_bac) * 100, 0)

labor_total_row = pd.DataFrame(
    {"%COMP":[labor_tot_pcmp],
     "BAC (K)":[round(labor_tot_bac/1000.0,1)],
     "EAC (K)":[round(labor_tot_eac/1000.0,1)],
     "VAC (K)":[round(labor_tot_vac/1000.0,1)]},
    index=["TOTAL"]
)
labor_tbl = pd.concat([labor_tbl, labor_total_row])

# ------------------ TABLE 2: COST PERFORMANCE (CPI: CTD & YTD) ---------------
cpi_ctd = safe_ratio(ctd["BCWP"], ctd["ACWP"])
cpi_ytd = safe_ratio(ytd["BCWP"], ytd["ACWP"])

cost_tbl = pd.DataFrame({
    "CTD": cpi_ctd.round(2),
    "YTD": cpi_ytd.round(2),
})
# TOTAL row = ratio of sums, not average of team ratios
tot_cpi_ctd = np.nan if ctd_all["ACWP"] == 0 else round(ctd_all["BCWP"]/ctd_all["ACWP"], 2)
tot_cpi_ytd = np.nan if ytd_all["ACWP"] == 0 else round(ytd_all["BCWP"]/ytd_all["ACWP"], 2)
cost_tbl.loc["TOTAL"] = [tot_cpi_ctd, tot_cpi_ytd]
cost_tbl.index.name = None

# ------------------ TABLE 3: SCHEDULE PERFORMANCE (SPI: CTD & YTD) -----------
spi_ctd = safe_ratio(ctd["BCWP"], ctd["BCWS"])
spi_ytd = safe_ratio(ytd["BCWP"], ytd["BCWS"])

sched_tbl = pd.DataFrame({
    "CTD": spi_ctd.round(2),
    "YTD": spi_ytd.round(2),
})
tot_spi_ctd = np.nan if ctd_all["BCWS"] == 0 else round(ctd_all["BCWP"]/ctd_all["BCWS"], 2)
tot_spi_ytd = np.nan if ytd_all["BCWS"] == 0 else round(ytd_all["BCWP"]/ytd_all["BCWS"], 2)
sched_tbl.loc["TOTAL"] = [tot_spi_ctd, tot_spi_ytd]
sched_tbl.index.name = None

# ------------------ SHOW / EXPORT -------------------------------------------
print(f"{PROGRAM}  |  Anchor: {ANCHOR:%Y-%m-%d}  |  YTD from: {year_start:%Y-%m-%d}")

print("\n=== Labor Hours Performance (paste into PPT) ===")
display(labor_tbl)

print("\n=== Cost Performance (CPI: CTD, YTD) ===")
display(cost_tbl)

print("\n=== Schedule Performance (SPI: CTD, YTD) ===")
display(sched_tbl)

# Optional Excel for easy copy/paste:
# with pd.ExcelWriter(f"EVMS_{PROGRAM}_{ANCHOR:%Y-%m-%d}.xlsx") as xw:
#     labor_tbl.to_excel(xw, sheet_name="Labor_Hours_Perf")
#     cost_tbl.to_excel(xw, sheet_name="Cost_Perf_CPI")
#     sched_tbl.to_excel(xw, sheet_name="Schedule_Perf_SPI")
# =============================================================================