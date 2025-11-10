# === Cost & Schedule Performance Tables by CHG# (CTD & YTD) ===================
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure DATE is datetime and define CTD/YTD windows
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
anchor = pd.to_datetime(df["DATE"].max())
ytd_start = datetime(anchor.year, 1, 1)

# ---- Helper to roll up hours by CHG# x COST-SET for a date window -----------
def rollup_by_costset(dframe):
    g = (dframe
         .groupby(["CHG#", "COST-SET"], dropna=False)["HOURS"]
         .sum()
         .unstack(fill_value=0.0))
    # guarantee required cost-sets exist
    for k in ["ACWP", "BCWP", "BCWS"]:
        if k not in g.columns:
            g[k] = 0.0
    return g[["ACWP", "BCWP", "BCWS"]].astype(float)

# CTD and YTD datasets
df_ctd = df[df["DATE"].notna() & (df["DATE"] <= anchor)]
df_ytd = df[df["DATE"].notna() & (df["DATE"] >= ytd_start) & (df["DATE"] <= anchor)]

ctd = rollup_by_costset(df_ctd)
ytd = rollup_by_costset(df_ytd)

# ===================== COST PERFORMANCE (CPI) =================================
# CPI = BCWP / ACWP
cpi_ctd = (ctd["BCWP"] / ctd["ACWP"].replace(0, np.nan)).round(2)
cpi_ytd = (ytd["BCWP"] / ytd["ACWP"].replace(0, np.nan)).round(2)

cost_performance_tbl = pd.DataFrame({
    "CTD": cpi_ctd,
    "YTD": cpi_ytd
})
cost_performance_tbl.index.name = None

# TOTAL row = ratio of sums (not average of ratios)
tot_ctd_acwp = ctd["ACWP"].sum()
tot_ctd_bcwp = ctd["BCWP"].sum()
tot_ytd_acwp = ytd["ACWP"].sum()
tot_ytd_bcwp = ytd["BCWP"].sum()
cost_performance_tbl.loc["TOTAL"] = [
    np.nan if np.isclose(tot_ctd_acwp, 0) else round(tot_ctd_bcwp / tot_ctd_acwp, 2),
    np.nan if np.isclose(tot_ytd_acwp, 0) else round(tot_ytd_bcwp / tot_ytd_acwp, 2),
]

# =================== SCHEDULE PERFORMANCE (SPI) ===============================
# SPI = BCWP / BCWS
spi_ctd = (ctd["BCWP"] / ctd["BCWS"].replace(0, np.nan)).round(2)
spi_ytd = (ytd["BCWP"] / ytd["BCWS"].replace(0, np.nan)).round(2)

schedule_performance_tbl = pd.DataFrame({
    "CTD": spi_ctd,
    "YTD": spi_ytd
})
schedule_performance_tbl.index.name = None

# TOTAL row = ratio of sums
tot_ctd_bcws = ctd["BCWS"].sum()
tot_ytd_bcws = ytd["BCWS"].sum()
schedule_performance_tbl.loc["TOTAL"] = [
    np.nan if np.isclose(tot_ctd_bcws, 0) else round(tot_ctd_bcwp / tot_ctd_bcws, 2),
    np.nan if np.isclose(tot_ytd_bcws, 0) else round(tot_ytd_bcwp / tot_ytd_bcws, 2),
]

print(f"Anchor: {anchor:%Y-%m-%d} | YTD start: {ytd_start:%Y-%m-%d}")
print("\n=== Cost Performance (CPI: CTD & YTD) ===")
display(cost_performance_tbl)

print("\n=== Schedule Performance (SPI: CTD & YTD) ===")
display(schedule_performance_tbl)
# ==============================================================================