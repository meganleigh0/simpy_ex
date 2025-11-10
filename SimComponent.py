# === schedule_performance_tbl (SPI by CTD and YTD for each CHG#) ==============
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure DATE is datetime (in-place, continuing from your notebook)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

# Use latest date in df as the anchor; YTD starts Jan 1 of that year
anchor = pd.to_datetime(df["DATE"].max())
ytd_start = datetime(anchor.year, 1, 1)

# ---- CTD rollup: sum HOURS by CHG# x COST-SET --------------------------------
ctd = (df[df["DATE"].notna()]
       .groupby(["CHG#", "COST-SET"], dropna=False)["HOURS"]
       .sum()
       .unstack(fill_value=0.0))
for k in ["BCWP", "BCWS"]:
    if k not in ctd.columns: ctd[k] = 0.0
ctd = ctd[["BCWP", "BCWS"]].astype(float)

# ---- YTD rollup --------------------------------------------------------------
ytd = (df[(df["DATE"] >= ytd_start) & df["DATE"].notna()]
       .groupby(["CHG#", "COST-SET"], dropna=False)["HOURS"]
       .sum()
       .unstack(fill_value=0.0))
for k in ["BCWP", "BCWS"]:
    if k not in ytd.columns: ytd[k] = 0.0
ytd = ytd[["BCWP", "BCWS"]].astype(float)

# ---- SPI = BCWP / BCWS -------------------------------------------------------
spi_ctd = (ctd["BCWP"] / ctd["BCWS"].replace(0, np.nan)).round(2)
spi_ytd = (ytd["BCWP"] / ytd["BCWS"].replace(0, np.nan)).round(2)

schedule_performance_tbl = pd.DataFrame({
    "CTD": spi_ctd,
    "YTD": spi_ytd
})
schedule_performance_tbl.index.name = None

# TOTAL row (ratio of sums, not average of ratios)
tot_spi_ctd = np.nan if np.isclose(ctd["BCWS"].sum(), 0) else round(ctd["BCWP"].sum() / ctd["BCWS"].sum(), 2)
tot_spi_ytd = np.nan if np.isclose(ytd["BCWS"].sum(), 0) else round(ytd["BCWP"].sum() / ytd["BCWS"].sum(), 2)
schedule_performance_tbl.loc["TOTAL"] = [tot_spi_ctd, tot_spi_ytd]

print(f"Anchor: {anchor:%Y-%m-%d} | YTD start: {ytd_start:%Y-%m-%d}")
display(schedule_performance_tbl)
# =============================================================================