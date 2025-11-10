# === evms_metrics_tbl: SPI & CPI as rows; CTD, 4WK, YTD as columns ===========
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Expect in-memory df with columns: DATE, COST-SET, HOURS (no renames)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
anchor = pd.to_datetime(df["DATE"].max())
ytd_start = datetime(anchor.year, 1, 1)
w4_start  = anchor - timedelta(weeks=4)

def sums_for_window(dframe, start=None, end=None):
    m = dframe["DATE"].notna()
    if start is not None: m &= dframe["DATE"] >= start
    if end   is not None: m &= dframe["DATE"] <= end
    g = (dframe.loc[m]
         .groupby("COST-SET", dropna=False)["HOURS"]
         .sum())
    # ensure required cost-sets exist
    for k in ["ACWP","BCWP","BCWS"]:
        if k not in g.index:
            g.loc[k] = 0.0
    # return as simple dict of floats
    s = g[["ACWP","BCWP","BCWS"]].astype(float).to_dict()
    return s

ctd = sums_for_window(df, end=anchor)
ytd = sums_for_window(df, start=ytd_start, end=anchor)
w4  = sums_for_window(df, start=w4_start,  end=anchor)

def r(num, den):
    return np.nan if np.isclose(den, 0) else num / den

# SPI = BCWP / BCWS
spi_ctd = r(ctd["BCWP"], ctd["BCWS"])
spi_4wk = r(w4["BCWP"],  w4["BCWS"])
spi_ytd = r(ytd["BCWP"], ytd["BCWS"])

# CPI = BCWP / ACWP
cpi_ctd = r(ctd["BCWP"], ctd["ACWP"])
cpi_4wk = r(w4["BCWP"],  w4["ACWP"])
cpi_ytd = r(ytd["BCWP"], ytd["ACWP"])

evms_metrics_tbl = pd.DataFrame(
    {
        "CTD":  [spi_ctd, cpi_ctd],
        "4WK":  [spi_4wk, cpi_4wk],
        "YTD":  [spi_ytd, cpi_ytd],
    },
    index=["SPI", "CPI"]
).round(2)

print(f"Anchor: {anchor:%Y-%m-%d} | YTD start: {ytd_start:%Y-%m-%d} | 4WK start: {w4_start:%Y-%m-%d}")
display(evms_metrics_tbl)
# =============================================================================