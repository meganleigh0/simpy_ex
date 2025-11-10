# === Metrics table from grouped_cum / grouped_4wk / grouped_status (ONE CELL) ===
import numpy as np
import pandas as pd

cols = ["ACWP","BCWP","BCWS","ETC"]

def total_series(df: pd.DataFrame, cols=cols):
    """Return TOTAL row for df over `cols`; if missing, compute the sum."""
    if len(df) == 0:
        return pd.Series({c: 0.0 for c in cols})
    last_idx = str(df.index[-1]).upper()
    if last_idx == "TOTAL" and set(cols).issubset(df.columns):
        s = df.iloc[-1][cols]
    else:
        s = df[cols].sum(numeric_only=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# Pull totals for each period
cum   = total_series(grouped_cum)
wk4   = total_series(grouped_4wk)
stat  = total_series(grouped_status)

periods = pd.DataFrame([cum, wk4, stat], index=["CUM","4Wk","Current status"])

def div(a, b):
    return np.where(np.isclose(b, 0.0), np.nan, a / b)

# Build metrics table skeleton
metrics = pd.DataFrame(
    index=["SPI","CPI","SV","SV%","CV","CV%","BAC","EAC","VAC","VAC%","BCWR","ETC","TCPI"],
    columns=["CUM","4Wk","Current status"],
    dtype=float
)

# Metrics calculated for each period (SPI/CPI/SV/SV%/CV/CV%)
for col, row in periods.iterrows():
    ACWP, BCWP, BCWS, ETCv = row["ACWP"], row["BCWP"], row["BCWS"], row["ETC"]
    SVv = BCWP - BCWS
    CVv = BCWP - ACWP
    metrics.loc["SPI", col] = div(BCWP, BCWS)
    metrics.loc["CPI", col] = div(BCWP, ACWP)
    metrics.loc["SV",  col] = SVv
    metrics.loc["SV%", col] = div(SVv, BCWS) * 100.0
    metrics.loc["CV",  col] = CVv
    metrics.loc["CV%", col] = div(CVv, BCWP) * 100.0

# CUM-only metrics (BAC/EAC/VAC/VAC%/BCWR/ETC/TCPI)
BAC  = cum["BCWS"]                     # BCWS (cumulative)
EAC  = cum["ACWP"] + cum["ETC"]        # ACWP_cum + ETC_cum
VAC  = BAC - EAC                       # BCWS - (ACWP + ETC)
VACp = (VAC / BAC * 100.0) if not np.isclose(BAC, 0.0) else np.nan
BCWR = BAC - cum["BCWP"]               # BCWS - BCWP
ETC_ = cum["ETC"]                      # cumulative ETC
TCPI = div((BAC - cum["BCWP"]), ETC_)  # (BCWS - BCWP) / ETC

metrics.loc["BAC", "CUM"]  = BAC
metrics.loc["EAC", "CUM"]  = EAC
metrics.loc["VAC", "CUM"]  = VAC
metrics.loc["VAC%", "CUM"] = VACp
metrics.loc["BCWR", "CUM"] = BCWR
metrics.loc["ETC", "CUM"]  = ETC_
metrics.loc["TCPI", "CUM"] = TCPI

# Optional: round like your sheet
rounded = metrics.copy()
rounded.loc[["SPI","CPI","SV%","CV%","VAC%","TCPI"]] = rounded.loc[["SPI","CPI","SV%","CV%","VAC%","TCPI"]].round(2)
rounded.loc[["SV","CV","BAC","EAC","VAC","BCWR","ETC"]] = rounded.loc[["SV","CV","BAC","EAC","VAC","BCWR","ETC"]].round(2)

rounded  # <-- final table