# ===== Dashboard metrics from summary totals (ONE CELL) =====
import numpy as np
import pandas as pd

# --- Helper to pull TOTAL from a grouped table ---
def _total_row(df: pd.DataFrame, cols=("ACWP","BCWP","BCWS","ETC")) -> pd.Series:
    if len(df)==0:
        return pd.Series({c:0.0 for c in cols})
    if str(df.index[-1]).upper()=="TOTAL":
        s = df.iloc[-1][list(cols)]
    else:
        s = df[list(cols)].sum(numeric_only=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# --- Build "summary" if not already defined (rows: period; cols: ACWP/BCWP/BCWS/ETC) ---
try:
    summary  # do we already have it?
except NameError:
    summary = pd.DataFrame(
        {
            "Status Period": _total_row(grouped_status),
            "Last 4 Weeks": _total_row(grouped_4wk),
            "Cumulative": _total_row(grouped_cum),
        }
    ).T

# standardize order and names to match your sheet headings
summary = summary[["ACWP","BCWP","BCWS","ETC"]].astype(float)
display_names = {"Cumulative":"CUM", "Last 4 Weeks":"4Wk", "Status Period":"Current status"}
summary_totals = summary.rename(index=display_names).loc[["CUM","4Wk","Current status"]]

# === Derived metrics (Excel-equivalent formulas) ===
# SPI = BCWP / BCWS
# CPI = BCWP / ACWP
# SV  = BCWP - BCWS
# SV% = SV / BCWS
# CV  = BCWP - ACWP
# CV% = CV / BCWP
def _safe_div(a, b):
    return np.where(np.isclose(b,0.0), np.nan, a/b)

spi = _safe_div(summary_totals["BCWP"], summary_totals["BCWS"])
cpi = _safe_div(summary_totals["BCWP"], summary_totals["ACWP"])
sv  = summary_totals["BCWP"] - summary_totals["BCWS"]
sv_pct = _safe_div(sv, summary_totals["BCWS"]) * 100.0
cv  = summary_totals["BCWP"] - summary_totals["ACWP"]
cv_pct = _safe_div(cv, summary_totals["BCWP"]) * 100.0

# CUM-only items (use CUM row like your sheet):
cum = summary_totals.loc["CUM"]
BAC  = cum["BCWS"]                       # =C3
EAC  = cum["ACWP"] + cum["ETC"]          # =A3 + D3
VAC  = BAC - EAC                         # =C3 - (A3 + D3)
VACp = (VAC / BAC * 100.0) if not np.isclose(BAC,0.0) else np.nan
BCWR = BAC - cum["BCWP"]                 # =C3 - B3
ETC_ = cum["ETC"]                        # just echo
TCPI = ((BAC - cum["BCWP"]) / (EAC - cum["ACWP"])) if not np.isclose(EAC - cum["ACWP"],0.0) else np.nan

# --- Assemble the dashboard-style tables ---
metrics_top = summary_totals.copy()  # ACWP/BCWP/BCWS/ETC blocks: CUM / 4Wk / Current status

metrics_mid = pd.DataFrame(
    {
        "CUM": [spi["CUM"], cpi["CUM"], sv["CUM"], sv_pct["CUM"], cv["CUM"], cv_pct["CUM"]],
        "4Wk": [spi["4Wk"], cpi["4Wk"], sv["4Wk"], sv_pct["4Wk"], cv["4Wk"], cv_pct["4Wk"]],
        "Current status": [spi["Current status"], cpi["Current status"], sv["Current status"],
                           sv_pct["Current status"], cv["Current status"], cv_pct["Current status"]],
    },
    index=["SPI","CPI","SV","SV%","CV","CV%"],
)

metrics_bottom = pd.DataFrame(
    {
        "CUM": [BAC, EAC, VAC, VACp, BCWR, ETC_, TCPI],
        "4Wk": [np.nan]*7,
        "Current status": [np.nan]*7,
    },
    index=["BAC","EAC","VAC","VAC%","BCWR","ETC","TCPI"],
)

# Optional rounding like your sheet
def _round(df):
    out = df.copy()
    for r in ["SPI","CPI","TCPI"]:
        if r in out.index: out.loc[r] = out.loc[r].astype(float).round(2)
    for r in ["SV%","CV%","VAC%"]:
        if r in out.index: out.loc[r] = out.loc[r].astype(float).round(2)
    for r in ["SV","CV","BAC","EAC","VAC","BCWR","ETC"]:
        if r in out.index: out.loc[r] = out.loc[r].astype(float).round(2)
    return out

metrics_mid = _round(metrics_mid)
metrics_bottom = _round(metrics_bottom)

# === Results ===
# 1) Top totals exactly like the yellow blocks in your sheet:
print("TOP TOTALS (match yellow CUM / 4Wk / Current status):")
display(metrics_top)

# 2) Derived metrics (SPI/CPI/SV/SV%/CV/CV%):
print("DERIVED METRICS:")
display(metrics_mid)

# 3) BAC/EAC/VAC/VAC%/BCWR/ETC/TCPI (CUM only):
print("CUM-ONLY METRICS:")
display(metrics_bottom)