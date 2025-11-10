# ===== EVMS / EMS METRICS PIPELINE (single cell) =====
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---- USER SETTINGS -----------------------------------------------------------
DATA_PATH   = "data/Cobra-XM30.xlsx"          # <-- your file
SHEET_NAME  = "tbl_Weekly Extract"            # <-- your sheet
PROGRAM     = "XM30"                          # label only (optional)
ANCHOR_DATE = None                             # None = today, or set e.g. datetime(2025, 10, 15)

# Company accounting month-end close days (month, day)
ACCOUNTING_CLOSE_MD = [
    (1, 26), (2, 23), (3, 30), (4, 27), (5, 25), (6, 29),
    (7, 27), (8, 24), (9, 28), (10, 26), (11, 23), (12, 31)
]

# ---- HELPERS -----------------------------------------------------------------
def as_dt(x):
    return pd.to_datetime(x, errors="coerce")

def prior_accounting_close(anchor: datetime) -> datetime:
    """Return the most recent accounting close date <= anchor."""
    candidates = []
    for y in (anchor.year - 1, anchor.year):
        for m, d in ACCOUNTING_CLOSE_MD:
            try:
                dt = datetime(y, m, d)
                if dt <= anchor:
                    candidates.append(dt)
            except ValueError:
                pass
    if not candidates:
        # Fallback far past date if nothing matched
        return datetime(anchor.year - 1, 1, 1)
    return max(candidates)

def safe_div(a, b):
    """Elementwise a/b with 0 -> NaN."""
    a = float(a)
    b = float(b)
    return np.nan if np.isclose(b, 0.0) else a / b

def period_totals(df: pd.DataFrame, start=None, end=None) -> pd.Series:
    """
    Sum HOURS by COST-SET for a filtered period.
    Returns Series with index ['ACWP','BCWP','BCWS','ETC'] (0.0 if missing).
    """
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["DATE"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["DATE"] <= pd.Timestamp(end)

    s = (df.loc[mask]
          .groupby("COST-SET", dropna=False)["HOURS"]
          .sum(min_count=1))
    return s.reindex(["ACWP","BCWP","BCWS","ETC"]).fillna(0.0)

def make_summary_tables(df: pd.DataFrame, anchor: datetime):
    """Build Cumulative, Last 4 Weeks, and Status Period totals (ACWP/BCWP/BCWS/ETC)."""
    four_weeks_ago = anchor - timedelta(weeks=4)
    close_date = prior_accounting_close(anchor)

    totals = {
        "Cumulative":    period_totals(df),
        "Last 4 Weeks":  period_totals(df, start=four_weeks_ago, end=anchor),
        "Status Period": period_totals(df, start=close_date + timedelta(days=0), end=anchor)
        # If you want strictly after close, use + timedelta(days=1)
    }
    summary = pd.DataFrame(totals).T[["ACWP","BCWP","BCWS","ETC"]].astype(float)
    return summary, close_date

def make_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build EVMS metrics for each period (rows = metrics, cols = periods).
    - SPI = BCWP / BCWS
    - CPI = BCWP / ACWP
    - SV  = BCWP - BCWS
    - SV% = SV / BCWS * 100
    - CV  = BCWP - ACWP
    - CV% = CV / BCWP * 100
    - BAC = BCWS (cumulative)
    - EAC = ACWP (cumulative) + ETC (cumulative)
    - VAC = BAC - EAC
    - VAC% = VAC / BAC * 100
    - BCWR = BCWS - BCWP  (cumulative)
    - ETC  = ETC (cumulative)
    - TCPI = (BAC - BCWP) / ETC  (cumulative)
    """
    periods = list(summary.index)
    out = pd.DataFrame(index=[
        "SPI","CPI","SV","SV%","CV","CV%","BAC","EAC","VAC","VAC%","BCWR","ETC","TCPI"
    ], columns=periods, dtype=float)

    # Per-period metrics
    for p in periods:
        acwp = summary.loc[p,"ACWP"]
        bcwp = summary.loc[p,"BCWP"]
        bcws = summary.loc[p,"BCWS"]
        etc  = summary.loc[p,"ETC"]

        spi = safe_div(bcwp, bcws)
        cpi = safe_div(bcwp, acwp)
        sv  = bcwp - bcws
        svp = np.nan if np.isclose(bcws,0) else (sv / bcws) * 100.0
        cv  = bcwp - acwp
        cvp = np.nan if np.isclose(bcwp,0) else (cv / bcwp) * 100.0

        out.loc["SPI", p] = spi
        out.loc["CPI", p] = cpi
        out.loc["SV",  p] = sv
        out.loc["SV%", p] = svp
        out.loc["CV",  p] = cv
        out.loc["CV%", p] = cvp

    # Cumulative-only metrics (BAC, EAC, VAC, VAC%, BCWR, ETC, TCPI)
    cum = summary.loc["Cumulative"]
    BAC  = cum["BCWS"]
    EAC  = cum["ACWP"] + cum["ETC"]
    VAC  = BAC - EAC
    VACp = np.nan if np.isclose(BAC,0) else (VAC / BAC) * 100.0
    BCWR = cum["BCWS"] - cum["BCWP"]
    TCPI = np.nan if np.isclose(cum["ETC"],0) else (BAC - cum["BCWP"]) / cum["ETC"]

    out.loc["BAC","Cumulative"]  = BAC
    out.loc["EAC","Cumulative"]  = EAC
    out.loc["VAC","Cumulative"]  = VAC
    out.loc["VAC%","Cumulative"] = VACp
    out.loc["BCWR","Cumulative"] = BCWR
    out.loc["ETC","Cumulative"]  = cum["ETC"]
    out.loc["TCPI","Cumulative"] = TCPI

    # Nice rounding for a copy you can paste
    rounded = out.copy()
    ratio_cols = ["SPI","CPI","TCPI"]
    pct_cols   = ["SV%","CV%","VAC%"]
    rounded.loc[ratio_cols] = rounded.loc[ratio_cols].round(2)
    rounded.loc[pct_cols]   = rounded.loc[pct_cols].round(1)
    # Hours-based quantities: integers look nicer for slides
    for r in ["SV","CV","BAC","EAC","VAC","BCWR","ETC"]:
        rounded.loc[r] = rounded.loc[r].round(0)

    return out, rounded

def make_slide_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build the PPT 'Labor Hours Performance' block (single total row):
      %COMP, BAC, EAC, VAC   (all in K hours except %COMP)
    """
    cum = summary.loc["Cumulative"]
    BAC = cum["BCWS"]
    EAC = cum["ACWP"] + cum["ETC"]
    VAC = BAC - EAC
    pct_comp = np.nan if np.isclose(BAC,0) else (cum["BCWP"] / BAC) * 100.0

    def K(x):  # thousands with one decimal
        return np.nan if pd.isna(x) else round(x / 1000.0, 1)

    slide = pd.DataFrame(
        {
            "%COMP": [round(pct_comp, 0)],
            "BAC (K)": [K(BAC)],
            "EAC (K)": [K(EAC)],
            "VAC (K)": [K(VAC)],
        },
        index=[f"{PROGRAM} Total"]
    )
    return slide

# ---- LOAD & RUN --------------------------------------------------------------
# Anchor date
anchor = (ANCHOR_DATE if isinstance(ANCHOR_DATE, datetime) else datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)

# Load exactly as-is (no renames)
xl = pd.ExcelFile(DATA_PATH)
weekly = xl.parse(SHEET_NAME)
weekly["DATE"] = as_dt(weekly["DATE"])

# Build period summaries and metrics
summary, close_date = make_summary_tables(weekly, anchor)
metrics_raw, metrics_rounded = make_metrics(summary)
slide_table = make_slide_table(summary)

# Optional: quick exports for pasting into PPT/Excel
# (Uncomment if you want files written.)
# with pd.ExcelWriter(f"EMS_{PROGRAM}_{anchor:%Y-%m-%d}.xlsx", engine="xlsxwriter") as xw:
#     summary.to_excel(xw, sheet_name="Summary_Totals")
#     metrics_rounded.to_excel(xw, sheet_name="Metrics_Table")
#     slide_table.to_excel(xw, sheet_name="Slide_Table")

# Display at the end of the cell so you can copy/paste directly from the notebook
print(f"Anchor Date: {anchor:%Y-%m-%d}  |  Prior Close: {close_date:%Y-%m-%d}")
print("\n=== SUMMARY TOTALS (HOURS) ===")
display(summary)

print("\n=== METRICS TABLE (rounded; paste to slide if desired) ===")
display(metrics_rounded)

print("\n=== SLIDE TABLE (Table 1: Labor Hours Performance) ===")
display(slide_table)
# ===== end cell =====