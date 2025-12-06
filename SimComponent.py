# ============================================
# XM30 EVMS VALIDATION – CPI/SPI CTD & LSD
# ============================================
import pandas as pd
import numpy as np
from datetime import datetime

XM30_PATH = "data/Cobra-XM30.xlsx"

# --- Accounting calendar closings (same as main pipeline) ---
ACCOUNTING_CLOSINGS = {
    (2025, 1): 26,
    (2025, 2): 23,
    (2025, 3): 30,
    (2025, 4): 27,
    (2025, 5): 25,
    (2025, 6): 29,
    (2025, 7): 27,
    (2025, 8): 24,
    (2025, 9): 28,
    (2025,10): 26,
    (2025,11): 23,
    (2025,12): 31,
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
        for c in df.columns
    })

def map_cost_sets(cost_cols):
    cleaned = {
        col: col.replace(" ", "").replace("-", "").replace("_", "").upper()
        for col in cost_cols
    }
    bcws = bcwp = acwp = etc = None
    for orig, clean in cleaned.items():
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig
        elif ("BCWP" in clean) or ("EARNED" in clean) or ("PROGRESS" in clean):
            bcwp = orig
        elif "ETC" in clean:
            etc = orig
    return bcws, bcwp, acwp, etc

def get_status_dates(dates: pd.Series):
    dates = pd.to_datetime(dates)
    max_date = dates.max()

    closing_dates = []
    for (year, month), day in ACCOUNTING_CLOSINGS.items():
        d = datetime(year, month, day)
        if d <= max_date:
            closing_dates.append(d)
    closing_dates = sorted(closing_dates)

    if len(closing_dates) >= 2:
        curr = closing_dates[-1]
        prev = closing_dates[-2]
    elif len(closing_dates) == 1:
        curr = prev = closing_dates[0]
    else:
        uniq = sorted(dates.unique())
        curr = uniq[-1]
        prev = uniq[-2] if len(uniq) > 1 else uniq[-1]
    return curr, prev

def get_row_on_or_before(df: pd.DataFrame, date: datetime):
    sub = df[df["DATE"] <= date]
    if sub.empty:
        return df.iloc[0]
    return sub.iloc[-1]

# --------------------------------------------
# 1) Load XM30 Cobra and build EV time series
# --------------------------------------------
xm = pd.read_excel(XM30_PATH)
xm = normalize_columns(xm)

required = {"DATE", "COSTSET", "HOURS"}
missing = required - set(xm.columns)
if missing:
    raise ValueError(f"XM30 missing columns after normalization: {missing}")

xm["DATE"] = pd.to_datetime(xm["DATE"])

# pivot to DATE x COSTSET
pivot = xm.pivot_table(
    index="DATE", columns="COSTSET", values="HOURS", aggfunc="sum"
).reset_index()

cost_cols = [c for c in pivot.columns if c != "DATE"]
bcws_col, bcwp_col, acwp_col, _ = map_cost_sets(cost_cols)

print("Mapped cost sets for XM30:")
print("  BCWS:", bcws_col)
print("  BCWP:", bcwp_col)
print("  ACWP:", acwp_col, "\n")

if any(c is None for c in [bcws_col, bcwp_col, acwp_col]):
    raise ValueError("One or more cost sets could not be mapped – check column names.")

BCWS = pivot[bcws_col].fillna(0.0)
BCWP = pivot[bcwp_col].fillna(0.0)
ACWP = pivot[acwp_col].fillna(0.0)

# Monthly indices
monthly_cpi = BCWP / ACWP.replace(0, np.nan)
monthly_spi = BCWP / BCWS.replace(0, np.nan)

# Cumulative indices
cum_bcws = BCWS.cumsum()
cum_bcwp = BCWP.cumsum()
cum_acwp = ACWP.cumsum()

cum_cpi = cum_bcwp / cum_acwp.replace(0, np.nan)
cum_spi = cum_bcwp / cum_bcws.replace(0, np.nan)

evdf_xm30 = pd.DataFrame({
    "DATE": pivot["DATE"],
    "BCWS": BCWS,
    "BCWP": BCWP,
    "ACWP": ACWP,
    "Monthly CPI": monthly_cpi,
    "Monthly SPI": monthly_spi,
    "Cumulative CPI": cum_cpi,
    "Cumulative SPI": cum_spi,
})

# --------------------------------------------
# 2) Determine CTD & LSD dates and values
# --------------------------------------------
curr_date, prev_date = get_status_dates(evdf_xm30["DATE"])
row_curr = get_row_on_or_before(evdf_xm30, curr_date)
row_prev = get_row_on_or_before(evdf_xm30, prev_date)

cpi_ctd = row_curr["Cumulative CPI"]
spi_ctd = row_curr["Cumulative SPI"]
cpi_lsd = row_prev["Cumulative CPI"]
spi_lsd = row_prev["Cumulative SPI"]

print(f"XM30 status dates:")
print(f"  CTD date (current): {curr_date.date()}")
print(f"  LSD date (previous): {prev_date.date()}\n")

print("Program-level CPI/SPI expected on slide:")
print(pd.DataFrame({
    "Metric": ["CPI", "SPI"],
    "CTD":    [cpi_ctd, spi_ctd],
    "LSD":    [cpi_lsd, spi_lsd],
}).round(4))
print()

# --------------------------------------------
# 3) Show last ~6 months of detailed EV data
# --------------------------------------------
print("Last 6 status rows for manual spot-checking (BCWS/BCWP/ACWP + indices):")
display(
    evdf_xm30.sort_values("DATE")
             .tail(6)
             .assign(
                 **{
                     "Monthly CPI": lambda d: d["Monthly CPI"].round(4),
                     "Monthly SPI": lambda d: d["Monthly SPI"].round(4),
                     "Cumulative CPI": lambda d: d["Cumulative CPI"].round(4),
                     "Cumulative SPI": lambda d: d["Cumulative SPI"].round(4),
                 }
             )
)
# Optional: save to Excel so you can compare vs manual calcs
evdf_xm30.to_excel("XM30_EVMS_Validation.xlsx", index=False)
print("\nSaved detailed XM30 EV series to XM30_EVMS_Validation.xlsx")