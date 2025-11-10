# === Color tables with EXACT palette + save CSV & styled Excel ===============
import os
import numpy as np

# ----- EXACT colors from your Threshold Key -----
# (RGB shown on your sheet -> HEX here)
HEX_NAVY   = "#1F497D"  # 031,073,125  (not used for thresholds; kept if you want headers)
HEX_BLUE   = "#8EB4E3"  # 142,180,227  (>= 1.05)
HEX_GREEN  = "#339966"  # 051,153,102  ([1.02, 1.05))
HEX_YELLOW = "#FFFF99"  # 255,255,153  ([0.98, 1.02))
HEX_RED    = "#C0504D"  # 192,080,077  (< 0.95)  (also used for 0.95–0.98 bucket below for consistency)

# SPI/CPI thresholds (from your key)
# Blue >= 1.05 | Green [1.02, 1.05) | Yellow [0.98, 1.02) | Orange-ish [0.95, 0.98) | Red < 0.95
# The key’s orange block matches Office "red accent" (192,80,77), so we use HEX_RED for both orange and red buckets.
def color_spi_cpi_exact(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    if v >= 1.05:   return f"background-color:{HEX_BLUE};color:#000000"
    if v >= 1.02:   return f"background-color:{HEX_GREEN};color:#000000"
    if v >= 0.98:   return f"background-color:{HEX_YELLOW};color:#000000"
    if v >= 0.95:   return f"background-color:{HEX_RED};color:#FFFFFF"   # 0.95–0.98
    return f"background-color:{HEX_RED};color:#FFFFFF"                    # <0.95

def style_and_save(df, name, subset_cols=None, fmt_cols=None, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    # 1) CSV (values only)
    df.to_csv(os.path.join(outdir, f"{name}.csv"), index=True)

    # 2) Styled Excel preserving colors
    sty = df.style
    if fmt_cols:
        sty = sty.format({c: "{:.2f}" for c in fmt_cols})
    if subset_cols:
        sty = sty.applymap(color_spi_cpi_exact, subset=subset_cols)
    try:
        sty.to_excel(os.path.join(outdir, f"{name}_styled.xlsx"), engine="openpyxl")
    except Exception as e:
        print(f"[warn] Could not write styled Excel for {name}: {e}")

    # 3) Show in notebook
    display(df.style.applymap(color_spi_cpi_exact, subset=subset_cols) if subset_cols else df.style)

# ---------------- Run for the tables that exist in your notebook --------------
# Cost Performance: CPI columns CTD/YTD
if 'cost_performance_tbl' in globals():
    style_and_save(
        cost_performance_tbl,
        name="cost_performance_tbl",
        subset_cols=["CTD", "YTD"],
        fmt_cols=["CTD", "YTD"]
    )

# Schedule Performance: SPI columns CTD/YTD
if 'schedule_performance_tbl' in globals():
    style_and_save(
        schedule_performance_tbl,
        name="schedule_performance_tbl",
        subset_cols=["CTD", "YTD"],
        fmt_cols=["CTD", "YTD"]
    )

# EVMS Metrics: rows = SPI/CPI; cols = CTD, 4WK, YTD  -> color all columns
if 'evms_metrics_tbl' in globals():
    style_and_save(
        evms_metrics_tbl,
        name="evms_metrics_tbl",
        subset_cols=list(evms_metrics_tbl.columns),
        fmt_cols=list(evms_metrics_tbl.columns)
    )

# Save labor tables as CSV too (no SPI/CPI coloring needed)
if 'labor_tbl' in globals():
    labor_tbl.to_csv(os.path.join("output", "labor_tbl.csv"), index=True)
if 'labor_monthly_tbl' in globals():
    labor_monthly_tbl.to_csv(os.path.join("output", "labor_monthly_tbl.csv"), index=True)

print("Saved CSVs (and styled XLSX copies with exact colors) to ./output")
# =============================================================================