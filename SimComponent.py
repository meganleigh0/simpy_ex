# === Style EVMS tables with exact colors; output ONLY styled Excel =============
import os
import numpy as np
import pandas as pd

# Exact palette from your Threshold Key (RGB -> HEX)
HEX_BLUE   = "#8EB4E3"   # 142,180,227  (>= 1.05 or >= +0.05)
HEX_GREEN  = "#339966"   # 051,153,102  ([1.02,1.05) or [+0.02,+0.05))
HEX_YELLOW = "#FFFF99"   # 255,255,153  ([0.98,1.02) or [-0.02,+0.02])
HEX_RED    = "#C0504D"   # 192,080,077  ([0.95,0.98) and <0.95 ; and VAC/BAC <-0.02)

# ----------------------- Threshold color functions -----------------------------
def color_spi_cpi_exact(x):
    """Color for ratios around 1.0 (SPI/CPI/BAC/EAC)."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    try: v = float(x)
    except Exception: return ""
    if v >= 1.05:   return f"background-color:{HEX_BLUE};color:#000000"
    if v >= 1.02:   return f"background-color:{HEX_GREEN};color:#000000"
    if v >= 0.98:   return f"background-color:{HEX_YELLOW};color:#000000"
    if v >= 0.95:   return f"background-color:{HEX_RED};color:#FFFFFF"   # 0.95–0.98
    return f"background-color:{HEX_RED};color:#FFFFFF"                    # <0.95

def color_vacbac_exact(x):
    """Color for VAC/BAC thresholds centered at 0."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    try: v = float(x)
    except Exception: return ""
    if v >= 0.05:   return f"background-color:{HEX_BLUE};color:#000000"
    if v >= 0.02:   return f"background-color:{HEX_GREEN};color:#000000"
    if v >= -0.02:  return f"background-color:{HEX_YELLOW};color:#000000"
    if v >= -0.05:  return f"background-color:{HEX_RED};color:#FFFFFF"   # -0.05 to -0.02
    return f"background-color:{HEX_RED};color:#FFFFFF"                    # < -0.05

# ----------------------- Styling / save helpers --------------------------------
def save_styled(df, name, styler, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_styled.xlsx")
    try:
        styler.to_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"[warn] Could not write styled Excel for {name}: {e}")
    display(styler)

# ----------------------- Cost Performance (CPI) --------------------------------
if 'cost_performance_tbl' in globals():
    # format + color both columns
    sty = (cost_performance_tbl
           .style
           .format({"CTD":"{:.2f}","YTD":"{:.2f}"})
           .applymap(color_spi_cpi_exact, subset=["CTD","YTD"]))
    save_styled(cost_performance_tbl, "cost_performance_tbl", sty)

# ----------------------- Schedule Performance (SPI) ----------------------------
if 'schedule_performance_tbl' in globals():
    sty = (schedule_performance_tbl
           .style
           .format({"CTD":"{:.2f}","YTD":"{:.2f}"})
           .applymap(color_spi_cpi_exact, subset=["CTD","YTD"]))
    save_styled(schedule_performance_tbl, "schedule_performance_tbl", sty)

# ----------------------- EVMS Metrics (SPI/CPI rows) ---------------------------
if 'evms_metrics_tbl' in globals():
    cols = list(evms_metrics_tbl.columns)  # ["CTD","4WK","YTD"]
    sty = (evms_metrics_tbl
           .style
           .format({c:"{:.2f}" for c in cols})
           .applymap(color_spi_cpi_exact, subset=cols))
    save_styled(evms_metrics_tbl, "evms_metrics_tbl", sty)

# ----------------------- Labor table: color VAC using VAC/BAC ------------------
if 'labor_tbl' in globals():
    # build a style DataFrame where only VAC (K) cells get colored by VAC/BAC
    def vac_column_style(df_):
        css = pd.DataFrame("", index=df_.index, columns=df_.columns)
        if "VAC (K)" in df_.columns and "BAC (K)" in df_.columns:
            vac = pd.to_numeric(df_["VAC (K)"], errors="coerce")
            bac = pd.to_numeric(df_["BAC (K)"], errors="coerce")
            vacbac = vac / bac.replace(0, np.nan)  # uses same units (K)
            colors = vacbac.apply(color_vacbac_exact)
            css.loc[:, "VAC (K)"] = colors.values
        return css

    sty = (labor_tbl
           .style
           .apply(vac_column_style, axis=None))
    save_styled(labor_tbl, "labor_tbl", sty)

# ----------------------- Monthly labor: color BOTH columns ---------------------
if 'labor_monthly_tbl' in globals():
    # BAC/EAC uses SPI/CPI thresholds; VAC/BAC uses VAC/BAC thresholds
    def apply_dual_thresholds(df_):
        css = pd.DataFrame("", index=df_.index, columns=df_.columns)
        if "BAC/EAC" in df_.columns:
            css["BAC/EAC"] = df_["BAC/EAC"].apply(color_spi_cpi_exact).values
        if "VAC/BAC" in df_.columns:
            css["VAC/BAC"] = df_["VAC/BAC"].apply(color_vacbac_exact).values
        return css

    # Ensure ratios display to 2 decimals
    fmt_cols = {c:"{:.2f}" for c in labor_monthly_tbl.columns if c in ["BAC/EAC","VAC/BAC"]}
    sty = (labor_monthly_tbl
           .style
           .format(fmt_cols)
           .apply(apply_dual_thresholds, axis=None))
    save_styled(labor_monthly_tbl, "labor_monthly_tbl", sty)

print("Styled tables displayed and saved to ./output as *_styled.xlsx (no CSVs).")
# =============================================================================