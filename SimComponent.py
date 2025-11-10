# === Style ONLY labor tables with exact colors; save styled Excel =============
import os
import numpy as np
import pandas as pd

# ---- Exact palette (from your key) -------------------------------------------
HEX_BLUE   = "#8EB4E3"   # 142,180,227
HEX_GREEN  = "#339966"   # 51,153,102
HEX_YELLOW = "#FFFF99"   # 255,255,153
HEX_RED    = "#C0504D"   # 192,80,77  (used for the two lowest bands)

# ---- Threshold mappers -------------------------------------------------------
def color_spi_cpi_exact(x):
    """For ratios centered at 1.0 (SPI/CPI/BAC/EAC style)."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    try: v = float(x)
    except Exception: return ""
    if v >= 1.05:   return f"background-color:{HEX_BLUE};color:#000000"
    if v >= 1.02:   return f"background-color:{HEX_GREEN};color:#000000"
    if v >= 0.98:   return f"background-color:{HEX_YELLOW};color:#000000"
    if v >= 0.95:   return f"background-color:{HEX_RED};color:#FFFFFF"   # 0.95–0.98
    return f"background-color:{HEX_RED};color:#FFFFFF"                    # < 0.95

def color_vacbac_exact(x):
    """For VAC/BAC thresholds centered at 0."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    try: v = float(x)
    except Exception: return ""
    if v >= 0.05:   return f"background-color:{HEX_BLUE};color:#000000"
    if v >= 0.02:   return f"background-color:{HEX_GREEN};color:#000000"
    if v >= -0.02:  return f"background-color:{HEX_YELLOW};color:#000000"
    if v >= -0.05:  return f"background-color:{HEX_RED};color:#FFFFFF"   # -0.05 to -0.02
    return f"background-color:{HEX_RED};color:#FFFFFF"                    # < -0.05

# ---- Helpers -----------------------------------------------------------------
def save_styled_only(name, styler, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_styled.xlsx")
    try:
        styler.to_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"[warn] Could not write styled Excel for {name}: {e}")
    display(styler)

def find_col(df, startswith_text):
    """Find first column whose name (upper/stripped) starts with text."""
    key = startswith_text.upper()
    for c in df.columns:
        if str(c).strip().upper().startswith(key):
            return c
    return None

# ---- Labor Hours table: color VAC by VAC/BAC ---------------------------------
if 'labor_tbl' in globals():
    # Accept either 'VAC' or 'VAC (K)', 'BAC' or 'BAC (K)'
    vac_col = find_col(labor_tbl, "VAC")
    bac_col = find_col(labor_tbl, "BAC")

    if vac_col and bac_col:
        # Build a per-cell CSS frame; color only the VAC column
        def vac_style(df_):
            css = pd.DataFrame("", index=df_.index, columns=df_.columns)
            vac = pd.to_numeric(df_[vac_col], errors="coerce")
            bac = pd.to_numeric(df_[bac_col], errors="coerce")
            ratio = vac / bac.replace(0, np.nan)  # K units cancel if used
            css[vac_col] = ratio.apply(color_vacbac_exact).values
            return css

        sty = labor_tbl.style.apply(vac_style, axis=None)
        save_styled_only("labor_tbl", sty)
    else:
        print("[warn] Could not find VAC/BAC columns in labor_tbl.")

# ---- Monthly labor table: color BOTH columns ---------------------------------
if 'labor_monthly_tbl' in globals():
    # Accept canonical names; tolerate case/spacing differences
    def match_col(df_, target):
        for c in df_.columns:
            if str(c).strip().replace(" ", "").upper() == target:
                return c
        return None

    bac_eac_col = match_col(labor_monthly_tbl, "BAC/EAC")
    vac_bac_col = match_col(labor_monthly_tbl, "VAC/BAC")

    def dual_thresholds(df_):
        css = pd.DataFrame("", index=df_.index, columns=df_.columns)
        if bac_eac_col in df_.columns:
            css[bac_eac_col] = df_[bac_eac_col].apply(color_spi_cpi_exact).values
        if vac_bac_col in df_.columns:
            css[vac_bac_col] = df_[vac_bac_col].apply(color_vacbac_exact).values
        return css

    fmt = {}
    if bac_eac_col: fmt[bac_eac_col] = "{:.2f}"
    if vac_bac_col: fmt[vac_bac_col] = "{:.2f}"

    sty = labor_monthly_tbl.style.format(fmt).apply(dual_thresholds, axis=None)
    save_styled_only("labor_monthly_tbl", sty)

print("Styled labor tables saved to ./output as *_styled.xlsx (no CSVs).")
# =============================================================================