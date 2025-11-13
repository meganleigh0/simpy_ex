import os
import numpy as np
import pandas as pd

from IPython.display import display

# Assumes color_spi_cpi_exact and color_vacbac_exact already exist in the notebook.
# They should both return CSS strings like 'background-color:#xxxxxx;color:#yyyyyy'

outdir = "pbi_exports"
os.makedirs(outdir, exist_ok=True)

def set_index(df, index_col="SUB_TEAM"):
    if index_col in df.columns:
        df = df.copy()
        df.set_index(index_col, inplace=True)
    return df

# -------------------
# Helper: write styled df to Excel
# -------------------
def save_styled_excel(styler, filepath):
    """
    Save a pandas Styler to an Excel file with formatting.
    Requires pandas >= 1.4 and openpyxl installed.
    """
    styler.to_excel(filepath, engine="openpyxl")
    print(f"Saved styled Excel: {filepath}")


# -------------------
# Cost Performance (CPI)
# -------------------
if "cost_performance_tbl" in globals():
    df = set_index(cost_performance_tbl)

    sty = (
        df.style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )

    display(sty)
    save_styled_excel(sty, os.path.join(outdir, "cost_performance_tbl.xlsx"))


# -------------------
# Schedule Performance (SPI)
# -------------------
if "schedule_performance_tbl" in globals():
    df = set_index(schedule_performance_tbl)

    sty = (
        df.style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )

    display(sty)
    save_styled_excel(sty, os.path.join(outdir, "schedule_performance_tbl.xlsx"))


# -------------------
# EVMS Metrics (SPI/CPI rows)
# -------------------
if "evms_metrics_tbl" in globals():
    df = set_index(evms_metrics_tbl)
    numeric_cols = df.select_dtypes(include=["number"]).columns

    sty = (
        df.style
        .format({c: "{:.2f}" for c in numeric_cols})
        .map(color_spi_cpi_exact, subset=numeric_cols)
    )

    display(sty)
    save_styled_excel(sty, os.path.join(outdir, "evms_metrics_tbl.xlsx"))


# -------------------
# Labor Hours (VAC/BAC)
# -------------------
def find_col(df, startswith_text):
    key = startswith_text.upper()
    for c in df.columns:
        if str(c).strip().upper().startswith(key):
            return c
    return None

if "labor_tbl" in globals():
    df = set_index(labor_tbl)
    vac_col = find_col(df, "VAC")
    bac_col = find_col(df, "BAC")

    if vac_col and bac_col:

        def vac_style_df(df_):
            css = pd.DataFrame("", index=df_.index, columns=df_.columns)
            vac = pd.to_numeric(df_[vac_col], errors="coerce")
            bac = pd.to_numeric(df_[bac_col], errors="coerce")
            ratio = vac / bac.replace(0, np.nan)
            css[vac_col] = ratio.apply(color_vacbac_exact).values
            return css

        sty = df.style.apply(vac_style_df, axis=None)
        display(sty)
        save_styled_excel(sty, os.path.join(outdir, "labor_tbl.xlsx"))
    else:
        print("[warn] VAC/BAC columns not found in labor_tbl")


# -------------------
# Monthly Labor (BAC/EAC & VAC/BAC)
# -------------------
def match_col(df, target):
    t = target.replace(" ", "").upper()
    for c in df.columns:
        if str(c).replace(" ", "").upper() == t:
            return c
    return None

if "labor_monthly_tbl" in globals():
    df = set_index(labor_monthly_tbl)
    bac_eac_col = match_col(df, "BAC/EAC")
    vac_bac_col = match_col(df, "VAC/BAC")

    def dual_style_df(df_):
        css = pd.DataFrame("", index=df_.index, columns=df_.columns)
        if bac_eac_col in df_.columns:
            css[bac_eac_col] = df_[bac_eac_col].apply(color_spi_cpi_exact).values
        if vac_bac_col in df_.columns:
            css[vac_bac_col] = df_[vac_bac_col].apply(color_vacbac_exact).values
        return css

    sty = df.style.apply(dual_style_df, axis=None)
    display(sty)
    save_styled_excel(sty, os.path.join(outdir, "labor_monthly_tbl.xlsx"))