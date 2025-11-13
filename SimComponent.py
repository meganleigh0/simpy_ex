import os
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor

# ---------- helpers for PPTX ----------

def hex_to_rgb_tuple(hex_color: str):
    """'#RRGGBB' -> (R,G,B)"""
    if not isinstance(hex_color, str):
        return None
    hc = hex_color.strip().lstrip("#")
    if len(hc) != 6:
        return None
    return tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))

def parse_css(css: str):
    """
    css like 'background-color:#4E9BE3;color:#000000'
    -> (bg_hex, text_hex)
    """
    if not isinstance(css, str) or "background-color" not in css:
        return None, None

    bg_hex, fg_hex = None, None
    for part in css.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k == "background-color":
            bg_hex = v
        elif k == "color":
            fg_hex = v
    return bg_hex, fg_hex

def add_df_slide(prs, title, df_display, css_df=None):
    """
    Add a slide with a table:
      df_display: DataFrame of strings/numbers to show
      css_df: DataFrame of css strings (same shape/index/cols as df_display)
    """
    rows, cols = df_display.shape
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = title

    left = Inches(0.3)
    top = Inches(1.2)
    width = Inches(9.0)
    height = Inches(0.8)

    table = slide.shapes.add_table(rows + 1, cols, left, top, width, height).table

    # headers
    for j, col in enumerate(df_display.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.bold = True

    # body
    for i, idx in enumerate(df_display.index):
        for j, col in enumerate(df_display.columns):
            value = df_display.loc[idx, col]
            if pd.isna(value):
                text = ""
            else:
                text = str(value)

            cell = table.cell(i + 1, j)
            cell.text = text

            bg_hex, fg_hex = None, None
            if css_df is not None and idx in css_df.index and col in css_df.columns:
                bg_hex, fg_hex = parse_css(css_df.loc[idx, col])

            if bg_hex:
                rgb = hex_to_rgb_tuple(bg_hex)
                if rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*rgb)

            if fg_hex:
                rgb = hex_to_rgb_tuple(fg_hex)
                if rgb:
                    for p in cell.text_frame.paragraphs:
                        for run in p.runs:
                            run.font.color.rgb = RGBColor(*rgb)

    return prs

# ---------- build the PowerPoint ----------

prs = Presentation()

# 1) Cost Performance (CPI) table -------------------------------------------
if "cost_performance_tbl" in globals():
    df = set_index(cost_performance_tbl.copy())

    # create CSS using same thresholds/colors as Styler
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in ["CTD", "YTD"]:
        if col in df.columns:
            css[col] = df[col].apply(color_spi_cpi_exact).values

    # format numbers as in Styler (.2f for CTD/YTD)
    df_display = df.copy()
    for col in ["CTD", "YTD"]:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce") \
                                  .map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    prs = add_df_slide(prs, "Cost Performance (CPI)", df_display, css)

# 2) Schedule Performance (SPI) table ---------------------------------------
if "schedule_performance_tbl" in globals():
    df = set_index(schedule_performance_tbl.copy())

    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in ["CTD", "YTD"]:
        if col in df.columns:
            css[col] = df[col].apply(color_spi_cpi_exact).values

    df_display = df.copy()
    for col in ["CTD", "YTD"]:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce") \
                                   .map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    prs = add_df_slide(prs, "Schedule Performance (SPI)", df_display, css)

# 3) EVMS metrics (SPI/CPI rows) -------------------------------------------
if "evms_metrics_tbl" in globals():
    df = set_index(evms_metrics_tbl.copy())

    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        css[col] = df[col].apply(color_spi_cpi_exact).values

    df_display = df.copy()
    for col in df_display.columns:
        df_display[col] = pd.to_numeric(df_display[col], errors="coerce") \
                               .map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    prs = add_df_slide(prs, "EVMS Metrics (SPI/CPI)", df_display, css)

# 4) Labor hours table – color VAC by VAC/BAC -------------------------------
if "labor_tbl" in globals():
    df = set_index(labor_tbl.copy())
    vac_col = find_col(df, "VAC")
    bac_col = find_col(df, "BAC")

    css = pd.DataFrame("", index=df.index, columns=df.columns)
    if vac_col and bac_col:
        vac = pd.to_numeric(df[vac_col], errors="coerce")
        bac = pd.to_numeric(df[bac_col], errors="coerce")
        bac = bac.replace(0, np.nan)
        ratio = vac / bac
        css[vac_col] = ratio.apply(color_vacbac_exact).values

    df_display = df.copy()  # keep original numeric formatting
    prs = add_df_slide(prs, "Labor Hours (VAC/BAC)", df_display, css)

# 5) Monthly labor table – color BAC/EAC and VAC/BAC -----------------------
def match_col(df, target):
    """Find the first column whose name starts with target (case-insensitive)."""
    key = target.upper()
    for c in df.columns:
        if str(c).strip().replace(" ", "").upper().startswith(key.replace(" ", "")):
            return c
    return None

if "labor_monthly_tbl" in globals():
    df = set_index(labor_monthly_tbl.copy())

    bac_eac_col = match_col(df, "BAC/EAC")
    vac_bac_col = match_col(df, "VAC/BAC")

    css = pd.DataFrame("", index=df.index, columns=df.columns)
    if bac_eac_col and bac_eac_col in df.columns:
        css[bac_eac_col] = df[bac_eac_col].apply(color_spi_cpi_exact).values
    if vac_bac_col and vac_bac_col in df.columns:
        css[vac_bac_col] = df[vac_bac_col].apply(color_vacbac_exact).values

    df_display = df.copy()
    for col in [bac_eac_col, vac_bac_col]:
        if col and col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce") \
                                   .map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    prs = add_df_slide(prs, "Monthly Labor (BAC/EAC & VAC/BAC)", df_display, css)

# ---------- save the PowerPoint ----------

os.makedirs("output", exist_ok=True)
ppt_path = os.path.join("output", "evms_tables.pptx")
prs.save(ppt_path)
print(f"Saved PowerPoint to: {ppt_path}")