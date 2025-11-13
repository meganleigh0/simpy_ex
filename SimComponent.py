import os
import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# -------------------------------------------------------------------
# Exact palette from your Threshold Key (RGB -> HEX)
# -------------------------------------------------------------------
HEX_COLORS = {
    "BLUE":   "#B4E3F6",
    "GREEN":  "#339966",
    "YELLOW": "#FFFF99",
    "RED":    "#CC5040",
}

# -------------------------------------------------------------------
# Threshold color functions (same logic you had before)
# -------------------------------------------------------------------
def get_color_style(value, thresholds, colors, text_colors):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    for threshold, color, text_color in zip(thresholds, colors, text_colors):
        if v >= threshold:
            return f"background-color:{color};color:{text_color}"
    return f"background-color:{colors[-1]};color:{text_colors[-1]}"

def color_spi_cpi_exact(v):
    thresholds = [1.05, 1.02, 0.98, 0.95]
    colors = [
        HEX_COLORS["BLUE"],
        HEX_COLORS["GREEN"],
        HEX_COLORS["YELLOW"],
        HEX_COLORS["RED"],
        HEX_COLORS["RED"],
    ]
    text_colors = ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"]
    return get_color_style(v, thresholds, colors, text_colors)

def color_vacbac_exact(v):
    thresholds = [0.05, 0.02, -0.02, -0.05]
    colors = [
        HEX_COLORS["BLUE"],
        HEX_COLORS["GREEN"],
        HEX_COLORS["YELLOW"],
        HEX_COLORS["RED"],
        HEX_COLORS["RED"],
    ]
    text_colors = ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"]
    return get_color_style(v, thresholds, colors, text_colors)

# -------------------------------------------------------------------
# Helpers for PPT color + style parsing
# -------------------------------------------------------------------
def hex_to_rgb(hex_color):
    """'#RRGGBB' -> (R, G, B)"""
    if not hex_color:
        return None
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def parse_css(style_str):
    """Parse 'background-color:#xxxxxx;color:#yyyyyy' -> (bg_hex, text_hex)."""
    bg = txt = None
    if not style_str:
        return bg, txt
    for part in style_str.split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "background-color":
            bg = val
        elif key == "color":
            txt = val
    return bg, txt

def add_table_slide(prs, df, title, fmt=None, cell_style_fn=None):
    """
    Add a slide with a table.

    cell_style_fn(row, col_name, value) -> (bg_hex, text_hex)  or (None, None)
    """
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    slide.shapes.title.text = title

    rows = df.shape[0] + 1              # header row
    cols = df.shape[1] + 1              # index column
    left = Inches(0.3)
    top = Inches(1.1)
    width = prs.slide_width - Inches(0.6)
    height = prs.slide_height - Inches(1.6)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # Header row
    table.cell(0, 0).text = df.index.name or ""
    for j, col_name in enumerate(df.columns, start=1):
        table.cell(0, j).text = str(col_name)

    # Body
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        table.cell(i, 0).text = str(idx)
        for j, col_name in enumerate(df.columns, start=1):
            val = row[col_name]

            # Text formatting
            if fmt and col_name in fmt:
                try:
                    text = fmt[col_name].format(val)
                except Exception:
                    text = "" if (pd.isna(val) or val is None) else str(val)
            else:
                text = "" if (pd.isna(val) or val is None) else str(val)

            cell = table.cell(i, j)
            cell.text = text

            # Cell coloring
            if cell_style_fn is not None:
                bg_hex, txt_hex = cell_style_fn(row, col_name, val)

                if bg_hex:
                    rgb = hex_to_rgb(bg_hex)
                    if rgb:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(*rgb)

                if txt_hex:
                    rgb = hex_to_rgb(txt_hex)
                    if rgb and cell.text_frame.paragraphs and cell.text_frame.paragraphs[0].runs:
                        run = cell.text_frame.paragraphs[0].runs[0]
                        run.font.color.rgb = RGBColor(*rgb)

# -------------------------------------------------------------------
# Set "SUB_TEAM" as the index for all DataFrames
# -------------------------------------------------------------------
def set_index(df, index_col="SUB_TEAM"):
    if index_col in df.columns:
        df = df.copy()
        df.set_index(index_col, inplace=True)
    return df

# -------------------------------------------------------------------
# Build the PowerPoint
# -------------------------------------------------------------------
outdir = "output"
os.makedirs(outdir, exist_ok=True)
pptx_path = os.path.join(outdir, "evms_tables.pptx")
prs = Presentation()

# -------------------------------------------------------------------
# Cost Performance (CPI)
# -------------------------------------------------------------------
if "cost_performance_tbl" in globals():
    cost_performance_tbl = set_index(cost_performance_tbl)

    # pandas styler (for notebook display only)
    sty = (
        cost_performance_tbl
        .style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )
    display(sty)

    def cp_style(row, col_name, value):
        if col_name in ("CTD", "YTD"):
            bg, txt = parse_css(color_spi_cpi_exact(value))
            return bg, txt
        return None, None

    add_table_slide(
        prs,
        cost_performance_tbl,
        title="Cost Performance (CPI)",
        fmt={"CTD": "{:.2f}", "YTD": "{:.2f}"},
        cell_style_fn=cp_style,
    )

# -------------------------------------------------------------------
# Schedule Performance (SPI)
# -------------------------------------------------------------------
if "schedule_performance_tbl" in globals():
    schedule_performance_tbl = set_index(schedule_performance_tbl)

    sty = (
        schedule_performance_tbl
        .style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )
    display(sty)

    def sp_style(row, col_name, value):
        if col_name in ("CTD", "YTD"):
            bg, txt = parse_css(color_spi_cpi_exact(value))
            return bg, txt
        return None, None

    add_table_slide(
        prs,
        schedule_performance_tbl,
        title="Schedule Performance (SPI)",
        fmt={"CTD": "{:.2f}", "YTD": "{:.2f}"},
        cell_style_fn=sp_style,
    )

# -------------------------------------------------------------------
# EVMS Metrics (SPI/CPI rows)
# -------------------------------------------------------------------
if "evms_metrics_tbl" in globals():
    evms_metrics_tbl = set_index(evms_metrics_tbl)
    cols = list(evms_metrics_tbl.columns)

    sty = (
        evms_metrics_tbl
        .style
        .format({c: "{:.2f}" for c in cols})
        .map(color_spi_cpi_exact, subset=cols)
    )
    display(sty)

    def evms_style(row, col_name, value):
        bg, txt = parse_css(color_spi_cpi_exact(value))
        return bg, txt

    add_table_slide(
        prs,
        evms_metrics_tbl,
        title="EVMS Metrics (SPI/CPI)",
        fmt={c: "{:.2f}" for c in cols},
        cell_style_fn=evms_style,
    )

# -------------------------------------------------------------------
# Helpers used by labor tables
# -------------------------------------------------------------------
def find_col(df, startswith_text):
    key = startswith_text.upper()
    for c in df.columns:
        if str(c).strip().upper().startswith(key):
            return c
    return None

# -------------------------------------------------------------------
# Labor Hours table: color VAC by VAC/BAC
# -------------------------------------------------------------------
if "labor_tbl" in globals():
    labor_tbl = set_index(labor_tbl)
    vac_col = find_col(labor_tbl, "VAC")
    bac_col = find_col(labor_tbl, "BAC")

    if vac_col and bac_col:

        def labor_style(row, col_name, value):
            if col_name == vac_col:
                vac = pd.to_numeric(row[vac_col], errors="coerce")
                bac = pd.to_numeric(row[bac_col], errors="coerce")
                if pd.isna(vac) or pd.isna(bac) or bac == 0:
                    return None, None
                ratio = vac / bac
                bg, txt = parse_css(color_vacbac_exact(ratio))
                return bg, txt
            return None, None

        # optional: show styled DataFrame in notebook
        def vac_style_df(df_):
            css = pd.DataFrame("", index=df_.index, columns=df_.columns)
            vac = pd.to_numeric(df_[vac_col], errors="coerce")
            bac = pd.to_numeric(df_[bac_col], errors="coerce")
            ratio = vac / bac.replace(0, np.nan)
            css[vac_col] = ratio.apply(color_vacbac_exact).values
            return css

        sty = labor_tbl.style.apply(vac_style_df, axis=None)
        display(sty)

        add_table