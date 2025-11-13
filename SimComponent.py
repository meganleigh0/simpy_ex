import os
import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor

# -----------------------------------------------------------
# Exact Palette
# -----------------------------------------------------------
HEX_COLORS = {
    "BLUE":   "#B4E3F6",
    "GREEN":  "#339966",
    "YELLOW": "#FFFF99",
    "RED":    "#CC5040",
}

# -----------------------------------------------------------
# Threshold Color Functions
# -----------------------------------------------------------
def get_color_style(value, thresholds, colors, text_colors):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        v = float(value)
    except:
        return ""
    for t, c, txt in zip(thresholds, colors, text_colors):
        if v >= t:
            return f"background-color:{c};color:{txt}"
    return f"background-color:{colors[-1]};color:{text_colors[-1]}"

def color_spi_cpi_exact(v):
    thresholds = [1.05, 1.02, 0.98, 0.95]
    colors     = [HEX_COLORS["BLUE"], HEX_COLORS["GREEN"], HEX_COLORS["YELLOW"], HEX_COLORS["RED"], HEX_COLORS["RED"]]
    text       = ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"]
    return get_color_style(v, thresholds, colors, text)

def color_vacbac_exact(v):
    thresholds = [0.05, 0.02, -0.02, -0.05]
    colors     = [HEX_COLORS["BLUE"], HEX_COLORS["GREEN"], HEX_COLORS["YELLOW"], HEX_COLORS["RED"], HEX_COLORS["RED"]]
    text       = ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"]
    return get_color_style(v, thresholds, colors, text)

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def hex_to_rgb(hex_color):
    if not hex_color:
        return None
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def parse_css(s):
    """Extract background/text from CSS string."""
    bg = txt = None
    if not s:
        return bg, txt
    parts = s.split(";")
    for p in parts:
        if ":" not in p:
            continue
        k,v = p.split(":",1)
        k=k.strip(); v=v.strip()
        if k=="background-color":
            bg=v
        elif k=="color":
            txt=v
    return bg, txt

# -----------------------------------------------------------
# PowerPoint Writer
# -----------------------------------------------------------
def add_table_slide(prs, df, title, fmt=None, style_fn=None):
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = title

    rows = df.shape[0] + 1
    cols = df.shape[1] + 1

    left = Inches(0.3)
    top  = Inches(1.0)
    width = prs.slide_width - Inches(0.6)
    height = prs.slide_height - Inches(1.4)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # header
    table.cell(0,0).text = df.index.name or ""
    for j,c in enumerate(df.columns, start=1):
        table.cell(0,j).text = str(c)

    # body
    for i,(idx,row) in enumerate(df.iterrows(), start=1):
        table.cell(i,0).text = str(idx)

        for j,c in enumerate(df.columns, start=1):
            val = row[c]

            # format text
            if fmt and c in fmt:
                try:
                    text = fmt[c].format(val)
                except:
                    text = "" if pd.isna(val) else str(val)
            else:
                text = "" if pd.isna(val) else str(val)

            cell = table.cell(i,j)
            cell.text = text

            if style_fn:
                bg_hex, txt_hex = style_fn(row,c,val)

                if bg_hex:
                    rgb = hex_to_rgb(bg_hex)
                    if rgb:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(*rgb)

                if txt_hex:
                    rgb = hex_to_rgb(txt_hex)
                    if rgb:
                        p = cell.text_frame.paragraphs[0]
                        r = p.runs[0]
                        r.font.color.rgb = RGBColor(*rgb)

# -----------------------------------------------------------
# Index Helper
# -----------------------------------------------------------
def set_index(df, col="SUB_TEAM"):
    if col in df.columns:
        df = df.copy()
        df.set_index(col, inplace=True)
    return df

# -----------------------------------------------------------
# Begin Output
# -----------------------------------------------------------
os.makedirs("output", exist_ok=True)
prs = Presentation()

# -----------------------------------------------------------
# Cost Performance
# -----------------------------------------------------------
if "cost_performance_tbl" in globals():
    df = set_index(cost_performance_tbl)

    def style_fn(row,c,val):
        if c in ("CTD","YTD"):
            return parse_css(color_spi_cpi_exact(val))
        return (None,None)

    add_table_slide(
        prs, df, "Cost Performance (CPI)",
        fmt={"CTD":"{:.2f}","YTD":"{:.2f}"},
        style_fn=style_fn
    )

# -----------------------------------------------------------
# Schedule Performance
# -----------------------------------------------------------
if "schedule_performance_tbl" in globals():
    df = set_index(schedule_performance_tbl)

    def style_fn(row,c,val):
        if c in ("CTD","YTD"):
            return parse_css(color_spi_cpi_exact(val))
        return (None,None)

    add_table_slide(
        prs, df, "Schedule Performance (SPI)",
        fmt={"CTD":"{:.2f}","YTD":"{:.2f}"},
        style_fn=style_fn
    )

# -----------------------------------------------------------
# EVMS Metrics (all cols SPI/CPI)
# -----------------------------------------------------------
if "evms_metrics_tbl" in globals():
    df = set_index(evms_metrics_tbl)
    cols = df.columns

    def style_fn(row,c,val):
        return parse_css(color_spi_cpi_exact(val))

    add_table_slide(
        prs, df, "EVMS Metrics",
        fmt={c:"{:.2f}" for c in cols},
        style_fn=style_fn
    )

# -----------------------------------------------------------
# Labor Table - VAC/BAC logic
# -----------------------------------------------------------
def find_col(df, key):
    key = key.upper()
    for c in df.columns:
        if str(c).upper().startswith(key):
            return c
    return None

if "labor_tbl" in globals():
    df = set_index(labor_tbl)
    vac_col = find_col(df,"VAC")
    bac_col = find_col(df,"BAC")

    if vac_col and bac_col:

        def style_fn(row,c,val):
            if c == vac_col:
                vac = pd.to_numeric(row[vac_col], errors="coerce")
                bac = pd.to_numeric(row[bac_col], errors="coerce")
                if pd.isna(vac) or pd.isna(bac) or bac==0:
                    return (None,None)
                ratio = vac / bac
                return parse_css(color_vacbac_exact(ratio))
            return (None,None)

        add_table_slide(
            prs, df, "Labor Hours (VAC/BAC)",
            fmt={vac_col:"{:.2f}", bac_col:"{:.2f}"},
            style_fn=style_fn
        )

# -----------------------------------------------------------
# Monthly Labor Table (colors BOTH columns)
# -----------------------------------------------------------
if "labor_monthly_tbl" in globals():
    df = set_index(labor_monthly_tbl)

    def match(df, text):
        return next((c for c in df.columns if text.upper() in str(c).upper()), None)

    bac_eac_col = match(df,"BAC/EAC")
    vac_bac_col = match(df,"VAC/BAC")

    def style_fn(row,c,val):
        if c == bac_eac_col:
            return parse_css(color_spi_cpi_exact(val))
        if c == vac_bac_col:
            return parse_css(color_vacbac_exact(val))
        return (None,None)

    fmt = {}
    if bac_eac_col: fmt[bac_eac_col] = "{:.2f}"
    if vac_bac_col: fmt[vac_bac_col] = "{:.2f}"

    add_table_slide(
        prs, df, "Monthly Labor Table",
        fmt=fmt,
        style_fn=style_fn
    )

# -----------------------------------------------------------
# Save PPTX
# -----------------------------------------------------------
pptx_path = "output/evms_styled_tables.pptx"
prs.save(pptx_path)
print("POWERPOINT CREATED →", pptx_path)