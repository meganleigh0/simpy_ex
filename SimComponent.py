import os
import numpy as np
import pandas as pd
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from IPython.display import display

# ========= Helper functions ===================================================

def hex_to_rgb(hex_color):
    """'#RRGGBB' -> (R, G, B) tuple."""
    if not hex_color:
        return None
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return None
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def parse_css(style_str):
    """Parse 'background-color:#xxxxxx;color:#yyyyyy' -> (bg_hex, text_hex)"""
    bg = txt = None
    if not style_str:
        return bg, txt

    for part in style_str.split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        if key == "background-color":
            bg = val
        elif key == "color":
            txt = val
    return bg, txt


def set_index(df, index_col="SUB_TEAM"):
    if index_col in df.columns:
        df = df.copy()
        df.set_index(index_col, inplace=True)
    return df


def add_comment_col(df, col_name="COMMENT"):
    df = df.copy()
    # Remove duplicates of "comment"
    for c in list(df.columns):
        if c.strip().upper() == "COMMENT" and c != col_name:
            df.drop(columns=[c], inplace=True)

    if col_name not in df.columns:
        df[col_name] = ""
    return df


def find_col(df, startswith_text):
    target = startswith_text.upper()
    for c in df.columns:
        if str(c).strip().upper().startswith(target):
            return c
    return None


def match_col(df, target):
    t = target.replace(" ", "").upper()
    for c in df.columns:
        if str(c).replace(" ", "").upper() == t:
            return c
    return None


# ------------------------------------------------------------------
# Add a metrics table slide (safe with theme layouts)
# ------------------------------------------------------------------
def add_df_slide(prs, title, df, fmt=None, cell_style=None, layout_index=5):

    layout_index = min(layout_index, len(prs.slide_layouts) - 1)
    slide = prs.slides.add_slide(prs.slide_layouts[layout_index])

    # --- TITLE HANDLING ---
    title_ph = None
    try:
        title_ph = slide.shapes.title
    except:
        title_ph = None

    if title_ph:
        title_ph.text = title
    else:
        # Add title text box manually
        tx = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3),
            prs.slide_width - Inches(1), Inches(0.6)
        )
        tf = tx.text_frame
        tf.text = title
        p = tf.paragraphs[0]
        p.font.size = Pt(32)
        p.font.bold = True

    # --- TABLE ---
    rows = df.shape[0] + 1
    cols = df.shape[1] + 1

    left = Inches(1.0)
    top = Inches(1.2)
    width = prs.slide_width - Inches(2.0)
    height = prs.slide_height - Inches(2.0)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # headers
    table.cell(0, 0).text = df.index.name or "SUB_TEAM"
    for j, col in enumerate(df.columns, start=1):
        table.cell(0, j).text = str(col)

    # body
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        table.cell(i, 0).text = str(idx)
        for j, col_name in enumerate(df.columns, start=1):
            val = row[col_name]

            # safe formatting
            if fmt and col_name in fmt:
                try:
                    txt = fmt[col_name].format(val)
                except:
                    txt = "" if pd.isna(val) else str(val)
            else:
                txt = "" if pd.isna(val) else str(val)

            cell = table.cell(i, j)
            cell.text = txt

            # colors
            if cell_style:
                bg_hex, txt_hex = cell_style(row, col_name, val)

                if bg_hex:
                    rgb = hex_to_rgb(bg_hex)
                    if rgb:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(*rgb)

                if txt_hex:
                    rgb = hex_to_rgb(txt_hex)
                    if rgb and cell.text_frame.paragraphs:
                        for p in cell.text_frame.paragraphs:
                            for r in p.runs:
                                r.font.color.rgb = RGBColor(*rgb)


# ------------------------------------------------------------------
# Create/Update title slide with TITLE + SUBTITLE
# ------------------------------------------------------------------
def set_title_slide(prs, title_text, subtitle_text):
    """Ensure a title slide with both title and subtitle filled."""

    if len(prs.slides) == 0:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
    else:
        slide = prs.slides[0]

    # ---- TITLE ----
    title_ph = None
    try:
        title_ph = slide.shapes.title
    except:
        title_ph = None

    if title_ph:
        title_ph.text = title_text
    else:
        tx = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3),
            prs.slide_width - Inches(1), Inches(0.8)
        )
        tf = tx.text_frame
        tf.text = title_text
        p = tf.paragraphs[0]
        p.font.size = Pt(40)
        p.font.bold = True

    # ---- SUBTITLE ----
    subtitle_ph = None
    try:
        subtitle_ph = slide.placeholders[1]
    except:
        subtitle_ph = None

    if subtitle_ph:
        subtitle_ph.text = subtitle_text
    else:
        tx = slide.shapes.add_textbox(
            Inches(0.5), Inches(1.0),
            prs.slide_width - Inches(1), Inches(0.7)
        )
        tf = tx.text_frame
        tf.text = subtitle_text
        p = tf.paragraphs[0]
        p.font.size = Pt(24)


# ========= Build the PowerPoint ==============================================

outdir = "output"
os.makedirs(outdir, exist_ok=True)
ppt_path = os.path.join(outdir, "evms_tables.pptx")

# --- load theme ---
theme_path = os.path.join("data", "theme.pptx")
if os.path.exists(theme_path):
    print(f"Using theme from: {theme_path}")
    prs = Presentation(theme_path)
else:
    print("Theme file not found. Using default PowerPoint template.")
    prs = Presentation()

# --- Title & Subtitle ---
title_text = "XM30 Weekly Metrics"
subtitle_text = datetime.today().strftime("%m/%d/%Y")
set_title_slide(prs, title_text, subtitle_text)

# ------------------------------------------------------------------
# COST PERFORMANCE (CPI)
# ------------------------------------------------------------------
if "cost_performance_tbl" in globals():
    df = add_comment_col(set_index(cost_performance_tbl))

    sty = df.style.format({"CTD": "{:.2f}", "YTD": "{:.2f}"}).map(color_spi_cpi_exact, subset=["CTD","YTD"])
    display(sty)

    def cpi_style(r, c, v):
        if c in ("CTD", "YTD"):
            return parse_css(color_spi_cpi_exact(v))
        return (None, None)

    fmt = {"CTD": "{:.2f}", "YTD": "{:.2f}"}
    add_df_slide(prs, "Cost Performance (CPI)", df, fmt, cpi_style)

# ------------------------------------------------------------------
# SCHEDULE PERFORMANCE (SPI)
# ------------------------------------------------------------------
if "schedule_performance_tbl" in globals():
    df = add_comment_col(set_index(schedule_performance_tbl))

    sty = df.style.format({"CTD": "{:.2f}", "YTD": "{:.2f}"}).map(color_spi_cpi_exact, subset=["CTD","YTD"])
    display(sty)

    def spi_style(r, c, v):
        if c in ("CTD", "YTD"):
            return parse_css(color_spi_cpi_exact(v))
        return (None, None)

    fmt = {"CTD": "{:.2f}", "YTD": "{:.2f}"}
    add_df_slide(prs, "Schedule Performance (SPI)", df, fmt, spi_style)

# ------------------------------------------------------------------
# EVMS METRICS
# ------------------------------------------------------------------
if "evms_metrics_tbl" in globals():
    df = add_comment_col(set_index(evms_metrics_tbl))
    num_cols = df.select_dtypes(include=["number"]).columns

    sty = df.style.format({c:"{:.2f}" for c in num_cols}).map(color_spi_cpi_exact, subset=num_cols)
    display(sty)

    def evms_style(r, c, v):
        if c in num_cols:
            return parse_css(color_spi_cpi_exact(v))
        return (None, None)

    fmt = {c:"{:.2f}" for c in num_cols}
    add_df_slide(prs, "EVMS Metrics (SPI/CPI)", df, fmt, evms_style)

# ------------------------------------------------------------------
# LABOR HOURS (VAC/BAC)
# ------------------------------------------------------------------
if "labor_tbl" in globals():
    df = add_comment_col(set_index(labor_tbl))
    vac_col = find_col(df, "VAC")
    bac_col = find_col(df, "BAC")

    if vac_col and bac_col:

        def labor_style(r, c, v):
            if c == vac_col:
                vac = pd.to_numeric(r[vac_col], errors="coerce")
                bac = pd.to_numeric(r[bac_col], errors="coerce")
                if pd.isna(vac) or pd.isna(bac) or bac == 0:
                    return (None, None)
                return parse_css(color_vacbac_exact(vac/bac))
            return (None, None)

        add_df_slide(prs, "Labor Hours (VAC/BAC)", df, fmt=None, cell_style=labor_style)
    else:
        print("[warn] Could not find VAC/BAC columns in labor_tbl")

# ------------------------------------------------------------------
# MONTHLY LABOR
# ------------------------------------------------------------------
if "labor_monthly_tbl" in globals():
    df = add_comment_col(set_index(labor_monthly_tbl))

    bac_eac_col = match_col(df, "BAC/EAC")
    vac_bac_col = match_col(df, "VAC/BAC")

    def dual_style(r, c, v):
        if c == bac_eac_col:
            return parse_css(color_spi_cpi_exact(v))
        if c == vac_bac_col:
            return parse_css(color_vacbac_exact(v))
        return (None, None)

    fmt = {}
    if bac_eac_col: fmt[bac_eac_col] = "{:.2f}"
    if vac_bac_col: fmt[vac_bac_col] = "{:.2f}"

    add_df_slide(prs, "Monthly Labor Table", df, fmt, dual_style)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
prs.save(ppt_path)
print(f"Saved PowerPoint to: {ppt_path}")