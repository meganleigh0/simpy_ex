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
    """
    Parse 'background-color:#xxxxxx;color:#yyyyyy'
    -> (bg_hex, text_hex)
    """
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


def set_index(df, index_col="SUB_TEAM"):
    """Ensure SUB_TEAM is the index if present."""
    if index_col in df.columns:
        df = df.copy()
        df.set_index(index_col, inplace=True)
    return df


def add_comment_col(df, col_name="COMMENT"):
    """Ensure exactly one COMMENT column exists at the far right."""
    df = df.copy()

    # remove any duplicate 'comment' columns with different casing
    for c in list(df.columns):
        if c.strip().upper() == "COMMENT" and c != col_name:
            df.drop(columns=[c], inplace=True)

    if col_name not in df.columns:
        df[col_name] = ""

    return df


def find_col(df, startswith_text):
    key = startswith_text.upper()
    for c in df.columns:
        if str(c).strip().upper().startswith(key):
            return c
    return None


def match_col(df, target):
    """Match full text, ignoring spaces & case (for 'BAC/EAC', 'VAC/BAC')."""
    t = target.replace(" ", "").upper()
    for c in df.columns:
        if str(c).replace(" ", "").upper() == t:
            return c
    return None


def add_df_slide(prs, title, df, fmt=None, cell_style=None, layout_index=5):
    """
    Add a slide with a (slightly smaller) table.
    Handles custom themes that may lack a title placeholder.
    """

    # pick a valid layout index
    layout_index = min(layout_index, len(prs.slide_layouts) - 1)
    slide = prs.slides.add_slide(prs.slide_layouts[layout_index])

    # ---- SAFE TITLE HANDLING ----
    title_placeholder = None
    try:
        title_placeholder = slide.shapes.title
    except Exception:
        title_placeholder = None

    if title_placeholder is not None:
        title_placeholder.text = title
    else:
        # create a manual title text box if theme layout has no title
        tx = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3),
            prs.slide_width - Inches(1.0), Inches(0.6)
        )
        tf = tx.text_frame
        tf.text = title
        p = tf.paragraphs[0]
        p.font.size = Pt(32)
        p.font.bold = True

    # ---- TABLE AREA ----
    rows = df.shape[0] + 1     # header
    cols = df.shape[1] + 1     # index column

    left = Inches(1.0)
    top = Inches(1.2)
    width = prs.slide_width - Inches(2.0)
    height = prs.slide_height - Inches(2.0)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # ---- HEADER ----
    table.cell(0, 0).text = df.index.name or "SUB_TEAM"
    for j, col in enumerate(df.columns, start=1):
        table.cell(0, j).text = str(col)

    # ---- BODY ----
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        table.cell(i, 0).text = str(idx)
        for j, col_name in enumerate(df.columns, start=1):
            val = row[col_name]

            # formatting (skip COMMENT / non-numeric safely)
            if fmt and col_name in fmt:
                try:
                    cell_text = fmt[col_name].format(val)
                except Exception:
                    cell_text = "" if pd.isna(val) else str(val)
            else:
                cell_text = "" if pd.isna(val) else str(val)

            cell = table.cell(i, j)
            cell.text = cell_text

            # conditional coloring
            if cell_style is not None:
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
                            for run in p.runs:
                                run.font.color.rgb = RGBColor(*rgb)


def set_title_slide(prs, title_text):
    """
    Ensure there is a title slide with the given text.
    If a slide already exists, use the first slide; otherwise, add one.
    """
    if len(prs.slides) > 0:
        slide = prs.slides[0]
    else:
        layout_index = 0
        layout_index = min(layout_index, len(prs.slide_layouts) - 1)
        slide = prs.slides.add_slide(prs.slide_layouts[layout_index])

    # Try to set the title placeholder; fallback to textbox.
    title_ph = None
    try:
        title_ph = slide.shapes.title
    except Exception:
        title_ph = None

    if title_ph is not None:
        title_ph.text = title_text
    else:
        tx = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3),
            prs.slide_width - Inches(1.0), Inches(0.8)
        )
        tf = tx.text_frame
        tf.text = title_text
        p = tf.paragraphs[0]
        p.font.size = Pt(36)
        p.font.bold = True


# ========= Build the PowerPoint ==============================================

outdir = "output"
os.makedirs(outdir, exist_ok=True)
ppt_path = os.path.join(outdir, "evms_tables.pptx")

# --- load theme presentation if available ---
theme_path = os.path.join("data", "theme.pptx")  # adjust if theme is elsewhere
if os.path.exists(theme_path):
    print(f"Using theme from: {theme_path}")
    prs = Presentation(theme_path)
else:
    print("Theme file not found, using default PowerPoint template.")
    prs = Presentation()

# --- create/update title slide with current date ---
today_str = datetime.today().strftime("%m/%d/%Y")
title_text = f"XM30 Weekly Metrics - {today_str}"
set_title_slide(prs, title_text)

# ----- Cost Performance (CPI) -----
if "cost_performance_tbl" in globals():
    df = set_index(cost_performance_tbl)
    df = add_comment_col(df)

    # Notebook display
    sty = (
        df.style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )
    display(sty)

    def cpi_style(row, col_name, value):
        if col_name in ("CTD", "YTD"):
            return parse_css(color_spi_cpi_exact(value))
        return (None, None)

    fmt = {"CTD": "{:.2f}", "YTD": "{:.2f}"}
    add_df_slide(prs, "Cost Performance (CPI)", df, fmt=fmt, cell_style=cpi_style)


# ----- Schedule Performance (SPI) -----
if "schedule_performance_tbl" in globals():
    df = set_index(schedule_performance_tbl)
    df = add_comment_col(df)

    sty = (
        df.style
        .format({"CTD": "{:.2f}", "YTD": "{:.2f}"})
        .map(color_spi_cpi_exact, subset=["CTD", "YTD"])
    )
    display(sty)

    def spi_style(row, col_name, value):
        if col_name in ("CTD", "YTD"):
            return parse_css(color_spi_cpi_exact(value))
        return (None, None)

    fmt = {"CTD": "{:.2f}", "YTD": "{:.2f}"}
    add_df_slide(prs, "Schedule Performance (SPI)", df, fmt=fmt, cell_style=spi_style)


# ----- EVMS Metrics (SPI/CPI rows) -----
if "evms_metrics_tbl" in globals():
    df = set_index(evms_metrics_tbl)
    df = add_comment_col(df)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    sty = (
        df.style
        .format({c: "{:.2f}" for c in numeric_cols})
        .map(color_spi_cpi_exact, subset=numeric_cols)
    )
    display(sty)

    def evms_style(row, col_name, value):
        if col_name in numeric_cols:
            return parse_css(color_spi_cpi_exact(value))
        return (None, None)

    fmt = {c: "{:.2f}" for c in numeric_cols}
    add_df_slide(prs, "EVMS Metrics (SPI/CPI)", df, fmt=fmt, cell_style=evms_style)


# ----- Labor Hours table: color VAC by VAC/BAC -----
if "labor_tbl" in globals():
    df = set_index(labor_tbl)
    df = add_comment_col(df)

    vac_col = find_col(df, "VAC")
    bac_col = find_col(df, "BAC")

    if vac_col and bac_col:

        def labor_style(row, col_name, value):
            if col_name == vac_col:
                vac = pd.to_numeric(row[vac_col], errors="coerce")
                bac = pd.to_numeric(row[bac_col], errors="coerce")
                if pd.isna(vac) or pd.isna(bac) or bac == 0:
                    return (None, None)
                ratio = vac / bac
                return parse_css(color_vacbac_exact(ratio))
            return (None, None)

        # Notebook display only – no numeric .format so COMMENT is safe
        def vac_style_df(df_):
            css = pd.DataFrame("", index=df_.index, columns=df_.columns)
            vac = pd.to_numeric(df_[vac_col], errors="coerce")
            bac = pd.to_numeric(df_[bac_col], errors="coerce")
            ratio = vac / bac.replace(0, np.nan)
            css[vac_col] = ratio.apply(color_vacbac_exact).values
            return css

        display(df.style.apply(vac_style_df, axis=None))

        add_df_slide(prs, "Labor Hours (VAC/BAC)", df, fmt=None, cell_style=labor_style)
    else:
        print("[warn] Could not find VAC/BAC columns in labor_tbl")


# ----- Monthly Labor table: BAC/EAC & VAC/BAC thresholds -----
if "labor_monthly_tbl" in globals():
    df = set_index(labor_monthly_tbl)
    df = add_comment_col(df)

    bac_eac_col = match_col(df, "BAC/EAC")
    vac_bac_col = match_col(df, "VAC/BAC")

    def dual_style(row, col_name, value):
        if bac_eac_col and col_name == bac_eac_col:
            return parse_css(color_spi_cpi_exact(value))
        if vac_bac_col and col_name == vac_bac_col:
            return parse_css(color_vacbac_exact(value))
        return (None, None)

    fmt = {}
    if bac_eac_col:
        fmt[bac_eac_col] = "{:.2f}"
    if vac_bac_col:
        fmt[vac_bac_col] = "{:.2f}"

    add_df_slide(prs, "Monthly Labor Table", df, fmt=fmt, cell_style=dual_style)


# ----- Save PowerPoint --------------------------------------------------------
prs.save(ppt_path)
print(f"Saved PowerPoint to: {ppt_path}")