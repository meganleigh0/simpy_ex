import os
import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.util import Inches
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


def add_df_slide(prs, title, df, fmt=None, cell_style=None):
    """
    Add a slide with a (slightly smaller) table.

    cell_style(row, col_name, value) -> (bg_hex, text_hex)
    """
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = title

    rows = df.shape[0] + 1        # +1 for header
    cols = df.shape[1] + 1        # +1 for index col

    # slightly smaller than full slide so it’s easy to see
    left = Inches(1.0)
    top = Inches(1.6)
    width = prs.slide_width - Inches(2.0)
    height = prs.slide_height - Inches(3.0)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # header row
    table.cell(0, 0).text = df.index.name or "SUB_TEAM"
    for j, col in enumerate(df.columns, start=1):
        table.cell(0, j).text = str(col)

    # data rows
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        table.cell(i, 0).text = str(idx)
        for j, col_name in enumerate(df.columns, start=1):
            val = row[col_name]

            # text formatting
            if fmt and col_name in fmt:
                try:
                    text = fmt[col_name].format(val)
                except Exception:
                    text = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
            else:
                text = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)

            cell = table.cell(i, j)
            cell.text = text

            # coloring
            if cell_style is not None:
                bg_hex, txt_hex = cell_style(row, col_name, val)

                if bg_hex:
                    rgb = hex_to_rgb(bg_hex)
                    if rgb:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(*rgb)

                if txt_hex and cell.text_frame.paragraphs:
                    for p in cell.text_frame.paragraphs:
                        for run in p.runs:
                            rgb_txt = hex_to_rgb(txt_hex)
                            if rgb_txt:
                                run.font.color.rgb = RGBColor(*rgb_txt)


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


# ========= Build the PowerPoint ==============================================

outdir = "output"
os.makedirs(outdir, exist_ok=True)
ppt_path = os.path.join(outdir, "evms_tables.pptx")
prs = Presentation()

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