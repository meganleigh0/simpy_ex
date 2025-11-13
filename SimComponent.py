# -------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------
import pandas as pd
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


# -------------------------------------------------------------
# HELPER: Convert HEX ("#FF0000") → RGB tuple
# -------------------------------------------------------------
def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))


# -------------------------------------------------------------
# APPLY YOUR COLOR FUNCTIONS DIRECTLY TO PPT CELLS
# -------------------------------------------------------------
def apply_threshold_to_cell(cell, value, mode):
    """
    mode = 'CPI_SPI', 'VACBAC', or 'DUAL'
    Applies your exact thresholds to a PowerPoint cell.
    """

    # Background + text colors returned by your functions
    try:
        if mode == "CPI_SPI":
            css = get_color_style(value, 
                                  [1.05, 1.02, 0.98, 0.95],
                                  [HEX_COLORS["BLUE"], HEX_COLORS["GREEN"], HEX_COLORS["YELLOW"], HEX_COLORS["RED"], HEX_COLORS["RED"]],
                                  ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"])

        elif mode == "VACBAC":
            css = get_color_style(value,
                                  [0.05, 0.02, -0.02, -0.05],
                                  [HEX_COLORS["BLUE"], HEX_COLORS["GREEN"], HEX_COLORS["YELLOW"], HEX_COLORS["RED"], HEX_COLORS["RED"]],
                                  ["#000000", "#000000", "#000000", "#FFFFFF", "#FFFFFF"])

        else:
            return  # no styling

    except:
        return

    if css is None:
        return

    # parse the returned CSS string
    # format: "background-color:#XXXXXX; color:#YYYYYY"
    if "background-color" in css:
        bg_hex = css.split("background-color:")[1].split(";")[0].strip()
        text_hex = css.split("color:")[1].strip()

        r, g, b = hex_to_rgb(bg_hex)
        tr, tg, tb = hex_to_rgb(text_hex)

        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(r, g, b)

        p = cell.text_frame.paragraphs[0]
        p.font.color.rgb = RGBColor(tr, tg, tb)


# -------------------------------------------------------------
# FUNCTION: Insert DataFrame into PPT with styling
# -------------------------------------------------------------
def df_to_ppt_table(slide, df, title_text, mode=None):

    df_clean = df.reset_index()

    title = slide.shapes.title
    title.text = title_text

    rows, cols = df_clean.shape

    left, top = Inches(0.4), Inches(1.2)
    width, height = Inches(9.1), Inches(0.8 + rows * 0.3)

    table = slide.shapes.add_table(rows + 1, cols, left, top, width, height).table

    # header
    for j in range(cols):
        cell = table.cell(0, j)
        cell.text = str(df_clean.columns[j])
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 230, 230)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # data
    for i in range(rows):
        for j in range(cols):
            val = df_clean.iloc[i, j]
            cell = table.cell(i+1, j)

            cell.text = "" if pd.isna(val) else str(val)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(10)

            # apply threshold if applicable
            if mode == "CPI_SPI" and df_clean.columns[j] in ["CTD", "YTD", "4WK"]:
                apply_threshold_to_cell(cell, val, "CPI_SPI")

            elif mode == "VACBAC" and df_clean.columns[j] in ["VAC/BAC", "VAC", "BAC"]:
                apply_threshold_to_cell(cell, val, "VACBAC")

    return table


# -------------------------------------------------------------
# MAIN FUNCTION: Add all your tables to PPT with styling
# -------------------------------------------------------------
def save_all_tables_to_ppt(output="Weekly_EVMS_Tables.pptx"):
    prs = Presentation()

    table_map = {
        "Cost Performance (CPI)": ("cost_performance_tbl", "CPI_SPI"),
        "Schedule Performance (SPI)": ("schedule_performance_tbl", "CPI_SPI"),
        "EVMS Metrics": ("evms_metrics_tbl", "CPI_SPI"),
        "Labor Table (VAC/BAC)": ("labor_tbl", "VACBAC"),
        "Monthly Labor Table": ("labor_monthly_tbl", "CPI_SPI")
    }

    for title_text, (var_name, mode) in table_map.items():
        if var_name in globals():

            df = globals()[var_name]

            slide = prs.slides.add_slide(prs.slide_layouts[5])
            df_to_ppt_table(slide, df, title_text, mode)

    prs.save(output)
    print(f"[OK] Styled PowerPoint saved → {output}")