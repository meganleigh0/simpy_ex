# ---------------------------
# IMPORTS (Complete + Correct)
# ---------------------------
import pandas as pd
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


# ---------------------------------------------
# FUNCTION: Insert Pandas DataFrame Into a Slide
# ---------------------------------------------
def df_to_ppt_table(slide, df, title_text):
    """
    Add a pandas DataFrame as a real PowerPoint table on a slide.
    """

    # --- Set slide title ---
    title = slide.shapes.title
    title.text = title_text

    # --- Prepare DataFrame ---
    df_clean = df.copy()
    df_clean = df_clean.reset_index()

    rows, cols = df_clean.shape

    # --- Table position & size ---
    left = Inches(0.4)
    top = Inches(1.2)
    width = Inches(9.1)
    height = Inches(0.8 + rows * 0.3)

    table = slide.shapes.add_table(rows + 1, cols, left, top, width, height).table

    # --- Set header formatting ---
    for j, col_name in enumerate(df_clean.columns):
        cell = table.cell(0, j)
        cell.text = str(col_name)

        # Header style
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 230, 230)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(10)
        p.alignment = PP_ALIGN.CENTER

    # --- Fill table with data ---
    for i in range(rows):
        for j in range(cols):
            val = df_clean.iloc[i, j]
            cell = table.cell(i + 1, j)

            cell.text = "" if pd.isna(val) else str(val)

            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            p.alignment = PP_ALIGN.CENTER

    return table


# ----------------------------------------------------------
# FUNCTION: Create PowerPoint and Add All Existing DataFrames
# ----------------------------------------------------------
def save_all_tables_to_ppt(output="EVMS_Tables.pptx"):
    prs = Presentation()

    # Table names expected to be in globals() after your pipeline
    table_map = {
        "Cost Performance (CPI)": "cost_performance_tbl",
        "Schedule Performance (SPI)": "schedule_performance_tbl",
        "EVMS Metrics": "evms_metrics_tbl",
        "Labor Table": "labor_tbl",
        "Monthly Labor Table": "labor_monthly_tbl"
    }

    for title_text, var_name in table_map.items():
        if var_name in globals():

            df = globals()[var_name]

            # Create slide
            slide_layout = prs.slide_layouts[5]  # Title Only
            slide = prs.slides.add_slide(slide_layout)

            # Add table
            df_to_ppt_table(slide, df, title_text)

    # Save PowerPoint
    prs.save(output)
    print(f"[OK] PowerPoint saved → {output}")


# -----------------------------------------
# RUN THIS TO GENERATE THE POWERPOINT FILE
# -----------------------------------------
save_all_tables_to_ppt("Weekly_EVMS_Tables.pptx")
