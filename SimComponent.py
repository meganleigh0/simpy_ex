import os
import tempfile
import pandas as pd
from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor
from openpyxl import load_workbook

# -------------------------------------------------------------------
# Helper: convert hex (#RRGGBB) → RGB tuple
# -------------------------------------------------------------------
def hex_to_rgb(hex_color):
    if not isinstance(hex_color, str):
        return None
    h = hex_color.replace("#", "")
    if len(h) != 6:
        return None
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# -------------------------------------------------------------------
# Extract exact Excel colors from Styler output
# -------------------------------------------------------------------
def get_excel_colors(styler):
    """
    - Writes Styler to a temporary Excel file
    - Opens it with openpyxl
    - Returns a matrix of (bg_hex, font_hex)
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    styler.to_excel(tmp.name, engine="openpyxl", index=True)

    wb = load_workbook(tmp.name)
    ws = wb.active

    colors = []
    for row in ws.iter_rows():
        row_colors = []
        for cell in row:
            # background fill color
            bg = None
            if cell.fill and cell.fill.fgColor and cell.fill.fgColor.rgb:
                c = cell.fill.fgColor.rgb
                bg = f"#{c[2:]}"  # remove alpha

            # font color
            fg = None
            if cell.font and cell.font.color and cell.font.color.rgb:
                c = cell.font.color.rgb
                fg = f"#{c[2:]}"

            row_colors.append((bg, fg))
        colors.append(row_colors)
    return colors


# -------------------------------------------------------------------
# Add fully colored table to PowerPoint slide
# -------------------------------------------------------------------
def add_table_slide(prs, title, df, excel_colors):
    rows, cols = df.shape
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title

    left = Inches(0.3)
    top = Inches(1.2)
    width = Inches(9.2)
    height = Inches(0.8)

    table = slide.shapes.add_table(rows + 1, cols + 1, left, top, width, height).table

    # Write header (SUB_TEAM + columns)
    table.cell(0, 0).text = "SUB_TEAM"
    for j, col in enumerate(df.columns):
        table.cell(0, j+1).text = str(col)
        for run in table.cell(0, j+1).text_frame.paragraphs[0].runs:
            run.font.bold = True

    # Fill table with values + colors
    for i, idx in enumerate(df.index):
        # First column = SUB_TEAM
        table.cell(i+1, 0).text = str(idx)

        for j, col in enumerate(df.columns):
            value = df.loc[idx, col]
            value = "" if pd.isna(value) else str(value)
            cell = table.cell(i+1, j+1)
            cell.text = value

            bg_hex, fg_hex = excel_colors[i+1][j+1]

            # apply background
            if bg_hex:
                rgb = hex_to_rgb(bg_hex)
                if rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*rgb)

            # apply font color
            if fg_hex:
                rgb = hex_to_rgb(fg_hex)
                if rgb:
                    for p in cell.text_frame.paragraphs:
                        for run in p.runs:
                            run.font.color.rgb = RGBColor(*rgb)

    return prs


# -------------------------------------------------------------------
# MASTER FUNCTION: save all tables to PowerPoint
# -------------------------------------------------------------------
def save_all_tables_to_powerpoint():
    prs = Presentation()
    os.makedirs("output", exist_ok=True)

    tables = {
        "Cost Performance (CPI)": "cost_performance_tbl",
        "Schedule Performance (SPI)": "schedule_performance_tbl",
        "EVMS Metrics": "evms_metrics_tbl",
        "Labor Table": "labor_tbl",
        "Monthly Labor Table": "labor_monthly_tbl",
    }

    for title, varname in tables.items():
        if varname not in globals():
            print(f"[WARN] Missing table: {varname}")
            continue

        df = globals()[varname].copy()

        # enforce SUB_TEAM as index
        if "SUB_TEAM" in df.columns:
            df = df.set_index("SUB_TEAM")

        # GET EXACT COLORS FROM STYLER
        sty = df.style
        excel_colors = get_excel_colors(sty)

        prs = add_table_slide(prs, title, df, excel_colors)

    out = "output/Weekly_EVMS_Tables.pptx"
    prs.save(out)
    print(f"\n✅ PowerPoint saved at: {out}")


# -------------------------------------------------------------------
# RUN IT
# -------------------------------------------------------------------
save_all_tables_to_powerpoint()