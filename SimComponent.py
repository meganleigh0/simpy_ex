from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

def df_to_ppt_table(slide, df, title_text):
    """Add a DataFrame as a formatted table onto a PowerPoint slide."""
    # --- Title box ---
    title = slide.shapes.title
    title.text = title_text

    # --- Add table shape ---
    rows, cols = df.shape
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(9)
    height = Inches(0.8 + rows * 0.3)

    table = slide.shapes.add_table(rows + 1, cols, left, top, width, height).table

    # --- Set column headers ---
    for j, col_name in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col_name)

        # Header formatting
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 230, 230)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # --- Fill table data ---
    for i in range(rows):
        for j in range(cols):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)
            cell.text = "" if pd.isna(val) else str(val)

            # Center text
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    return table


def save_all_tables_to_ppt(output="styled_tables.pptx"):
    """
    Collect all known tables in globals(), detect which exist,
    extract their styled dataframes, and export to PowerPoint.
    """
    prs = Presentation()

    possible_tables = {
        "Cost Performance (CPI)": "cost_performance_tbl",
        "Schedule Performance (SPI)": "schedule_performance_tbl",
        "EVMS Metrics": "evms_metrics_tbl",
        "Labor Hours Table": "labor_tbl",
        "Monthly Labor Table": "labor_monthly_tbl"
    }

    for slide_title, tbl_name in possible_tables.items():
        if tbl_name in globals():

            df = globals()[tbl_name]

            # Remove index → becomes first column
            if isinstance(df.index, pd.MultiIndex):
                df_clean = df.reset_index()
            else:
                df_clean = df.reset_index()

            # --- Create new slide ---
            slide_layout = prs.slide_layouts[5]  # title + content
            slide = prs.slides.add_slide(slide_layout)

            # --- Add DataFrame to slide ---
            df_to_ppt_table(slide, df_clean, slide_title)

    prs.save(output)
    print(f"[OK] PowerPoint saved → {output}")


# --- RUN THIS TO GENERATE THE POWERPOINT ---
save_all_tables_to_ppt("Weekly_EVMS_Tables.pptx")