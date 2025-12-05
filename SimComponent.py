# ============================================================
# EVMS Chart + CPI/SPI CTD vs LSD Metrics Slide (per program)
# - Uses Cobra exports
# - Uses accounting calendar to define status periods
# - Colors SPI/CPI cells by GDLS threshold key
# ============================================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS"     : "data/Cobra-Abrams STS.xlsx",
    "XM30"           : "data/Cobra-XM30.xlsx",
}

THEME_PATH = "data/Theme.pptx"       # your PPT theme (optional)
OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accounting calendar: 2025 GDLS accounting period closings
# (read from your screenshot). If your Cobra data is a different year,
# this map will simply be ignored and the code will fall back to the
# last two dates in the Cobra file.
ACCOUNTING_CLOSINGS = {
    (2025, 1): 26,
    (2025, 2): 23,
    (2025, 3): 30,
    (2025, 4): 27,
    (2025, 5): 25,
    (2025, 6): 29,
    (2025, 7): 27,
    (2025, 8): 24,
    (2025, 9): 28,
    (2025,10): 26,
    (2025,11): 23,
    (2025,12): 31,
}

# Color palette from GDLS Threshold Key
COLOR_BLUE_LIGHT = RGBColor(142, 180, 227)  # best (>=1.05)
COLOR_GREEN      = RGBColor(51, 153, 102)
COLOR_YELLOW     = RGBColor(255, 255, 153)
COLOR_RED        = RGBColor(192, 80, 77)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match EVMS logic."""
    return df.rename(columns={
        c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
        for c in df.columns
    })

def map_cost_sets(cost_cols):
    """Map Cobra COST-SET-like values to BCWS / BCWP / ACWP."""
    cleaned = {
        col: col.replace(" ", "").replace("-", "").replace("_", "").upper()
        for col in cost_cols
    }
    bcws = bcwp = acwp = None
    for orig, clean in cleaned.items():
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig
        elif ("BCWP" in clean) or ("EARNED" in clean) or ("PROGRESS" in clean):
            bcwp = orig
    return bcws, bcwp, acwp

def compute_ev_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build EVMS CPI/SPI time series at the program level.
    Expects normalized columns: DATE, COSTSET, HOURS.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE",
        columns="COSTSET",
        values="HOURS",
        aggfunc="sum"
    ).reset_index()

    cost_cols = [c for c in pivot.columns if c != "DATE"]
    bcws_col, bcwp_col, acwp_col = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing cost sets {missing}. Found: {cost_cols}")

    BCWS = pivot[bcws_col].replace(0, np.nan)
    BCWP = pivot[bcwp_col].replace(0, np.nan)
    ACWP = pivot[acwp_col].replace(0, np.nan)

    out = pd.DataFrame({"DATE": pivot["DATE"]})

    # Monthly indices
    out["Monthly CPI"] = BCWP / ACWP
    out["Monthly SPI"] = BCWP / BCWS

    # Cumulative indices
    out["Cumulative CPI"] = BCWP.cumsum() / ACWP.cumsum()
    out["Cumulative SPI"] = BCWP.cumsum() / BCWS.cumsum()

    return out

def get_status_dates(dates: pd.Series):
    """
    Use accounting calendar to get:
      - current status date (latest closing before or on max Cobra date)
      - previous status date (prior closing)
    Fallback: last two dates in the data if the accounting map doesn't apply.
    """
    dates = pd.to_datetime(dates)
    max_date = dates.max()

    # All known closing dates that are <= max Cobra date
    closing_dates = []
    for (year, month), day in ACCOUNTING_CLOSINGS.items():
        d = datetime(year, month, day)
        if d <= max_date:
            closing_dates.append(d)
    closing_dates = sorted(closing_dates)

    if len(closing_dates) >= 2:
        curr_close = closing_dates[-1]
        prev_close = closing_dates[-2]
    elif len(closing_dates) == 1:
        curr_close = closing_dates[0]
        prev_close = curr_close
    else:
        # Fallback: use last two actual EVMS dates
        uniq = sorted(dates.unique())
        curr_close = uniq[-1]
        prev_close = uniq[-2] if len(uniq) > 1 else uniq[-1]

    return curr_close, prev_close

def get_row_for_status(evdf: pd.DataFrame, target_date: datetime):
    """Return the last EV row on or before the target_date."""
    sub = evdf[evdf["DATE"] <= target_date]
    if sub.empty:
        # Fallback if nothing <= target_date – just use first row
        return evdf.iloc[0]
    return sub.iloc[-1]

def build_evms_metrics_table(evdf: pd.DataFrame) -> pd.DataFrame:
    """
    Build the EVMS metrics table with CPI/SPI for:
      - CTD (current status period)
      - LSD (last status period)
    Using cumulative CPI/SPI up to each status date.
    """
    curr_date, prev_date = get_status_dates(evdf["DATE"])

    row_curr = get_row_for_status(evdf, curr_date)
    row_prev = get_row_for_status(evdf, prev_date)

    cpi_ctd = row_curr["Cumulative CPI"]
    spi_ctd = row_curr["Cumulative SPI"]
    cpi_lsd = row_prev["Cumulative CPI"]
    spi_lsd = row_prev["Cumulative SPI"]

    metrics = pd.DataFrame({
        "Metric": ["CPI", "SPI"],
        "CTD":    [cpi_ctd, spi_ctd],
        "LSD":    [cpi_lsd, spi_lsd],
    })

    return metrics, curr_date, prev_date

def spi_cpi_color(val: float):
    """
    Color by SPI/CPI/BEI thresholds (from your Threshold Key):
      Blue   >= 1.05
      Green  1.05 > X >= 0.98
      Yellow 0.98 > X >= 0.95
      Red    < 0.95
    """
    if pd.isna(val):
        return None
    if val >= 1.05:
        return COLOR_BLUE_LIGHT
    elif val >= 0.98:
        return COLOR_GREEN
    elif val >= 0.95:
        return COLOR_YELLOW
    else:
        return COLOR_RED

def create_evms_plot(program: str, evdf: pd.DataFrame) -> go.Figure:
    """EVMS plot (Monthly + Cumulative CPI/SPI) with performance bands."""
    fig = go.Figure()

    # background bands
    fig.add_hrect(y0=0.90, y1=0.97, fillcolor="red",      opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.97, y1=1.00, fillcolor="yellow",   opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.00, y1=1.07, fillcolor="green",    opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.07, y1=1.20, fillcolor="lightblue",opacity=0.25, line_width=0)

    # monthly indices
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Monthly CPI"],
        mode="markers", name="Monthly CPI",
        marker=dict(symbol="diamond", size=10, color="yellow")
    ))
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Monthly SPI"],
        mode="markers", name="Monthly SPI",
        marker=dict(size=8, color="black")
    ))

    # cumulative indices
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Cumulative CPI"],
        mode="lines", name="Cumulative CPI",
        line=dict(color="blue", width=4)
    ))
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Cumulative SPI"],
        mode="lines", name="Cumulative SPI",
        line=dict(color="darkgrey", width=4)
    ))

    fig.update_layout(
        title=f"{program} EVMS Trend",
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=600,
    )

    return fig

def add_title_box(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(24)

def add_evms_metrics_table(slide, df: pd.DataFrame,
                           left=0.8, top=5.2, width=8.0):
    """Add CPI/SPI CTD/LSD table with threshold-based cell colors."""
    rows, cols = df.shape  # rows=2, cols=3
    shape = slide.shapes.add_table(
        rows + 1, cols,
        Inches(left), Inches(top),
        Inches(width), Inches(0.8 + 0.3 * rows)
    )
    table = shape.table

    # Header row
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(12)

    # Body
    for i in range(rows):
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)

            if isinstance(val, (float, np.floating)):
                cell.text = "" if pd.isna(val) else f"{val:.3f}"
            else:
                cell.text = "" if pd.isna(val) else str(val)

            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)

            # Color only numeric CPI/SPI cells (CTD & LSD columns)
            if col in ("CTD", "LSD"):
                rgb = spi_cpi_color(val)
                if rgb is not None:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb
                    p.font.color.rgb = RGBColor(0, 0, 0)

# ------------------------------------------------------------
# MAIN: one-slide deck per program
# ------------------------------------------------------------

for program, path in cobra_files.items():
    print(f"Processing {program} …")

    # Load Cobra export and normalize
    df = pd.read_excel(path)
    df = normalize_columns(df)

    # Expect these columns (original 'COST-SET', 'DATE', 'HOURS')
    required = {"DATE", "COSTSET", "HOURS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{program}: missing columns {missing} after normalization.")

    # EV time series
    evdf = compute_ev_time_series(df)

    # Metrics table (CPI/SPI CTD & LSD) using accounting calendar
    metrics_df, curr_date, prev_date = build_evms_metrics_table(evdf)

    # EVMS figure
    fig = create_evms_plot(program, evdf)
    img_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(img_path, scale=3)

    # PPT
    prs = Presentation(THEME_PATH) if os.path.exists(THEME_PATH) else Presentation()
    try:
        blank = prs.slide_layouts[6]
    except IndexError:
        blank = prs.slide_layouts[1]

    slide = prs.slides.add_slide(blank)
    add_title_box(
        slide,
        f"{program} – EVMS Trend "
        f"(CTD as of {curr_date.date()}, LSD as of {prev_date.date()})"
    )

    # Chart at top
    slide.shapes.add_picture(
        img_path,
        Inches(0.5), Inches(1.0),
        width=Inches(9.0)
    )

    # Metrics table below
    add_evms_metrics_table(slide, metrics_df, left=0.5, top=5.3, width=9.0)

    out_pptx = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Slide.pptx")
    prs.save(out_pptx)
    print(f"   → Saved {out_pptx}")

print("ALL PROGRAM EVMS METRIC SLIDES COMPLETE ✓")