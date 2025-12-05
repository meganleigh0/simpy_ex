# ============================================================
# EVMS Slide – EV Plot + CPI/SPI CTD & LSD Metrics (per program)
# Uses Cobra exports + GDLS accounting calendar thresholds
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
# CONFIG – update these paths/programs for your environment
# ------------------------------------------------------------

cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS"     : "data/Cobra-Abrams STS.xlsx",
    "XM30"           : "data/Cobra-XM30.xlsx",
}

THEME_PATH = "data/Theme.pptx"   # your PPT template (optional)
OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accounting calendar (screenshot – DETROIT area 2025)
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

# Color palette from the GDLS dashboard guideline
COLOR_BLUE_LIGHT = RGBColor(142, 180, 227)  # x >= 1.05
COLOR_GREEN      = RGBColor( 51, 153, 102)  # 1.05 > x >= 0.98
COLOR_YELLOW     = RGBColor(255, 255, 153)  # 0.98 > x >= 0.95
COLOR_RED        = RGBColor(192,  80,  77)  # x < 0.95

# ------------------------------------------------------------
# Helpers: EVMS calculations
# ------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
        for c in df.columns
    })

def map_cost_sets(cost_cols):
    """
    Map Cobra COST-SET column labels to BCWS / BCWP / ACWP using fuzzy text.
    """
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

def compute_ev_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time series with Monthly & Cumulative CPI/SPI.

    IMPORTANT FIX:
      - Use raw hours for cumulative sums (zeros allowed)
      - Only treat zero as NaN when it's the *denominator* for a ratio
        so CTD values don't go NaN just because the current month has 0 hours.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE", columns="COSTSET", values="HOURS", aggfunc="sum"
    ).reset_index()

    cost_cols = [c for c in pivot.columns if c != "DATE"]
    bcws_col, bcwp_col, acwp_col = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing cost sets {missing} in COSTSET values: {cost_cols}")

    BCWS_raw = pivot[bcws_col].fillna(0.0)
    BCWP_raw = pivot[bcwp_col].fillna(0.0)
    ACWP_raw = pivot[acwp_col].fillna(0.0)

    # Monthly indices (treat zero denominator as NaN)
    monthly_cpi = BCWP_raw / ACWP_raw.replace(0, np.nan)
    monthly_spi = BCWP_raw / BCWS_raw.replace(0, np.nan)

    # Cumulative indices: cumulative sums first, then protect against /0
    cum_bcws = BCWS_raw.cumsum()
    cum_bcwp = BCWP_raw.cumsum()
    cum_acwp = ACWP_raw.cumsum()

    cumulative_cpi = cum_bcwp / cum_acwp.replace(0, np.nan)
    cumulative_spi = cum_bcwp / cum_bcws.replace(0, np.nan)

    out = pd.DataFrame({
        "DATE": pivot["DATE"],
        "Monthly CPI": monthly_cpi,
        "Monthly SPI": monthly_spi,
        "Cumulative CPI": cumulative_cpi,
        "Cumulative SPI": cumulative_spi,
    })
    return out

def get_status_dates(dates: pd.Series):
    """
    Get current and last status-period dates using accounting closings.
    Fallback: last two dates in the EV series.
    """
    dates = pd.to_datetime(dates)
    max_date = dates.max()

    closing_dates = []
    for (year, month), day in ACCOUNTING_CLOSINGS.items():
        d = datetime(year, month, day)
        if d <= max_date:
            closing_dates.append(d)

    closing_dates = sorted(closing_dates)

    if len(closing_dates) >= 2:
        curr = closing_dates[-1]
        prev = closing_dates[-2]
    elif len(closing_dates) == 1:
        curr = prev = closing_dates[0]
    else:
        uniq = sorted(dates.unique())
        curr = uniq[-1]
        prev = uniq[-2] if len(uniq) > 1 else uniq[-1]

    return curr, prev

def get_row_on_or_before(evdf: pd.DataFrame, date: datetime):
    """Last EV row on or before a given date."""
    sub = evdf[evdf["DATE"] <= date]
    if sub.empty:
        return evdf.iloc[0]
    return sub.iloc[-1]

def build_evms_metrics(evdf: pd.DataFrame):
    """
    Build 2x3 metrics table:
        Metric | CTD | LSD
        CPI    | ..  | ..
        SPI    | ..  | ..
    Both CTD and LSD are 'to date' (cumulative) using accounting status periods.
    """
    curr_date, prev_date = get_status_dates(evdf["DATE"])

    row_curr = get_row_on_or_before(evdf, curr_date)
    row_prev = get_row_on_or_before(evdf, prev_date)

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

def spi_cpi_color(x: float):
    """
    Apply SPI/CPI/BEI thresholds:
        Blue   x >= 1.05
        Green  1.05 > x >= 0.98
        Yellow 0.98 > x >= 0.95
        Red    x < 0.95
    """
    if pd.isna(x):
        return None
    if x >= 1.05:
        return COLOR_BLUE_LIGHT
    elif x >= 0.98:
        return COLOR_GREEN
    elif x >= 0.95:
        return COLOR_YELLOW
    else:
        return COLOR_RED

# ------------------------------------------------------------
# Helpers: Plot + PowerPoint
# ------------------------------------------------------------

def create_evms_plot(program: str, evdf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Background performance zones (same for CPI/SPI)
    fig.add_hrect(y0=0.90, y1=0.95, fillcolor="red",      opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.95, y1=0.98, fillcolor="yellow",   opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.98, y1=1.05, fillcolor="green",    opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.05, y1=1.20, fillcolor="lightblue",opacity=0.25, line_width=0)

    # Monthly indices (markers)
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

    # Cumulative indices (lines)
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

    # NOTE: leave chart title off – the slide title will carry it
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=400,
        margin=dict(l=60, r=20, t=20, b=60)
    )
    return fig

def find_blank_layout(prs: Presentation):
    """Try to find an actual 'Blank' layout; otherwise use first layout."""
    for layout in prs.slide_layouts:
        if "blank" in layout.name.lower():
            return layout
    return prs.slide_layouts[0]

def add_title(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3),
                                   Inches(9.0), Inches(0.6))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(24)

def add_evms_metrics_table(slide, df: pd.DataFrame,
                           left_in=0.6, top_in=4.9, width_in=8.8):
    rows, cols = df.shape
    shape = slide.shapes.add_table(
        rows + 1, cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(0.8 + 0.3 * rows)
    )
    table = shape.table

    # Header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(12)

    # Body with color coding for CTD/LSD numeric cells
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

            if col in ("CTD", "LSD"):
                rgb = spi_cpi_color(val)
                if rgb is not None:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb
                    p.font.color.rgb = RGBColor(0, 0, 0)

# ------------------------------------------------------------
# MAIN – build one-slide PPT per program
# ------------------------------------------------------------

for program, path in cobra_files.items():
    print(f"Processing {program} ...")

    cobra = pd.read_excel(path)
    cobra = normalize_columns(cobra)

    # Expect original columns: 'DATE', 'COST-SET', 'HOURS'
    required = {"DATE", "COSTSET", "HOURS"}
    missing = required - set(cobra.columns)
    if missing:
        raise ValueError(f"{program}: missing columns after normalization: {missing}")

    evdf = compute_ev_timeseries(cobra)
    metrics_df, curr_date, prev_date = build_evms_metrics(evdf)

    # Debug print so you can quickly confirm values
    print(f"{program} EVMS Metrics:")
    print(metrics_df)

    # Plot
    fig = create_evms_plot(program, evdf)
    img_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(img_path, scale=3)

    # PPT
    prs = Presentation(THEME_PATH) if os.path.exists(THEME_PATH) else Presentation()
    blank_layout = find_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)

    title_text = (f"{program} EVMS Trend "
                  f"(CTD as of {curr_date.date()}, "
                  f"LSD as of {prev_date.date()})")
    add_title(slide, title_text)

    # Chart
    slide.shapes.add_picture(img_path, Inches(0.6), Inches(1.1),
                             width=Inches(8.8))

    # Metrics table under the chart
    add_evms_metrics_table(slide, metrics_df,
                           left_in=0.6, top_in=4.9, width_in=8.8)

    out_pptx = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Slide.pptx")
    prs.save(out_pptx)
    print(f"   → Saved {out_pptx}")

print("ALL PROGRAM EVMS SLIDES COMPLETE ✓")