# ============================================================
# EVMS Slide – Chart + CPI/SPI/BEI CTD & LSD Metrics + RCCA box
# ============================================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ---------------- CONFIG ----------------

cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS"     : "data/Cobra-Abrams STS.xlsx",
    "XM30"           : "data/Cobra-XM30.xlsx",
}

PENSKE_PATH = "data/OpenPlan_Activity-Penske.xlsx"
THEME_PATH  = "data/Theme.pptx"
OUTPUT_DIR  = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accounting calendar closings (2025 – Detroit area)
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

# Threshold colors
COLOR_BLUE_LIGHT = RGBColor(142, 180, 227)  # x >= 1.05
COLOR_GREEN      = RGBColor( 51, 153, 102)  # 1.05 > x >= 0.98
COLOR_YELLOW     = RGBColor(255, 255, 153)  # 0.98 > x >= 0.95
COLOR_RED        = RGBColor(192,  80,  77)  # x < 0.95

# ---------------- SHARED HELPERS ----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
        for c in df.columns
    })

def map_cost_sets(cost_cols):
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

def spi_cpi_color(x: float):
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

# ---------------- EVMS (Cobra) ----------------

def compute_ev_timeseries(df: pd.DataFrame) -> pd.DataFrame:
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

    monthly_cpi = BCWP_raw / ACWP_raw.replace(0, np.nan)
    monthly_spi = BCWP_raw / BCWS_raw.replace(0, np.nan)

    cum_bcws = BCWS_raw.cumsum()
    cum_bcwp = BCWP_raw.cumsum()
    cum_acwp = ACWP_raw.cumsum()

    cumulative_cpi = cum_bcwp / cum_acwp.replace(0, np.nan)
    cumulative_spi = cum_bcwp / cum_bcws.replace(0, np.nan)

    return pd.DataFrame({
        "DATE": pivot["DATE"],
        "Monthly CPI": monthly_cpi,
        "Monthly SPI": monthly_spi,
        "Cumulative CPI": cumulative_cpi,
        "Cumulative SPI": cumulative_spi,
    })

def get_status_dates(dates: pd.Series):
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
    sub = evdf[evdf["DATE"] <= date]
    if sub.empty:
        return evdf.iloc[0]
    return sub.iloc[-1]

# ---------------- BEI (OpenPlan Penske) ----------------

PENSKE = pd.read_excel(PENSKE_PATH)
PENSKE = normalize_columns(PENSKE)

def compute_bei_for_program(program_name: str,
                            status_date: datetime,
                            prev_status_date: datetime) -> tuple[float, float]:
    df = PENSKE.copy()

    # filter by PROGRAM if available
    if "PROGRAM" in df.columns:
        mask = df["PROGRAM"].astype(str).str.contains(program_name, case=False, na=False)
        if mask.any():
            df = df[mask]

    base_col = next((c for c in df.columns if "BASELINEFINISH" in c), None)
    act_col  = next((c for c in df.columns if "ACTUALFINISH"   in c), None)
    if base_col is None or act_col is None:
        return np.nan, np.nan

    df[base_col] = pd.to_datetime(df[base_col], errors="coerce")
    df[act_col]  = pd.to_datetime(df[act_col],  errors="coerce")

    lev_col = next((c for c in df.columns if "LEVTYPE" in c), None)
    if lev_col is not None:
        df = df[~df[lev_col].isin(["A", "B"])]

    def _bei(as_of):
        denom = df[df[base_col] <= as_of]
        if denom.empty:
            return np.nan
        numer = denom[denom[act_col].notna() & (denom[act_col] <= as_of)]
        return len(numer) / len(denom)

    return _bei(status_date), _bei(prev_status_date)

# ---------------- Plot & PPT helpers ----------------

def create_evms_plot(program: str, evdf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(y0=0.90, y1=0.95, fillcolor="red",      opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.95, y1=0.98, fillcolor="yellow",   opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.98, y1=1.05, fillcolor="green",    opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.05, y1=1.20, fillcolor="lightblue",opacity=0.25, line_width=0)

    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Monthly CPI"],
        mode="markers", name="Monthly CPI",
        marker=dict(symbol="diamond", size=7, color="yellow")
    ))
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Monthly SPI"],
        mode="markers", name="Monthly SPI",
        marker=dict(size=6, color="black")
    ))
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Cumulative CPI"],
        mode="lines", name="Cumulative CPI",
        line=dict(color="blue", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=evdf["DATE"], y=evdf["Cumulative SPI"],
        mode="lines", name="Cumulative SPI",
        line=dict(color="darkgrey", width=3)
    ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=360,
        margin=dict(l=60, r=20, t=20, b=50)
    )
    return fig

def get_blank_layout(prs: Presentation):
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
                           left_in, top_in, width_in):
    rows, cols = df.shape
    shape = slide.shapes.add_table(
        rows + 1, cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(0.8 + 0.3 * rows)
    )
    table = shape.table

    # header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(12)

    # body
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

def add_rcca_box(slide, left_in=0.5, top_in=4.9, width_in=9.0, height_in=1.6):
    """Root Cause / Corrective Actions textbox."""
    box = slide.shapes.add_textbox(
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(height_in)
    )
    tf = box.text_frame
    tf.text = "Root Cause / Corrective Actions:"
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(14)
    tf.add_paragraph()  # blank line for user text

# ---------------- MAIN: one slide per program ----------------

for program, path in cobra_files.items():
    print(f"Processing {program} ...")

    cobra = pd.read_excel(path)
    cobra = normalize_columns(cobra)

    required = {"DATE", "COSTSET", "HOURS"}
    missing = required - set(cobra.columns)
    if missing:
        raise ValueError(f"{program}: missing columns {missing} after normalization.")

    evdf = compute_ev_timeseries(cobra)
    curr_date, prev_date = get_status_dates(evdf["DATE"])

    row_curr = get_row_on_or_before(evdf, curr_date)
    row_prev = get_row_on_or_before(evdf, prev_date)
    cpi_ctd = row_curr["Cumulative CPI"]
    spi_ctd = row_curr["Cumulative SPI"]
    cpi_lsd = row_prev["Cumulative CPI"]
    spi_lsd = row_prev["Cumulative SPI"]

    bei_ctd, bei_lsd = compute_bei_for_program(program, curr_date, prev_date)

    metrics_df = pd.DataFrame({
        "Metric": ["CPI", "SPI", "BEI"],
        "CTD":    [cpi_ctd, spi_ctd, bei_ctd],
        "LSD":    [cpi_lsd, spi_lsd, bei_lsd],
    })

    # Print dates (for your notes) but do NOT put in PPT
    print(f"  CTD date: {curr_date.date()}, LSD date: {prev_date.date()}")
    print(metrics_df, "\n")

    # EVMS plot
    fig = create_evms_plot(program, evdf)
    img_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(img_path, scale=3)

    # PowerPoint slide
    prs = Presentation(THEME_PATH) if os.path.exists(THEME_PATH) else Presentation()
    blank_layout = get_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)

    # Title WITHOUT dates
    add_title(slide, f"{program} EVMS Trend Overview")

    # Layout: chart on left, table on right, RCCA box across bottom
    # Chart size tuned so X-axis is visible and leaves room below
    slide.shapes.add_picture(
        img_path,
        Inches(0.5),  # left
        Inches(1.0),  # top
        width=Inches(5.5)
    )

    # Metrics table on the right
    add_evms_metrics_table(
        slide,
        metrics_df,
        left_in=6.2,
        top_in=1.0,
        width_in=3.5,
    )

    # RCCA box along bottom (full width)
    add_rcca_box(slide, left_in=0.5, top_in=4.7, width_in=9.2, height_in=1.7)

    out_pptx = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Slide.pptx")
    prs.save(out_pptx)
    print(f"  → Saved: {out_pptx}")

print("ALL PROGRAM EVMS SLIDES COMPLETE ✓")