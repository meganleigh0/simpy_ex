# ============================================================
# EVMS Deck per Program
#   Slide 1: EVMS Trend Overview (chart + CPI/SPI/BEI metrics + RCCA)
#   Slide 2: Labor Hours Perf, Cost/Sched/BEI Perf, Program Manpower + RC Box
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

HOURS_PER_FTE = 173.33  # approx 9/80 monthly hours

# Threshold colors (from dashboard key)
COLOR_BLUE_LIGHT = RGBColor(142, 180, 227)  # best / highest
COLOR_GREEN      = RGBColor( 51, 153, 102)
COLOR_YELLOW     = RGBColor(255, 255, 153)
COLOR_RED        = RGBColor(192,  80,  77)

# ============================================================
# SHARED HELPERS
# ============================================================

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
    bcws = bcwp = acwp = etc = None
    for orig, clean in cleaned.items():
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig
        elif ("BCWP" in clean) or ("EARNED" in clean) or ("PROGRESS" in clean):
            bcwp = orig
        elif "ETC" in clean:
            etc = orig
    return bcws, bcwp, acwp, etc

def spi_cpi_color(x: float):
    """SPI / CPI / BEI thresholds."""
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

def vac_color_from_ratio(r: float):
    """VAC/BAC thresholds."""
    if pd.isna(r):
        return None
    # VAC/BAC thresholds: >= +0.05 blue; +0.05 > x >= -0.02 green;
    # -0.02 > x >= -0.05 yellow; < -0.05 red
    if r >= 0.05:
        return COLOR_BLUE_LIGHT
    elif r >= -0.02:
        return COLOR_GREEN
    elif r >= -0.05:
        return COLOR_YELLOW
    else:
        return COLOR_RED

def manpower_var_color(r: float):
    """Program manpower %Var thresholds based on Actual/Demand."""
    if pd.isna(r):
        return None
    # >=110% red; 110–105% yellow; 105–90% green; 90–85% yellow; <85% red
    if r >= 1.10:
        return COLOR_RED
    elif r >= 1.05:
        return COLOR_YELLOW
    elif r >= 0.90:
        return COLOR_GREEN
    elif r >= 0.85:
        return COLOR_YELLOW
    else:
        return COLOR_RED

def get_status_dates(dates: pd.Series):
    """Status dates from accounting calendar; fallback to last two EV dates."""
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

# ============================================================
# EVMS FROM COBRA
# ============================================================

def compute_ev_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE", columns="COSTSET", values="HOURS", aggfunc="sum"
    ).reset_index()

    cost_cols = [c for c in pivot.columns if c != "DATE"]
    bcws_col, bcwp_col, acwp_col, _ = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing cost sets {missing}; found {cost_cols}")

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

# ============================================================
# BEI FROM OPENPLAN PENSKE
# ============================================================

PENSKE = pd.read_excel(PENSKE_PATH)
PENSKE = normalize_columns(PENSKE)

def compute_bei_for_program(program_name: str,
                            status_date: datetime,
                            prev_status_date: datetime) -> tuple[float, float]:
    df = PENSKE.copy()

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

# ============================================================
# TABLE BUILDERS FOR SLIDE 2
# ============================================================

def build_labor_hours_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labor Hours Performance by Sub Team:
        %COMP = BCWP / BAC
        BAC   = BCWS
        EAC   = ACWP + ETC (if ETC cost-set exists, else ACWP)   [assumption]
        VAC   = BAC - EAC
    """
    if "SUBTEAM" not in df.columns:
        raise ValueError("Labor table requires SUBTEAM column.")

    pivot = df.pivot_table(
        index="SUBTEAM", columns="COSTSET", values="HOURS", aggfunc="sum"
    ).fillna(0)

    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(pivot.columns)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing cost sets for labor table: {missing}")

    BAC = pivot[bcws_col]
    BCWP = pivot[bcwp_col]
    ACWP = pivot[acwp_col]
    ETC  = pivot[etc_col] if etc_col in pivot.columns else 0.0

    EAC = ACWP + ETC
    VAC = BAC - EAC
    pct_comp = np.where(BAC != 0, BCWP / BAC, np.nan)

    out = pd.DataFrame({
        "Sub Team": pivot.index,
        "%COMP": pct_comp,
        "BAC": BAC,
        "EAC": EAC,
        "VAC": VAC,
    }).reset_index(drop=True)

    return out

def build_manpower_table(df: pd.DataFrame,
                         curr_date: datetime,
                         prev_date: datetime) -> pd.DataFrame:
    """Program Manpower: Demand, Actual, %Var, Last Mo, Next Mo (FTEs)."""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index=df["DATE"].dt.to_period("M"),
        columns="COSTSET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0)

    bcws_col, _, acwp_col, _ = map_cost_sets(pivot.columns)
    if bcws_col is None or acwp_col is None or pivot.empty:
        return pd.DataFrame(columns=["Demand", "Actual", "% Var", "Last Mo", "Next Mo"])

    demand_hrs = pivot[bcws_col]
    actual_hrs = pivot[acwp_col]

    demand_fte = demand_hrs / HOURS_PER_FTE
    actual_fte = actual_hrs / HOURS_PER_FTE

    status_period = curr_date.to_period("M")
    last_period   = prev_date.to_period("M")
    next_period   = status_period + 1

    demand = demand_fte.get(status_period, np.nan)
    actual = actual_fte.get(status_period, np.nan)
    last   = demand_fte.get(last_period,   np.nan)
    nxt    = demand_fte.get(next_period,   np.nan)

    pct_var = (actual / demand) if (demand and not pd.isna(demand) and demand != 0) else np.nan

    return pd.DataFrame({
        "Row": ["Program Manpower"],
        "Demand": [demand],
        "Actual": [actual],
        "% Var": [pct_var],
        "Last Mo": [last],
        "Next Mo": [nxt],
    })

# ============================================================
# PLOTTING & PPT HELPERS
# ============================================================

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

def add_title(slide, text, left=0.5, top=0.3):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(9.0), Inches(0.6))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(24)

def add_simple_table(slide, df: pd.DataFrame,
                     left_in, top_in, width_in) -> object:
    """Create a table for a DataFrame; return pptx table object."""
    rows, cols = df.shape
    shape = slide.shapes.add_table(
        rows + 1, cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(0.6 + 0.28 * rows)
    )
    table = shape.table

    # header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(11)

    # body
    for i in range(rows):
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)

            if isinstance(val, (float, np.floating)):
                # % columns display as percent
                if "%COMP" in col.upper() or "VAR" in col.upper():
                    cell.text = "" if pd.isna(val) else f"{val:.1%}"
                else:
                    cell.text = "" if pd.isna(val) else f"{val:,.1f}"
            else:
                cell.text = "" if pd.isna(val) else str(val)

            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)

    return table

def add_rcca_box(slide, label="Root Cause / Corrective Actions",
                 left_in=0.5, top_in=4.7, width_in=9.0, height_in=1.6):
    box = slide.shapes.add_textbox(
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(height_in)
    )
    tf = box.text_frame
    tf.text = label + ":"
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(14)
    tf.add_paragraph()  # blank line for notes

# ============================================================
# MAIN LOOP – Build decks
# ============================================================

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

    # Program-level CPI/SPI CTD & LSD
    row_curr = get_row_on_or_before(evdf, curr_date)
    row_prev = get_row_on_or_before(evdf, prev_date)
    cpi_ctd = row_curr["Cumulative CPI"]
    spi_ctd = row_curr["Cumulative SPI"]
    cpi_lsd = row_prev["Cumulative CPI"]
    spi_lsd = row_prev["Cumulative SPI"]

    # BEI CTD/LSD
    bei_ctd, bei_lsd = compute_bei_for_program(program, curr_date, prev_date)

    metrics_df = pd.DataFrame({
        "Metric": ["CPI", "SPI", "BEI"],
        "CTD":    [cpi_ctd, spi_ctd, bei_ctd],
        "LSD":    [cpi_lsd, spi_lsd, bei_lsd],
    })

    # DEBUG info (not used on slides)
    print(f"  CTD date: {curr_date.date()}, LSD date: {prev_date.date()}")
    print(metrics_df, "\n")

    # EVMS plot
    fig = create_evms_plot(program, evdf)
    img_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(img_path, scale=3)

    # Labor & manpower tables for slide 2
    labor_tbl = build_labor_hours_table(cobra)
    manpower_tbl = build_manpower_table(cobra, curr_date, prev_date)

    # Cost vs Schedule/BEI tables (program-level)
    cost_tbl = metrics_df[metrics_df["Metric"] == "CPI"].reset_index(drop=True)
    sched_tbl = metrics_df[metrics_df["Metric"].isin(["SPI", "BEI"])].reset_index(drop=True)

    # --- PowerPoint deck per program ---
    prs = Presentation(THEME_PATH) if os.path.exists(THEME_PATH) else Presentation()
    blank_layout = get_blank_layout(prs)

    # ---------------------------------------------------
    # Slide 1 – EVMS Trend Overview
    # ---------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    add_title(slide1, f"{program} EVMS Trend Overview")

    # chart on left
    slide1.shapes.add_picture(
        img_path,
        Inches(0.5), Inches(1.0),
        width=Inches(5.5)
    )

    # metrics table on right
    tbl1 = add_simple_table(slide1, metrics_df, left_in=6.2, top_in=1.0, width_in=3.5)
    # color CTD/LSD cells
    for i in range(1, len(metrics_df) + 1):
        for col_name, col_idx in (("CTD", 1), ("LSD", 2)):
            val = metrics_df.iloc[i-1][col_name]
            rgb = spi_cpi_color(val)
            if rgb is not None:
                cell = tbl1.cell(i, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb

    add_rcca_box(slide1)

    # ---------------------------------------------------
    # Slide 2 – Tables (Labor, Cost/Sched/BEI, Manpower) + RC box
    # ---------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_title(slide2, f"{program} EVMS Detail – Tables")

    # Labor Hours Performance – across top
    tbl_labor = add_simple_table(slide2, labor_tbl,
                                 left_in=0.5, top_in=0.9, width_in=9.0)
    # VAC coloring by VAC/BAC
    columns = list(labor_tbl.columns)
    if "VAC" in columns and "BAC" in columns:
        vac_idx = columns.index("VAC")
        bac_idx = columns.index("BAC")
        for i in range(len(labor_tbl)):
            vac = labor_tbl.iloc[i]["VAC"]
            bac = labor_tbl.iloc[i]["BAC"]
            ratio = vac / bac if (not pd.isna(bac) and bac != 0) else np.nan
            rgb = vac_color_from_ratio(ratio)
            if rgb is not None:
                cell = tbl_labor.cell(i+1, vac_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb

    # Cost Performance (CPI) – bottom left
    tbl_cost = add_simple_table(slide2, cost_tbl,
                                left_in=0.5, top_in=3.0, width_in=3.5)
    for i in range(1, len(cost_tbl)+1):
        for col_name, col_idx in (("CTD", 1), ("LSD", 2)):
            val = cost_tbl.iloc[i-1][col_name]
            rgb = spi_cpi_color(val)
            if rgb is not None:
                cell = tbl_cost.cell(i, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb

    # Schedule + BEI Performance – bottom middle
    tbl_sched = add_simple_table(slide2, sched_tbl,
                                 left_in=4.1, top_in=3.0, width_in=3.5)
    for i in range(1, len(sched_tbl)+1):
        for col_name, col_idx in (("CTD", 1), ("LSD", 2)):
            val = sched_tbl.iloc[i-1][col_name]
            rgb = spi_cpi_color(val)
            if rgb is not None:
                cell = tbl_sched.cell(i, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb

    # Program Manpower – bottom right
    tbl_mp = add_simple_table(slide2, manpower_tbl,
                              left_in=7.7, top_in=3.0, width_in=2.0)
    if "% Var" in manpower_tbl.columns:
        var_idx = list(manpower_tbl.columns).index("% Var")
        for i in range(len(manpower_tbl)):
            var_ratio = manpower_tbl.iloc[i]["% Var"]
            rgb = manpower_var_color(var_ratio)
            if rgb is not None:
                cell = tbl_mp.cell(i+1, var_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb

    # Root cause / comments box across bottom
    add_rcca_box(slide2, label="Comments / Root Cause & Corrective Actions",
                 left_in=0.5, top_in=4.9, width_in=9.0, height_in=1.5)

    # Save deck
    out_pptx = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Deck.pptx")
    prs.save(out_pptx)
    print(f"  → Saved deck: {out_pptx}")

print("ALL PROGRAM EVMS DECKS COMPLETE ✓")