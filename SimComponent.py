# ============================================================
# EVMS Pipeline - Multi-Program EV Plot + Tables + PPT Decks
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
# CONFIG: update paths / program names as needed
# ------------------------------------------------------------

cobra_files = {
    # Program name           # Cobra export path
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS"     : "data/Cobra-Abrams STS.xlsx",
    "XM30"           : "data/Cobra-XM30.xlsx",
}

OPENPLAN_PATH = "data/OpenPlan_Activity-Penske.xlsx"  # BEI source
THEME_PATH     = "data/Theme.pptx"                    # optional PPT theme

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ASSUMPTIONS (for you to confirm with the team):
# 1) BAC = total BCWS hours by Sub-Team.
# 2) EAC = ACWP + ETC hours (if ETC cost set exists); else EAC = ACWP.
# 3) VAC = BAC - EAC and VAC% = VAC / BAC. VAC cells are color-coded from VAC%.
#       <= -3% red, -3–0% yellow, 0–+3% green, >+3% light blue.
# 4) EV Cost/Schedule CTD metrics use last row of Cumulative CPI/SPI.
#    LSD metrics use last row of Monthly CPI/SPI.
# 5) BEI CTD/LSD are computed from OpenPlan using Baseline Finish vs Actual Finish.
#    Status date = max Cobra DATE for that program.
# 6) Program Manpower is calculated at the program level in FTE using:
#       FTE = Hours / 173.33  (40-hr 9/80 schedule approximation).
#    Demand = BCWS hours, Actual = ACWP hours for the status month.
# ------------------------------------------------------------


# ------------------------- Helpers --------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, upper, remove spaces / hyphens / underscores."""
    df = df.rename(columns={c: c.strip().upper()
                              .replace(" ", "")
                              .replace("-", "")
                              .replace("_", "") for c in df.columns})
    return df


def map_cost_sets(cost_cols):
    """
    Map Cobra COST-SET labels to BCWS / BCWP / ACWP / ETC using fuzzy text matching.
    Works even when column names are weird, as long as they contain key words.
    """
    cleaned = {col: col.replace(" ", "").replace("-", "").replace("_", "").upper()
               for col in cost_cols}

    bcws = bcwp = acwp = etc = None

    for orig, clean in cleaned.items():
        # ACWP detection
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig

        # BCWS detection (Budget / Plan)
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig

        # BCWP detection (Earned / Progress)
        elif ("BCWP" in clean) or ("PROGRESS" in clean) or ("EARNED" in clean):
            bcwp = orig

        # ETC detection
        elif "ETC" in clean:
            etc = orig

    return bcws, bcwp, acwp, etc


def compute_ev_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Monthly & Cumulative CPI / SPI time-series at the program level.
    Expects normalized Cobra dataframe with DATE, COSTSET, HOURS.
    """
    if "COSTSET" not in df.columns:
        raise ValueError("Missing COSTSET column after normalization.")

    # Ensure DATE is datetime
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE",
        columns="COSTSET",
        values="HOURS",
        aggfunc="sum"
    ).reset_index()

    cost_cols = [c for c in pivot.columns if c != "DATE"]
    bcws_col, bcwp_col, acwp_col, _ = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None:
        missing.append("BCWS")
    if bcwp_col is None:
        missing.append("BCWP")
    if acwp_col is None:
        missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing required cost sets: {missing}, found: {cost_cols}")

    # Replace zeros with NaN to avoid divide-by-zero noise
    BCWS = pivot[bcws_col].replace(0, np.nan)
    BCWP = pivot[bcwp_col].replace(0, np.nan)
    ACWP = pivot[acwp_col].replace(0, np.nan)

    out = pd.DataFrame({"DATE": pivot["DATE"]})
    out["Monthly CPI"] = BCWP / ACWP
    out["Monthly SPI"] = BCWP / BCWS

    out["Cumulative CPI"] = BCWP.cumsum() / ACWP.cumsum()
    out["Cumulative SPI"] = BCWP.cumsum() / BCWS.cumsum()

    return out


def create_evms_plot(program: str, evdf: pd.DataFrame) -> go.Figure:
    """Build EVMS CPI/SPI plot with background performance zones."""

    fig = go.Figure()

    # Background performance bands for CPI/SPI
    fig.add_hrect(y0=0.90, y1=0.97, fillcolor="red",      opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.97, y1=1.00, fillcolor="yellow",   opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.00, y1=1.07, fillcolor="green",    opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.07, y1=1.20, fillcolor="lightblue",opacity=0.25, line_width=0)

    # Monthly CPI / SPI (markers)
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

    # Cumulative CPI / SPI (lines)
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


def build_labor_hours_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labor Hours Performance by Sub-Team with %COMP, BAC, EAC, VAC.
    BAC = total BCWS; EAC = ACWP + ETC (if ETC present, else ACWP).
    """
    if "SUBTEAM" not in df.columns:
        raise ValueError("Expected SUBTEAM column after normalization for Labor table.")

    pivot = df.pivot_table(
        index="SUBTEAM",
        columns="COSTSET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0)

    cost_cols = list(pivot.columns)
    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None:
        missing.append("BCWS")
    if bcwp_col is None:
        missing.append("BCWP")
    if acwp_col is None:
        missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing cost sets for Labor table: {missing}")

    BAC = pivot[bcws_col]
    if etc_col is not None and etc_col in pivot.columns:
        ETC = pivot[etc_col]
    else:
        ETC = 0.0  # assumption: no ETC column, treat ETC as 0

    ACWP = pivot[acwp_col]

    EAC = ACWP + ETC
    VAC = BAC - EAC
    PCT_COMP = np.where(BAC != 0, pivot[bcwp_col] / BAC, np.nan)

    out = pd.DataFrame({
        "Sub Team": pivot.index,
        "%COMP": PCT_COMP,
        "BAC": BAC,
        "EAC": EAC,
        "VAC": VAC,
    }).reset_index(drop=True)

    return out


def compute_bei_from_openplan(openplan_df: pd.DataFrame,
                              status_date: pd.Timestamp,
                              program_name: str | None = None) -> tuple[float, float]:
    """
    Compute BEI CTD & LSD from OpenPlan activity file.
    BEI = (Total tasks completed) / (Tasks with Baseline Finish <= time_now).

    status_date is typically the last Cobra status DATE for that program.
    If a PROGRAM column exists, we try to filter rows for that program name.
    """
    df = normalize_columns(openplan_df.copy())

    # Try to filter by program if possible
    if program_name and "PROGRAM" in df.columns:
        mask = df["PROGRAM"].astype(str).str.contains(program_name, case=False, na=False)
        if mask.any():
            df = df[mask]

    # Identify baseline/actual finish columns
    base_col = None
    act_col = None
    for c in df.columns:
        if "BASELINEFINISH" in c and base_col is None:
            base_col = c
        if ("ACTUALFINISH" in c or "ACTFINISH" in c) and act_col is None:
            act_col = c

    if base_col is None or act_col is None:
        # If columns are missing, return NaNs so table still builds
        return np.nan, np.nan

    df[base_col] = pd.to_datetime(df[base_col], errors="coerce")
    df[act_col] = pd.to_datetime(df[act_col], errors="coerce")

    # CTD as of status_date
    denom_ctd = df[df[base_col] <= status_date]
    numer_ctd = denom_ctd[
        denom_ctd[act_col].notna() & (denom_ctd[act_col] <= status_date)
    ]
    bei_ctd = (len(numer_ctd) / len(denom_ctd)) if len(denom_ctd) else np.nan

    # LSD = previous month-end
    status_period = status_date.to_period("M")
    prev_period_end = (status_period - 1).to_timestamp("M")

    denom_lsd = df[df[base_col] <= prev_period_end]
    numer_lsd = denom_lsd[
        denom_lsd[act_col].notna() & (denom_lsd[act_col] <= prev_period_end)
    ]
    bei_lsd = (len(numer_lsd) / len(denom_lsd)) if len(denom_lsd) else np.nan

    return bei_ctd, bei_lsd


def build_cost_and_schedule_tables(evdf: pd.DataFrame,
                                   bei_ctd: float,
                                   bei_lsd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cost Performance: CPI CTD / LSD
    Schedule Performance: SPI CTD / LSD + BEI CTD / LSD.
    """
    cpi_ctd = evdf["Cumulative CPI"].iloc[-1]
    cpi_lsd = evdf["Monthly CPI"].iloc[-1]
    spi_ctd = evdf["Cumulative SPI"].iloc[-1]
    spi_lsd = evdf["Monthly SPI"].iloc[-1]

    cost_df = pd.DataFrame({
        "Metric": ["CPI"],
        "CTD": [cpi_ctd],
        "LSD": [cpi_lsd],
    })

    sched_df = pd.DataFrame({
        "Metric": ["SPI", "BEI"],
        "CTD": [spi_ctd, bei_ctd],
        "LSD": [spi_lsd, bei_lsd],
    })

    return cost_df, sched_df


def build_manpower_table(df: pd.DataFrame,
                         hours_per_fte: float = 173.33) -> pd.DataFrame:
    """
    Program manpower table for the status month:
    Demand FTE, Actual FTE, %Var, Last Month Demand FTE, Next Month Demand FTE.
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index=df["DATE"].dt.to_period("M"),
        columns="COSTSET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0)

    cost_cols = list(pivot.columns)
    bcws_col, _, acwp_col, _ = map_cost_sets(cost_cols)

    if bcws_col is None or acwp_col is None:
        # If we can't find BCWS / ACWP, return empty table to avoid hard fail.
        return pd.DataFrame(columns=["Period", "Demand FTE", "Actual FTE",
                                     "% Var", "Last Month Demand FTE",
                                     "Next Month Demand FTE"])

    demand_hrs = pivot[bcws_col]
    actual_hrs = pivot[acwp_col]

    periods = demand_hrs.index.sort_values()
    status_period = periods[-1]
    last_period = periods[-2] if len(periods) > 1 else status_period
    next_period = status_period + 1

    demand_fte = demand_hrs / hours_per_fte
    actual_fte = actual_hrs / hours_per_fte

    status_demand = demand_fte.loc[status_period]
    status_actual = actual_fte.loc[status_period]
    last_demand = demand_fte.loc[last_period]
    next_demand = demand_fte.get(next_period, np.nan)

    pct_var = ((status_actual - status_demand) / status_demand
               if status_demand != 0 else np.nan)

    out = pd.DataFrame({
        "Period": [str(status_period)],
        "Demand FTE": [status_demand],
        "Actual FTE": [status_actual],
        "% Var": [pct_var],
        "Last Month Demand FTE": [last_demand],
        "Next Month Demand FTE": [next_demand],
    })

    return out


# ------------- PowerPoint helpers ---------------------------

def vac_rgb(vac_val, bac_val):
    """Color for VAC cell based on VAC% thresholds."""
    if pd.isna(vac_val) or pd.isna(bac_val) or bac_val == 0:
        return None
    vac_pct = vac_val / bac_val
    if vac_pct <= -0.03:
        return RGBColor(192, 0, 0)      # red
    elif vac_pct <= 0:
        return RGBColor(255, 192, 0)    # yellow
    elif vac_pct <= 0.03:
        return RGBColor(0, 176, 80)     # green
    else:
        return RGBColor(0, 112, 192)    # light blue


def add_title_box(slide, text):
    """Add a slide title at the top."""
    tx_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3),
                                      Inches(9), Inches(0.6))
    tf = tx_box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(24)


def add_df_table(slide, df: pd.DataFrame,
                 top_in=1.0, left_in=0.5, width_in=9.0,
                 vac_color: bool = False):
    """Render a pandas DataFrame as a PowerPoint table."""
    rows, cols = df.shape
    table_shape = slide.shapes.add_table(
        rows + 1, cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(0.6 + 0.3 * rows)
    )
    table = table_shape.table

    # Header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        para = cell.text_frame.paragraphs[0]
        para.font.bold = True
        para.font.size = Pt(12)

    # Body
    for i in range(rows):
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)

            if isinstance(val, (float, np.floating)):
                if "%COMP" in col.upper() or "% VAR" in col.upper():
                    cell.text = "" if np.isnan(val) else f"{val:.1%}"
                else:
                    cell.text = "" if np.isnan(val) else f"{val:,.1f}"
            else:
                cell.text = "" if pd.isna(val) else str(val)

            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(11)

    # VAC color coding (Labor table)
    if vac_color and "VAC" in df.columns and "BAC" in df.columns:
        vac_idx = list(df.columns).index("VAC")
        bac_idx = list(df.columns).index("BAC")
        for i in range(rows):
            vac_val = df.iloc[i, vac_idx]
            bac_val = df.iloc[i, bac_idx]
            rgb = vac_rgb(vac_val, bac_val)
            if rgb is not None:
                cell = table.cell(i + 1, vac_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb
                # Make text stand out
                cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 0, 0)


# ============================================================
# MAIN LOOP – build outputs for each program
# ============================================================

openplan_raw = pd.read_excel(OPENPLAN_PATH)

for program, path in cobra_files.items():
    print(f"Processing {program} …")

    # --- Load & normalize Cobra ---
    cobra_df = pd.read_excel(path)
    cobra_df = normalize_columns(cobra_df)

    # Ensure required columns exist
    required_cols = {"DATE", "COSTSET", "HOURS"}
    missing_cols = required_cols - set(cobra_df.columns)
    if missing_cols:
        raise ValueError(f"{program}: Cobra file missing columns {missing_cols}")

    cobra_df["DATE"] = pd.to_datetime(cobra_df["DATE"])

    # --- EV metrics & status date ---
    evdf = compute_ev_metrics(cobra_df)
    status_date = evdf["DATE"].max()

    # --- BEI (CTD & LSD) from OpenPlan ---
    bei_ctd, bei_lsd = compute_bei_from_openplan(
        openplan_raw, status_date=status_date, program_name=program
    )

    # --- Tables ---
    labor_tbl   = build_labor_hours_table(cobra_df)
    cost_tbl, sched_tbl = build_cost_and_schedule_tables(evdf, bei_ctd, bei_lsd)
    manpower_tbl = build_manpower_table(cobra_df)

    # --- EVMS Plot ---
    fig = create_evms_plot(program, evdf)
    png_path  = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    html_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.html")
    fig.write_image(png_path, scale=3)
    fig.write_html(html_path)

    # --- PowerPoint Deck ---
    if os.path.exists(THEME_PATH):
        prs = Presentation(THEME_PATH)
    else:
        prs = Presentation()

    # Try to use blank layout; fall back if not available
    try:
        blank_layout = prs.slide_layouts[6]
    except IndexError:
        blank_layout = prs.slide_layouts[1]

    # Slide 1 – EVMS Plot
    slide1 = prs.slides.add_slide(blank_layout)
    add_title_box(slide1, f"{program} – EVMS Trend")
    slide1.shapes.add_picture(png_path,
                              Inches(0.5), Inches(1.1),
                              width=Inches(9.0))

    # Slide 2 – Labor Hours Performance
    slide2 = prs.slides.add_slide(blank_layout)
    add_title_box(slide2, f"{program} – Labor Hours Performance")
    add_df_table(slide2, labor_tbl, top_in=1.1, vac_color=True)

    # Slide 3 – Cost Performance
    slide3 = prs.slides.add_slide(blank_layout)
    add_title_box(slide3, f"{program} – Cost Performance (CPI)")
    add_df_table(slide3, cost_tbl, top_in=1.1)

    # Slide 4 – Schedule Performance (SPI + BEI)
    slide4 = prs.slides.add_slide(blank_layout)
    add_title_box(slide4, f"{program} – Schedule Performance (SPI & BEI)")
    add_df_table(slide4, sched_tbl, top_in=1.1)

    # Slide 5 – Program Manpower
    slide5 = prs.slides.add_slide(blank_layout)
    add_title_box(slide5, f"{program} – Program Manpower")
    add_df_table(slide5, manpower_tbl, top_in=1.1)

    # Save PPTX
    pptx_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Deck.pptx")
    prs.save(pptx_path)

    print(f"✓ Completed EVMS outputs for {program}")
    print(f"   PNG:  {png_path}")
    print(f"   HTML: {html_path}")
    print(f"   PPTX: {pptx_path}")

print("ALL PROGRAM EVMS COMPLETE ✓")
print("\nNOTE: Please validate assumptions for BAC/EAC/VAC, FTE hours, and BEI logic with the team on Monday.")