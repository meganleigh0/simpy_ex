# ================================================
# XM30 EVMS + BEI → PowerPoint (Theme.pptx)
# All in ONE cell, simplified & de-duplicated
# ================================================
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ----------------------------
# CONFIG
# ----------------------------
PROGRAM          = "XM30"

COBRA_PATH       = "data/Cobra-XM30.xlsx"
COBRA_SHEET      = "tbl_Weekly Extract"   # adjust if needed

PENSKE_PATH      = "data/OpenPlan_Activity-Penske.xlsx"

THEME_PATH       = "Theme.pptx"           # must exist, no generic fallback
OUTPUT_PPTX      = f"{PROGRAM}_EVMS_Update.pptx"
EV_PLOT_PATH     = "evms_chart.png"

# Cobra column names (tall table)
DATE_COL    = "DATE"
GROUP_COL   = "SUB_TEAM"
COSTSET_COL = "COST-SET"
HOURS_COL   = "HOURS"
COST_SETS   = ["ACWP", "BCWP", "BCWS", "ETC"]

# Penske column names (from your screenshot)
P_BF_COL    = "Baseline Finish"
P_AF_COL    = "Actual Finish"
P_TYPE_COL  = "Activity_Type"
P_ID_COL    = "Activity ID"
P_GROUP_COL = "SubTeam"   # can change to Planner / CAM later

# Snapshot date (anchor)
SNAPSHOT_DATE = datetime.now()

# ================================================
# 1. LOAD DATA
# ================================================
cobra = pd.read_excel(COBRA_PATH, sheet_name=COBRA_SHEET)
cobra[DATE_COL] = pd.to_datetime(cobra[DATE_COL], errors="coerce")
cobra = cobra[cobra[DATE_COL].notna() & (cobra[DATE_COL] <= SNAPSHOT_DATE)].copy()

penske = pd.read_excel(PENSKE_PATH)
penske[P_BF_COL] = pd.to_datetime(penske[P_BF_COL], errors="coerce")
penske[P_AF_COL] = pd.to_datetime(penske[P_AF_COL], errors="coerce")

# ================================================
# 2. GENERIC HELPERS
# ================================================
def safe_div(num, den):
    return np.nan if (den is None or np.isclose(den, 0)) else num / den

def cost_rollup(df, start=None, end=None, by_group=True):
    """Sum ACWP / BCWP / BCWS / ETC; either by SUB_TEAM or overall."""
    m = df[DATE_COL].notna()
    if start is not None:
        m &= df[DATE_COL] >= start
    if end is not None:
        m &= df[DATE_COL] <= end

    group_fields = [GROUP_COL, COSTSET_COL] if by_group else [COSTSET_COL]
    g = df.loc[m].groupby(group_fields)[HOURS_COL].sum()

    if by_group:
        wide = g.unstack(fill_value=0.0)
    else:
        wide = g

    # Ensure all expected cost sets exist
    if by_group:
        for cs in COST_SETS:
            if cs not in wide.columns:
                wide[cs] = 0.0
        return wide[COST_SETS].astype(float)
    else:
        out = {cs: float(wide.get(cs, 0.0)) for cs in COST_SETS}
        return out

# ================================================
# 3. LABOR & PERFORMANCE TABLES
# ================================================
# CTD and YTD windows
ytd_start = datetime(SNAPSHOT_DATE.year, 1, 1)
w4_start  = SNAPSHOT_DATE - timedelta(weeks=4)

# By SUB_TEAM (for tables)
ctd_by_group = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=True)
ytd_by_group = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=True)

# Overall sums (for EVMS metrics)
ctd_tot = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=False)
ytd_tot = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=False)
w4_tot  = cost_rollup(cobra, start=w4_start, end=SNAPSHOT_DATE, by_group=False)

# ---- Labor Hours table (BAC, ACWP, BCWP, ETC, EAC, VAC, %COMP) ----
BAC = ctd_by_group["BCWS"]
ACW = ctd_by_group["ACWP"]
BCW = ctd_by_group["BCWP"]
ETC = ctd_by_group["ETC"]

EAC = ACW + ETC
VAC = BAC - EAC
PCOMP = np.where(BAC.eq(0), np.nan, (BCW / BAC) * 100)

labor_tbl = pd.DataFrame(
    {
        "BAC": BAC,
        "ACWP": ACW,
        "BCWP": BCW,
        "ETC": ETC,
        "EAC": EAC,
        "VAC": VAC,
        "%COMP": np.round(PCOMP, 1),
    }
)
labor_tbl.index.name = GROUP_COL

# Add TOTAL row
tot_bac = ctd_tot["BCWS"]
tot_acw = ctd_tot["ACWP"]
tot_bcw = ctd_tot["BCWP"]
tot_etc = ctd_tot["ETC"]
tot_eac = tot_acw + tot_etc
tot_vac = tot_bac - tot_eac
tot_pcomp = np.nan if np.isclose(tot_bac, 0) else round((tot_bcw / tot_bac) * 100, 1)

labor_tbl.loc["TOTAL"] = [tot_bac, tot_acw, tot_bcw, tot_etc, tot_eac, tot_vac, tot_pcomp]
labor_tbl_reset = labor_tbl.reset_index().rename(columns={GROUP_COL: "SUB_TEAM"})

# ---- Monthly Labor Ratios table (BAC/EAC, VAC/BAC) ----
labor_monthly_tbl = pd.DataFrame(
    {
        "SUB_TEAM": labor_tbl_reset["SUB_TEAM"],
        "BAC/EAC": np.where(labor_tbl_reset["EAC"].eq(0),
                            np.nan,
                            labor_tbl_reset["BAC"] / labor_tbl_reset["EAC"]),
        "VAC/BAC": np.where(labor_tbl_reset["BAC"].eq(0),
                            np.nan,
                            labor_tbl_reset["VAC"] / labor_tbl_reset["BAC"]),
    }
).set_index("SUB_TEAM")

# ---- Cost & Schedule Performance tables (CPI/SPI – YTD & CTD) ----
def build_cpi_spi(ctd_df, ytd_df):
    # By SUB_TEAM
    cpi_ctd = np.where(ctd_df["ACWP"].eq(0), np.nan, ctd_df["BCWP"] / ctd_df["ACWP"])
    cpi_ytd = np.where(ytd_df["ACWP"].eq(0), np.nan, ytd_df["BCWP"] / ytd_df["ACWP"])
    spi_ctd = np.where(ctd_df["BCWS"].eq(0), np.nan, ctd_df["BCWP"] / ctd_df["BCWS"])
    spi_ytd = np.where(ytd_df["BCWS"].eq(0), np.nan, ytd_df["BCWP"] / ytd_df["BCWS"])

    cost_tbl = pd.DataFrame({"YTD": cpi_ytd, "CTD": cpi_ctd})
    sched_tbl = pd.DataFrame({"YTD": spi_ytd, "CTD": spi_ctd})

    # TOTAL rows
    ctd_sums = ctd_df.sum()
    ytd_sums = ytd_df.sum()
    tot_cpi_ctd = safe_div(ctd_sums["BCWP"], ctd_sums["ACWP"])
    tot_cpi_ytd = safe_div(ytd_sums["BCWP"], ytd_sums["ACWP"])
    tot_spi_ctd = safe_div(ctd_sums["BCWP"], ctd_sums["BCWS"])
    tot_spi_ytd = safe_div(ytd_sums["BCWP"], ytd_sums["BCWS"])

    cost_tbl.loc["TOTAL"] = [tot_cpi_ytd, tot_cpi_ctd]
    sched_tbl.loc["TOTAL"] = [tot_spi_ytd, tot_spi_ctd]

    return cost_tbl.round(2), sched_tbl.round(2)

cost_performance_tbl, schedule_performance_tbl = build_cpi_spi(ctd_by_group, ytd_by_group)

# ---- EVMS Metrics table (SPI/CPI – 4WK, YTD, CTD, program level) ----
spi_4wk = safe_div(w4_tot["BCWP"], w4_tot["BCWS"])
spi_ytd = safe_div(ytd_tot["BCWP"], ytd_tot["BCWS"])
spi_ctd = safe_div(ctd_tot["BCWP"], ctd_tot["BCWS"])

cpi_4wk = safe_div(w4_tot["BCWP"], w4_tot["ACWP"])
cpi_ytd = safe_div(ytd_tot["BCWP"], ytd_tot["ACWP"])
cpi_ctd = safe_div(ctd_tot["BCWP"], ctd_tot["ACWP"])

evms_metrics_tbl = pd.DataFrame(
    {
        "4WK": [spi_4wk, cpi_4wk],
        "YTD": [spi_ytd, cpi_ytd],
        "CTD": [spi_ctd, cpi_ctd],
    },
    index=["SPI", "CPI"],
).round(2)

# ================================================
# 4. EVMS PLOT (MONTHLY SPI/CPI & CUMULATIVE)
# ================================================
month_totals = (
    cobra.groupby([cobra[DATE_COL].dt.to_period("M"), COSTSET_COL])[HOURS_COL]
    .sum()
    .unstack(fill_value=0.0)
)
for cs in COST_SETS:
    if cs not in month_totals.columns:
        month_totals[cs] = 0.0
ev = month_totals.sort_index()

ev["CPI_month"] = np.where(ev["ACWP"].eq(0), np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"].eq(0), np.nan, ev["BCWP"] / ev["BCWS"])

ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

ev["CPI_cum"] = np.where(ev["ACWP_cum"].eq(0), np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"].eq(0), np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

ev_plot_df = ev[["CPI_month", "SPI_month", "CPI_cum", "SPI_cum"]].copy()
ev_plot_df["Month"] = ev_plot_df.index.to_timestamp("M").strftime("%b-%y")

fig = go.Figure()

# Performance bands
fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#FF0000", opacity=0.2, line_width=0, layer="below")
fig.add_hrect(y0=0.95, y1=1.05, fillcolor="#FFC000", opacity=0.2, line_width=0, layer="below")
fig.add_hrect(y0=1.05, y1=1.20, fillcolor="#00B050", opacity=0.2, line_width=0, layer="below")

x = ev_plot_df["Month"]

fig.add_trace(go.Scatter(
    x=x, y=ev_plot_df["CPI_month"],
    mode="markers", name="Monthly CPI",
    marker=dict(symbol="diamond", size=8)
))
fig.add_trace(go.Scatter(
    x=x, y=ev_plot_df["SPI_month"],
    mode="markers", name="Monthly SPI",
    marker=dict(symbol="circle", size=8)
))
fig.add_trace(go.Scatter(
    x=x, y=ev_plot_df["CPI_cum"],
    mode="lines", name="Cumulative CPI",
    line=dict(width=3)
))
fig.add_trace(go.Scatter(
    x=x, y=ev_plot_df["SPI_cum"],
    mode="lines", name="Cumulative SPI",
    line=dict(width=3, dash="dash")
))

fig.update_layout(
    title="EV Indices (SPI / CPI)",
    xaxis_title="Month",
    yaxis_title="Index",
    yaxis=dict(range=[0.9, 1.2]),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=40, r=20, t=60, b=80),
)

# Export PNG using kaleido (no noisy message; either works or the slide will show a text note)
ev_image_ok = False
try:
    img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
    with open(EV_PLOT_PATH, "wb") as f:
        f.write(img_bytes)
    ev_image_ok = True
except Exception as e:
    # If this fails even with kaleido installed, you may need to restart the kernel/cluster.
    ev_image_ok = False

# ================================================
# 5. BEI TABLE (Penske)
# ================================================
# BEI assumptions (TODO: confirm w/ SME next meeting):
# - Baseline field: P_BF_COL ('Baseline Finish')
# - Completed: P_AF_COL ('Actual Finish') not null
# - Exclude Activity_Type in ['A', 'B'] (LOE, Milestones)
# - Grouping: P_GROUP_COL ('SubTeam')
bei_filter = (
    penske[P_BF_COL].notna()
    & (penske[P_BF_COL] <= SNAPSHOT_DATE)
    & (~penske[P_TYPE_COL].isin(["A", "B"]))
)

penske_be = penske.loc[bei_filter].copy()

total_tasks = (
    penske_be.groupby(P_GROUP_COL)[P_ID_COL]
    .count()
    .rename("Tasks w/ Baseline Finish ≤ Snapshot")
)

completed_tasks = (
    penske_be[penske_be[P_AF_COL].notna()]
    .groupby(P_GROUP_COL)[P_ID_COL]
    .count()
    .rename("Completed Tasks")
)

bei_tbl = pd.concat([total_tasks, completed_tasks], axis=1).fillna(0)
bei_tbl["BEI"] = np.where(
    bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"].eq(0),
    np.nan,
    bei_tbl["Completed Tasks"] / bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"],
)
bei_tbl = bei_tbl.reset_index().rename(columns={P_GROUP_COL: "SubTeam"})
bei_tbl["BEI"] = bei_tbl["BEI"].round(2)

# ================================================
# 6. COLOR HELPERS (TABLE CELLS)
# ================================================
def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def spi_cpi_cell_color(val):
    """Color for SPI/CPI-style index values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    v = float(val)
    if v < 0.90:
        bg, fg = "#C00000", "#FFFFFF"
    elif v < 0.95:
        bg, fg = "#FFC000", "#000000"
    elif v <= 1.05:
        bg, fg = "#00B050", "#FFFFFF"
    else:
        bg, fg = "#1F4E79", "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

def vac_cell_color(val):
    """Color for VAC amounts or VAC/BAC ratios."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    v = float(val)
    if v < 0:
        bg, fg = "#C00000", "#FFFFFF"   # overrun risk
    elif v == 0:
        bg, fg = "#FFC000", "#000000"
    else:
        bg, fg = "#00B050", "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

# ================================================
# 7. POWERPOINT – USING Theme.pptx ONLY
# ================================================
if not os.path.exists(THEME_PATH):
    raise FileNotFoundError(
        f"Theme file '{THEME_PATH}' not found. "
        "Place Theme.pptx in the working directory or update THEME_PATH."
    )

prs = Presentation(THEME_PATH)

def add_table_slide(prs, title, df, fmt=None, color_cols=None, layout_idx=5):
    """
    Adds a slide with a title and a table for df.
    fmt: dict of col -> format string (e.g., {:.2f})
    color_cols: dict of col -> color_func(value) -> (bg_rgb, text_rgb)
    """
    if fmt is None:
        fmt = {}
    if color_cols is None:
        color_cols = {}

    layout_idx = min(layout_idx, len(prs.slide_layouts) - 1)
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])

    # Title
    if slide.shapes.title:
        slide.shapes.title.text = title

    rows, cols = df.shape
    rows += 1  # header row

    left = Inches(0.4)
    top = Inches(1.0)
    width = prs.slide_width - Inches(0.8)
    height = prs.slide_height - Inches(1.4)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # Headers
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(12)

    # Body
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            # formatting
            fmt_str = fmt.get(col)
            if fmt_str and pd.api.types.is_numeric_dtype(type(val)):
                try:
                    text = fmt_str.format(val)
                except Exception:
                    text = "" if pd.isna(val) else str(val)
            else:
                text = "" if pd.isna(val) else str(val)

            cell = table.cell(i, j)
            cell.text = text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)

            # conditional colors
            color_fn = color_cols.get(col)
            if color_fn:
                bg_rgb, fg_rgb = color_fn(val)
                if bg_rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*bg_rgb)
                if fg_rgb:
                    p.font.color.rgb = RGBColor(*fg_rgb)

    return slide

# ---- Slide 1: EVMS Plot ----
layout_idx_plot = 5 if len(prs.slide_layouts) > 5 else 1
slide_plot = prs.slides.add_slide(prs.slide_layouts[layout_idx_plot])
if slide_plot.shapes.title:
    slide_plot.shapes.title.text = "EV Indices (SPI / CPI)"

if ev_image_ok and os.path.exists(EV_PLOT_PATH):
    left = Inches(0.5)
    top = Inches(1.2)
    width = prs.slide_width - Inches(1.0)
    slide_plot.shapes.add_picture(EV_PLOT_PATH, left, top, width=width)
else:
    # If image couldn't be created, put a simple note (no noisy prints)
    tb = slide_plot.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1))
    tb.text_frame.text = "EVMS chart image not available (image export error)."

# ---- Slide 2: EVMS Metrics ----
evms_fmt = {"4WK": "{:.2f}", "YTD": "{:.2f}", "CTD": "{:.2f}"}
evms_color_cols = {"4WK": spi_cpi_cell_color, "YTD": spi_cpi_cell_color, "CTD": spi_cpi_cell_color}
add_table_slide(prs, "EVMS Metrics (SPI & CPI – 4WK / YTD / CTD)", evms_metrics_tbl, fmt=evms_fmt, color_cols=evms_color_cols)

# ---- Slide 3: Cost Performance (CPI) ----
cpi_fmt = {"YTD": "{:.2f}", "CTD": "{:.2f}"}
cpi_colors = {"YTD": spi_cpi_cell_color, "CTD": spi_cpi_cell_color}
add_table_slide(prs, "Cost Performance Index (CPI – YTD / CTD)", cost_performance_tbl, fmt=cpi_fmt, color_cols=cpi_colors)

# ---- Slide 4: Schedule Performance (SPI) ----
spi_fmt = {"YTD": "{:.2f}", "CTD": "{:.2f}"}
spi_colors = {"YTD": spi_cpi_cell_color, "CTD": spi_cpi_cell_color}
add_table_slide(prs, "Schedule Performance Index (SPI – YTD / CTD)", schedule_performance_tbl, fmt=spi_fmt, color_cols=spi_colors)

# ---- Slide 5: Labor Hours ----
labor_fmt = {
    "BAC": "{:,.0f}",
    "ACWP": "{:,.0f}",
    "BCWP": "{:,.0f}",
    "ETC": "{:,.0f}",
    "EAC": "{:,.0f}",
    "VAC": "{:,.0f}",
    "%COMP": "{:.1f}",
}
labor_colors = {"VAC": vac_cell_color}
add_table_slide(prs, "Labor Hours (BAC / ACWP / BCWP / ETC / EAC / VAC / %COMP)", labor_tbl_reset, fmt=labor_fmt, color_cols=labor_colors)

# ---- Slide 6: Monthly Labor Ratios ----
labor_monthly_fmt = {"BAC/EAC": "{:.2f}", "VAC/BAC": "{:.2f}"}
labor_monthly_colors = {"VAC/BAC": vac_cell_color}
add_table_slide(prs, "Monthly Labor Ratios (BAC/EAC & VAC/BAC)", labor_monthly_tbl.reset_index(), fmt=labor_monthly_fmt, color_cols=labor_monthly_colors)

# ---- Slide 7: BEI ----
bei_fmt = {
    "Tasks w/ Baseline Finish ≤ Snapshot": "{:.0f}",
    "Completed Tasks": "{:.0f}",
    "BEI": "{:.2f}",
}
add_table_slide(prs, "Baseline Execution Index (BEI)", bei_tbl, fmt=bei_fmt)

# Save
prs.save(OUTPUT_PPTX)
print(f"PowerPoint saved: {OUTPUT_PPTX}")