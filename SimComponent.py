# =========================================================
# XM30 EVMS + BEI – Full Presentation Generator (Clean Final)
# =========================================================

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PROGRAM = "XM30"

COBRA_PATH = "data/Cobra-XM30.xlsx"
COBRA_SHEET = "tbl_Weekly Extract"

PENSKE_PATH = "data/OpenPlan_Activity-Penske.xlsx"

THEME_PATH = "data/Theme.pptx"
OUTPUT_PPTX = f"{PROGRAM}_EVMS_Update.pptx"

EV_PLOT_HTML = "evms_chart.html"  # no PNG used

DATE_COL = "DATE"
GROUP_COL = "SUB_TEAM"
COSTSET_COL = "COST-SET"
HOURS_COL = "HOURS"
COST_SETS = ["ACWP", "BCWP", "BCWS", "ETC"]

# Penske columns
P_BF_COL = "Baseline Finish"
P_AF_COL = "Actual Finish"
P_TYPE_COL = "Activity_Type"
P_ID_COL = "Activity ID"
P_GROUP_COL = "SubTeam"

SNAPSHOT_DATE = datetime.now()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
cobra = pd.read_excel(COBRA_PATH, sheet_name=COBRA_SHEET)
cobra[DATE_COL] = pd.to_datetime(cobra[DATE_COL], errors="coerce")
cobra = cobra[cobra[DATE_COL].notna() & (cobra[DATE_COL] <= SNAPSHOT_DATE)].copy()

penske = pd.read_excel(PENSKE_PATH)
penske[P_BF_COL] = pd.to_datetime(penske[P_BF_COL], errors="coerce")
penske[P_AF_COL] = pd.to_datetime(penske[P_AF_COL], errors="coerce")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def safe_div(num, den):
    return np.nan if den is None or np.isclose(den, 0) else num / den

def cost_rollup(df, start=None, end=None, by_group=True):
    m = df[DATE_COL].notna()
    if start is not None: m &= df[DATE_COL] >= start
    if end is not None:   m &= df[DATE_COL] <= end

    if by_group:
        g = df.loc[m].groupby([GROUP_COL, COSTSET_COL])[HOURS_COL].sum()
        wide = g.unstack(fill_value=0.0)
        for cs in COST_SETS:
            if cs not in wide.columns:
                wide[cs] = 0.0
        return wide[COST_SETS].astype(float)
    else:
        g = df.loc[m].groupby(COSTSET_COL)[HOURS_COL].sum()
        return {cs: float(g.get(cs, 0.0)) for cs in COST_SETS}

# ---------------------------------------------------------
# LABOR TABLES
# ---------------------------------------------------------
ytd_start = datetime(SNAPSHOT_DATE.year, 1, 1)
w4_start = SNAPSHOT_DATE - timedelta(weeks=4)

ctd_by_group = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=True)
ytd_by_group = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=True)

ctd_tot = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=False)
ytd_tot = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=False)
w4_tot = cost_rollup(cobra, start=w4_start, end=SNAPSHOT_DATE, by_group=False)

# Labor Hours
BAC = ctd_by_group["BCWS"]
ACW = ctd_by_group["ACWP"]
BCW = ctd_by_group["BCWP"]
ETC = ctd_by_group["ETC"]
EAC = ACW + ETC
VAC = BAC - EAC
PCOMP = np.where(BAC.eq(0), np.nan, BCW / BAC * 100)

labor_tbl = pd.DataFrame({
    "SUB_TEAM": BAC.index,
    "BAC": BAC.values,
    "ACWP": ACW.values,
    "BCWP": BCW.values,
    "ETC": ETC.values,
    "EAC": EAC.values,
    "VAC": VAC.values,
    "%COMP": np.round(PCOMP, 1),
})

# TOTAL row
tot_bac = ctd_tot["BCWS"]
tot_acw = ctd_tot["ACWP"]
tot_bcw = ctd_tot["BCWP"]
tot_etc = ctd_tot["ETC"]
tot_eac = tot_acw + tot_etc
tot_vac = tot_bac - tot_eac
tot_pcomp = safe_div(tot_bcw, tot_bac) * 100 if tot_bac else np.nan

labor_tbl.loc[len(labor_tbl)] = [
    "TOTAL", tot_bac, tot_acw, tot_bcw, tot_etc, tot_eac, tot_vac, round(tot_pcomp,1)
]

# Monthly Labor Ratios
labor_monthly_tbl = labor_tbl[["SUB_TEAM"]].copy()
labor_monthly_tbl["BAC/EAC"] = np.where(
    labor_tbl["EAC"].eq(0), np.nan, labor_tbl["BAC"] / labor_tbl["EAC"]
)
labor_monthly_tbl["VAC/BAC"] = np.where(
    labor_tbl["BAC"].eq(0), np.nan, labor_tbl["VAC"] / labor_tbl["BAC"]
)

# ---------------------------------------------------------
# CPI / SPI BY SUBTEAM
# ---------------------------------------------------------
def build_cpi_spi(ctd_df, ytd_df):
    cost_tbl = pd.DataFrame({
        "SUB_TEAM": ctd_df.index,
        "YTD": np.where(ytd_df["ACWP"].eq(0), np.nan, ytd_df["BCWP"]/ytd_df["ACWP"]),
        "CTD": np.where(ctd_df["ACWP"].eq(0), np.nan, ctd_df["BCWP"]/ctd_df["ACWP"]),
    })

    sched_tbl = pd.DataFrame({
        "SUB_TEAM": ctd_df.index,
        "YTD": np.where(ytd_df["BCWS"].eq(0), np.nan, ytd_df["BCWP"]/ytd_df["BCWS"]),
        "CTD": np.where(ctd_df["BCWS"].eq(0), np.nan, ctd_df["BCWP"]/ctd_df["BCWS"]),
    })

    # Add TOTAL rows
    cost_tbl.loc[len(cost_tbl)] = [
        "TOTAL",
        safe_div(ytd_df["BCWP"].sum(), ytd_df["ACWP"].sum()),
        safe_div(ctd_df["BCWP"].sum(), ctd_df["ACWP"].sum())
    ]

    sched_tbl.loc[len(sched_tbl)] = [
        "TOTAL",
        safe_div(ytd_df["BCWP"].sum(), ytd_df["BCWS"].sum()),
        safe_div(ctd_df["BCWP"].sum(), ctd_df["BCWS"].sum())
    ]

    return cost_tbl.round(2), sched_tbl.round(2)

cost_performance_tbl, schedule_performance_tbl = build_cpi_spi(ctd_by_group, ytd_by_group)

# ---------------------------------------------------------
# EVMS METRICS (PROGRAM LEVEL)
# ---------------------------------------------------------
evms_metrics_tbl = pd.DataFrame({
    "Metric": ["SPI", "CPI"],
    "4WK": [ safe_div(w4_tot["BCWP"], w4_tot["BCWS"]),
             safe_div(w4_tot["BCWP"], w4_tot["ACWP"]) ],

    "YTD": [ safe_div(ytd_tot["BCWP"], ytd_tot["BCWS"]),
             safe_div(ytd_tot["BCWP"], ytd_tot["ACWP"]) ],

    "CTD": [ safe_div(ctd_tot["BCWP"], ctd_tot["BCWS"]),
             safe_div(ctd_tot["BCWP"], ctd_tot["ACWP"]) ],
}).round(2)

# ---------------------------------------------------------
# EVMS CHART (HTML ONLY)
# ---------------------------------------------------------
# Monthly SPI/CPI
month_totals = cobra.groupby([
    cobra[DATE_COL].dt.to_period("M"),
    COSTSET_COL
])[HOURS_COL].sum().unstack(fill_value=0.0)

for cs in COST_SETS:
    if cs not in month_totals.columns:
        month_totals[cs] = 0.0

ev = month_totals.sort_index()
ev["CPI_month"] = safe_div(ev["BCWP"], ev["ACWP"])
ev["SPI_month"] = safe_div(ev["BCWP"], ev["BCWS"])
ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()
ev["CPI_cum"] = safe_div(ev["BCWP_cum"], ev["ACWP_cum"])
ev["SPI_cum"] = safe_div(ev["BCWP_cum"], ev["BCWS_cum"])
ev["Month"] = ev.index.to_timestamp().strftime("%b-%y")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_month"], name="Monthly CPI", mode="markers"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_month"], name="Monthly SPI", mode="markers"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_cum"], name="Cumulative CPI", mode="lines"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_cum"], name="Cumulative SPI", mode="lines"))

fig.update_layout(title="EV Indices (SPI / CPI)", template="plotly_white")
fig.write_html(EV_PLOT_HTML)

# ---------------------------------------------------------
# BEI TABLE
# ---------------------------------------------------------
# SME NOTE:
#   - Completed = Actual Finish not null
#   - Exclude LOE/Milestones (A, B)
#   - Baseline Finish <= snapshot

bei_filter = (
    penske[P_BF_COL].notna() &
    (penske[P_BF_COL] <= SNAPSHOT_DATE) &
    (~penske[P_TYPE_COL].isin(["A", "B"]))
)
penske_be = penske.loc[bei_filter].copy()

tasks_total = penske_be.groupby(P_GROUP_COL)[P_ID_COL].count()
tasks_done = penske_be[penske_be[P_AF_COL].notna()].groupby(P_GROUP_COL)[P_ID_COL].count()

bei_tbl = pd.DataFrame({
    "SubTeam": tasks_total.index,
    "Tasks w/ Baseline Finish ≤ Snapshot": tasks_total.values,
    "Completed Tasks": tasks_done.reindex(tasks_total.index).fillna(0).values
})
bei_tbl["BEI"] = safe_div(
    bei_tbl["Completed Tasks"],
    bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"]
)
bei_tbl["BEI"] = bei_tbl["BEI"].round(2)

# ---------------------------------------------------------
# POWERPOINT BUILDING
# ---------------------------------------------------------

# Load theme
prs = Presentation(THEME_PATH)

# Find the first layout with a title placeholder (real title slide)
title_layout_idx = None
for idx, layout in enumerate(prs.slide_layouts):
    for shp in layout.placeholders:
        if shp.placeholder_format.type == 0:  # TITLE
            title_layout_idx = idx
            break
    if title_layout_idx is not None:
        break

if title_layout_idx is None:
    title_layout_idx = 0  # fallback

def add_table_slide(title, df):
    layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[1]
    slide = prs.slides.add_slide(layout)

    if slide.shapes.title:
        slide.shapes.title.text = title

    rows, cols = df.shape
    rows += 1

    left = Inches(0.4)
    top = Inches(1.2)
    width = prs.slide_width - Inches(0.8)
    height = prs.slide_height - Inches(1.8)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # Header row
    for j, colname in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(colname)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(11)

    # Rows
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, colname in enumerate(df.columns):
            cell = table.cell(i, j)
            val = row[colname]
            text = "" if pd.isna(val) else str(val)
            cell.text = text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)

# ---------------------------------------------------------
# SLIDE 1 – TITLE SLIDE
# ---------------------------------------------------------
title_slide = prs.slides.add_slide(prs.slide_layouts[title_layout_idx])

# Title
if title_slide.shapes.title:
    title_slide.shapes.title.text = PROGRAM
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(0.6), Inches(8), Inches(1))
    tx.text_frame.text = PROGRAM

# Subtitle or fallback textbox
subtitle = None
for shp in title_slide.placeholders:
    if shp.placeholder_format.type == 1:
        subtitle = shp
        break

if subtitle:
    subtitle.text = SNAPSHOT_DATE.strftime("%Y-%m-%d")
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(5), Inches(0.6))
    tx.text_frame.text = SNAPSHOT_DATE.strftime("%Y-%m-%d")

# ---------------------------------------------------------
# SLIDE 2 – EVMS METRICS + LINK TO HTML
# ---------------------------------------------------------
layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[1]
slide2 = prs.slides.add_slide(layout)

if slide2.shapes.title:
    slide2.shapes.title.text = "EVMS Metrics Table"

# Table
add_table_slide("EVMS Metrics Table", evms_metrics_tbl)

# Add hyperlink textbox
tx = slide2.shapes.add_textbox(Inches(1), Inches(6.2), Inches(6), Inches(0.5))
p = tx.text_frame.paragraphs[0]
p.text = "Click here to open the EVMS Chart (HTML)"
r = p.add_run()
r.hyperlink.address = os.path.abspath(EV_PLOT_HTML)

# ---------------------------------------------------------
# SLIDE 3 – COST PERFORMANCE INDEX
# ---------------------------------------------------------
add_table_slide("Cost Performance Index Table", cost_performance_tbl)

# ---------------------------------------------------------
# SLIDE 4 – SCHEDULE PERFORMANCE INDEX
# ---------------------------------------------------------
add_table_slide("Schedule Performance Index Table", schedule_performance_tbl)

# ---------------------------------------------------------
# SLIDE 5 – LABOR HOURS
# ---------------------------------------------------------
add_table_slide("Labor Hours Table", labor_tbl)

# ---------------------------------------------------------
# SLIDE 6 – MONTHLY LABOR RATIOS
# ---------------------------------------------------------
add_table_slide("Monthly Labor Ratios Table", labor_monthly_tbl)

# ---------------------------------------------------------
# SLIDE 7 – BEI TABLE
# ---------------------------------------------------------
add_table_slide("Baseline Execution Index (BEI) Table", bei_tbl)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
prs.save(OUTPUT_PPTX)
print("Presentation saved:", OUTPUT_PPTX)
print("EVMS HTML chart saved:", EV_PLOT_HTML)