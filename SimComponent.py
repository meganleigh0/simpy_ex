# =========================================================
# XM30 EVMS + BEI – Full Presentation with Color Coding
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
EV_PLOT_HTML = "evms_chart.html"

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
def cost_rollup(df, start=None, end=None, by_group=True):
    mask = df[DATE_COL].notna()
    if start is not None:
        mask &= df[DATE_COL] >= start
    if end is not None:
        mask &= df[DATE_COL] <= end

    if by_group:
        g = (
            df.loc[mask]
              .groupby([GROUP_COL, COSTSET_COL])[HOURS_COL]
              .sum()
              .unstack(fill_value=0.0)
        )
        for cs in COST_SETS:
            if cs not in g.columns:
                g[cs] = 0.0
        return g[COST_SETS].astype(float)
    else:
        g = df.loc[mask].groupby(COSTSET_COL)[HOURS_COL].sum()
        return {cs: float(g.get(cs, 0.0)) for cs in COST_SETS}

def safe_div_scalar(num, den):
    if den is None or np.isclose(den, 0):
        return np.nan
    return num / den

# ---------------------------------------------------------
# LABOR & PERFORMANCE TABLES
# ---------------------------------------------------------
ytd_start = datetime(SNAPSHOT_DATE.year, 1, 1)
w4_start = SNAPSHOT_DATE - timedelta(weeks=4)

ctd_by_group = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=True)
ytd_by_group = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=True)

ctd_tot = cost_rollup(cobra, end=SNAPSHOT_DATE, by_group=False)
ytd_tot = cost_rollup(cobra, start=ytd_start, end=SNAPSHOT_DATE, by_group=False)
w4_tot  = cost_rollup(cobra, start=w4_start, end=SNAPSHOT_DATE, by_group=False)

# ---- Labor Hours Table ----
BAC = ctd_by_group["BCWS"]
ACW = ctd_by_group["ACWP"]
BCW = ctd_by_group["BCWP"]
ETC = ctd_by_group["ETC"]

EAC = ACW + ETC
VAC = BAC - EAC
PCOMP = np.where(BAC == 0, np.nan, BCW / BAC * 100.0)

labor_tbl = pd.DataFrame(
    {
        "SUB_TEAM": BAC.index,
        "BAC": BAC.values,
        "ACWP": ACW.values,
        "BCWP": BCW.values,
        "ETC": ETC.values,
        "EAC": EAC.values,
        "VAC": VAC.values,
        "%COMP": np.round(PCOMP, 1),
    }
)

# TOTAL row
tot_bac = ctd_tot["BCWS"]
tot_acw = ctd_tot["ACWP"]
tot_bcw = ctd_tot["BCWP"]
tot_etc = ctd_tot["ETC"]
tot_eac = tot_acw + tot_etc
tot_vac = tot_bac - tot_eac
tot_pcomp = np.nan if tot_bac == 0 else round(tot_bcw / tot_bac * 100.0, 1)

labor_tbl.loc[len(labor_tbl)] = [
    "TOTAL", tot_bac, tot_acw, tot_bcw, tot_etc, tot_eac, tot_vac, tot_pcomp
]

# ---- Monthly Labor Ratios Table ----
labor_monthly_tbl = labor_tbl[["SUB_TEAM"]].copy()
labor_monthly_tbl["BAC/EAC"] = np.where(
    labor_tbl["EAC"] == 0, np.nan, labor_tbl["BAC"] / labor_tbl["EAC"]
)
labor_monthly_tbl["VAC/BAC"] = np.where(
    labor_tbl["BAC"] == 0, np.nan, labor_tbl["VAC"] / labor_tbl["BAC"]
)

# ---- CPI & SPI Tables by SUB_TEAM ----
def build_cpi_spi(ctd_df, ytd_df):
    ytd_df = ytd_df.reindex(ctd_df.index)

    cpi_ytd = np.where(ytd_df["ACWP"] == 0, np.nan, ytd_df["BCWP"] / ytd_df["ACWP"])
    cpi_ctd = np.where(ctd_df["ACWP"] == 0, np.nan, ctd_df["BCWP"] / ctd_df["ACWP"])
    spi_ytd = np.where(ytd_df["BCWS"] == 0, np.nan, ytd_df["BCWP"] / ytd_df["BCWS"])
    spi_ctd = np.where(ctd_df["BCWS"] == 0, np.nan, ctd_df["BCWP"] / ctd_df["BCWS"])

    cost_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ctd_df.index,
            "YTD": np.round(cpi_ytd, 2),
            "CTD": np.round(cpi_ctd, 2),
        }
    )
    sched_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ctd_df.index,
            "YTD": np.round(spi_ytd, 2),
            "CTD": np.round(spi_ctd, 2),
        }
    )

    y_ACWP_sum = ytd_df["ACWP"].sum()
    y_BCWP_sum = ytd_df["BCWP"].sum()
    y_BCWS_sum = ytd_df["BCWS"].sum()
    c_ACWP_sum = ctd_df["ACWP"].sum()
    c_BCWP_sum = ctd_df["BCWP"].sum()
    c_BCWS_sum = ctd_df["BCWS"].sum()

    cost_total_ytd = np.nan if y_ACWP_sum == 0 else y_BCWP_sum / y_ACWP_sum
    cost_total_ctd = np.nan if c_ACWP_sum == 0 else c_BCWP_sum / c_ACWP_sum
    sched_total_ytd = np.nan if y_BCWS_sum == 0 else y_BCWP_sum / y_BCWS_sum
    sched_total_ctd = np.nan if c_BCWS_sum == 0 else c_BCWP_sum / c_BCWS_sum

    cost_tbl.loc[len(cost_tbl)] = [
        "TOTAL",
        round(cost_total_ytd, 2) if not np.isnan(cost_total_ytd) else np.nan,
        round(cost_total_ctd, 2) if not np.isnan(cost_total_ctd) else np.nan,
    ]
    sched_tbl.loc[len(sched_tbl)] = [
        "TOTAL",
        round(sched_total_ytd, 2) if not np.isnan(sched_total_ytd) else np.nan,
        round(sched_total_ctd, 2) if not np.isnan(sched_total_ctd) else np.nan,
    ]

    return cost_tbl, sched_tbl

cost_performance_tbl, schedule_performance_tbl = build_cpi_spi(ctd_by_group, ytd_by_group)

# ---- EVMS Metrics (Program Level) ----
spi_4wk = np.nan if w4_tot["BCWS"] == 0 else w4_tot["BCWP"] / w4_tot["BCWS"]
cpi_4wk = np.nan if w4_tot["ACWP"] == 0 else w4_tot["BCWP"] / w4_tot["ACWP"]
spi_ytd = np.nan if ytd_tot["BCWS"] == 0 else ytd_tot["BCWP"] / ytd_tot["BCWS"]
cpi_ytd = np.nan if ytd_tot["ACWP"] == 0 else ytd_tot["BCWP"] / ytd_tot["ACWP"]
spi_ctd = np.nan if ctd_tot["BCWS"] == 0 else ctd_tot["BCWP"] / ctd_tot["BCWS"]
cpi_ctd = np.nan if ctd_tot["ACWP"] == 0 else ctd_tot["BCWP"] / ctd_tot["ACWP"]

evms_metrics_tbl = pd.DataFrame(
    {
        "Metric": ["SPI", "CPI"],
        "4WK": [spi_4wk, cpi_4wk],
        "YTD": [spi_ytd, cpi_ytd],
        "CTD": [spi_ctd, cpi_ctd],
    }
).round(2)

# ---------------------------------------------------------
# EVMS CHART (HTML) WITH COLOR BANDS
# ---------------------------------------------------------
month_totals = (
    cobra.groupby([cobra[DATE_COL].dt.to_period("M"), COSTSET_COL])[HOURS_COL]
    .sum()
    .unstack(fill_value=0.0)
)

for cs in COST_SETS:
    if cs not in month_totals.columns:
        month_totals[cs] = 0.0

ev = month_totals.sort_index().copy()

ev["CPI_month"] = np.where(ev["ACWP"] == 0, np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"] == 0, np.nan, ev["BCWP"] / ev["BCWS"])

ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

ev["CPI_cum"] = np.where(ev["ACWP_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

ev["Month"] = ev.index.to_timestamp().strftime("%b-%y")

fig = go.Figure()

# performance bands (same thresholds as SPI/CPI color)
fig.add_hrect(y0=0.0,  y1=0.90, fillcolor="#C00000", opacity=0.2, line_width=0, layer="below")  # red
fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#FFC000", opacity=0.2, line_width=0, layer="below")  # yellow
fig.add_hrect(y0=0.95, y1=1.05, fillcolor="#00B050", opacity=0.2, line_width=0, layer="below")  # green
fig.add_hrect(y0=1.05, y1=1.20, fillcolor="#1F4E79", opacity=0.15, line_width=0, layer="below") # blue

fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_month"], name="Monthly CPI",
                         mode="markers", marker=dict(symbol="diamond", size=8)))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_month"], name="Monthly SPI",
                         mode="markers", marker=dict(symbol="circle", size=8)))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_cum"], name="Cumulative CPI",
                         mode="lines", line=dict(width=3)))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_cum"], name="Cumulative SPI",
                         mode="lines", line=dict(width=3, dash="dash")))

fig.update_layout(
    title="EV Indices (SPI / CPI)",
    xaxis_title="Month",
    yaxis_title="Index",
    yaxis=dict(range=[0.8, 1.2]),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=40, r=20, t=60, b=80),
)

fig.write_html(EV_PLOT_HTML)

# ---------------------------------------------------------
# BEI TABLE (Penske)
# ---------------------------------------------------------
# Assumptions (leave comment for SME):
# - Baseline Finish <= snapshot
# - Completed when Actual Finish not null
# - Exclude Activity_Type in ['A', 'B'] (LOE / Milestones)
bei_mask = (
    penske[P_BF_COL].notna()
    & (penske[P_BF_COL] <= SNAPSHOT_DATE)
    & (~penske[P_TYPE_COL].isin(["A", "B"]))
)
penske_be = penske.loc[bei_mask].copy()

tasks_total = penske_be.groupby(P_GROUP_COL)[P_ID_COL].count()
tasks_done = penske_be[penske_be[P_AF_COL].notna()].groupby(P_GROUP_COL)[P_ID_COL].count()
tasks_done = tasks_done.reindex(tasks_total.index).fillna(0)

bei_tbl = pd.DataFrame(
    {
        "SubTeam": tasks_total.index,
        "Tasks w/ Baseline Finish ≤ Snapshot": tasks_total.values,
        "Completed Tasks": tasks_done.values,
    }
)
den = bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"].astype(float).values
num = bei_tbl["Completed Tasks"].astype(float).values
bei_vals = np.where(den == 0, np.nan, num / den)
bei_tbl["BEI"] = np.round(bei_vals, 2)

# ---------------------------------------------------------
# COLOR HELPERS FOR TABLES
# ---------------------------------------------------------
HEX = {
    "GREEN": "#00B050",
    "YELLOW": "#FFC000",
    "RED": "#C00000",
    "BLUE": "#1F4E79",
}

def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def spi_cpi_color(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    v = float(val)
    if v < 0.90:
        bg, fg = HEX["RED"], "#FFFFFF"
    elif v < 0.95:
        bg, fg = HEX["YELLOW"], "#000000"
    elif v <= 1.05:
        bg, fg = HEX["GREEN"], "#FFFFFF"
    else:
        bg, fg = HEX["BLUE"], "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

def vac_color(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    v = float(val)
    if v < 0:
        bg, fg = HEX["RED"], "#FFFFFF"
    elif v == 0:
        bg, fg = HEX["YELLOW"], "#000000"
    else:
        bg, fg = HEX["GREEN"], "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

def bei_color(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    v = float(val)
    if v < 0.90:
        bg, fg = HEX["RED"], "#FFFFFF"
    elif v < 0.95:
        bg, fg = HEX["YELLOW"], "#000000"
    else:
        bg, fg = HEX["GREEN"], "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

# ---------------------------------------------------------
# POWERPOINT BUILD
# ---------------------------------------------------------
prs = Presentation(THEME_PATH)

# Title slide layout: first layout that has a TITLE placeholder
title_layout_idx = None
for idx, layout in enumerate(prs.slide_layouts):
    if any(ph.placeholder_format.type == 0 for ph in layout.placeholders):
        title_layout_idx = idx
        break
if title_layout_idx is None:
    title_layout_idx = 0

# Table slide layout: use a content layout (often index 1)
table_layout_idx = 1 if len(prs.slide_layouts) > 1 else 0

def add_table_slide(title, df, color_cols=None):
    """Add a slide with title and df table. color_cols maps column->color_fn."""
    if color_cols is None:
        color_cols = {}

    slide = prs.slides.add_slide(prs.slide_layouts[table_layout_idx])

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

    # Data rows
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, colname in enumerate(df.columns):
            val = row[colname]
            text = "" if pd.isna(val) else str(val)
            cell = table.cell(i, j)
            cell.text = text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)

            color_fn = color_cols.get(colname)
            if color_fn:
                bg_rgb, fg_rgb = color_fn(val)
                if bg_rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*bg_rgb)
                if fg_rgb:
                    p.font.color.rgb = RGBColor(*fg_rgb)

    return slide

# ---- Slide 1: Title ----
title_slide = prs.slides.add_slide(prs.slide_layouts[title_layout_idx])

if title_slide.shapes.title:
    title_slide.shapes.title.text = PROGRAM
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(0.6), Inches(8), Inches(1))
    tx.text_frame.text = PROGRAM

subtitle_shape = None
for shp in title_slide.placeholders:
    if shp.placeholder_format.type == 1:  # SUBTITLE
        subtitle_shape = shp
        break

subtitle_text = SNAPSHOT_DATE.strftime("%Y-%m-%d")
if subtitle_shape:
    subtitle_shape.text = subtitle_text
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(4), Inches(0.6))
    tx.text_frame.text = subtitle_text

# ---- Slide 2: EVMS Metrics + link ----
evms_colors = {"4WK": spi_cpi_color, "YTD": spi_cpi_color, "CTD": spi_cpi_color}
slide2 = add_table_slide("EVMS Metrics Table", evms_metrics_tbl, color_cols=evms_colors)

# Add single link textbox ONLY on this slide
html_abs_path = os.path.abspath(EV_PLOT_HTML)
link_box = slide2.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(7.5), Inches(0.4))
pf = link_box.text_frame
p = pf.paragraphs[0]
p.text = "Click here to open the EVMS Chart (HTML)"
run = p.runs[0]
run.font.color.rgb = RGBColor(0, 0, 192)
run.underline = True
run.hyperlink.address = html_abs_path

# ---- Slide 3: Cost Performance Index Table ----
cpi_colors = {"YTD": spi_cpi_color, "CTD": spi_cpi_color}
add_table_slide("Cost Performance Index Table", cost_performance_tbl, color_cols=cpi_colors)

# ---- Slide 4: Schedule Performance Index Table ----
spi_colors = {"YTD": spi_cpi_color, "CTD": spi_cpi_color}
add_table_slide("Schedule Performance Index Table", schedule_performance_tbl, color_cols=spi_colors)

# ---- Slide 5: Labor Hours Table ----
labor_colors = {"VAC": vac_color}
add_table_slide("Labor Hours Table", labor_tbl, color_cols=labor_colors)

# ---- Slide 6: Monthly Labor Ratios Table ----
ratio_colors = {"VAC/BAC": vac_color}
add_table_slide("Monthly Labor Ratios Table", labor_monthly_tbl, color_cols=ratio_colors)

# ---- Slide 7: BEI Table ----
bei_colors = {"BEI": bei_color}
add_table_slide("Baseline Execution Index (BEI) Table", bei_tbl, color_cols=bei_colors)

# SAVE
prs.save(OUTPUT_PPTX)
print("Presentation saved:", OUTPUT_PPTX)
print("EVMS HTML chart saved:", EV_PLOT_HTML)