# =========================================================
# XM30 EVMS + BEI – Single-Cell PPT Generator (with colors)
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

def round_df(df, decimals=2, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    num_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    df[num_cols] = df[num_cols].round(decimals)
    return df

# color helpers
HEX = {
    "GREEN": "#00B050",
    "YELLOW": "#FFC000",
    "RED": "#C00000",
    "BLUE": "#1F4E79",
}

def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def safe_color_val(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    return float(v)

def spi_cpi_color(val):
    v = safe_color_val(val)
    if v is None:
        return None, None
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
    v = safe_color_val(val)
    if v is None:
        return None, None
    if v < 0:
        bg, fg = HEX["RED"], "#FFFFFF"
    elif v == 0:
        bg, fg = HEX["YELLOW"], "#000000"
    else:
        bg, fg = HEX["GREEN"], "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

def bei_color(val):
    v = safe_color_val(val)
    if v is None:
        return None, None
    if v < 0.90:
        bg, fg = HEX["RED"], "#FFFFFF"
    elif v < 0.95:
        bg, fg = HEX["YELLOW"], "#000000"
    else:
        bg, fg = HEX["GREEN"], "#FFFFFF"
    return hex_to_rgb(bg), hex_to_rgb(fg)

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
        "%COMP": PCOMP,
    }
)

# TOTAL row
tot_bac = ctd_tot["BCWS"]
tot_acw = ctd_tot["ACWP"]
tot_bcw = ctd_tot["BCWP"]
tot_etc = ctd_tot["ETC"]
tot_eac = tot_acw + tot_etc
tot_vac = tot_bac - tot_eac
tot_pcomp = np.nan if tot_bac == 0 else tot_bcw / tot_bac * 100.0

labor_tbl.loc[len(labor_tbl)] = [
    "TOTAL", tot_bac, tot_acw, tot_bcw, tot_etc, tot_eac, tot_vac, tot_pcomp
]

labor_tbl = round_df(labor_tbl, 2, exclude_cols=["SUB_TEAM"])

# ---- Monthly Labor Ratios Table ----
labor_monthly_tbl = pd.DataFrame(
    {
        "SUB_TEAM": labor_tbl["SUB_TEAM"],
        "BAC/EAC": np.where(
            labor_tbl["EAC"] == 0, np.nan, labor_tbl["BAC"] / labor_tbl["EAC"]
        ),
        "VAC/BAC": np.where(
            labor_tbl["BAC"] == 0, np.nan, labor_tbl["VAC"] / labor_tbl["BAC"]
        ),
    }
)
labor_monthly_tbl = round_df(labor_monthly_tbl, 2, exclude_cols=["SUB_TEAM"])

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
            "YTD": cpi_ytd,
            "CTD": cpi_ctd,
        }
    )
    sched_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ctd_df.index,
            "YTD": spi_ytd,
            "CTD": spi_ctd,
        }
    )

    # Totals
    y_ACWP_sum = ytd_df["ACWP"].sum()
    y_BCWP_sum = ytd_df["BCWP"].sum()
    y_BCWS_sum = ytd_df["BCWS"].sum()
    c_ACWP_sum = ctd_df["ACWP"].sum()
    c_BCWP_sum = ctd_df["BCWP"].sum()
    c_BCWS_sum = ctd_df["BCWS"].sum()

    cost_total_ytd = safe_div_scalar(y_BCWP_sum, y_ACWP_sum)
    cost_total_ctd = safe_div_scalar(c_BCWP_sum, c_ACWP_sum)
    sched_total_ytd = safe_div_scalar(y_BCWP_sum, y_BCWS_sum)
    sched_total_ctd = safe_div_scalar(c_BCWP_sum, c_BCWS_sum)

    cost_tbl.loc[len(cost_tbl)] = ["TOTAL", cost_total_ytd, cost_total_ctd]
    sched_tbl.loc[len(sched_tbl)] = ["TOTAL", sched_total_ytd, sched_total_ctd]

    cost_tbl = round_df(cost_tbl, 2, exclude_cols=["SUB_TEAM"])
    sched_tbl = round_df(sched_tbl, 2, exclude_cols=["SUB_TEAM"])
    return cost_tbl, sched_tbl

cost_performance_tbl, schedule_performance_tbl = build_cpi_spi(ctd_by_group, ytd_by_group)

# ---- EVMS Metrics (Program Level) ----
spi_4wk = safe_div_scalar(w4_tot["BCWP"],  w4_tot["BCWS"])
cpi_4wk = safe_div_scalar(w4_tot["BCWP"],  w4_tot["ACWP"])
spi_ytd = safe_div_scalar(ytd_tot["BCWP"], ytd_tot["BCWS"])
cpi_ytd = safe_div_scalar(ytd_tot["BCWP"], ytd_tot["ACWP"])
spi_ctd = safe_div_scalar(ctd_tot["BCWP"], ctd_tot["BCWS"])
cpi_ctd = safe_div_scalar(ctd_tot["BCWP"], ctd_tot["ACWP"])

evms_metrics_tbl = pd.DataFrame(
    {
        "Metric": ["SPI", "CPI"],
        "4WK": [spi_4wk, cpi_4wk],
        "YTD": [spi_ytd, cpi_ytd],
        "CTD": [spi_ctd, cpi_ctd],
    }
)
evms_metrics_tbl = round_df(evms_metrics_tbl, 2, exclude_cols=["Metric"])

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
fig.add_hrect(y0=0.0,  y1=0.90, fillcolor="#C00000", opacity=0.2, line_width=0, layer="below")
fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#FFC000", opacity=0.2, line_width=0, layer="below")
fig.add_hrect(y0=0.95, y1=1.05, fillcolor="#00B050", opacity=0.2, line_width=0, layer="below")
fig.add_hrect(y0=1.05, y1=1.20, fillcolor="#1F4E79", opacity=0.15, line_width=0, layer="below")

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
bei_mask = (
    penske[P_BF_COL].notna()
    & (penske[P_BF_COL] <= SNAPSHOT_DATE)
    & (~penske[P_TYPE_COL].isin(["A", "B"]))   # SME assumption
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
bei_tbl["BEI"] = bei_vals
bei_tbl = round_df(bei_tbl, 2, exclude_cols=["SubTeam"])

# ---------------------------------------------------------
# POWERPOINT BUILD
# ---------------------------------------------------------
prs = Presentation(THEME_PATH)

# --- Title slide: use existing first slide if present ---
if len(prs.slides) > 0:
    title_slide = prs.slides[0]
else:
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])

# program title in title placeholder
if title_slide.shapes.title:
    title_slide.shapes.title.text = PROGRAM
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(0.6), Inches(8), Inches(1))
    tx.text_frame.text = PROGRAM

# date in subtitle placeholder (or textbox if missing)
subtitle_shape = None
for shp in title_slide.placeholders:
    if shp is not title_slide.shapes.title and shp.has_text_frame:
        subtitle_shape = shp
        break

subtitle_text = SNAPSHOT_DATE.strftime("%Y-%m-%d")
if subtitle_shape:
    subtitle_shape.text = subtitle_text
else:
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(4), Inches(0.6))
    tx.text_frame.text = subtitle_text

# --- choose a "blank-ish" layout for data slides ---
blank_layout_idx = None
for idx, layout in enumerate(prs.slide_layouts):
    if len(layout.placeholders) == 0:
        blank_layout_idx = idx
        break
if blank_layout_idx is None:
    # fallback: use same as title layout but we will delete placeholders
    blank_layout_idx = title_slide.slide_layout.slide_layout_id

def add_clean_table_slide(title, df, color_cols=None):
    """Add slide with no extra text boxes: blank layout + title textbox + table."""
    if color_cols is None:
        color_cols = {}

    slide = prs.slides.add_slide(prs.slide_layouts[blank_layout_idx])

    # remove any placeholders that might exist on this layout
    for shp in list(slide.shapes):
        if shp.is_placeholder:
            slide.shapes._spTree.remove(shp._element)

    # title box
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(0.3),
                                  prs.slide_width - Inches(1.2), Inches(0.6))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.bold = True
    p.font.size = Pt(24)

    # table
    rows, cols = df.shape
    rows += 1
    left = Inches(0.4)
    top = Inches(1.1)
    width = prs.slide_width - Inches(0.8)
    height = prs.slide_height - Inches(1.6)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # header
    for j, colname in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(colname)
        hp = cell.text_frame.paragraphs[0]
        hp.font.bold = True
        hp.font.size = Pt(11)

    # body
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, colname in enumerate(df.columns):
            val = row[colname]
            txt = "" if pd.isna(val) else str(val)
            cell = table.cell(i, j)
            cell.text = txt
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)

            fn = color_cols.get(colname)
            if fn:
                bg_rgb, fg_rgb = fn(val)
                if bg_rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*bg_rgb)
                if fg_rgb:
                    p.font.color.rgb = RGBColor(*fg_rgb)

    return slide

# ---- Slide 2: EVMS Metrics + EV chart link ----
evms_colors = {"4WK": spi_cpi_color, "YTD": spi_cpi_color, "CTD": spi_cpi_color}
slide2 = add_clean_table_slide("EVMS Metrics Table", evms_metrics_tbl, evms_colors)

# hyperlink at bottom; this is the ONLY extra textbox on data slides
html_abs_path = os.path.abspath(EV_PLOT_HTML)
link_box = slide2.shapes.add_textbox(Inches(0.8), Inches(6.2),
                                     Inches(7.5), Inches(0.4))
p = link_box.text_frame.paragraphs[0]
p.text = "Click here to open the EVMS HTML Chart"
run = p.runs[0]
run.font.color.rgb = RGBColor(0, 0, 192)
run.underline = True
run.hyperlink.address = html_abs_path

# ---- Slide 3: Cost Performance Index ----
cpi_colors = {"YTD": spi_cpi_color, "CTD": spi_cpi_color}
add_clean_table_slide("Cost Performance Index Table", cost_performance_tbl, cpi_colors)

# ---- Slide 4: Schedule Performance Index ----
spi_colors = {"YTD": spi_cpi_color, "CTD": spi_cpi_color}
add_clean_table_slide("Schedule Performance Index Table", schedule_performance_tbl, spi_colors)

# ---- Slide 5: Labor Hours ----
labor_colors = {"VAC": vac_color}
add_clean_table_slide("Labor Hours Table", labor_tbl, labor_colors)

# ---- Slide 6: Monthly Labor Ratios ----
ratio_colors = {"BAC/EAC": spi_cpi_color, "VAC/BAC": vac_color}
add_clean_table_slide("Monthly Labor Ratios Table", labor_monthly_tbl, ratio_colors)

# ---- Slide 7: BEI ----
bei_colors = {"BEI": bei_color}
add_clean_table_slide("Baseline Execution Index (BEI) Table", bei_tbl, bei_colors)

# SAVE
prs.save(OUTPUT_PPTX)
print("Presentation saved:", OUTPUT_PPTX)
print("EVMS HTML chart saved:", EV_PLOT_HTML)