# XM30 EVMS + BEI PIPELINE  (one cell)

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from IPython.display import display

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
PROGRAM          = "XM30"
COBRA_PATH       = "data/Cobra-XM30.xlsx"
COBRA_SHEET_NAME = "tbl_Weekly Extract"   # tall table with DATE, SUB_TEAM, COST-SET, HOURS
PENSKE_PATH      = "data/OpenPlan_Activity-Penske.xlsx"
THEME_PATH       = "Theme.pptx"
OUTPUT_PPTX      = f"{PROGRAM}_EVMS_Update.pptx"

GROUP_COL        = "SUB_TEAM"
DATE_COL         = "DATE"
COSTSET_COL      = "COST-SET"
HOURS_COL        = "HOURS"
ANCHOR           = datetime.now()         # snapshot date

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
xl_cobra = pd.ExcelFile(COBRA_PATH)
df = xl_cobra.parse(COBRA_SHEET_NAME)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df[df[DATE_COL].notna()].copy()

# Expected cost set labels
COST_SETS = ["ACWP", "BCWP", "BCWS", "ETC"]

# -------------------------------------------------------------------
# HELPER: wide table by SUB_TEAM x COST-SET (up to ANCHOR)
# -------------------------------------------------------------------
def rollup_costs(dframe, start=None, end=None):
    m = dframe[DATE_COL].notna()
    if start is not None:
        m &= dframe[DATE_COL] >= start
    if end is not None:
        m &= dframe[DATE_COL] <= end

    g = (
        dframe.loc[m]
        .groupby([GROUP_COL, COSTSET_COL], dropna=False)[HOURS_COL]
        .sum()
        .unstack(fill_value=0.0)
    )

    # Ensure all expected cost sets exist
    for k in COST_SETS:
        if k not in g.columns:
            g[k] = 0.0

    return g[COST_SETS].astype(float)

# -------------------------------------------------------------------
# LABOR TABLE (HOURS, BAC/EAC, VAC)
# -------------------------------------------------------------------
g_ctd = rollup_costs(df, end=ANCHOR)

BAC = g_ctd["BCWS"]
EAC = g_ctd["ACWP"] + g_ctd["ETC"]
VAC = BAC - EAC
XCOMP = np.where(BAC.eq(0), np.nan, (g_ctd["BCWP"] / BAC) * 100)

labor_tbl = pd.DataFrame(
    {
        "BAC": BAC,
        "ACWP": g_ctd["ACWP"],
        "BCWP": g_ctd["BCWP"],
        "ETC": g_ctd["ETC"],
        "EAC": EAC,
        "VAC": VAC,
        "%COMP": np.round(XCOMP, 1),
    }
)

labor_tbl.index.name = GROUP_COL

# Totals row
tot = g_ctd.sum()
tot_bac = tot["BCWS"]
tot_eac = tot["ACWP"] + tot["ETC"]
tot_vac = tot_bac - tot_eac
tot_pcmp = np.nan if np.isclose(tot_bac, 0) else round((tot["BCWP"] / tot_bac) * 100, 1)

labor_tbl.loc["TOTAL"] = [
    tot_bac,
    tot["ACWP"],
    tot["BCWP"],
    tot["ETC"],
    tot_eac,
    tot_vac,
    tot_pcmp,
]

labor_tbl_reset = labor_tbl.reset_index().rename(columns={GROUP_COL: "SUB_TEAM"})

# -------------------------------------------------------------------
# MONTHLY LABOR TABLE (RATIOS)
# -------------------------------------------------------------------
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
)
labor_monthly_tbl.set_index("SUB_TEAM", inplace=True)

# -------------------------------------------------------------------
# COST & SCHEDULE PERFORMANCE TABLES (CPI, SPI – CTD & YTD)
# -------------------------------------------------------------------
ytd_start = datetime(ANCHOR.year, 1, 1)

g_ytd = rollup_costs(df, start=ytd_start, end=ANCHOR)

def build_cpi_spi_tables(ctd, ytd):
    # CPI
    cpi_ctd = np.where(ctd["ACWP"].eq(0), np.nan, ctd["BCWP"] / ctd["ACWP"])
    cpi_ytd = np.where(ytd["ACWP"].eq(0), np.nan, ytd["BCWP"] / ytd["ACWP"])
    cost_performance_tbl = pd.DataFrame({"YTD": cpi_ytd, "CTD": cpi_ctd})

    # TOTAL row
    tot_ctd = ctd.sum()
    tot_ytd = ytd.sum()
    tot_ctd_cpi = np.nan if np.isclose(tot_ctd["ACWP"], 0) else round(tot_ctd["BCWP"] / tot_ctd["ACWP"], 2)
    tot_ytd_cpi = np.nan if np.isclose(tot_ytd["ACWP"], 0) else round(tot_ytd["BCWP"] / tot_ytd["ACWP"], 2)
    cost_performance_tbl.loc["TOTAL"] = [tot_ytd_cpi, tot_ctd_cpi]

    # SPI
    spi_ctd = np.where(ctd["BCWS"].eq(0), np.nan, ctd["BCWP"] / ctd["BCWS"])
    spi_ytd = np.where(ytd["BCWS"].eq(0), np.nan, ytd["BCWP"] / ytd["BCWS"])
    schedule_performance_tbl = pd.DataFrame({"YTD": spi_ytd, "CTD": spi_ctd})

    tot_ctd_spi = np.nan if np.isclose(tot_ctd["BCWS"], 0) else round(tot_ctd["BCWP"] / tot_ctd["BCWS"], 2)
    tot_ytd_spi = np.nan if np.isclose(tot_ytd["BCWS"], 0) else round(tot_ytd["BCWP"] / tot_ytd["BCWS"], 2)
    schedule_performance_tbl.loc["TOTAL"] = [tot_ytd_spi, tot_ctd_spi]

    return cost_performance_tbl, schedule_performance_tbl

cost_performance_tbl, schedule_performance_tbl = build_cpi_spi_tables(g_ctd, g_ytd)

# -------------------------------------------------------------------
# EVMS METRICS TABLE (4WK, YTD, CTD – SPI & CPI, program level)
# -------------------------------------------------------------------
w4_start = ANCHOR - timedelta(weeks=4)

def sums_for_window(dframe, start=None, end=None):
    m = dframe[DATE_COL].notna()
    if start is not None:
        m &= dframe[DATE_COL] >= start
    if end is not None:
        m &= dframe[DATE_COL] <= end

    g = (
        dframe.loc[m]
        .groupby(COSTSET_COL, dropna=False)[HOURS_COL]
        .sum()
    )
    # ensure keys
    s = {k: float(g.get(k, 0.0)) for k in COST_SETS}
    return s

ctd = sums_for_window(df, end=ANCHOR)
ytd = sums_for_window(df, start=ytd_start, end=ANCHOR)
w4  = sums_for_window(df, start=w4_start, end=ANCHOR)

def safe_ratio(num, den):
    return np.nan if np.isclose(den, 0) else num / den

# SPI = BCWP / BCWS
spi_ctd = safe_ratio(ctd["BCWP"], ctd["BCWS"])
spi_ytd = safe_ratio(ytd["BCWP"], ytd["BCWS"])
spi_4wk = safe_ratio(w4["BCWP"], w4["BCWS"])

# CPI = BCWP / ACWP
cpi_ctd = safe_ratio(ctd["BCWP"], ctd["ACWP"])
cpi_ytd = safe_ratio(ytd["BCWP"], ytd["ACWP"])
cpi_4wk = safe_ratio(w4["BCWP"], w4["ACWP"])

evms_metrics_tbl = pd.DataFrame(
    {
        "4WK": [spi_4wk, cpi_4wk],
        "YTD": [spi_ytd, cpi_ytd],
        "CTD": [spi_ctd, cpi_ctd],
    },
    index=["SPI", "CPI"],
).round(2)

# -------------------------------------------------------------------
# EVMS PLOT (Monthly SPI/CPI and cumulative)
# -------------------------------------------------------------------
cobra = xl_cobra.parse(COBRA_SHEET_NAME)
cobra[DATE_COL] = pd.to_datetime(cobra[DATE_COL], errors="coerce")
cobra = cobra[cobra[DATE_COL].notna()].copy()
cobra = cobra[cobra[DATE_COL] <= ANCHOR].copy()

month_totals = (
    cobra.groupby([cobra[DATE_COL].dt.to_period("M"), COSTSET_COL])[HOURS_COL]
    .sum()
    .unstack(fill_value=0.0)
)
for k in COST_SETS:
    if k not in month_totals.columns:
        month_totals[k] = 0.0

month_totals_index = month_totals.index.to_timestamp("M")
ev = month_totals.copy().sort_index()

# Monthly SPI / CPI
ev["CPI_month"] = np.where(ev["ACWP"].eq(0), np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"].eq(0), np.nan, ev["BCWP"] / ev["BCWS"])

# Cumulative
ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

ev["CPI_cum"] = np.where(ev["ACWP_cum"].eq(0), np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"].eq(0), np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

evms_tbl = ev[["CPI_month", "SPI_month", "CPI_cum", "SPI_cum"]].copy()
evms_tbl.index = month_totals_index
evms_tbl_reset = evms_tbl.reset_index().rename(columns={"index": "Month"})
evms_tbl_reset["Month"] = evms_tbl_reset[DATE_COL].dt.strftime("%b-%y")

# Plot
x_vals = evms_tbl_reset["Month"]

fig = go.Figure()

# Background performance bands
fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#ff4444", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=0.95, y1=1.05, fillcolor="#ff6633", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=1.05, y1=1.20, fillcolor="#66cc66", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=0.0,  y1=0.90, fillcolor="#99ccff", opacity=0.3, line_width=0, layer="below")

# Monthly CPI
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl_reset["CPI_month"],
        mode="markers",
        name="Monthly CPI",
        marker=dict(symbol="diamond", size=10, color="#f4b183"),
    )
)

# Monthly SPI
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl_reset["SPI_month"],
        mode="markers",
        name="Monthly SPI",
        marker=dict(symbol="circle", size=10, color="#000000"),
    )
)

# Cumulative CPI
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl_reset["CPI_cum"],
        mode="lines",
        name="Cumulative CPI",
        line=dict(color="#0050b3", width=3),
    )
)

# Cumulative SPI
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl_reset["SPI_cum"],
        mode="lines",
        name="Cumulative SPI",
        line=dict(color="#7f7fff", width=3),
    )
)

fig.update_layout(
    title="EV Indices",
    xaxis_title="Month",
    yaxis_title="Index",
    yaxis=dict(range=[0.9, 1.2]),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=40, r=20, t=60, b=80),
)

# Save chart to PNG for PowerPoint (requires 'kaleido' installed)
EV_PLOT_PATH = "evms_chart.png"
try:
    fig.write_image(EV_PLOT_PATH, width=1200, height=700, scale=2)
except Exception as e:
    print("WARNING: Could not export EV plot image. Install 'kaleido' if you need it in PowerPoint.")
    EV_PLOT_PATH = None

# -------------------------------------------------------------------
# BEI TABLE (Penske OpenPlan data)
# -------------------------------------------------------------------
penske_data = pd.read_excel(PENSKE_PATH)

# Assumptions for BEI (PLEASE CONFIRM WITH SME – SEE COMMENT BELOW):
# - Baseline date column: 'Baseline Finish'
# - Task is considered Completed when 'Actual Finish' is not null
# - Exclude Activity_Type 'A' (LOE) and 'B' (Milestone)
# - Grouping for BEI done by 'SubTeam'
#
# TODO (FOR SME): Confirm that BEI definition here matches contract guidance:
#   BEI = (Completed tasks) / (Total tasks with a Baseline Finish on/before snapshot date),
#   where completion is based on non-null 'Actual Finish' and excluding Activity_Type in ['A','B'].

penske_data["Baseline Finish"] = pd.to_datetime(penske_data["Baseline Finish"], errors="coerce")
penske_data["Actual Finish"] = pd.to_datetime(penske_data["Actual Finish"], errors="coerce")

penske_filtered = penske_data[
    (penske_data["Baseline Finish"].notna())
    & (penske_data["Baseline Finish"] <= ANCHOR)
    & (~penske_data["Activity_Type"].isin(["A", "B"]))
].copy()

group_field = "SubTeam"  # can change to 'Planner', 'CAM', etc. if preferred

total_tasks = (
    penske_filtered.groupby(group_field)["Activity ID"]
    .count()
    .rename("Tasks w/ Baseline Finish ≤ Snapshot")
)

completed_tasks = (
    penske_filtered[penske_filtered["Actual Finish"].notna()]
    .groupby(group_field)["Activity ID"]
    .count()
    .rename("Completed Tasks")
)

bei_tbl = pd.concat([total_tasks, completed_tasks], axis=1).fillna(0)
bei_tbl["BEI"] = np.where(
    bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"].eq(0),
    np.nan,
    bei_tbl["Completed Tasks"] / bei_tbl["Tasks w/ Baseline Finish ≤ Snapshot"],
)
bei_tbl = bei_tbl.reset_index().rename(columns={group_field: "SubTeam"})
bei_tbl["BEI"] = bei_tbl["BEI"].round(2)

# -------------------------------------------------------------------
# COLOR HELPERS FOR TABLES
# -------------------------------------------------------------------
HEX_COLORS = {
    "BLUE":   "#3E4E8F",
    "GREEN":  "#339966",
    "YELLOW": "#FFCC00",
    "RED":    "#FF0000",
    "WHITE":  "#FFFFFF",
    "BLACK":  "#000000",
}

def hex_to_rgb(hex_color):
    if not hex_color:
        return None
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return None
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_spi_cpi_color(value):
    """Return fill/text RGB tuples for SPI/CPI-like indices."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    v = float(value)
    # thresholds: <0.9 red, 0.9-0.95 yellow, 0.95-1.05 green, >1.05 blue
    if v < 0.9:
        bg = HEX_COLORS["RED"]
        txt = HEX_COLORS["WHITE"]
    elif v < 0.95:
        bg = HEX_COLORS["YELLOW"]
        txt = HEX_COLORS["BLACK"]
    elif v <= 1.05:
        bg = HEX_COLORS["GREEN"]
        txt = HEX_COLORS["WHITE"]
    else:
        bg = HEX_COLORS["BLUE"]
        txt = HEX_COLORS["WHITE"]
    return hex_to_rgb(bg), hex_to_rgb(txt)

def get_vac_color(value):
    """Color VAC / VAC ratios (negative bad, positive good)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    v = float(value)
    if v < 0:   # overrun risk
        bg = HEX_COLORS["RED"]
        txt = HEX_COLORS["WHITE"]
    elif v == 0:
        bg = HEX_COLORS["YELLOW"]
        txt = HEX_COLORS["BLACK"]
    else:
        bg = HEX_COLORS["GREEN"]
        txt = HEX_COLORS["WHITE"]
    return hex_to_rgb(bg), hex_to_rgb(txt)

# -------------------------------------------------------------------
# POWERPOINT HELPERS
# -------------------------------------------------------------------
def add_table_slide(prs, title, df, layout_index=5, fmt=None, color_map=None):
    """
    Add a slide with a title and a table built from df.
    fmt: optional dict col_name -> format string, e.g. {"SPI": "{:.2f}"}
    color_map: optional dict col_name -> color_function(value)->(bg_rgb, txt_rgb)
    """
    if fmt is None:
        fmt = {}
    if color_map is None:
        color_map = {}

    layout_index = min(layout_index, len(prs.slide_layouts) - 1)
    slide = prs.slides.add_slide(prs.slide_layouts[layout_index])

    # Title
    try:
        title_ph = slide.shapes.title
    except Exception:
        title_ph = None

    if title_ph is not None:
        title_ph.text = title

    rows, cols = df.shape
    rows += 1  # header row

    left = Inches(0.5)
    top = Inches(1.2)
    width = prs.slide_width - Inches(1.0)
    height = prs.slide_height - Inches(1.8)

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # headers
    for j, col_name in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col_name)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(12)

    # body
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col_name in enumerate(df.columns):
            val = row[col_name]
            fmt_str = fmt.get(col_name)
            try:
                if fmt_str and (isinstance(val, (int, float, np.number)) or pd.api.types.is_number(val)):
                    txt = fmt_str.format(val)
                else:
                    txt = "" if pd.isna(val) else str(val)
            except Exception:
                txt = "" if pd.isna(val) else str(val)

            cell = table.cell(i, j)
            cell.text = txt
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)

            # conditional color
            color_fn = color_map.get(col_name)
            if color_fn is not None:
                bg_rgb, txt_rgb = color_fn(val)
                if bg_rgb is not None:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*bg_rgb)
                if txt_rgb is not None:
                    p.font.color.rgb = RGBColor(*txt_rgb)

    return slide

# -------------------------------------------------------------------
# BUILD POWERPOINT
# -------------------------------------------------------------------
if os.path.exists(THEME_PATH):
    prs = Presentation(THEME_PATH)
else:
    print(f"WARNING: Theme file '{THEME_PATH}' not found. Using default PowerPoint template.")
    prs = Presentation()

# 1) TITLE SLIDE (using theme's title layout)
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1] if len(slide.placeholders) > 1 else None

if title is not None:
    title.text = f"{PROGRAM} EVMS Update"
if subtitle is not None:
    subtitle.text = f"Snapshot as of {ANCHOR.strftime('%Y-%m-%d')}"

# 2) EVMS PLOT SLIDE
layout_index_plot = 5 if len(prs.slide_layouts) > 5 else 1
plot_slide = prs.slides.add_slide(prs.slide_layouts[layout_index_plot])
try:
    plot_title = plot_slide.shapes.title
except Exception:
    plot_title = None
if plot_title is not None:
    plot_title.text = "EV Indices (SPI/CPI)"

if EV_PLOT_PATH and os.path.exists(EV_PLOT_PATH):
    left = Inches(0.5)
    top = Inches(1.0)
    width = prs.slide_width - Inches(1.0)
    plot_slide.shapes.add_picture(EV_PLOT_PATH, left, top, width=width)
else:
    # If no image, drop a simple note
    tx_box = plot_slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1))
    tf = tx_box.text_frame
    tf.text = "EVMS plot image could not be generated (check 'kaleido' installation)."

# 3) EVMS METRICS TABLE SLIDE
evms_fmt = {"4WK": "{:.2f}", "YTD": "{:.2f}", "CTD": "{:.2f}"}
evms_color_map = {"4WK": get_spi_cpi_color, "YTD": get_spi_cpi_color, "CTD": get_spi_cpi_color}
add_table_slide(prs, "EVMS Metrics (SPI & CPI – 4WK/YTD/CTD)", evms_metrics_tbl, fmt=evms_fmt, color_map=evms_color_map)

# 4) COST PERFORMANCE TABLE (CPI)
cpi_fmt = {"YTD": "{:.2f}", "CTD": "{:.2f}"}
cpi_color_map = {"YTD": get_spi_cpi_color, "CTD": get_spi_cpi_color}
add_table_slide(prs, "Cost Performance Index (CPI)", cost_performance_tbl, fmt=cpi_fmt, color_map=cpi_color_map)

# 5) SCHEDULE PERFORMANCE TABLE (SPI)
spi_fmt = {"YTD": "{:.2f}", "CTD": "{:.2f}"}
spi_color_map = {"YTD": get_spi_cpi_color, "CTD": get_spi_cpi_color}
add_table_slide(prs, "Schedule Performance Index (SPI)", schedule_performance_tbl, fmt=spi_fmt, color_map=spi_color_map)

# 6) LABOR HOURS TABLE
labor_fmt = {
    "BAC": "{:,.0f}",
    "ACWP": "{:,.0f}",
    "BCWP": "{:,.0f}",
    "ETC": "{:,.0f}",
    "EAC": "{:,.0f}",
    "VAC": "{:,.0f}",
    "%COMP": "{:.1f}",
}
labor_color_map = {"VAC": get_vac_color}
add_table_slide(prs, "Labor Hours (BAC/ACWP/BCWP/ETC/EAC/VAC)", labor_tbl_reset, fmt=labor_fmt, color_map=labor_color_map)

# 7) MONTHLY LABOR HOURS TABLE (RATIOS)
labor_monthly_fmt = {"BAC/EAC": "{:.2f}", "VAC/BAC": "{:.2f}"}
labor_monthly_color_map = {"VAC/BAC": get_vac_color}
add_table_slide(prs, "Monthly Labor Ratios (BAC/EAC & VAC/BAC)", labor_monthly_tbl.reset_index(), fmt=labor_monthly_fmt, color_map=labor_monthly_color_map)

# 8) BEI TABLE SLIDE
bei_fmt = {
    "Tasks w/ Baseline Finish ≤ Snapshot": "{:.0f}",
    "Completed Tasks": "{:.0f}",
    "BEI": "{:.2f}",
}
add_table_slide(prs, "Baseline Execution Index (BEI)", bei_tbl, fmt=bei_fmt)

# -------------------------------------------------------------------
# SAVE PRESENTATION
# -------------------------------------------------------------------
prs.save(OUTPUT_PPTX)
print(f"PowerPoint saved to: {OUTPUT_PPTX}")

# Optional: display key tables in notebook
print("\nEVMS Metrics Table:")
display(evms_metrics_tbl)

print("\nCost Performance Table (CPI):")
display(cost_performance_tbl)

print("\nSchedule Performance Table (SPI):")
display(schedule_performance_tbl)

print("\nLabor Hours Table:")
display(labor_tbl_reset)

print("\nMonthly Labor Ratios Table:")
display(labor_monthly_tbl)

print("\nBEI Table:")
display(bei_tbl)