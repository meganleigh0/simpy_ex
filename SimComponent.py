# ================================================================
# FULL EVMS AUTOMATED DASHBOARD GENERATOR
# ================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import os

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

PROGRAMS = {
    "Abrams_STS": {
        "COBRA_PATH": "data/Cobra-Abrams STS.xlsx",
        "SHEET": "CAP_Extract",   # or your “tbl_Weekly Extract”
    },
    "Abrams_STS_2022": {
        "COBRA_PATH": "data/Cobra-Abrams STS 2022.xlsx",
        "SHEET": "CAP_Extract",
    },
    "XM30": {
        "COBRA_PATH": "data/Cobra-XM30.xlsx",
        "SHEET": "CAP_Extract",
    }
}

OPENPLAN_PATH = "data/OpenPlan_Activity-Penske.xlsx"
THEME_PATH = "data/theme.pptx"

DATE_COL = "DATE"
GROUP_COL = "SUB_TEAM"
P_ID = "Activity ID"

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNAPSHOT_DATE = datetime.now().replace(day=1)

# ---------------------------------------------------------------
# SME COLOR PALETTE
# ---------------------------------------------------------------

COLORS = {
    "blue":   RGBColor(31, 73, 125),
    "ltblue": RGBColor(142, 180, 227),
    "green":  RGBColor(51, 153, 102),
    "yellow": RGBColor(255, 255, 153),
    "red":    RGBColor(192, 80, 77),
}

# SME THRESHOLDS (SPI, CPI, BEI)
def color_perf(val):
    if pd.isna(val): 
        return None
    if val >= 1.05: return "blue"
    if 1.05 > val >= 0.98: return "green"
    if 0.98 > val >= 0.95: return "yellow"
    if val < 0.95: return "red"
    return None

# VAC thresholds
def color_vac(v):
    if v >= 0.055: return "blue"
    if 0.055 > v >= -0.02: return "green"
    if -0.02 > v >= -0.05: return "yellow"
    if v < -0.05: return "red"
    return None

# ---------------------------------------------------------------
# LOAD OPENPLAN FOR BEI
# ---------------------------------------------------------------
op = pd.read_excel(OPENPLAN_PATH)

op["Baseline Finish"] = pd.to_datetime(op["Baseline Finish"], errors="coerce")
op["Actual Finish"]   = pd.to_datetime(op["Actual Finish"], errors="coerce")
op["Activity_Type"]   = op["Activity_Type"].fillna("")

# Only A/B tasks
op = op[op["Activity_Type"].isin(["A", "B"])]

# ---------------------------------------------------------------
# FUNCTIONS TO COMPUTE EV / CPI / SPI
# ---------------------------------------------------------------

def compute_ev(cobra_df):
    g = cobra_df.groupby([GROUP_COL, "COST-SET"])["HOURS"].sum().unstack(fill_value=0)

    for k in ["ACWP", "BCWP", "BCWS", "ETC"]:
        if k not in g.columns:
            g[k] = 0.0

    g = g[["ACWP", "BCWP", "BCWS", "ETC"]]

    BAC = g["BCWS"]
    CTD_SPI = g["BCWP"] / g["BCWS"].replace(0, np.nan)
    CTD_CPI = g["BCWP"] / g["ACWP"].replace(0, np.nan)

    EAC  = g["ACWP"] + g["ETC"]
    VAC  = BAC - EAC
    PCOMP = g["BCWP"] / BAC.replace(0, np.nan)

    ev = pd.DataFrame({
        "BAC": BAC,
        "EAC": EAC,
        "VAC": VAC,
        "%COMP": PCOMP,
        "SPI_CTD": CTD_SPI,
        "CPI_CTD": CTD_CPI
    })
    return ev.reset_index()

# ---------------------------------------------------------------
# COMPUTE BEI FOR PROGRAM
# ---------------------------------------------------------------
def compute_bei(program_name):
    df = op[op["Program"] == program_name].copy()

    # CTD BEI
    df_ctd = df[df["Baseline Finish"] <= SNAPSHOT_DATE]
    total_tasks = df_ctd.groupby("SubTeam")[P_ID].count()
    completed = df_ctd[df_ctd["Actual Finish"].notna()].groupby("SubTeam")[P_ID].count()
    bei_ctd = (completed / total_tasks).fillna(0)

    # LSP BEI
    LSP = SNAPSHOT_DATE - timedelta(days=30)
    df_lsp = df[df["Baseline Finish"] <= LSP]
    total_lsp = df_lsp.groupby("SubTeam")[P_ID].count()
    comp_lsp = df_lsp[df_lsp["Actual Finish"].notna()].groupby("SubTeam")[P_ID].count()
    bei_lsp = (comp_lsp / total_lsp).fillna(0)

    out = pd.DataFrame({
        "SUB_TEAM": bei_ctd.index,
        "BEI_CTD": bei_ctd.values,
        "BEI_LSP": bei_lsp.reindex(bei_ctd.index).fillna(0).values
    })
    return out

# ---------------------------------------------------------------
# BUILD EV TREND PLOT
# ---------------------------------------------------------------
def make_ev_plot(df_dates, program):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_dates["DATE"], y=df_dates["SPI"], mode="lines+markers",
        name="SPI"
    ))

    fig.add_trace(go.Scatter(
        x=df_dates["DATE"], y=df_dates["CPI"], mode="lines+markers",
        name="CPI"
    ))

    # SME bands
    bands = [
        (1.05, 2.0, "blue"),
        (0.98, 1.05, "green"),
        (0.95, 0.98, "yellow"),
        (0.0,  0.95, "red")
    ]

    for low, high, col in bands:
        fig.add_shape(
            type="rect",
            x0=min(df_dates["DATE"]), x1=max(df_dates["DATE"]),
            y0=low, y1=high,
            fillcolor=f"rgb({COLORS[col].rgb[0]},{COLORS[col].rgb[1]},{COLORS[col].rgb[2]})",
            opacity=0.15, layer="below", line_width=0,
        )

    fig.update_layout(
        title=f"{program} – EV Trend",
        yaxis_title="Index",
        xaxis_title="Date",
        template="plotly_white"
    )
    plot_path = f"temp_{program}_ev_plot.png"
    fig.write_image(plot_path, scale=3)
    return plot_path

# ---------------------------------------------------------------
# APPLY COLORING TO PPT TABLE CELLS
# ---------------------------------------------------------------
def apply_color(cell, value, func):
    col = func(value)
    if col:
        rgb = COLORS[col]
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb

# ---------------------------------------------------------------
# MAIN LOOP — PROCESS EACH PROGRAM
# ---------------------------------------------------------------
for program, cfg in PROGRAMS.items():

    print(f"\nProcessing → {program}")

    # Load Cobra
    cobra = pd.read_excel(cfg["COBRA_PATH"], sheet_name=cfg["SHEET"])
    cobra[DATE_COL] = pd.to_datetime(cobra[DATE_COL], errors="coerce")
    cobra = cobra[cobra[DATE_COL] <= SNAPSHOT_DATE]

    # Compute EV
    ev = compute_ev(cobra)

    # Load SPI/CPI over time
    df_dates = cobra.groupby(DATE_COL).apply(
        lambda x: pd.Series({
            "SPI": x[x["COST-SET"]=="BCWP"]["HOURS"].sum() /
                   x[x["COST-SET"]=="BCWS"]["HOURS"].sum() if x[x["COST-SET"]=="BCWS"]["HOURS"].sum()>0 else np.nan,
            "CPI": x[x["COST-SET"]=="BCWP"]["HOURS"].sum() /
                   x[x["COST-SET"]=="ACWP"]["HOURS"].sum() if x[x["COST-SET"]=="ACWP"]["HOURS"].sum()>0 else np.nan,
        })
    ).reset_index()

    # Compute BEI
    bei = compute_bei(program)

    # Merge BEI with EV
    ev = ev.merge(bei, on="SUB_TEAM", how="left")

    # EV Metrics Summary Table
    summary_tbl = pd.DataFrame({
        "Metric": ["SPI","CPI","BEI","% Complete"],
        "CTD": [
            ev["SPI_CTD"].mean(),
            ev["CPI_CTD"].mean(),
            ev["BEI_CTD"].mean(),
            ev["%COMP"].mean()
        ],
        "LSP": [
            ev["SPI_CTD"].median(),   # assumption: LSP approximated by median
            ev["CPI_CTD"].median(),
            ev["BEI_LSP"].mean(),
            ev["%COMP"].median()
        ]
    })

    # Build Plot
    plot_path = make_ev_plot(df_dates, program)

    # -----------------------------------------------------------
    # BUILD POWERPOINT OUTPUT
    # -----------------------------------------------------------
    prs = Presentation(THEME_PATH)

    # ================== SLIDE 1: EV METRICS ==================
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = f"{program} – EV Metrics"

    rows, cols = summary_tbl.shape
    table = slide.shapes.add_table(rows+1, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(1.5)).table

    for j, c in enumerate(summary_tbl.columns):
        table.cell(0,j).text = c

    for i in range(rows):
        for j in range(cols):
            val = summary_tbl.iloc[i,j]
            cell = table.cell(i+1,j)
            cell.text = f"{val:.2f}" if isinstance(val,(int,float)) else str(val)
            if j>0:
                apply_color(cell, val, color_perf)

    # ================== SLIDE 2: COST PERFORMANCE ===============
    cost_tbl = ev[["SUB_TEAM","CPI_CTD"]].copy()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = f"{program} – Cost Performance"

    rows, cols = cost_tbl.shape
    table = slide.shapes.add_table(rows+1, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(4)).table

    for j, c in enumerate(cost_tbl.columns):
        table.cell(0,j).text = c

    for i in range(rows):
        for j in range(cols):
            val = cost_tbl.iloc[i,j]
            cell = table.cell(i+1,j)
            cell.text = f"{val:.2f}" if isinstance(val,(int,float)) else str(val)
            if j>0:
                apply_color(cell, val, color_perf)

    # ================== SLIDE 3: SCHEDULE PERFORMANCE ==========
    sched_tbl = ev[["SUB_TEAM","SPI_CTD","BEI_CTD"]].copy()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = f"{program} – Schedule Performance"

    rows, cols = sched_tbl.shape
    table = slide.shapes.add_table(rows+1, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(4)).table

    for j, c in enumerate(sched_tbl.columns):
        table.cell(0,j).text = c

    for i in range(rows):
        for j in range(cols):
            val = sched_tbl.iloc[i,j]
            cell = table.cell(i+1,j)
            cell.text = f"{val:.2f}" if isinstance(val,(int,float)) else str(val)
            if j>0:
                apply_color(cell, val, color_perf)

    # ================== SLIDE 4: LABOR HOURS ========================
    labor_tbl = ev[["SUB_TEAM","%COMP","BAC","EAC","VAC"]]
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = f"{program} – Labor Hours Performance"

    rows, cols = labor_tbl.shape
    table = slide.shapes.add_table(rows+1, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(4)).table

    for j, c in enumerate(labor_tbl.columns):
        table.cell(0,j).text = c

    for i in range(rows):
        for j in range(cols):
            val = labor_tbl.iloc[i,j]
            cell = table.cell(i+1,j)
            if isinstance(val,(int,float)):
                cell.text = f"{val:.2f}"
            else:
                cell.text = str(val)
            if labor_tbl.columns[j] == "VAC":
                apply_color(cell, val/(labor_tbl["BAC"].iloc[i] if labor_tbl["BAC"].iloc[i]!=0 else 1), color_vac)

    # ================== SLIDE 5: EV TREND ============================
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = f"{program} – EV Trend"

    slide.shapes.add_picture(plot_path, Inches(0.5), Inches(1.3), width=Inches(9))

    # SAVE OUTPUT
    out_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_path)
    print(f"Saved → {out_path}")

print("\nALL EVMS DASHBOARDS COMPLETE ✓")