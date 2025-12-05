# =====================================================================
# GDLS EVMS DASHBOARD GENERATOR (FINAL FULL VERSION)
# Includes:
#   • Hours-based Cobra EV calculations
#   • BEI from OpenPlan
#   • Last Month / LSP Month / Next Month manpower demand
#   • GDLS threshold coloring
#   • 3-slide PPT per program
# =====================================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# =====================================================================
# CONFIGURATION
# =====================================================================

DATA_DIR = "data"

cobra_files = {
    "Abrams_STS_2022": os.path.join(DATA_DIR, "Cobra-Abrams STS 2022.xlsx"),
    "Abrams_STS":      os.path.join(DATA_DIR, "Cobra-Abrams STS.xlsx"),
    "XM30":            os.path.join(DATA_DIR, "Cobra-XM30.xlsx")
}

openplan_path = os.path.join(DATA_DIR, "OpenPlan_Activity-Penske.xlsx")

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Program name matching for OpenPlan
PROGRAM_MAP = {
    "Abrams_STS_2022": "ABRAMS",
    "Abrams_STS":      "ABRAMS",
    "XM30":            "XM30"
}

# =====================================================================
# ACCOUNTING CLOSE CALENDAR
# (Used to determine LSP)
# =====================================================================

ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27}
}

# =====================================================================
# COLOR PALETTE (GDLS Official)
# =====================================================================

BLUE   = RGBColor(31, 73, 125)
TEAL   = RGBColor(142, 180, 227)
GREEN  = RGBColor(51, 153, 102)
YELLOW = RGBColor(255, 255, 153)
RED    = RGBColor(192, 80, 77)

# =====================================================================
# THRESHOLD LOGIC (STRICT GDLS)
# =====================================================================

def color_spi_cpi_bei(v):
    if pd.isna(v): return None
    if v >= 1.055: return BLUE
    if v >= 1.02: return TEAL
    if v >= 0.975: return GREEN
    if v >= 0.945: return YELLOW
    return RED

def color_vac_ratio(v):
    if pd.isna(v): return None
    if v >= 0.055: return BLUE
    if v >= 0.025: return TEAL
    if v >= -0.025: return GREEN
    if v >= -0.055: return YELLOW
    return RED

def color_manpower(v):
    if pd.isna(v): return None
    if v >= 1.095: return RED
    if v >= 1.055: return BLUE
    if v >= 0.895: return GREEN
    if v >= 0.855: return YELLOW
    return RED

# =====================================================================
# UTILITIES
# =====================================================================

def normalize(df):
    return df.rename(columns=lambda c: c.strip().upper().replace(" ", "_").replace("-", "_"))

def find_lsp_date(dates):
    maxd = dates.max()
    y, m = maxd.year, maxd.month

    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        c = ACCOUNTING_CLOSE[y][m]
        cutoff = pd.Timestamp(year=y, month=m, day=c)
        eligible = dates[dates <= cutoff]
        return eligible.max()
    return dates.max()

# =====================================================================
# COBRA EV COMPUTATION
# =====================================================================

def compute_ev_from_cobra(df_raw):
    df = normalize(df_raw)

    # Locate required fields
    cost_col = next((c for c in df.columns if "COST_SET" in c), None)
    date_col = next((c for c in df.columns if "DATE" in c), None)
    hours_col = next((c for c in df.columns if "HOURS" in c), None)

    if None in (cost_col, date_col, hours_col):
        raise ValueError("Missing COST_SET / DATE / HOURS")

    df = df.rename(columns={
        cost_col: "COST_SET",
        date_col: "DATE",
        hours_col: "HOURS"
    })

    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE",
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum"
    ).sort_index().fillna(0)

    # Cost set detection
    def pick(colset, keys):
        for k in keys:
            for c in colset:
                if k in c.upper():
                    return c
        return None

    cols = pivot.columns
    BCWS = pick(cols, ["BCWS", "BUDGET"])
    BCWP = pick(cols, ["BCWP", "PROGRESS", "EARNED"])
    ACWP = pick(cols, ["ACWP", "ACTUAL"])
    ETC  = pick(cols, ["ETC"])

    ev = pd.DataFrame(index=pivot.index)
    ev["BCWS"] = pivot.get(BCWS, 0)
    ev["BCWP"] = pivot.get(BCWP, 0)
    ev["ACWP"] = pivot.get(ACWP, 0)
    ev["ETC"]  = pivot.get(ETC, 0)

    # Cumulative
    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    lsp = find_lsp_date(ev.index)

    # CTD
    BCWS_CTD = ev["BCWS_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]
    ETC_TOT  = ev["ETC"].sum()

    EAC = ACWP_CTD + ETC_TOT
    VAC = BCWS_CTD - EAC

    SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
    PCT_COMP = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

    # LSP
    row_lsp = ev.loc[lsp]
    SPI_LSP = row_lsp["BCWP"] / row_lsp["BCWS"] if row_lsp["BCWS"] else np.nan
    CPI_LSP = row_lsp["BCWP"] / row_lsp["ACWP"] if row_lsp["ACWP"] else np.nan

    summary = {
        "BAC": BCWS_CTD,
        "EAC": EAC,
        "VAC": VAC,
        "ETC": ETC_TOT,
        "SPI_CTD": SPI_CTD,
        "CPI_CTD": CPI_CTD,
        "SPI_LSP": SPI_LSP,
        "CPI_LSP": CPI_LSP,
        "PCT_COMP": PCT_COMP,
        "LSP_DATE": lsp,
        "EV_TABLE": ev
    }

    return ev, summary

# =====================================================================
# BEI COMPUTATION FROM OPENPLAN
# =====================================================================

def compute_bei(open_df, program):
    df = normalize(open_df.copy())
    key = PROGRAM_MAP[program]

    df = df[df["PROGRAM"].str.contains(key, case=False, na=False)]
    if df.empty:
        return np.nan, np.nan

    base_col = next((c for c in df.columns if "BASELINE_FINISH" in c), None)
    act_col  = next((c for c in df.columns if "ACTUAL_FINISH" in c), None)
    status_col = next((c for c in df.columns if "STATUS" in c or "DATA_DATE" in c), None)

    if not base_col or not status_col:
        return np.nan, np.nan

    df[base_col] = pd.to_datetime(df[base_col])
    if act_col:
        df[act_col] = pd.to_datetime(df[act_col])
    df[status_col] = pd.to_datetime(df[status_col])

    lsp = df[status_col].max()

    denom = df[df[base_col] <= lsp]      # all tasks baselined by LSP
    if act_col:
        num = denom[(denom[act_col].notna()) & (denom[act_col] <= lsp)]
    else:
        num = denom.copy()

    if len(denom) == 0:
        return np.nan, np.nan

    bei = len(num) / len(denom)
    return bei, bei

# =====================================================================
# MANPOWER (DEMAND / ACTUAL / NEXT)
# =====================================================================

WORKING_HOURS = 980   # Placeholder – replace with your month-specific SHC table later.

def compute_manpower_demand(ev):
    df = ev.copy()
    df["MONTH"] = df.index.to_period("M")

    monthly = df.groupby("MONTH").agg({
        "BCWS": "sum",
        "ACWP": "sum"
    })

    monthly["Demand"] = monthly["BCWS"] / WORKING_HOURS
    monthly["Actual"] = monthly["ACWP"] / WORKING_HOURS
    monthly["PctVar"] = monthly["Actual"] / monthly["Demand"]

    # Most recent month = LSP Month
    this_month = monthly.index.max()
    last_month = this_month - 1

    next_month = this_month + 1
    ETC_next = df["ETC"].iloc[-1] / WORKING_HOURS

    return {
        "Last_Demand": monthly.loc[last_month, "Demand"] if last_month in monthly.index else np.nan,
        "This_Demand": monthly.loc[this_month, "Demand"],
        "Next_Demand": ETC_next,
        "This_PctVar": monthly.loc[this_month, "PctVar"],
        "This_Actual": monthly.loc[this_month, "Actual"]
    }

# =====================================================================
# PPT HELPERS
# =====================================================================

def add_table(slide, rows, cols, left, top, width, height):
    return slide.shapes.add_table(rows, cols, left, top, width, height).table

def set_bg(cell, color):
    if color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = color

# =====================================================================
# EV TREND CHART
# =====================================================================

def build_chart(program, ev):
    fig = go.Figure()

    fig.add_hrect(y0=0.945,y1=0.975,fillcolor="yellow",opacity=.2,line_width=0)
    fig.add_hrect(y0=0.975,y1=1.02,fillcolor="green",opacity=.2,line_width=0)
    fig.add_hrect(y0=1.02,y1=1.055,fillcolor="lightblue",opacity=.2,line_width=0)
    fig.add_hrect(y0=1.055,y1=1.2,fillcolor="blue",opacity=.15,line_width=0)
    fig.add_hrect(y0=0,y1=0.945,fillcolor="red",opacity=.15,line_width=0)

    ev["SPI_M"] = ev["BCWP"] / ev["BCWS"].replace(0, np.nan)
    ev["CPI_M"] = ev["BCWP"] / ev["ACWP"].replace(0, np.nan)
    ev["SPI_CUM"] = ev["BCWP_CUM"] / ev["BCWS_CUM"].replace(0, np.nan)
    ev["CPI_CUM"] = ev["BCWP_CUM"] / ev["ACWP_CUM"].replace(0, np.nan)

    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_CUM"], mode="lines", name="SPI (Cum)", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_CUM"], mode="lines", name="CPI (Cum)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_M"], mode="markers", name="SPI (M)", marker=dict(color="black")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_M"], mode="markers", name="CPI (M)", marker=dict(color="gold")))

    fig.update_layout(title=f"{program} EV Trend", template="simple_white", yaxis=dict(range=[0.9,1.2]))
    return fig

# =====================================================================
# MAIN EXECUTION
# =====================================================================

openplan_df = pd.read_excel(openplan_path)

for program, path in cobra_files.items():
    print(f"\nProcessing → {program}")

    cobra_df = pd.read_excel(path)
    ev, summary = compute_ev_from_cobra(cobra_df)

    # BEI
    bei_ctd, bei_lsp = compute_bei(openplan_df, program)
    summary["BEI_CTD"] = bei_ctd
    summary["BEI_LSP"] = bei_lsp

    # Manpower
    mp = compute_manpower_demand(summary["EV_TABLE"])

    # PPT
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # ---------------- SLIDE 1 ----------------
    s1 = prs.slides.add_slide(blank)
    t1 = s1.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.4))
    t1.text = f"{program} — EV Metrics"

    tbl = add_table(s1, 5, 4, Inches(0.3), Inches(0.7), Inches(7), Inches(2))
    headers = ["Metric", "CTD", "LSP", "Comments"]
    for i, h in enumerate(headers):
        tbl.cell(0,i).text = h

    metrics = [
        ("SPI", summary["SPI_CTD"], summary["SPI_LSP"]),
        ("CPI", summary["CPI_CTD"], summary["CPI_LSP"]),
        ("BEI", summary["BEI_CTD"], summary["BEI_LSP"]),
        ("% Complete", summary["PCT_COMP"], summary["PCT_COMP"])
    ]

    for r,(label,ctd,lsp) in enumerate(metrics, start=1):
        tbl.cell(r,0).text = label
        tbl.cell(r,1).text = f"{ctd:.3f}" if not pd.isna(ctd) else ""
        tbl.cell(r,2).text = f"{lsp:.3f}" if not pd.isna(lsp) else ""
        set_bg(tbl.cell(r,1), color_spi_cpi_bei(ctd))
        set_bg(tbl.cell(r,2), color_spi_cpi_bei(lsp))

    # ---------------- SLIDE 2 ----------------
    s2 = prs.slides.add_slide(blank)
    t2 = s2.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.4))
    t2.text = f"{program} — Labor & Manpower"

    tbl2 = add_table(s2, 4, 5, Inches(0.3), Inches(0.7), Inches(7), Inches(2))
    for i,h in enumerate(["", "Last Month", "This Month", "Next Month", "%Var"]):
        tbl2.cell(0,i).text = h

    tbl2.cell(1,0).text = "Demand"
    tbl2.cell(1,1).text = f"{mp['Last_Demand']:.2f}" if mp['Last_Demand']==mp['Last_Demand'] else ""
    tbl2.cell(1,2).text = f"{mp['This_Demand']:.2f}"
    tbl2.cell(1,3).text = f"{mp['Next_Demand']:.2f}"

    tbl2.cell(2,0).text = "Actual"
    tbl2.cell(2,2).text = f"{mp['This_Actual']:.2f}"

    tbl2.cell(3,0).text = "%Var"
    tbl2.cell(3,2).text = f"{mp['This_PctVar']:.2f}"
    set_bg(tbl2.cell(3,2), color_manpower(mp['This_PctVar']))

    # ---------------- SLIDE 3 ----------------
    s3 = prs.slides.add_slide(blank)
    fig = build_chart(program, ev)
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_chart.png")
    fig.write_image(chart_path, scale=3)
    s3.shapes.add_picture(chart_path, Inches(0.3), Inches(0.8), width=Inches(9))

    out = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out)
    print("Saved:", out)

print("\nALL EVMS DASHBOARDS COMPLETE ✔")