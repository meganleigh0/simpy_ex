# =====================================================================
# GDLS EVMS DASHBOARD GENERATOR (FINAL CLEAN VERSION)
# =====================================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# =====================================================================
# CONFIG
# =====================================================================

DATA_DIR = "data"

cobra_files = {
    "Abrams_STS_2022": os.path.join(DATA_DIR, "Cobra-Abrams STS 2022.xlsx"),
    "Abrams_STS":      os.path.join(DATA_DIR, "Cobra-Abrams STS.xlsx"),
    "XM30":            os.path.join(DATA_DIR, "Cobra-XM30.xlsx")
}

openplan_path = os.path.join(DATA_DIR, "OpenPlan_Activity-Penske.xlsx")

PROGRAM_NAME_MAP = {
    "Abrams_STS_2022": "ABRAMS STS",
    "Abrams_STS":      "ABRAMS STS",
    "XM30":            "XM30"
}

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# ACCOUNTING CLOSE DATES (2020 + 2024)
# =====================================================================

ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27}
}

# =====================================================================
# COLORS (GDLS Threshold Palette)
# =====================================================================

COLOR_BLUE   = RGBColor(31, 73, 125)
COLOR_TEAL   = RGBColor(142, 180, 227)
COLOR_GREEN  = RGBColor(51, 153, 102)
COLOR_YELLOW = RGBColor(255, 255, 153)
COLOR_RED    = RGBColor(192, 80, 77)

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def normalize(df):
    return df.rename(columns=lambda c: c.strip().upper().replace(" ", "_").replace("-", "_"))

def safe_lsp_date(dates, lsp):
    """Return the closest Cobra reporting date <= LSP."""
    eligible = dates[dates <= lsp]
    return eligible.max() if not eligible.empty else dates.max()

def find_lsp_cutoff(date_series):
    dmax = date_series.max()
    y, m = dmax.year, dmax.month
    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        close_day = ACCOUNTING_CLOSE[y][m]
        close_date = pd.Timestamp(year=y, month=m, day=close_day)
        eligible = date_series[date_series <= close_date]
        return eligible.max() if not eligible.empty else date_series.max()
    else:
        return date_series.max()

# =====================================================================
# COST-SET DETECTION
# =====================================================================

def map_cost_sets(cols):
    cleaned = {c: c.replace("_","").upper() for c in cols}
    bcws = bcwp = acwp = etc = None

    for orig, clean in cleaned.items():
        if ("BCWS" in clean) or ("BUDGET" in clean):
            bcws = orig
        if ("BCWP" in clean) or ("PROGRESS" in clean) or ("EARNED" in clean):
            bcwp = orig
        if ("ACWP" in clean) or ("ACTUAL" in clean and "FINISH" not in clean):
            acwp = orig
        if ("ETC" in clean):
            etc = orig

    return bcws, bcwp, acwp, etc

# =====================================================================
# EV COMPUTATION
# =====================================================================

def compute_ev(df_raw):
    df = normalize(df_raw)

    # Required fields
    col_costset = next(c for c in df.columns if "COST_SET" in c)
    col_date    = next(c for c in df.columns if "DATE" in c)
    col_hours   = next(c for c in df.columns if "HOURS" in c)

    df.rename(columns={
        col_costset: "COST_SET",
        col_date:    "DATE",
        col_hours:   "HOURS"
    }, inplace=True)

    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index="DATE",
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum"
    ).sort_index()

    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(pivot.columns)

    ev = pd.DataFrame(index=pivot.index)
    ev["BCWS"] = pivot[bcws_col].fillna(0)
    ev["BCWP"] = pivot[bcwp_col].fillna(0)
    ev["ACWP"] = pivot[acwp_col].fillna(0)
    ev["ETC"]  = pivot[etc_col] if etc_col else 0

    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    # CTD
    BCWS_CTD = ev["BCWS_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]

    EAC = ACWP_CTD + ev["ETC"].iloc[-1]
    VAC = BCWS_CTD - EAC

    SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
    PCT_COMP_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

    # LSP
    lsp_date = find_lsp_cutoff(ev.index)
    closest = safe_lsp_date(ev.index, lsp_date)
    row = ev.loc[closest]

    SPI_LSP = row["BCWP"] / row["BCWS"] if row["BCWS"] else np.nan
    CPI_LSP = row["BCWP"] / row["ACWP"] if row["ACWP"] else np.nan

    summary = dict(
        BAC=BCWS_CTD,
        EAC=EAC,
        VAC=VAC,
        ETC=ev["ETC"].iloc[-1],
        SPI_CTD=SPI_CTD,
        CPI_CTD=CPI_CTD,
        SPI_LSP=SPI_LSP,
        CPI_LSP=CPI_LSP,
        PCT_COMP_CTD=PCT_COMP_CTD,
        LSP_DATE=closest
    )

    return ev, summary

# =====================================================================
# BEI COMPUTATION (OPENPLAN)
# =====================================================================

def compute_bei(df_op, program_key):

    df = normalize(df_op.copy())
    pname = PROGRAM_NAME_MAP[program_key]

    df = df[df["PROGRAM"] == pname]
    if df.empty:
        print(f"⚠ WARNING: No OpenPlan records for {pname}")
        return np.nan, np.nan

    # Required BEI columns
    col_base = "BASELINE_FINISH"
    col_act  = "ACTUAL_FINISH"
    col_status = "WEEKEND_DATE" if "WEEKEND_DATE" in df.columns else None

    if col_base not in df.columns:
        return np.nan, np.nan

    df[col_base] = pd.to_datetime(df[col_base])
    if col_act in df.columns:
        df[col_act] = pd.to_datetime(df[col_act])
    if col_status:
        df[col_status] = pd.to_datetime(df[col_status])
        lsp = df[col_status].max()
    else:
        lsp = df[col_base].max()

    denom = df[df[col_base] <= lsp]
    num = denom[denom[col_act].notna()] if col_act in df.columns else denom

    bei = len(num) / len(denom) if len(denom) > 0 else np.nan
    return bei, bei

# =====================================================================
# COLORING FUNCTIONS
# =====================================================================

def idx_color(v):
    if pd.isna(v): return None
    if v >= 1.055: return COLOR_BLUE
    if v >= 1.02:  return COLOR_TEAL
    if v >= 0.975: return COLOR_GREEN
    if v >= 0.945: return COLOR_YELLOW
    return COLOR_RED

def vac_color(v):
    if pd.isna(v): return None
    if v >= 0.055: return COLOR_BLUE
    if v >= 0.025: return COLOR_GREEN
    if v >= -0.025: return COLOR_YELLOW
    if v >= -0.055: return COLOR_TEAL
    return COLOR_RED

# =====================================================================
# ADD TABLE TO PPT
# =====================================================================

def add_table(slide, rows, cols, left, top, width, height, headers=None):
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    if headers:
        for j, hdr in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = hdr
            cell.text_frame.paragraphs[0].font.bold = True
    return table

def set_bg(cell, color):
    if color:
        fill = cell.fill
        fill.solid()
        fill.fore_color.rgb = color

# =====================================================================
# EV TREND CHART
# =====================================================================

def make_ev_chart(program, ev):
    fig = go.Figure()

    # Threshold bands
    fig.add_hrect(y0=0.945, y1=0.975, fillcolor="yellow", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.975, y1=1.02, fillcolor="green", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.02, y1=1.055, fillcolor="lightblue", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.055, y1=1.2, fillcolor="rgb(200,220,255)", opacity=0.2, line_width=0)

    fig.add_trace(go.Scatter(x=ev.index, y=ev["BCWP_CUM"]/ev["BCWS_CUM"], mode="lines", name="SPI (Cum)"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["BCWP_CUM"]/ev["ACWP_CUM"], mode="lines", name="CPI (Cum)"))

    fig.update_layout(title=f"{program} EVMS Trend", template="simple_white")
    return fig

# =====================================================================
# MAIN PROGRAM LOOP
# =====================================================================

openplan_df = pd.read_excel(openplan_path)

for program, path in cobra_files.items():

    print(f"\nProcessing → {program}")

    cobra_df = pd.read_excel(path)
    ev, summary = compute_ev(cobra_df)

    bei_ctd, bei_lsp = compute_bei(openplan_df, program)
    summary["BEI_CTD"] = bei_ctd
    summary["BEI_LSP"] = bei_lsp

    prs = Presentation()
    blank = prs.slide_layouts[6]

    # ================================================================
    # SLIDE 1 — EVMS Metrics
    # ================================================================
    s1 = prs.slides.add_slide(blank)
    title = s1.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title.text_frame.text = f"{program} — EV Metrics"

    t = add_table(
        s1, rows=5, cols=4,
        left=Inches(0.3), top=Inches(0.8),
        width=Inches(7), height=Inches(2),
        headers=["Metric", "CTD", "LSP", "Comments"]
    )

    metrics = [
        ("SPI", summary["SPI_CTD"], summary["SPI_LSP"]),
        ("CPI", summary["CPI_CTD"], summary["CPI_LSP"]),
        ("BEI", summary["BEI_CTD"], summary["BEI_LSP"]),
        ("% Complete", summary["PCT_COMP_CTD"], summary["PCT_COMP_CTD"])
    ]

    for i, (label, ctd, lsp) in enumerate(metrics, start=1):
        t.cell(i,0).text = label
        t.cell(i,1).text = "" if pd.isna(ctd) else f"{ctd:.2f}"
        t.cell(i,2).text = "" if pd.isna(lsp) else f"{lsp:.2f}"
        set_bg(t.cell(i,1), idx_color(ctd))
        set_bg(t.cell(i,2), idx_color(lsp))

    # ================================================================
    # SLIDE 2 — Labor & Manpower Demand
    # ================================================================
    s2 = prs.slides.add_slide(blank)
    title2 = s2.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title2.text_frame.text = f"{program} — Labor & Manpower"

    # Demand Calculation (simple version)
    BAC = summary["BAC"]
    VAC = summary["VAC"]
    Actual = BAC - VAC

    t2 = add_table(
        s2, rows=2, cols=5,
        left=Inches(0.3), top=Inches(0.8),
        width=Inches(7), height=Inches(1),
        headers=["Last Month", "This Month", "Next Month", "%Var", "Comments"]
    )

    # Placeholders (update when SME confirms)
    last_month = BAC * 0.9
    this_month = BAC
    next_month = BAC * 1.05

    pct_var = Actual / BAC if BAC else np.nan

    vals = [last_month, this_month, next_month, pct_var]
    for j, v in enumerate(vals):
        t2.cell(1,j).text = "" if pd.isna(v) else f"{v:,.1f}"
    set_bg(t2.cell(1,3), idx_color(pct_var))

    # ================================================================
    # SLIDE 3 — Trend Chart
    # ================================================================
    s3 = prs.slides.add_slide(blank)
    title3 = s3.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title3.text_frame.text = f"{program} — EVMS Trend"

    fig = make_ev_chart(program, ev)
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_trend.png")
    fig.write_image(chart_path, scale=3)
    s3.shapes.add_picture(chart_path, Inches(0.3), Inches(0.8), width=Inches(9))

    # SAVE
    out_file = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_file)
    print(f"✔ Saved → {out_file}")

print("\nALL EVMS DASHBOARDS COMPLETE ✔")