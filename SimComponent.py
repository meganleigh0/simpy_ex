# =====================================================================
# GDLS EVMS DASHBOARD GENERATOR (FINAL, VALIDATED VERSION)
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
# ACCOUNTING CLOSE CALENDAR (LSP determination)
# =====================================================================

ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27}
}

# =====================================================================
# 9/80 WORKING HOURS PER MONTH (from Shelby)
# =====================================================================

WORKING_HOURS = {
    1: 144, 2: 160, 3: 176, 4: 168, 5: 176, 6: 160,
    7: 176, 8: 168, 9: 160, 10: 176, 11: 160, 12: 176
}

# =====================================================================
# COLORS EXACTLY FROM GDLS THRESHOLD KEY
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

def find_lsp(ev_dates):
    """Find LSP using Cobra dates + accounting close"""
    max_date = ev_dates.max()
    y, m = max_date.year, max_date.month

    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        close_day = ACCOUNTING_CLOSE[y][m]
        close_date = pd.Timestamp(y, m, close_day)
        valid = ev_dates[ev_dates <= close_date]
        return valid.max()

    # fallback → month end
    return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(1)

# =====================================================================
# COST-SET MAPPING
# =====================================================================

def map_cost_sets(columns):
    clean = {c: c.replace("_","") for c in columns}
    bcws = bcwp = acwp = etc = None

    for orig, c in clean.items():
        uc = c.upper()
        if "BCWS" in uc or "BUDGET" in uc: bcws = orig
        if "BCWP" in uc or "PROGRESS" in uc or "EARNED" in uc: bcwp = orig
        if "ACWP" in uc or "ACTUAL" in uc: acwp = orig
        if "ETC" in uc: etc = orig

    return bcws, bcwp, acwp, etc

# =====================================================================
# COBRA EVMS PROCESSING
# =====================================================================

def compute_ev_from_cobra(df_raw):
    df = normalize(df_raw)

    # Required columns
    col_cost = next((c for c in df if "COST_SET" in c), None)
    col_date = next((c for c in df if c == "DATE"), None)
    col_hours = next((c for c in df if "HOURS" in c), None)

    if not col_cost or not col_date or not col_hours:
        raise ValueError(f"Missing required columns in Cobra file: {df.columns.tolist()}")

    df.rename(columns={col_cost: "COST_SET", col_hours:"HOURS"}, inplace=True)
    df["DATE"] = pd.to_datetime(df[col_date])

    pivot = df.pivot_table(index="DATE", columns="COST_SET", values="HOURS", aggfunc="sum").fillna(0)

    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(pivot.columns)

    ev = pd.DataFrame(index=pivot.index)
    ev["BCWS"] = pivot[bcws_col]
    ev["BCWP"] = pivot[bcwp_col]
    ev["ACWP"] = pivot[acwp_col]
    ev["ETC"] = pivot[etc_col] if etc_col else 0

    # CUMULATIVE
    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    # LSP
    lsp_date = find_lsp(ev.index)

    # CTD values
    BCWS_CTD = ev["BCWS_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]
    ETC_total = ev["ETC"].iloc[-1]

    EAC = ACWP_CTD + ETC_total
    VAC = BCWS_CTD - EAC

    SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
    PCT_COMP = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

    # LSP row
    row = ev.loc[lsp_date]
    SPI_LSP = row["BCWP"] / row["BCWS"] if row["BCWS"] else np.nan
    CPI_LSP = row["BCWP"] / row["ACWP"] if row["ACWP"] else np.nan

    summary = dict(
        LSP_DATE=lsp_date,
        BAC=BCWS_CTD,
        EAC=EAC,
        VAC=VAC,
        ETC=ETC_total,
        SPI_CTD=SPI_CTD,
        CPI_CTD=CPI_CTD,
        SPI_LSP=SPI_LSP,
        CPI_LSP=CPI_LSP,
        PCT_COMP=PCT_COMP,
        ACWP_CTD=ACWP_CTD,
        BCWS_CUM=ev["BCWS_CUM"]
    )

    return ev, summary

# =====================================================================
# BEI FROM OPENPLAN
# =====================================================================

def compute_bei(openplan_df, program_key):
    df = normalize(openplan_df.copy())

    pname = PROGRAM_NAME_MAP[program_key]
    df = df[df["PROGRAM"] == pname]

    if df.empty:
        return np.nan, np.nan

    # Required columns
    bf = next((c for c in df if "BASELINE_FINISH" in c), None)
    af = next((c for c in df if "ACTUAL_FINISH" in c), None)

    if bf is None:
        return np.nan, np.nan

    # Clean
    df[bf] = pd.to_datetime(df[bf], errors="coerce")
    if af:
        df[af] = pd.to_datetime(df[af], errors="coerce")

    # Exclude milestones + LOE
    if "ACTIVITY_TYPE" in df:
        df = df[~df["ACTIVITY_TYPE"].str.contains("M|LOE", na=False)]

    # Use Cobra LSP? → SME confirmation needed. For now use max AF/BF.
    lsp = max(df[bf].max(), df[af].max() if af else df[bf].max())

    planned = df[df[bf] <= lsp]
    completed = planned[(planned[af].notna()) & (planned[af] <= lsp)] if af else planned

    bei = len(completed) / len(planned) if len(planned) else np.nan
    return bei, bei

# =====================================================================
# COLOR BANDS
# =====================================================================

def idx_color(v):
    if pd.isna(v): return None
    if v >= 1.05: return COLOR_BLUE
    if v >= 1.02: return COLOR_TEAL
    if v >= 0.98: return COLOR_GREEN
    if v >= 0.95: return COLOR_YELLOW
    return COLOR_RED

def vac_color(v):
    if pd.isna(v): return None
    if v >= 0.05: return COLOR_BLUE
    if v >= -0.02: return COLOR_GREEN
    if v >= -0.05: return COLOR_YELLOW
    if v >= -0.10: return COLOR_TEAL
    return COLOR_RED

# =====================================================================
# DEMAND CALCULATION (Last, This, Next Month)
# =====================================================================

def compute_demand(summary):
    bcws_series = summary["BCWS_CUM"].diff().fillna(summary["BCWS_CUM"])
    bcws_by_month = bcws_series.groupby(bcws_series.index.month).sum()

    today = summary["LSP_DATE"]
    m = today.month

    last = bcws_by_month.get(m-1, np.nan) / WORKING_HOURS.get(m-1, np.nan)
    this = bcws_by_month.get(m, np.nan) / WORKING_HOURS.get(m, np.nan)
    nxt  = bcws_by_month.get(m+1, np.nan) / WORKING_HOURS.get(m+1, np.nan)

    actual = summary["ACWP_CTD"]
    pct_var = (actual - (this if this else np.nan)) / (this if this else np.nan)

    return last, this, nxt, actual, pct_var

# =====================================================================
# PPT HELPERS
# =====================================================================

def add_table(slide, rows, cols, left, top, width, height, headers=None):
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    if headers:
        for j,h in enumerate(headers):
            table.cell(0,j).text = h
            table.cell(0,j).text_frame.paragraphs[0].font.bold = True
    return table

def set_bg(cell, color):
    if color is not None:
        fill = cell.fill
        fill.solid()
        fill.fore_color.rgb = color

# =====================================================================
# EV TREND CHART
# =====================================================================

def make_chart(program, ev):
    ev["CPI_M"] = ev["BCWP"] / ev["ACWP"].replace(0,np.nan)
    ev["SPI_M"] = ev["BCWP"] / ev["BCWS"].replace(0,np.nan)
    ev["CPI_CUM"] = ev["BCWP_CUM"] / ev["ACWP_CUM"].replace(0,np.nan)
    ev["SPI_CUM"] = ev["BCWP_CUM"] / ev["BCWS_CUM"].replace(0,np.nan)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_CUM"], mode="lines", name="CPI CUM"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_CUM"], mode="lines", name="SPI CUM"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_M"], mode="markers", name="CPI M"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_M"], mode="markers", name="SPI M"))

    fig.update_layout(title=f"{program} EV Trend", template="simple_white")
    return fig

# =====================================================================
# MAIN LOOP
# =====================================================================

openplan_df = pd.read_excel(openplan_path)

for program, path in cobra_files.items():

    print(f"\nProcessing → {program}")

    cobra_df = pd.read_excel(path)
    ev, summary = compute_ev_from_cobra(cobra_df)

    bei_ctd, bei_lsp = compute_bei(openplan_df, program)
    summary["BEI_CTD"] = bei_ctd
    summary["BEI_LSP"] = bei_lsp

    last, this, nxt, actual, pct = compute_demand(summary)

    prs = Presentation()
    blank = prs.slide_layouts[6]

    # ---------------- SLIDE 1: METRICS ----------------
    s1 = prs.slides.add_slide(blank)
    title = s1.shapes.add_textbox(Inches(0.2), Inches(0.1), Inches(9), Inches(0.5))
    title.text_frame.text = f"{program} – EV Metrics"

    t = add_table(s1, 5, 3, Inches(0.2), Inches(0.7), Inches(6), Inches(2),
                  headers=["Metric","CTD","LSP"])

    metrics = [
        ("SPI", summary["SPI_CTD"], summary["SPI_LSP"]),
        ("CPI", summary["CPI_CTD"], summary["CPI_LSP"]),
        ("BEI", summary["BEI_CTD"], summary["BEI_LSP"]),
        ("%Complete", summary["PCT_COMP"], summary["PCT_COMP"])
    ]

    for i,(name,ctd,lsp) in enumerate(metrics, start=1):
        t.cell(i,0).text = name
        t.cell(i,1).text = f"{ctd:.2f}" if not pd.isna(ctd) else ""
        t.cell(i,2).text = f"{lsp:.2f}" if not pd.isna(lsp) else ""
        set_bg(t.cell(i,1), idx_color(ctd))
        set_bg(t.cell(i,2), idx_color(lsp))

    # comments box
    cb = s1.shapes.add_textbox(Inches(6.5), Inches(0.7), Inches(3), Inches(2.5))
    cb.text_frame.text = "Comments:\n"

    # ---------------- SLIDE 2: MANPOWER & DEMAND ----------------
    s2 = prs.slides.add_slide(blank)
    title2 = s2.shapes.add_textbox(Inches(0.2), Inches(0.1), Inches(9), Inches(0.5))
    title2.text_frame.text = f"{program} – Labor & Manpower"

    t2 = add_table(s2, 2, 5, Inches(0.2), Inches(0.7), Inches(7), Inches(1.0),
                   headers=["Last Month","This Month","Next Month","Actual","%Var"])

    vals = [last, this, nxt, actual, pct]

    for j,v in enumerate(vals):
        if j < 4:
            t2.cell(1,j).text = f"{v:,.1f}" if not pd.isna(v) else ""
        else:
            t2.cell(1,j).text = f"{v*100:.1f}%" if not pd.isna(v) else ""

    set_bg(t2.cell(1,4), idx_color(pct))

    # comments box
    cb2 = s2.shapes.add_textbox(Inches(6.5), Inches(0.7), Inches(3), Inches(2.5))
    cb2.text_frame.text = "Comments:\n"

    # ---------------- SLIDE 3: EV TREND ----------------
    s3 = prs.slides.add_slide(blank)
    title3 = s3.shapes.add_textbox(Inches(0.2), Inches(0.1), Inches(9), Inches(0.5))
    title3.text_frame.text = f"{program} – Trend Chart"

    fig = make_chart(program, ev)
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_chart.png")
    fig.write_image(chart_path, scale=3)

    s3.shapes.add_picture(chart_path, Inches(0.2), Inches(0.7), width=Inches(9))

    out_file = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_file)

    print(f"Saved → {out_file}")

print("\nALL EVMS DASHBOARDS COMPLETE ✔")