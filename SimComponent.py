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
# ACCOUNTING CLOSE CALENDAR
# =====================================================================

ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27}
}

# =====================================================================
# COLOR PALETTE (GDLS)
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
    """Normalize column names across files."""
    return df.rename(columns=lambda c: c.strip().upper().replace(" ", "_").replace("-", "_"))


def find_lsp(ev_index):
    """
    Determine LSP date safely:
    → Use accounting close date if available
    → Otherwise use last valid Cobra date ≤ end of month
    → Always return a date that EXISTS in ev_index
    """
    max_date = ev_index.max()
    y, m = max_date.year, max_date.month

    # Option 1: Accounting close exists
    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        close_day = ACCOUNTING_CLOSE[y][m]
        close_date = pd.Timestamp(year=y, month=m, day=close_day)
        valid = ev_index[ev_index <= close_date]
        if len(valid) > 0:
            return valid.max()

    # Option 2: fallback — last day of month
    end_month = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(1)
    valid = ev_index[ev_index <= end_month]
    return valid.max()


def map_cost_sets(cols):
    """Auto-detect BCWS, BCWP, ACWP, ETC columns."""
    cleaned = {c: c.replace("_", "").upper() for c in cols}
    bcws = bcwp = acwp = etc = None

    for orig, clean in cleaned.items():
        if "BUDGET" in clean or "BCWS" in clean:
            bcws = orig
        if "BCWP" in clean or "PROGRESS" in clean or "EARNED" in clean:
            bcwp = orig
        if "ACWP" in clean or "ACTUAL" in clean:
            acwp = orig
        if "ETC" in clean:
            etc = orig

    return bcws, bcwp, acwp, etc


# =====================================================================
# EV CALCULATION (FIXED + LSP SAFE + MONTHLY SP/CPI ADDED)
# =====================================================================

def compute_ev_from_cobra(df_raw):
    df = normalize(df_raw)

    # Identify columns
    required = ["COST_SET", "DATE", "HOURS"]
    colmap = {}
    for r in required:
        for c in df.columns:
            if r in c:
                colmap[r] = c
                break

    if len(colmap) < 3:
        raise ValueError(f"Could not locate COST_SET/DATE/HOURS. Found: {df.columns.tolist()}")

    df.rename(columns={
        colmap["COST_SET"]: "COST_SET",
        colmap["DATE"]: "DATE",
        colmap["HOURS"]: "HOURS"
    }, inplace=True)

    df["DATE"] = pd.to_datetime(df["DATE"])

    # Pivot
    pivot = df.pivot_table(
        index="DATE",
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0).sort_index()

    # Map cost sets
    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(pivot.columns)

    ev = pd.DataFrame(index=pivot.index)
    ev["BCWS"] = pivot[bcws_col]
    ev["BCWP"] = pivot[bcwp_col]
    ev["ACWP"] = pivot[acwp_col]
    ev["ETC"]  = pivot[etc_col] if etc_col else 0

    # cumulative
    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    # monthly index values
    ev["SPI_M"] = ev["BCWP"] / ev["BCWS"].replace(0, np.nan)
    ev["CPI_M"] = ev["BCWP"] / ev["ACWP"].replace(0, np.nan)

    ev["SPI_CUM"] = ev["BCWP_CUM"] / ev["BCWS_CUM"].replace(0, np.nan)
    ev["CPI_CUM"] = ev["BCWP_CUM"] / ev["ACWP_CUM"].replace(0, np.nan)

    # LSP — SAFE VERSION
    lsp_date = find_lsp(ev.index)

    # CTD metrics
    BCWS_CTD = ev["BCWS_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]
    ETC_total = ev["ETC"].iloc[-1]

    EAC = ACWP_CTD + ETC_total
    VAC = BCWS_CTD - EAC

    SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
    PCT_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

    # LSP row (SAFE)
    valid = ev.index[ev.index <= lsp_date]
    lsp_effective = valid.max()

    row_lsp = ev.loc[lsp_effective]
    SPI_LSP = row_lsp["BCWP"] / row_lsp["BCWS"] if row_lsp["BCWS"] else np.nan
    CPI_LSP = row_lsp["BCWP"] / row_lsp["ACWP"] if row_lsp["ACWP"] else np.nan

    summary = dict(
        BAC=BCWS_CTD,
        EAC=EAC,
        VAC=VAC,
        ETC=ETC_total,
        SPI_CTD=SPI_CTD,
        CPI_CTD=CPI_CTD,
        SPI_LSP=SPI_LSP,
        CPI_LSP=CPI_LSP,
        PCT_COMP_CTD=PCT_CTD,
        PCT_COMP_LSP=PCT_CTD,
        LSP_DATE=lsp_effective
    )

    return ev, summary


# =====================================================================
# OPENPLAN BEI COMPUTATION
# =====================================================================

def compute_bei(openplan_df, program_key):

    df = normalize(openplan_df.copy())
    prog = PROGRAM_NAME_MAP[program_key]

    df = df[df["PROGRAM"].str.upper() == prog.upper()]
    if df.empty:
        return np.nan, np.nan

    col_base = next(c for c in df.columns if "BASELINE_FINISH" in c)
    col_status = next(c for c in df.columns if "STATUS" in c or "DATA_DATE" in c)

    df[col_base] = pd.to_datetime(df[col_base])
    df[col_status] = pd.to_datetime(df[col_status])

    lsp = df[col_status].max()

    denom = df[df[col_base] <= lsp]
    num = denom  # assume completed tasks not provided

    bei = len(num) / len(denom) if len(denom) else np.nan
    return bei, bei


# =====================================================================
# COLOR FUNCTIONS
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
    if v >= 0.02: return COLOR_GREEN
    if v >= -0.02: return COLOR_YELLOW
    if v >= -0.05: return COLOR_TEAL
    return COLOR_RED


# =====================================================================
# PPT HELPERS
# =====================================================================

def add_table(slide, rows, cols, left, top, width, height, headers=None):
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    if headers:
        for j,h in enumerate(headers):
            cell = table.cell(0,j)
            cell.text = h
            cell.text_frame.paragraphs[0].font.bold = True
    return table

def set_bg(cell, color):
    if color is None: return
    fill = cell.fill
    fill.solid()
    fill.fore_color.rgb = color


# =====================================================================
# EVMS TREND CHART
# =====================================================================

def make_ev_chart(program, ev):
    fig = go.Figure()

    fig.add_hrect(y0=0.90, y1=0.95, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.95, y1=0.98, fillcolor="yellow", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.98, y1=1.02, fillcolor="green", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.02, y1=1.05, fillcolor="lightblue", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.05, y1=1.20, fillcolor="rgb(200,220,255)", opacity=0.2, line_width=0)

    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_CUM"], mode='lines',
                             line=dict(color="blue", width=3), name="Cumulative CPI"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_CUM"], mode='lines',
                             line=dict(color="gray", width=3), name="Cumulative SPI"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["CPI_M"], mode='markers',
                             marker=dict(color="gold"), name="Monthly CPI"))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["SPI_M"], mode='markers',
                             marker=dict(color="black"), name="Monthly SPI"))

    fig.update_layout(title=f"{program} EVMS Trend",
                      yaxis=dict(range=[0.9,1.2]),
                      template="simple_white")
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

    # ---------------------------------------------------------
    # BUILD POWERPOINT
    # ---------------------------------------------------------
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # SLIDE 1 – EVMS METRICS
    s1 = prs.slides.add_slide(blank)
    title = s1.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title.text_frame.text = f"{program} – EVMS Metrics"

    t = add_table(s1, rows=5, cols=3,
                  left=Inches(0.3), top=Inches(0.8),
                  width=Inches(6), height=Inches(2.2),
                  headers=["Metric","CTD","LSP"])

    metrics = [
        ("SPI (hrs)", summary["SPI_CTD"], summary["SPI_LSP"]),
        ("CPI (hrs)", summary["CPI_CTD"], summary["CPI_LSP"]),
        ("BEI", summary["BEI_CTD"], summary["BEI_LSP"]),
        ("% Complete", summary["PCT_COMP_CTD"], summary["PCT_COMP_LSP"])
    ]

    for i,(label,ctd,lsp) in enumerate(metrics, start=1):
        t.cell(i,0).text = label
        t.cell(i,1).text = "" if pd.isna(ctd) else f"{ctd:.2f}"
        t.cell(i,2).text = "" if pd.isna(lsp) else f"{lsp:.2f}"
        set_bg(t.cell(i,1), idx_color(ctd))
        set_bg(t.cell(i,2), idx_color(lsp))

    cb = s1.shapes.add_textbox(Inches(6.5), Inches(0.8), Inches(3), Inches(2.4))
    cb.text_frame.text = "Comments (RC/CA):\n"

    footer = s1.shapes.add_textbox(Inches(0.3), Inches(3.0), Inches(4), Inches(0.3))
    footer.text_frame.text = f"LSP: {summary['LSP_DATE'].date()}"

    # SLIDE 2 – LABOR & MANPOWER
    s2 = prs.slides.add_slide(blank)
    title2 = s2.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title2.text_frame.text = f"{program} – Labor & Manpower"

    # Labor
    t2 = add_table(s2, rows=2, cols=5,
                   left=Inches(0.3), top=Inches(0.8),
                   width=Inches(6.5), height=Inches(1),
                   headers=["BAC","EAC","VAC","ETC","%Complete"])

    BAC = summary["BAC"]
    EAC = summary["EAC"]
    VAC = summary["VAC"]
    ETC = summary["ETC"]
    pct = summary["PCT_COMP_CTD"]

    vals = [BAC, EAC, VAC, ETC, pct*100 if pct else np.nan]
    for j,v in enumerate(vals):
        t2.cell(1,j).text = "" if pd.isna(v) else f"{v:,.1f}"

    set_bg(t2.cell(1,2), vac_color(VAC/BAC if BAC else np.nan))

    # Manpower
    actual = BAC - VAC
    pct_var = actual / BAC if BAC else np.nan

    t3 = add_table(s2, rows=2, cols=4,
                   left=Inches(0.3), top=Inches(1.9),
                   width=Inches(6.5), height=Inches(1),
                   headers=["Demand","Actual","%Var","Next"])

    t3.cell(1,0).text = f"{BAC:,.0f}"
    t3.cell(1,1).text = f"{actual:,.0f}"
    t3.cell(1,2).text = "" if pd.isna(pct_var) else f"{pct_var*100:,.1f}%"
    set_bg(t3.cell(1,2), idx_color(pct_var))

    s2.shapes.add_textbox(Inches(6.5), Inches(0.8), Inches(3), Inches(2.5)).text_frame.text = "Comments (RC/CA):\n"

    # SLIDE 3 – TREND CHART
    s3 = prs.slides.add_slide(blank)
    title3 = s3.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title3.text_frame.text = f"{program} – EVMS Trend"

    fig = make_ev_chart(program, ev)
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(chart_path, scale=3)
    s3.shapes.add_picture(chart_path, Inches(0.3), Inches(0.8), width=Inches(9))

    # SAVE PPT
    out_file = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_file)
    print(f"Saved → {out_file}")

print("\nALL EVMS DASHBOARDS COMPLETE ✔")