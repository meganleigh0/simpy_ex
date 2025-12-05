# =====================================================================
# GDLS MASTER EVMS PIPELINE – CPI/SPI/BEI + LABOR HOURS + MANPOWER
# Produces complete 5-slide EVMS dashboard PPT for each program
# =====================================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# =====================================================================
# PATHS / INPUTS
# =====================================================================

DATA_DIR = "data"

PROGRAM_FILES = {
    "Abrams_STS_2022": "Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS":      "Cobra-Abrams STS.xlsx",
    "XM30":            "Cobra-XM30.xlsx"
}

OPENPLAN_PATH = os.path.join(DATA_DIR, "OpenPlan_Activity-Penske.xlsx")

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# COLOR PALETTE (GDLS THRESHOLD KEY)
# =====================================================================

BLUE      = RGBColor(31,73,125)
LT_BLUE   = RGBColor(142,180,227)
GREEN     = RGBColor(51,153,102)
YELLOW    = RGBColor(255,255,153)
RED       = RGBColor(192,80,77)
GRAY      = RGBColor(220,220,220)

# =====================================================================
# THRESHOLD LOGIC
# =====================================================================

def idx_color(x):
    """Color for SPI/CPI/BEI thresholds"""
    if pd.isna(x): return None
    if x >= 1.05: return BLUE
    if x >= 1.02: return LT_BLUE
    if x >= 0.98: return GREEN
    if x >= 0.95: return YELLOW
    return RED

def vac_color(x):
    """Color for VAC/BAC thresholds"""
    if pd.isna(x): return None
    if x >= 0.05: return BLUE
    if x >= 0.02: return GREEN
    if x >= -0.02: return YELLOW
    if x >= -0.05: return LT_BLUE
    return RED

# =====================================================================
# NORMALIZE COLUMN NAMES
# =====================================================================

def clean(df):
    df = df.rename(columns=lambda c: c.strip().upper().replace(" ","_"))
    return df

# =====================================================================
# FIND LSP DATE (last status period)
# =====================================================================

def find_lsp(df):
    dates = pd.to_datetime(df["DATE"])
    return dates.max()

# =====================================================================
# COMPUTE EV (BCWS / BCWP / ACWP / ETC)
# =====================================================================

def compute_ev(df_raw):
    df = clean(df_raw.copy())
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["DATE"].notna()]

    pivot = df.pivot_table(
        index="DATE",
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum",
        fill_value=0
    ).sort_index()

    # map cost-sets
    cols = {c: c.replace("_","") for c in pivot.columns}
    def match(k): return next((orig for orig,new in cols.items() if k in new), None)

    bcws = match("BCWS")
    bcwp = match("BCWP") or match("PROGRESS")
    acwp = match("ACWP")
    etc  = match("ETC")

    ev = pd.DataFrame()
    ev["BCWS"] = pivot[bcws]
    ev["BCWP"] = pivot[bcwp]
    ev["ACWP"] = pivot[acwp]
    ev["ETC"]  = pivot[etc] if etc else 0

    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    lsp = ev.index.max()

    # CTD metrics
    BCWS_CTD = ev["BCWS_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]
    ETC_TOT  = ev["ETC"].iloc[-1]

    EAC = ACWP_CTD + ETC_TOT
    VAC = BCWS_CTD - EAC

    SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
    PCT_COMP = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

    # LSP metrics
    row = ev.loc[lsp]
    SPI_LSP = row["BCWP"] / row["BCWS"] if row["BCWS"] else np.nan
    CPI_LSP = row["BCWP"] / row["ACWP"] if row["ACWP"] else np.nan

    summary = dict(
        LSP=lsp,
        SPI_CTD=SPI_CTD, CPI_CTD=CPI_CTD,
        SPI_LSP=SPI_LSP, CPI_LSP=CPI_LSP,
        BEI_CTD=np.nan, BEI_LSP=np.nan,   # filled after BEI
        PCT_COMP=PCT_COMP,
        BAC=BCWS_CTD, EAC=EAC, VAC=VAC
    )

    return ev, summary

# =====================================================================
# COMPUTE BEI (from OpenPlan)
# =====================================================================

def compute_bei(openplan, program_name, lsp):
    df = clean(openplan.copy())
    df = df[df["PROGRAM"].str.upper() == program_name.upper()]

    # required fields
    if "BASELINE_FINISH" not in df.columns: return np.nan, np.nan
    if "ACTUAL_FINISH" not in df.columns: return np.nan, np.nan

    df["BASELINE_FINISH"] = pd.to_datetime(df["BASELINE_FINISH"], errors="coerce")
    df["ACTUAL_FINISH"]   = pd.to_datetime(df["ACTUAL_FINISH"], errors="coerce")

    # exclude milestones & LOE if present
    if "ACTIVITY_TYPE" in df.columns:
        df = df[~df["ACTIVITY_TYPE"].isin(["M","LOE"])]

    denom = df[df["BASELINE_FINISH"] <= lsp]
    num   = denom[denom["ACTUAL_FINISH"].notna() & (denom["ACTUAL_FINISH"] <= lsp)]

    bei = len(num) / len(denom) if len(denom) else np.nan
    return bei, bei

# =====================================================================
# SUBTEAM EV TABLES (CPI/SPI)
# =====================================================================

def build_subteam_ev(df_raw):
    df = clean(df_raw.copy())
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["DATE"].notna()]

    # aggregate by subteam + cost-set + date
    g = df.groupby(["SUB_TEAM","COST_SET","DATE"])["HOURS"].sum().unstack("COST_SET").fillna(0)

    out = []
    for sub in g.index.unique():
        x = g.loc[sub].sort_index()
        BCWS = x.get("BCWS", pd.Series([0]))
        BCWP = x.get("BCWP", pd.Series([0]))
        ACWP = x.get("ACWP", pd.Series([0]))

        SPI_LSP = BCWP.iloc[-1] / BCWS.iloc[-1] if BCWS.iloc[-1] else np.nan
        CPI_LSP = BCWP.iloc[-1] / ACWP.iloc[-1] if ACWP.iloc[-1] else np.nan

        SPI_CTD = BCWP.sum() / BCWS.sum() if BCWS.sum() else np.nan
        CPI_CTD = BCWP.sum() / ACWP.sum() if ACWP.sum() else np.nan

        out.append([sub, SPI_LSP, SPI_CTD, CPI_LSP, CPI_CTD])

    df_out = pd.DataFrame(out, columns=["SUB_TEAM","SPI_LSP","SPI_CTD","CPI_LSP","CPI_CTD"])
    return df_out

# =====================================================================
# LABOR HOURS PERFORMANCE TABLE
# =====================================================================

def labor_hours_table(df_raw):
    df = clean(df_raw.copy())
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    g = df.groupby(["SUB_TEAM","COST_SET"])["HOURS"].sum().unstack("COST_SET").fillna(0)

    BAC = g["BCWS"]
    EAC = g["ACWP"] + g.get("ETC",0)
    VAC = BAC - EAC
    PCOMP = np.where(BAC==0, np.nan, (g["BCWP"] / BAC) * 100)

    tbl = pd.DataFrame({
        "SUB_TEAM": g.index,
        "%COMP": PCOMP.round(1),
        "BAC": BAC.round(1),
        "EAC": EAC.round(1),
        "VAC": VAC.round(1),
    })

    return tbl

# =====================================================================
# MANPOWER TABLE
# =====================================================================

def manpower_table(ev):
    """Demand & Actual based on monthly BCWS/ACWP divided by SHC=980hrs"""
    SHC = 980
    ev2 = ev.copy()
    ev2["MONTH"] = ev2.index.to_period("M")

    m = ev2.groupby("MONTH")[["BCWS","ACWP"]].sum()

    m["DEMAND"] = m["BCWS"] / SHC
    m["ACTUAL"] = m["ACWP"] / SHC

    last = m.iloc[-2] if len(m)>=2 else m.iloc[-1]
    now  = m.iloc[-1]
    nxt  = np.nan

    pct_var = (now["ACTUAL"] - now["DEMAND"]) / now["DEMAND"] if now["DEMAND"] else np.nan

    out = pd.DataFrame({
        "Last Month Demand":[last["DEMAND"]],
        "This Month Demand":[now["DEMAND"]],
        "Next Month Demand":[nxt],
        "Actual":[now["ACTUAL"]],
        "%Var":[pct_var]
    })

    return out.round(2)

# =====================================================================
# PLOT
# =====================================================================

def make_ev_plot(ev, program):
    fig = go.Figure()

    # threshold bands
    bands = [
        (1.05, 2.0, "lightblue"),
        (1.02, 1.05, "blue"),
        (0.98, 1.02, "green"),
        (0.95, 0.98, "yellow"),
        (0.0, 0.95, "red")
    ]
    for low,high,col in bands:
        fig.add_hrect(y0=low, y1=high, fillcolor=col, opacity=0.15, line_width=0)

    fig.add_trace(go.Scatter(x=ev.index, y=ev["BCWP_CUM"]/ev["ACWP_CUM"], mode="lines",
                             name="CPI (Cum)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["BCWP_CUM"]/ev["BCWS_CUM"], mode="lines",
                             name="SPI (Cum)", line=dict(color="black")))
    fig.update_layout(title=f"{program} EVMS Trend", yaxis=dict(range=[0.8,1.2]))
    path = os.path.join(OUTPUT_DIR, f"{program}_EV_plot.png")
    fig.write_image(path, scale=3)
    return path

# =====================================================================
# PPT TABLE HELPER
# =====================================================================

def add_table(slide, df, left, top, width):
    rows, cols = df.shape
    shape = slide.shapes.add_table(rows+1, cols, left, top, width, Inches(0.2))
    tbl = shape.table
    # headers
    for j, c in enumerate(df.columns):
        cell = tbl.cell(0,j)
        cell.text = c
        cell.text_frame.paragraphs[0].font.bold = True
    # body
    for i in range(rows):
        for j in range(cols):
            val = df.iloc[i,j]
            cell = tbl.cell(i+1,j)
            cell.text = "" if pd.isna(val) else str(val)
            # color
            if df.columns[j] in ["SPI_LSP","SPI_CTD","CPI_LSP","CPI_CTD","BEI_LSP","BEI_CTD"]:
                bg = idx_color(float(val))
                if bg: cell.fill.solid(); cell.fill.fore_color.rgb = bg
            if df.columns[j]=="VAC":
                bg = vac_color(float(val/df.iloc[i]["BAC"]) if df.iloc[i]["BAC"] else np.nan)
                if bg: cell.fill.solid(); cell.fill.fore_color.rgb = bg
    return tbl

# =====================================================================
# MAIN LOOP
# =====================================================================

openplan = pd.read_excel(OPENPLAN_PATH)

for program, file in PROGRAM_FILES.items():

    print(f"\nProcessing → {program}")
    cobra = pd.read_excel(os.path.join(DATA_DIR, file))

    # EV + summary
    ev, summary = compute_ev(cobra)

    # BEI
    bei_ctd, bei_lsp = compute_bei(openplan, program, summary["LSP"])
    summary["BEI_CTD"] = bei_ctd
    summary["BEI_LSP"] = bei_lsp

    # labor hours
    labor_tbl = labor_hours_table(cobra)

    # subteam ev
    st_ev = build_subteam_ev(cobra)

    # schedule table merges SPI/BEI
    sched = st_ev[["SUB_TEAM","SPI_LSP","SPI_CTD"]].copy()
    sched["BEI_LSP"] = bei_lsp
    sched["BEI_CTD"] = bei_ctd

    # cost table
    cost = st_ev[["SUB_TEAM","CPI_LSP","CPI_CTD"]].copy()

    # manpower
    mp = manpower_table(ev)

    # EV summary table
    summary_tbl = pd.DataFrame({
        "Metric":["SPI","CPI","BEI","% Complete"],
        "CTD":[summary["SPI_CTD"],summary["CPI_CTD"],summary["BEI_CTD"],summary["PCT_COMP"]],
        "LSP":[summary["SPI_LSP"],summary["CPI_LSP"],summary["BEI_LSP"],summary["PCT_COMP"]],
    })

    # Build PPT
    prs = Presentation()

    # Slide 1 – EV Summary
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    add_table(s1, summary_tbl.round(3), Inches(0.3), Inches(0.3), Inches(8))

    # Slide 2 – Labor Hours
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    add_table(s2, labor_tbl, Inches(0.3), Inches(0.3), Inches(9))

    # Slide 3 – Cost Performance
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    add_table(s3, cost.round(3), Inches(0.3), Inches(0.3), Inches(8))

    # Slide 4 – Schedule Performance
    s4 = prs.slides.add_slide(prs.slide_layouts[6])
    add_table(s4, sched.round(3), Inches(0.3), Inches(0.3), Inches(8))

    # Slide 5 – EV Plot
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    plot_path = make_ev_plot(ev, program)
    s5.shapes.add_picture(plot_path, Inches(0.3), Inches(0.3), width=Inches(9))

    # SAVE
    out = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out)
    print("Saved →", out)

print("\nALL EVMS DASHBOARDS COMPLETE ✔")
