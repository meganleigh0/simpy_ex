# ===================================================================
#  GDLS – EVMS DASHBOARD GENERATOR (Hardened Version)
#  Supports: SPI, CPI, BEI, Labor, Cost, Schedule, LSP Logic
# ===================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor
import plotly.graph_objects as go

# ===============================================================
# CONFIG
# ===============================================================
DATA_DIR = "data"

PROGRAM_FILES = {
    "Abrams_STS":       "Cobra-Abrams STS.xlsx",
    "Abrams_STS_2022":  "Cobra-Abrams STS 2022.xlsx",
    "XM30":             "Cobra-XM30.xlsx"
}

OPENPLAN_FILE = "OpenPlan_Activity-Penske.xlsx"
OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNAPSHOT_DATE = datetime.now().date()

# ===============================================================
# COLORS (GDLS SPEC)
# ===============================================================
BLUE   = RGBColor(31, 73, 125)
TEAL   = RGBColor(142, 180, 227)
GREEN  = RGBColor(51, 153, 102)
YELLOW = RGBColor(255, 255, 153)
RED    = RGBColor(192, 80, 77)


# ===============================================================
# AUTO-DETECT SHEET NAME
# ===============================================================

def detect_cobra_sheet(path):
    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        if any(k in s.lower() for k in ["extract", "weekly", "ev", "cap"]):
            return s
    return xl.sheet_names[0]


# ===============================================================
# LOAD COBRA
# ===============================================================

def load_cobra(path):
    sheet = detect_cobra_sheet(path)
    print(f"   → Using Cobra sheet: {sheet}")
    
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # Detect DATE column
    date_cols = [c for c in df.columns if "DATE" in c.upper()]
    if not date_cols:
        raise ValueError(f"No DATE column in {path}")

    df.rename(columns={date_cols[0]: "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    return df


# ===============================================================
# EV CALC (CTD or LSP)
# ===============================================================

def compute_ev(df):
    if "SUB_TEAM" not in df.columns:
        raise ValueError("SUB_TEAM column missing in Cobra")

    if "COST-SET" not in df.columns:
        raise ValueError("COST-SET column missing in Cobra")

    g = df.groupby(["SUB_TEAM", "COST-SET"])["HOURS"].sum().unstack(fill_value=0)

    # Ensure cost set cols exist
    for col in ["BCWS", "BCWP", "ACWP", "ETC"]:
        if col not in g.columns:
            g[col] = 0

    g["SPI"] = np.where(g["BCWS"] == 0, np.nan, g["BCWP"] / g["BCWS"])
    g["CPI"] = np.where(g["ACWP"] == 0, np.nan, g["BCWP"] / g["ACWP"])
    g["%COMP"] = np.where(g["BCWS"] == 0, np.nan, (g["BCWP"] / g["BCWS"]) * 100)

    return g.reset_index()


# ===============================================================
# GET LSP ROWS
# ===============================================================

def get_lsp(df):
    lsp = df["DATE"].max()
    return df[df["DATE"] == lsp]


# ===============================================================
# FORCE DATE COLUMNS IN OPENPLAN
# ===============================================================

def fix_date_column(df, col):
    if col not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    tmp = df[col].astype(str)
    return pd.to_datetime(tmp, errors="coerce")


# ===============================================================
# BEI CALC
# ===============================================================

def compute_bei(openplan, program, snapshot):
    df = openplan.copy()
    df = df[df["Program"] == program]

    if df.empty:
        print(f"⚠ No OpenPlan rows for {program}")
        return pd.DataFrame({"SUB_TEAM": [], "BEI": []})

    df["Baseline Finish"] = fix_date_column(df, "Baseline Finish")
    df["Actual Finish"]   = fix_date_column(df, "Actual Finish")

    if "Activity_Type" not in df.columns:
        print("⚠ Missing Activity_Type → BEI unavailable")
        return pd.DataFrame({"SUB_TEAM": df["SubTeam"], "BEI": np.nan})

    df = df[df["Activity_Type"].isin(["A","B"])]

    baseline = df[df["Baseline Finish"] <= pd.to_datetime(snapshot)]
    complete = df[df["Actual Finish"] <= pd.to_datetime(snapshot)]

    base_ct = baseline.groupby("SubTeam")["Activity ID"].count()
    comp_ct = complete.groupby("SubTeam")["Activity ID"].count().reindex(base_ct.index, fill_value=0)

    bei = pd.DataFrame({
        "SUB_TEAM": base_ct.index,
        "Baseline Tasks": base_ct.values,
        "Completed Tasks": comp_ct.values
    })

    bei["BEI"] = np.where(
        bei["Baseline Tasks"] == 0,
        np.nan,
        bei["Completed Tasks"] / bei["Baseline Tasks"]
    )

    return bei


# ===============================================================
# COLOR RULES
# ===============================================================

def color_index(v):
    if pd.isna(v): return None
    if v >= 1.05: return BLUE
    if 1.05 > v >= 0.98: return TEAL
    if 0.98 > v >= 0.95: return GREEN
    if 0.95 > v >= 0.90: return YELLOW
    return RED

def color_vac(v):
    if pd.isna(v): return None
    if v >= 0.05: return BLUE
    if 0.05 > v >= -0.02: return GREEN
    if -0.02 > v >= -0.05: return YELLOW
    if -0.05 > v >= -0.10: return TEAL
    return RED


# ===============================================================
# TABLE BUILDER
# ===============================================================

def add_table(slide, df, color_fn=None):
    rows, cols = df.shape
    table = slide.shapes.add_table(
        rows+1, cols,
        Inches(0.3), Inches(1.0),
        Inches(9), Inches(0.5*(rows+1))
    ).table

    # Header
    for c, col in enumerate(df.columns):
        cell = table.cell(0,c)
        cell.text = col
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255,255,255)

    # Body
    for r in range(rows):
        for c in range(cols):
            v = df.iloc[r,c]
            cell = table.cell(r+1,c)
            cell.text = "" if pd.isna(v) else str(round(v,3))

            if color_fn and isinstance(v, (int,float)):
                rgb = color_fn(v)
                if rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb


# ===============================================================
# MAIN EXECUTION LOOP
# ===============================================================

openplan = pd.read_excel(os.path.join(DATA_DIR, OPENPLAN_FILE))

for program, cobra_file in PROGRAM_FILES.items():

    print(f"\nProcessing → {program}")
    cobra = load_cobra(os.path.join(DATA_DIR, cobra_file))

    ev_ctd = compute_ev(cobra)
    ev_lsp = compute_ev(get_lsp(cobra))

    bei = compute_bei(openplan, program, SNAPSHOT_DATE)
    ev_ctd = ev_ctd.merge(bei[["SUB_TEAM","BEI"]], on="SUB_TEAM", how="left")

    # Labor Table
    labor = ev_ctd[["SUB_TEAM","%COMP","BCWS","ACWP","ETC"]].copy()
    labor["BAC"] = labor["BCWS"]
    labor["EAC"] = labor["ACWP"] + labor["ETC"]
    labor["VAC"] = labor["BAC"] - labor["EAC"]
    labor = labor[["SUB_TEAM","%COMP","BAC","EAC","VAC"]]

    # Cost Performance
    cost_tbl = pd.DataFrame({
        "SUB_TEAM": ev_ctd["SUB_TEAM"],
        "CPI CTD": ev_ctd["CPI"],
        "CPI LSP": ev_lsp["CPI"].values,
        "Comments": ""
    })

    # Schedule Performance
    sched_tbl = pd.DataFrame({
        "SUB_TEAM": ev_ctd["SUB_TEAM"],
        "SPI CTD": ev_ctd["SPI"],
        "SPI LSP": ev_lsp["SPI"].values,
        "BEI CTD": ev_ctd["BEI"],
        "Comments": ""
    })

    # EV Summary
    summary_tbl = pd.DataFrame({
        "Metric": ["SPI","CPI","BEI","%COMP"],
        "CTD": [ev_ctd["SPI"].mean(),
                ev_ctd["CPI"].mean(),
                ev_ctd["BEI"].mean(),
                ev_ctd["%COMP"].mean()],
        "LSP": [ev_lsp["SPI"].mean(),
                ev_lsp["CPI"].mean(),
                ev_ctd["BEI"].mean(),
                ev_lsp["%COMP"].mean()],
        "Comments": ["","","",""]
    })

    # PowerPoint
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # Slide 1 – Summary
    s = prs.slides.add_slide(blank)
    s.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.4)).text = f"{program} – EVMS Metrics"
    add_table(s, summary_tbl, color_index)

    # Slide 2 – Cost
    s = prs.slides.add_slide(blank)
    s.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.4)).text = f"{program} – Cost Performance (CPI)"
    add_table(s, cost_tbl, color_index)

    # Slide 3 – Schedule
    s = prs.slides.add_slide(blank)
    s.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.4)).text = f"{program} – Schedule Performance (SPI / BEI)"
    add_table(s, sched_tbl, color_index)

    # Slide 4 – Labor
    s = prs.slides.add_slide(blank)
    s.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.4)).text = f"{program} – Labor Hours Performance"
    add_table(s, labor, color_vac)

    # Save
    out = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out)
    print(f"Saved → {out}")

print("\n✔ ALL EVMS DASHBOARDS COMPLETE")