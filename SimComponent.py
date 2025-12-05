# =======================================================================
# EVMS DASHBOARD – Full Automated Pipeline (Auto-Sheet Version)
# =======================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objs as go
from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor


# =======================================================================
# CONFIG
# =======================================================================

DATA_DIR = "data"
OPENPLAN_FILE = "OpenPlan_Activity-Penske.xlsx"

PROGRAM_FILES = {
    "Abrams_STS":       "Cobra-Abrams STS.xlsx",
    "Abrams_STS_2022":  "Cobra-Abrams STS 2022.xlsx",
    "XM30":             "Cobra-XM30.xlsx"
}

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNAPSHOT_DATE = datetime.now().date()


# =======================================================================
# AUTO-DETECT COBRA SHEET
# =======================================================================

def detect_cobra_sheet(path):
    """
    Returns the sheet name most likely containing the EVMS export.
    We match based on common patterns that appear in Cobra reports.
    """
    xl = pd.ExcelFile(path)
    candidates = [s for s in xl.sheet_names 
                  if any(k in s.lower() for k in 
                         ["extract", "weekly", "tbl", "ev", "export", "report"])]
    
    if len(candidates) == 0:
        print(f"⚠ No EVMS-like sheet detected for {path}. Using first sheet.")
        return xl.sheet_names[0]
    
    return candidates[0]


# =======================================================================
# LOAD COBRA (WITH AUTO-SHEET)
# =======================================================================

def load_cobra(path):
    sheet = detect_cobra_sheet(path)
    print(f"   → Using Cobra sheet: {sheet}")

    df = pd.read_excel(path, sheet_name=sheet)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "DATE" not in df.columns:
        raise ValueError(f"'DATE' column not found in {path}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df.dropna(subset=["DATE"])


# =======================================================================
# EV METRICS (SPI, CPI, %COMP)
# =======================================================================

def compute_ev(df):
    g = df.groupby(["SUB_TEAM", "COST-SET"])["HOURS"].sum().unstack(fill_value=0)

    for col in ["BCWP", "BCWS", "ACWP", "ETC"]:
        if col not in g.columns:
            g[col] = 0

    g["SPI"] = np.where(g["BCWS"] == 0, np.nan, g["BCWP"] / g["BCWS"])
    g["CPI"] = np.where(g["ACWP"] == 0, np.nan, g["BCWP"] / g["ACWP"])
    g["%COMP"] = np.where(g["BCWS"] == 0, np.nan, (g["BCWP"] / g["BCWS"]) * 100)

    return g.reset_index()


# =======================================================================
# GET LAST STATUS PERIOD (LSP)
# =======================================================================

def get_lsp(df):
    last_date = df["DATE"].max()
    return df[df["DATE"] == last_date]


# =======================================================================
# BEI CALC
# =======================================================================

def compute_bei(openplan, program, snapshot):
    df = openplan.copy()
    df = df[df["Program"] == program]

    df["Baseline Finish"] = pd.to_datetime(df["Baseline Finish"], errors="coerce")
    df["Actual Finish"] = pd.to_datetime(df["Actual Finish"], errors="coerce")

    df = df[df["Activity_Type"].isin(["A", "B"])]

    baseline = df[df["Baseline Finish"] <= snapshot]
    complete = df[df["Actual Finish"] <= snapshot]

    baseline_count = baseline.groupby("SubTeam")["Activity ID"].count()
    complete_count = complete.groupby("SubTeam")["Activity ID"].count()

    bei = pd.DataFrame({
        "SUB_TEAM": baseline_count.index,
        "Baseline Tasks": baseline_count.values,
        "Completed Tasks": complete_count.reindex(baseline_count.index).fillna(0).values
    })

    bei["BEI"] = np.where(
        bei["Baseline Tasks"] == 0, np.nan,
        bei["Completed Tasks"] / bei["Baseline Tasks"]
    )

    return bei


# =======================================================================
# COLOR RULES
# =======================================================================

def color_spi_cpi(v):
    if pd.isna(v): return None
    if v >= 1.05: return RGBColor(31, 73, 125)
    if 1.05 > v >= 0.98: return RGBColor(142, 180, 227)
    if 0.98 > v >= 0.95: return RGBColor(51, 153, 102)
    if 0.95 > v >= 0.90: return RGBColor(255, 255, 153)
    return RGBColor(192, 80, 77)

def color_vac(v):
    if pd.isna(v): return None
    if v >= 0.05: return RGBColor(31, 73, 125)
    if 0.05 > v >= -0.02: return RGBColor(142, 180, 227)
    if -0.02 > v >= -0.05: return RGBColor(255, 255, 153)
    if -0.05 > v >= -0.10: return RGBColor(255, 204, 153)
    return RGBColor(192, 80, 77)


# =======================================================================
# POWERPOINT TABLE HELPER
# =======================================================================

def add_table(slide, df, title, color_fn=None):

    rows, cols = df.shape
    left, top = Inches(0.3), Inches(1.0)
    width, height = Inches(9), Inches(0.3*(rows+1))

    table = slide.shapes.add_table(rows+1, cols, left, top, width, height).table

    # Header
    for c, col in enumerate(df.columns):
        cell = table.cell(0, c)
        cell.text = col
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(31, 73, 125)
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

    # Body
    for r in range(rows):
        for c in range(cols):
            val = df.iloc[r, c]
            cell = table.cell(r+1, c)
            cell.text = "" if pd.isna(val) else str(np.round(val, 3))

            if color_fn and isinstance(val, (int, float)):
                rgb = color_fn(val)
                if rgb:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb


# =======================================================================
# MAIN LOOP
# =======================================================================

openplan = pd.read_excel(os.path.join(DATA_DIR, OPENPLAN_FILE))

for program, cobra_file in PROGRAM_FILES.items():

    print(f"\nProcessing → {program}")

    cobra_path = os.path.join(DATA_DIR, cobra_file)
    cobra = load_cobra(cobra_path)

    # CTD EV
    ev_ctd = compute_ev(cobra)

    # LSP EV
    ev_lsp = compute_ev(get_lsp(cobra))

    # BEI
    bei = compute_bei(openplan, program, SNAPSHOT_DATE)
    bei.index = bei["SUB_TEAM"]

    # Merge BEI into CTD table
    ev_ctd = ev_ctd.merge(bei[["BEI"]], on="SUB_TEAM", how="left")

    # Labor Table
    labor = ev_ctd[["SUB_TEAM", "%COMP", "BCWS", "ACWP", "ETC"]].copy()
    labor["BAC"] = labor["BCWS"]
    labor["EAC"] = labor["ACWP"] + labor["ETC"]
    labor["VAC"] = labor["BAC"] - labor["EAC"]
    labor = labor[["SUB_TEAM", "%COMP", "BAC", "EAC", "VAC"]]

    # Cost Table
    cost_tbl = pd.DataFrame({
        "SUB_TEAM": ev_ctd["SUB_TEAM"],
        "CPI CTD": ev_ctd["CPI"],
        "CPI LSP": ev_lsp["CPI"].values,
        "Comments": ""
    })

    # Schedule Table
    sched_tbl = pd.DataFrame({
        "SUB_TEAM": ev_ctd["SUB_TEAM"],
        "SPI CTD": ev_ctd["SPI"],
        "SPI LSP": ev_lsp["SPI"].values,
        "BEI CTD": ev_ctd["BEI"],
        "Comments": ""
    })

    # EV Summary Table
    summary_tbl = pd.DataFrame({
        "Metric": ["SPI", "CPI", "BEI", "% Complete"],
        "CTD": [
            ev_ctd["SPI"].mean(),
            ev_ctd["CPI"].mean(),
            ev_ctd["BEI"].mean(),
            ev_ctd["%COMP"].mean()
        ],
        "LSP": [
            ev_lsp["SPI"].mean(),
            ev_lsp["CPI"].mean(),
            ev_ctd["BEI"].mean(),
            ev_lsp["%COMP"].mean()
        ],
        "Comments": ["", "", "", ""]
    })

    # Build PPT
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # Slide 1 – EV Metrics
    slide = prs.slides.add_slide(blank)
    tx = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.5))
    tx.text = f"{program} – EVMS Metrics"
    add_table(slide, summary_tbl, "Metrics", color_spi_cpi)

    # Slide 2 – Labor Hours
    slide = prs.slides.add_slide(blank)
    tx = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.5))
    tx.text = f"{program} – Labor Hours Performance"
    add_table(slide, labor, "Labor", color_vac)

    # Slide 3 – Cost Performance
    slide = prs.slides.add_slide(blank)
    tx = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.5))
    tx.text = f"{program} – Cost Performance"
    add_table(slide, cost_tbl, "Cost", color_spi_cpi)

    # Slide 4 – Schedule Performance
    slide = prs.slides.add_slide(blank)
    tx = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(8), Inches(0.5))
    tx.text = f"{program} – Schedule Performance"
    add_table(slide, sched_tbl, "Schedule", color_spi_cpi)

    # Save file
    out = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out)

    print(f"✔ Saved → {out}")

print("\nALL EVMS DASHBOARDS COMPLETE ✔")