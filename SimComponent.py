# ============================================================
# EVMS DASHBOARD – Full Automated Pipeline (Final Version)
# Programs: Abrams_STS, Abrams_STS_2022, XM30
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
OPENPLAN_FILE = "OpenPlan_Activity-Penske.xlsx"

PROGRAM_FILES = {
    "Abrams_STS":       "Cobra-Abrams STS.xlsx",
    "Abrams_STS_2022":  "Cobra-Abrams STS 2022.xlsx",
    "XM30":             "Cobra-XM30.xlsx"
}

COBRA_SHEET = "tbl_Weekly Extract"

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNAPSHOT_DATE = datetime.now().date()


# ============================================================
# COLOR RULES (Using thresholds exactly from your image)
# ============================================================

# For SPI, CPI, BEI (ratio-based)
def color_spi_cpi(val):
    if pd.isna(val): return None
    if val >= 1.05: return RGBColor(31, 73, 125)     # blue
    if 1.05 > val >= 0.98: return RGBColor(142, 180, 227)  # light blue
    if 0.98 > val >= 0.95: return RGBColor(51, 153, 102)   # green
    if 0.95 > val >= 0.90: return RGBColor(255, 255, 153)  # yellow
    return RGBColor(192, 80, 77)  # red


# For VAC thresholds
def color_vac(val):
    if pd.isna(val): return None
    if val >= 0.05: return RGBColor(31, 73, 125)      # blue
    if 0.05 > val >= -0.02: return RGBColor(142, 180, 227)
    if -0.02 > val >= -0.05: return RGBColor(255, 255, 153)
    if -0.05 > val >= -0.10: return RGBColor(255, 204, 153)
    return RGBColor(192, 80, 77)  # red


# ============================================================
# CLEAN COBRA INPUT
# ============================================================

def load_cobra(path):
    df = pd.read_excel(path, sheet_name=COBRA_SHEET)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df.dropna(subset=["DATE"])


# ============================================================
# GET LAST STATUS PERIOD (LSP)
# ============================================================

def get_lsp(df):
    last = df["DATE"].max()
    df_lsp = df[df["DATE"] == last]
    return last, df_lsp


# ============================================================
# COMPUTE EV METRICS (SPI, CPI, BEI, %COMPLETE)
# ============================================================

def compute_ev_metrics(df):
    g = df.groupby(["SUB_TEAM", "COST-SET"])["HOURS"].sum().unstack(fill_value=0)

    for col in ["BCWP", "BCWS", "ACWP"]:
        if col not in g.columns:
            g[col] = 0

    g["SPI"] = np.where(g["BCWS"] == 0, np.nan, g["BCWP"] / g["BCWS"])
    g["CPI"] = np.where(g["ACWP"] == 0, np.nan, g["BCWP"] / g["ACWP"])

    # Percent Complete
    g["%COMP"] = np.where(
        g["BCWS"] == 0, np.nan, (g["BCWP"] / g["BCWS"]) * 100
    )

    return g.reset_index()


# ============================================================
# COMPUTE BEI FROM OPENPLAN
# ============================================================

def compute_bei(openplan, snapshot, program):
    op = openplan.copy()

    # Filter to program and valid tasks
    op = op[op["Program"] == program]
    op = op[op["Activity_Type"].isin(["A", "B"])]

    op["Baseline Finish"] = pd.to_datetime(op["Baseline Finish"], errors="coerce")
    op["Actual Finish"] = pd.to_datetime(op["Actual Finish"], errors="coerce")

    baseline_tasks = op[op["Baseline Finish"] <= snapshot]
    completed_tasks = op[op["Actual Finish"] <= snapshot]

    total_baseline = baseline_tasks.groupby("SubTeam")["Activity ID"].count()
    total_completed = completed_tasks.groupby("SubTeam")["Activity ID"].count()

    bei = pd.DataFrame({
        "SubTeam": total_baseline.index,
        "Baseline Tasks": total_baseline.values,
        "Completed Tasks": total_completed.reindex(total_baseline.index).fillna(0).values
    })

    bei["BEI"] = np.where(
        bei["Baseline Tasks"] == 0, np.nan,
        bei["Completed Tasks"] / bei["Baseline Tasks"]
    )

    return bei


# ============================================================
# LABOR HOURS PERFORMANCE TABLE
# ============================================================

def compute_labor(cobra_ev):
    df = cobra_ev.copy()
    df["BAC"] = df["BCWS"]
    df["EAC"] = df["ACWP"] + df.get("ETC", 0)
    df["VAC"] = df["BAC"] - df["EAC"]
    return df[["SUB_TEAM", "%COMP", "BAC", "EAC", "VAC"]]


# ============================================================
# COST PERFORMANCE TABLE (CPI CTD + CPI LSP)
# ============================================================

def make_cpi_table(ev_ctd, ev_lsp):
    df = pd.DataFrame()
    df["SUB_TEAM"] = ev_ctd["SUB_TEAM"]
    df["CPI CTD"] = ev_ctd["CPI"]
    df["CPI LSP"] = ev_lsp["CPI"]
    df["Comments"] = ""
    return df


# ============================================================
# SCHEDULE PERFORMANCE TABLE (SPI + BEI)
# ============================================================

def make_spi_bei_table(ev_ctd, ev_lsp, bei):
    df = pd.DataFrame()
    df["SUB_TEAM"] = ev_ctd["SUB_TEAM"]
    df["SPI CTD"] = ev_ctd["SPI"]
    df["SPI LSP"] = ev_lsp["SPI"]
    df["BEI CTD"] = bei.groupby("SubTeam")["BEI"].max().reindex(ev_ctd["SUB_TEAM"]).values
    df["BEI LSP"] = df["BEI CTD"]  # If separate needed, replace
    df["Comments"] = ""
    return df


# ============================================================
# EV TREND PLOT
# ============================================================

def make_ev_plot(ev_dates, program):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ev_dates["DATE"], y=ev_dates["SPI"], name="SPI (CTD)",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=ev_dates["DATE"], y=ev_dates["CPI"], name="CPI (CTD)",
        line=dict(width=3)
    ))

    fig.update_layout(
        title=f"{program} – EVMS Trend",
        template="plotly_white",
        yaxis=dict(range=[0, 2])
    )

    out = os.path.join(OUTPUT_DIR, f"{program}_EV_Trend.png")
    fig.write_image(out, scale=3)
    return out


# ============================================================
# ADD TABLE TO POWERPOINT WITH COLORING
# ============================================================

def add_table_to_slide(slide, df, title, color_rule):
    rows, cols = df.shape
    table = slide.shapes.add_table(rows+1, cols, Inches(0.3), Inches(1.0), Inches(9), Inches(0.3*(rows+1))).table

    # Write header
    for c, col in enumerate(df.columns):
        table.cell(0, c).text = col
        table.cell(0, c).text_frame.paragraphs[0].font.bold = True
        table.cell(0, c).fill.solid()
        table.cell(0, c).fill.fore_color.rgb = RGBColor(31, 73, 125)
        table.cell(0, c).text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)

    # Write body
    for r in range(rows):
        for c in range(cols):
            val = df.iloc[r, c]
            cell = table.cell(r+1, c)
            cell.text = "" if pd.isna(val) else str(round(val, 3) if isinstance(val, float) else val)

            # Apply color if rule provided
            if color_rule and isinstance(val, (int, float)):
                color = color_rule(val)
                if color:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = color


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

openplan = pd.read_excel(os.path.join(DATA_DIR, OPENPLAN_FILE))

for program, cobra_file in PROGRAM_FILES.items():
    print(f"\nProcessing → {program}")

    cobra_path = os.path.join(DATA_DIR, cobra_file)
    cobra = load_cobra(cobra_path)

    # CTD EV
    ev_ctd = compute_ev_metrics(cobra)

    # LSP EV
    lsp_date, cobra_lsp = get_lsp(cobra)
    ev_lsp = compute_ev_metrics(cobra_lsp)

    # BEI
    bei = compute_bei(openplan, SNAPSHOT_DATE, program)

    # Labor Hours
    labor = compute_labor(ev_ctd)

    # Cost Performance
    cpi_tbl = make_cpi_table(ev_ctd, ev_lsp)

    # Schedule Performance
    spi_tbl = make_spi_bei_table(ev_ctd, ev_lsp, bei)

    # EV Trend Plot
    plot_path = make_ev_plot(ev_ctd.rename(columns={"DATE":"DATE"}).assign(DATE=cobra["DATE"]), program)

    # Build PowerPoint
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # Slide 1 — EVMS Summary / Metrics
    slide = prs.slides.add_slide(blank)
    slide.shapes.title = slide.shapes.title or slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.5))
    slide.shapes.title.text = f"{program} – EVMS Metrics"

    summary_tbl = pd.DataFrame({
        "Metric": ["SPI", "CPI", "BEI", "% Complete"],
        "CTD": [
            ev_ctd["SPI"].mean(),
            ev_ctd["CPI"].mean(),
            bei["BEI"].mean(),
            ev_ctd["%COMP"].mean()
        ],
        "LSP": [
            ev_lsp["SPI"].mean(),
            ev_lsp["CPI"].mean(),
            bei["BEI"].mean(),
            ev_lsp["%COMP"].mean()
        ],
        "Comments": ["", "", "", ""]
    })

    add_table_to_slide(slide, summary_tbl, "EV Summary", color_spi_cpi)

    # Slide 2 — Labor Hours Performance
    slide = prs.slides.add_slide(blank)
    slide.shapes.title = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.5))
    slide.shapes.title.text = f"{program} – Labor Hours Performance"
    add_table_to_slide(slide, labor, "Labor", color_vac)

    # Slide 3 — Cost Performance
    slide = prs.slides.add_slide(blank)
    slide.shapes.title = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.5))
    slide.shapes.title.text = f"{program} – Cost Performance"
    add_table_to_slide(slide, cpi_tbl, "CPI", color_spi_cpi)

    # Slide 4 — Schedule Performance
    slide = prs.slides.add_slide(blank)
    slide.shapes.title = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.5))
    slide.shapes.title.text = f"{program} – Schedule Performance"
    add_table_to_slide(slide, spi_tbl, "SPI/BEI", color_spi_cpi)

    # Slide 5 — Trend Plot
    slide = prs.slides.add_slide(blank)
    slide.shapes.title = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.5))
    slide.shapes.title.text = f"{program} – EVMS Trend"
    slide.shapes.add_picture(plot_path, Inches(0.5), Inches(1.0), width=Inches(8))

    # Save file
    out = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out)
    print(f"✔ Saved → {out}")

print("\nALL EVMS DASHBOARDS COMPLETE ✔")