import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ============================================================
# USER CONFIG
# ============================================================

DATA_DIR = "data"
OUTPUT = "EVMS_Dashboard_Output.pptx"

PROGRAM_FILES = {
    "Abrams_STS": "Cobra-Abrams STS.xlsx",
    "XM30": "Cobra-XM30.xlsx",
}

OPENPLAN_FILE = "OpenPlan_Activity-Penske.xlsx"

COLOR_THRESHOLDS = {
    "GOOD": RGBColor(0, 128, 0),
    "WARN": RGBColor(255, 204, 0),
    "BAD": RGBColor(204, 0, 0),
}

# ============================================================
# LOADERS
# ============================================================

def load_cobra_auto(path):
    """
    Reads Cobra export when sheet name varies.
    Automatically selects the sheet containing "Extract" or "Weekly".
    """
    xl = pd.ExcelFile(path)
    sheet = None
    for s in xl.sheet_names:
        if ("extract" in s.lower()) or ("weekly" in s.lower()):
            sheet = s
            break
    if sheet is None:
        sheet = xl.sheet_names[0]

    df = xl.parse(sheet)
    df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed")]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def load_openplan(path):
    df = pd.read_excel(path)
    for c in ["Baseline Finish", "Actual Finish"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ============================================================
# EV CALCULATOR (ACWP, BCWP, BCWS, ETC)
# ============================================================

def compute_ev(cobra_df):

    df = cobra_df.copy()
    df = df[df["DATE"].notna()]   # remove blanks
    df = df[df["HOURS"].notna()]  # needed

    pivot = df.pivot_table(
        index=["SUB_TEAM", "DATE"],
        columns="COST-SET",
        values="HOURS",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # CTD is max date
    max_date = pivot["DATE"].max()
    ctd = pivot[pivot["DATE"] == max_date].copy()

    # LSP = previous period
    prev_date = (pivot[pivot["DATE"] < max_date]["DATE"].max())
    lsp = pivot[pivot["DATE"] == prev_date].copy()

    # rename cost sets if missing
    for col in ["ACWP", "BCWP", "BCWS", "ETC"]:
        if col not in ctd.columns:
            ctd[col] = 0
        if col not in lsp.columns:
            lsp[col] = 0

    # computed values
    def add_metrics(df):
        df["SPI"] = np.where(df["BCWS"] == 0, np.nan, df["BCWP"] / df["BCWS"])
        df["CPI"] = np.where(df["ACWP"] == 0, np.nan, df["BCWP"] / df["ACWP"])
        df["%COMP"] = np.where(df["BCWS"] == 0, np.nan, (df["BCWP"] / df["BCWS"]) * 100)
        df["BAC"] = df["BCWS"]
        df["EAC"] = df["ACWP"] + df["ETC"]
        df["VAC"] = df["BAC"] - df["EAC"]
        return df

    return add_metrics(ctd), add_metrics(lsp)


# ============================================================
# BEI CALCULATOR
# ============================================================

def compute_bei(openplan_df, subteam_col="SubTeam"):
    df = openplan_df.copy()

    # total tasks: baseline finish <= snapshot date
    SNAPSHOT = df["Baseline Finish"].max()

    t_total = df[(df["Baseline Finish"].notna()) &
                 (df["Baseline Finish"] <= SNAPSHOT)]

    t_done = df[df["Actual Finish"].notna()]

    total = t_total.groupby(subteam_col).size()
    done = t_done.groupby(subteam_col).size()

    all_idx = total.index.union(done.index)

    total = total.reindex(all_idx, fill_value=0)
    done = done.reindex(all_idx, fill_value=0)

    bei = done / total.replace(0, np.nan)
    bei_tbl = pd.DataFrame({"SUB_TEAM": all_idx, "BEI": bei.values})
    return bei_tbl


# ============================================================
# MERGE EV (CTD/LSP) + BEI
# ============================================================

def merge_ev(ev_ctd, ev_lsp, bei_tbl):
    ev_ctd = ev_ctd.rename(columns={
        "SPI": "SPI_CTD",
        "CPI": "CPI_CTD",
        "%COMP": "%COMP_CTD"
    })
    ev_lsp = ev_lsp.rename(columns={
        "SPI": "SPI_LSP",
        "CPI": "CPI_LSP",
        "%COMP": "%COMP_LSP"
    })

    ev = ev_ctd.merge(ev_lsp[["SUB_TEAM", "SPI_LSP", "CPI_LSP", "%COMP_LSP"]],
                      on="SUB_TEAM", how="left")

    ev = ev.merge(bei_tbl, on="SUB_TEAM", how="left")
    return ev


# ============================================================
# MANPOWER (DEMAND TABLE)
# ============================================================

def compute_manpower(cobra_df):
    # Assume monthly demand = sum(BCWS) per month
    cobra_df["MONTH"] = cobra_df["DATE"].dt.to_period("M")

    m = cobra_df.groupby("MONTH")["HOURS"].sum()

    months = sorted(m.index)

    if len(months) < 3:
        return pd.DataFrame({"Metric": ["Demand", "Actual"], "Last Month": 0, "This Month": 0, "Next Month": 0})

    lm, tm, nm = months[-3], months[-2], months[-1]

    tbl = pd.DataFrame({
        "Metric": ["Demand"],
        "Last Month": [m[lm]],
        "This Month": [m[tm]],
        "Next Month": [m[nm]],
        "%Var": [((m[tm] - m[lm]) / m[lm] * 100) if m[lm] != 0 else np.nan]
    })
    return tbl


# ============================================================
# COLORING FOR PPT
# ============================================================

def color_cell(cell, val):
    if pd.isna(val):
        return
    if val >= 0.98:
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_THRESHOLDS["GOOD"]
    elif val >= 0.90:
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_THRESHOLDS["WARN"]
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_THRESHOLDS["BAD"]


# ============================================================
# BUILD POWERPOINT
# ============================================================

def add_table_slide(prs, title, df):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title_box = slide.shapes.title
    title_box.text = title

    rows, cols = df.shape[0] + 1, df.shape[1]
    table = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.2),
                                   Inches(9), Inches(5)).table

    for j, col in enumerate(df.columns):
        table.cell(0, j).text = col

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iloc[i, j]
            cell = table.cell(i + 1, j)
            txt = "" if pd.isna(val) else str(val)
            cell.text = txt

            # apply coloring to numeric performance columns
            if df.columns[j] in ["CPI CTD", "CPI LSP", "SPI CTD", "SPI LSP", "BEI", "%COMP", "VAC"]:
                if isinstance(val, (int, float)):
                    color_cell(cell, val)

    return slide


# ============================================================
# RUN PIPELINE
# ============================================================

openplan = load_openplan(os.path.join(DATA_DIR, OPENPLAN_FILE))
bei_tbl = compute_bei(openplan)

prs = Presentation()

for program, file in PROGRAM_FILES.items():

    cobra = load_cobra_auto(os.path.join(DATA_DIR, file))
    ev_ctd, ev_lsp = compute_ev(cobra)
    ev = merge_ev(ev_ctd, ev_lsp, bei_tbl)

    # COST PERFORMANCE TABLE
    cost_tbl = ev[["SUB_TEAM", "CPI_LSP", "CPI_CTD"]].copy()
    cost_tbl["Comments"] = ""

    # SCHEDULE PERFORMANCE TABLE
    sched_tbl = ev[["SUB_TEAM", "SPI_LSP", "SPI_CTD", "BEI"]].copy()
    sched_tbl["Comments"] = ""

    # LABOR HOURS TABLE
    labor_tbl = ev[["SUB_TEAM", "%COMP_CTD", "BAC", "EAC", "VAC"]].copy()
    labor_tbl.rename(columns={"%COMP_CTD": "%COMP"}, inplace=True)
    labor_tbl["Comments"] = ""

    # MANPOWER TABLE
    manpower_tbl = compute_manpower(cobra)

    # EV SUMMARY TABLE
    summary_tbl = pd.DataFrame({
        "Metric": ["SPI", "CPI", "BEI", "%COMP"],
        "CTD": [
            ev["SPI_CTD"].mean(),
            ev["CPI_CTD"].mean(),
            ev["BEI"].mean(),
            ev["%COMP_CTD"].mean()
        ],
        "LSP": [
            ev["SPI_LSP"].mean(),
            ev["CPI_LSP"].mean(),
            ev["BEI"].mean(),
            ev["%COMP_LSP"].mean()
        ],
        "Comments": ""
    })

    # ADD SLIDES
    add_table_slide(prs, f"{program} – EV Summary", summary_tbl)
    add_table_slide(prs, f"{program} – Cost Performance", cost_tbl)
    add_table_slide(prs, f"{program} – Schedule Performance", sched_tbl)
    add_table_slide(prs, f"{program} – Labor Hours Performance", labor_tbl)
    add_table_slide(prs, f"{program} – Demand Table", manpower_tbl)

prs.save(OUTPUT)

print("\nEVMS DASHBOARD COMPLETE →", OUTPUT)