# =====================================================================
# GDLS EVMS DASHBOARD – COST / SCHEDULE / EVMS / LABOR TABLES + CHART
# =====================================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = "data"

cobra_files = {
    "Abrams_STS_2022": os.path.join(DATA_DIR, "Cobra-Abrams STS 2022.xlsx"),
    "Abrams_STS":      os.path.join(DATA_DIR, "Cobra-Abrams STS.xlsx"),
    "XM30":            os.path.join(DATA_DIR, "Cobra-XM30.xlsx"),
}

openplan_path = os.path.join(DATA_DIR, "OpenPlan_Activity-Penske.xlsx")

PROGRAM_NAME_MAP = {
    "Abrams_STS_2022": "ABRAMS STS",
    "Abrams_STS":      "ABRAMS STS",
    "XM30":            "XM30",
}

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accounting period close dates (from your calendars)
ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27},
}

# GDLS color palette
COLOR_BLUE   = RGBColor(31, 73, 125)
COLOR_TEAL   = RGBColor(142, 180, 227)
COLOR_GREEN  = RGBColor(51, 153, 102)
COLOR_YELLOW = RGBColor(255, 255, 153)
COLOR_RED    = RGBColor(192, 80, 77)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().upper().replace(" ", "_").replace("-", "_"))

def find_lsp_cutoff(dates: pd.Series) -> pd.Timestamp:
    dmax = dates.max()
    y, m = dmax.year, dmax.month
    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        close_day = ACCOUNTING_CLOSE[y][m]
        close_date = pd.Timestamp(year=y, month=m, day=close_day)
        eligible = dates[dates <= close_date]
        if not eligible.empty:
            return eligible.max()
    return dmax

def map_cost_sets(cols):
    cleaned = {c: c.replace("_", "").upper() for c in cols}
    bcws = bcwp = acwp = etc = None
    for orig, clean in cleaned.items():
        if "BCWS" in clean or "BUDGET" in clean:
            bcws = orig
        if "BCWP" in clean or "EARNED" in clean or "PROGRESS" in clean:
            bcwp = orig
        if "ACWP" in clean or ("ACTUAL" in clean and "FINISH" not in clean):
            acwp = orig
        if "ETC" in clean:
            etc = orig
    return bcws, bcwp, acwp, etc

# SPI / CPI / BEI thresholds
def idx_color(v):
    if pd.isna(v): return None
    if v >= 1.055: return COLOR_BLUE
    if v >= 1.02:  return COLOR_TEAL
    if v >= 0.975: return COLOR_GREEN
    if v >= 0.945: return COLOR_YELLOW
    return COLOR_RED

# VAC/BAC thresholds
def vac_color(v):
    if pd.isna(v): return None
    if v >= 0.055: return COLOR_BLUE
    if v >= 0.025: return COLOR_GREEN
    if v >= -0.025: return COLOR_YELLOW
    if v >= -0.055: return COLOR_TEAL
    return COLOR_RED

# PowerPoint helpers
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
    if color is not None:
        fill = cell.fill
        fill.solid()
        fill.fore_color.rgb = color

# ---------------------------------------------------------------------
# EV CALC – BY SUB_TEAM + PROGRAM
# ---------------------------------------------------------------------

def compute_ev_from_cobra(df_raw: pd.DataFrame):
    df = normalize(df_raw)

    if "PLUG" in df.columns:
        df = df[df["PLUG"] == "HOURS"]

    if "SUB_TEAM" not in df.columns:
        raise ValueError("Cobra file must contain SUB_TEAM column")
    if "DATE" not in df.columns:
        raise ValueError("Cobra file must contain DATE column")

    df["DATE"] = pd.to_datetime(df["DATE"])

    pivot = df.pivot_table(
        index=["DATE", "SUB_TEAM"],
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0.0).reset_index()

    cost_cols = [c for c in pivot.columns if c not in ["DATE", "SUB_TEAM"]]
    bcws_c, bcwp_c, acwp_c, etc_c = map_cost_sets(cost_cols)

    col_map = {}
    if bcws_c: col_map[bcws_c] = "BCWS"
    if bcwp_c: col_map[bcwp_c] = "BCWP"
    if acwp_c: col_map[acwp_c] = "ACWP"
    if etc_c:  col_map[etc_c]  = "ETC"
    pivot = pivot.rename(columns=col_map)
    for col in ["BCWS", "BCWP", "ACWP", "ETC"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot = pivot[["DATE", "SUB_TEAM", "BCWS", "BCWP", "ACWP", "ETC"]]
    pivot.sort_values(["SUB_TEAM", "DATE"], inplace=True)

    # cumulative per subteam
    for col in ["BCWS", "BCWP", "ACWP"]:
        pivot[f"{col}_CUM"] = pivot.groupby("SUB_TEAM")[col].cumsum()

    # program time series for trend + program CTD
    prog_ts = pivot.groupby("DATE")[["BCWS", "BCWP", "ACWP"]].sum()
    prog_ts["BCWS_CUM"] = prog_ts["BCWS"].cumsum()
    prog_ts["BCWP_CUM"] = prog_ts["BCWP"].cumsum()
    prog_ts["ACWP_CUM"] = prog_ts["ACWP"].cumsum()

    # program LSP
    lsp_date = find_lsp_cutoff(pivot["DATE"])

    # per-subteam CTD & LSP indices
    sub_rows = []
    for st, grp in pivot.groupby("SUB_TEAM"):
        last = grp.iloc[-1]
        bcws_ctd = last["BCWS_CUM"]
        bcwp_ctd = last["BCWP_CUM"]
        acwp_ctd = last["ACWP_CUM"]
        etc_last = last["ETC"]

        eac = acwp_ctd + etc_last
        vac = bcws_ctd - eac

        spi_ctd = bcwp_ctd / bcws_ctd if bcws_ctd else np.nan
        cpi_ctd = bcwp_ctd / acwp_ctd if acwp_ctd else np.nan
        pct_comp = bcwp_ctd / bcws_ctd if bcws_ctd else np.nan

        # LSP row for that subteam (closest <= lsp_date)
        lsp_grp = grp[grp["DATE"] <= lsp_date]
        if not lsp_grp.empty:
            lsp_row = lsp_grp.iloc[-1]
            spi_lsp = lsp_row["BCWP"] / lsp_row["BCWS"] if lsp_row["BCWS"] else np.nan
            cpi_lsp = lsp_row["BCWP"] / lsp_row["ACWP"] if lsp_row["ACWP"] else np.nan
        else:
            spi_lsp = cpi_lsp = np.nan

        sub_rows.append({
            "SUB_TEAM": st,
            "BAC": bcws_ctd,
            "EAC": eac,
            "VAC": vac,
            "ACWP_CTD": acwp_ctd,
            "SPI_CTD": spi_ctd,
            "CPI_CTD": cpi_ctd,
            "SPI_LSP": spi_lsp,
            "CPI_LSP": cpi_lsp,
            "PCT_COMP": pct_comp,
        })

    sub_df = pd.DataFrame(sub_rows).set_index("SUB_TEAM")

    # program CTD from prog_ts
    prog_last = prog_ts.iloc[-1]
    bcws_prog = prog_last["BCWS_CUM"]
    bcwp_prog = prog_last["BCWP_CUM"]
    acwp_prog = prog_last["ACWP_CUM"]

    spi_ctd_prog = bcwp_prog / bcws_prog if bcws_prog else np.nan
    cpi_ctd_prog = bcwp_prog / acwp_prog if acwp_prog else np.nan
    pct_comp_prog = bcwp_prog / bcws_prog if bcws_prog else np.nan

    # program LSP indices from program curve
    prog_lsp = prog_ts[prog_ts.index <= lsp_date]
    if not prog_lsp.empty:
        lsp_row = prog_lsp.iloc[-1]
        spi_lsp_prog = lsp_row["BCWP_CUM"] / lsp_row["BCWS_CUM"] if lsp_row["BCWS_CUM"] else np.nan
        cpi_lsp_prog = lsp_row["BCWP_CUM"] / lsp_row["ACWP_CUM"] if lsp_row["ACWP_CUM"] else np.nan
    else:
        spi_lsp_prog = cpi_lsp_prog = np.nan

    prog_summary = dict(
        LSP_DATE=lsp_date,
        BAC=bcws_prog,
        ACWP_CTD=acwp_prog,
        SPI_CTD=spi_ctd_prog,
        CPI_CTD=cpi_ctd_prog,
        SPI_LSP=spi_lsp_prog,
        CPI_LSP=cpi_lsp_prog,
        PCT_COMP=pct_comp_prog,
        prog_ts=prog_ts,
    )

    return pivot, sub_df, prog_summary

# ---------------------------------------------------------------------
# BEI FROM OPENPLAN – using CTD / YTD fields
# ---------------------------------------------------------------------

def compute_bei_from_openplan(openplan_norm: pd.DataFrame, program_key: str):
    pname = PROGRAM_NAME_MAP[program_key]
    df = openplan_norm[openplan_norm["PROGRAM"] == pname].copy()
    if df.empty:
        print(f"⚠ BEI: no OpenPlan rows for {pname}")
        return np.nan, np.nan, pd.DataFrame()

    # Expected columns after normalize()
    for col in ["CTD_ACTUAL", "CTD_BASE", "YTD_ACTUAL", "YTD_BASE"]:
        if col not in df.columns:
            print(f"⚠ BEI: missing {col} for {pname}")
            return np.nan, np.nan, pd.DataFrame()

    # program totals
    ctd_act = df["CTD_ACTUAL"].sum()
    ctd_base = df["CTD_BASE"].sum()
    ytd_act = df["YTD_ACTUAL"].sum()
    ytd_base = df["YTD_BASE"].sum()

    bei_ctd_prog = ctd_act / ctd_base if ctd_base else np.nan
    bei_lsp_prog = ytd_act / ytd_base if ytd_base else bei_ctd_prog

    # subteam BEI
    sub_col = None
    if "SUBTEAM" in df.columns:
        sub_col = "SUBTEAM"
    elif "SUB_TEAM" in df.columns:
        sub_col = "SUB_TEAM"

    bei_sub = []
    if sub_col:
        for st, grp in df.groupby(sub_col):
            ctd_act = grp["CTD_ACTUAL"].sum()
            ctd_base = grp["CTD_BASE"].sum()
            ytd_act = grp["YTD_ACTUAL"].sum()
            ytd_base = grp["YTD_BASE"].sum()

            bei_ctd = ctd_act / ctd_base if ctd_base else np.nan
            bei_lsp = ytd_act / ytd_base if ytd_base else bei_ctd
            bei_sub.append({"SUB_TEAM": st, "BEI_CTD": bei_ctd, "BEI_LSP": bei_lsp})

    bei_sub_df = pd.DataFrame(bei_sub).set_index("SUB_TEAM") if bei_sub else pd.DataFrame()

    return bei_ctd_prog, bei_lsp_prog, bei_sub_df

# ---------------------------------------------------------------------
# EVMS TREND CHART
# ---------------------------------------------------------------------

def make_evms_chart(program, prog_ts: pd.DataFrame):
    df = prog_ts.copy()
    df["SPI_CUM"] = df["BCWP_CUM"] / df["BCWS_CUM"].replace(0, np.nan)
    df["CPI_CUM"] = df["BCWP_CUM"] / df["ACWP_CUM"].replace(0, np.nan)
    df["SPI_M"] = df["BCWP"] / df["BCWS"].replace(0, np.nan)
    df["CPI_M"] = df["BCWP"] / df["ACWP"].replace(0, np.nan)

    fig = go.Figure()

    # threshold bands (same as dashboard)
    fig.add_hrect(y0=0.90,  y1=0.945, fillcolor="red",    opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.945, y1=0.975, fillcolor="yellow", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.975, y1=1.02,  fillcolor="green",  opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.02,  y1=1.055, fillcolor="lightblue", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.055, y1=1.20,  fillcolor="rgb(200,220,255)", opacity=0.2, line_width=0)

    fig.add_trace(go.Scatter(x=df.index, y=df["CPI_CUM"], mode="lines", name="CPI (Cum)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SPI_CUM"], mode="lines", name="SPI (Cum)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["CPI_M"],   mode="markers", name="CPI (M)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SPI_M"],   mode="markers", name="SPI (M)"))

    fig.update_layout(
        title=f"{program} EVMS Trend",
        yaxis=dict(range=[0.9, 1.2], title="EV Indices"),
        xaxis_title="Month",
        template="simple_white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom")
    )
    return fig

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

openplan_raw = pd.read_excel(openplan_path)
openplan_norm = normalize(openplan_raw)

for program, cobra_path in cobra_files.items():
    print(f"\nProcessing → {program} from {cobra_path}")

    cobra_raw = pd.read_excel(cobra_path)
    ts, sub_ev, prog_summary = compute_ev_from_cobra(cobra_raw)

    bei_ctd_prog, bei_lsp_prog, bei_sub = compute_bei_from_openplan(openplan_norm, program)

    # merge BEI onto subteam EV table
    perf = sub_ev.join(bei_sub, how="left")

    # ----------------- PROGRAM EVMS METRICS TABLE --------------------
    evms_metrics = pd.DataFrame(
        {
            "CTD": [
                prog_summary["SPI_CTD"],
                prog_summary["CPI_CTD"],
                bei_ctd_prog,
                prog_summary["PCT_COMP"],
            ],
            "LSP": [
                prog_summary["SPI_LSP"],
                prog_summary["CPI_LSP"],
                bei_lsp_prog,
                prog_summary["PCT_COMP"],   # % complete at LSP ~ CTD
            ],
        },
        index=["SPI", "CPI", "BEI", "% Complete"]
    )

    # ----------------- COST PERFORMANCE TABLE (SLIDE 1) --------------
    cost_perf = perf[["CPI_LSP", "CPI_CTD", "BEI_LSP", "BEI_CTD"]].copy()

    # ----------------- SCHEDULE PERFORMANCE TABLE (SLIDE 2) ----------
    sched_perf = perf[["SPI_LSP", "SPI_CTD", "BEI_LSP", "BEI_CTD"]].copy()

    # ----------------- LABOR HOURS PERFORMANCE (SLIDE 4) -------------
    labor_perf = perf[["PCT_COMP", "BAC", "EAC", "VAC"]].copy()

    # -----------------------------------------------------------------
    # BUILD POWERPOINT
    # -----------------------------------------------------------------
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # === SLIDE 1: COST PERFORMANCE (CPI + BEI, by Subteam) ===========
    s1 = prs.slides.add_slide(blank)
    t1 = s1.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    t1.text_frame.text = f"{program} – Cost Performance"

    tbl1 = add_table(
        s1,
        rows=cost_perf.shape[0] + 1,
        cols=5,
        left=Inches(0.3),
        top=Inches(0.8),
        width=Inches(9),
        height=Inches(3),
        headers=["Subteam", "CPI LSP", "CPI CTD", "BEI LSP", "BEI CTD"]
    )

    for i, (st, row) in enumerate(cost_perf.iterrows(), start=1):
        tbl1.cell(i, 0).text = str(st)
        for j, col in enumerate(["CPI_LSP", "CPI_CTD", "BEI_LSP", "BEI_CTD"], start=1):
            val = row.get(col, np.nan)
            tbl1.cell(i, j).text = "" if pd.isna(val) else f"{val:.2f}"
            set_bg(tbl1.cell(i, j), idx_color(val))

    # === SLIDE 2: SCHEDULE PERFORMANCE (SPI + BEI, by Subteam) =======
    s2 = prs.slides.add_slide(blank)
    t2 = s2.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    t2.text_frame.text = f"{program} – Schedule Performance"

    tbl2 = add_table(
        s2,
        rows=sched_perf.shape[0] + 1,
        cols=6,
        left=Inches(0.3),
        top=Inches(0.8),
        width=Inches(9),
        height=Inches(3),
        headers=["Subteam", "SPI LSP", "SPI CTD", "BEI LSP", "BEI CTD", "Comments"]
    )

    for i, (st, row) in enumerate(sched_perf.iterrows(), start=1):
        tbl2.cell(i, 0).text = str(st)
        for j, col in enumerate(["SPI_LSP", "SPI_CTD", "BEI_LSP", "BEI_CTD"], start=1):
            val = row.get(col, np.nan)
            tbl2.cell(i, j).text = "" if pd.isna(val) else f"{val:.2f}"
            set_bg(tbl2.cell(i, j), idx_color(val))
        # comments column left blank for RC/CA

    # === SLIDE 3: EVMS METRICS + TREND PLOT ==========================
    s3 = prs.slides.add_slide(blank)
    t3 = s3.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    t3.text_frame.text = f"{program} – EVMS Metrics & Trend"

    # EVMS metrics table (top-left)
    tbl3 = add_table(
        s3,
        rows=evms_metrics.shape[0] + 1,
        cols=3,
        left=Inches(0.3),
        top=Inches(0.7),
        width=Inches(4),
        height=Inches(1.5),
        headers=["Metric", "CTD", "LSP"]
    )
    for i, (metric, row) in enumerate(evms_metrics.iterrows(), start=1):
        tbl3.cell(i, 0).text = metric
        for j, col in enumerate(["CTD", "LSP"], start=1):
            val = row[col]
            txt = f"{val:.2f}" if metric != "% Complete" and not pd.isna(val) else \
                  (f"{val*100:,.1f}%" if metric == "% Complete" and not pd.isna(val) else "")
            tbl3.cell(i, j).text = txt
            if metric in ["SPI", "CPI", "BEI"]:
                set_bg(tbl3.cell(i, j), idx_color(val))

    # EVMS chart (right/below)
    fig = make_evms_chart(program, prog_summary["prog_ts"])
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Trend.png")
    fig.write_image(chart_path, scale=3)
    s3.shapes.add_picture(chart_path, Inches(4.5), Inches(0.7), width=Inches(5))

    # === SLIDE 4: LABOR HOURS PERFORMANCE ============================
    s4 = prs.slides.add_slide(blank)
    t4 = s4.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    t4.text_frame.text = f"{program} – Labor Hours Performance"

    tbl4 = add_table(
        s4,
        rows=labor_perf.shape[0] + 1,
        cols=6,
        left=Inches(0.3),
        top=Inches(0.8),
        width=Inches(9),
        height=Inches(3),
        headers=["Subteam", "%Comp", "BAC", "EAC", "VAC", "Comments"]
    )

    for i, (st, row) in enumerate(labor_perf.iterrows(), start=1):
        tbl4.cell(i, 0).text = str(st)
        pct = row["PCT_COMP"]
        tbl4.cell(i, 1).text = "" if pd.isna(pct) else f"{pct*100:,.1f}%"
        tbl4.cell(i, 2).text = f"{row['BAC']:,.0f}"
        tbl4.cell(i, 3).text = f"{row['EAC']:,.0f}"
        tbl4.cell(i, 4).text = f"{row['VAC']:,.0f}"
        vac_ratio = row["VAC"] / row["BAC"] if row["BAC"] else np.nan
        set_bg(tbl4.cell(i, 4), vac_color(vac_ratio))
        # comments column left blank

    # SAVE PPT
    out_file = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_file)
    print(f"Saved → {out_file}")

print("\nALL EVMS DASHBOARDS COMPLETE ✅")