# =====================================================================
# GDLS EVMS DASHBOARD GENERATOR (MULTI-TABLE, PER-SUBTEAM, FINAL)
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

# Accounting close (from your 2020, 2024 calendars)
ACCOUNTING_CLOSE = {
    2020: {1:20,2:21,3:30,4:20,5:25,6:26,7:27,8:24,9:28,10:19,11:23,12:29},
    2024: {1:15,2:21,3:29,4:19,5:27,6:26,7:26,8:23,9:30,10:18,11:22,12:27},
}

# 9/80 working hours per month (approx; tweak if needed)
WORKING_HOURS = {
    1: 144, 2: 160, 3: 176, 4: 168, 5: 176, 6: 160,
    7: 176, 8: 168, 9: 160, 10: 176, 11: 160, 12: 176,
}

# Colors from GDLS palette
COLOR_BLUE   = RGBColor(31, 73, 125)
COLOR_TEAL   = RGBColor(142, 180, 227)
COLOR_GREEN  = RGBColor(51, 153, 102)
COLOR_YELLOW = RGBColor(255, 255, 153)
COLOR_RED    = RGBColor(192, 80, 77)

# =====================================================================
# HELPERS
# =====================================================================

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().upper().replace(" ", "_").replace("-", "_"))

def find_lsp_cutoff(dates: pd.Series) -> pd.Timestamp:
    dmax = dates.max()
    y, m = dmax.year, dmax.month
    if y in ACCOUNTING_CLOSE and m in ACCOUNTING_CLOSE[y]:
        close_day = ACCOUNTING_CLOSE[y][m]
        close_date = pd.Timestamp(year=y, month=m, day=close_day)
        eligible = dates[dates <= close_date]
        return eligible.max() if not eligible.empty else dates.max()
    return dates.max()

def safe_lsp_date(dates: pd.Series, lsp: pd.Timestamp) -> pd.Timestamp:
    eligible = dates[dates <= lsp]
    return eligible.max() if not eligible.empty else dates.max()

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

# Threshold colors
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

def manpower_color(ratio):
    """Program Manpower thresholds based on Actual/Demand ratio."""
    if pd.isna(ratio): return None
    pct = ratio * 100
    if pct >= 110 or pct < 85: return COLOR_RED
    if pct >= 105: return COLOR_BLUE
    if pct >= 90:  return COLOR_GREEN
    if pct >= 85:  return COLOR_YELLOW
    return COLOR_RED

# PPT helpers
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

# =====================================================================
# EV COMPUTATION BY SUBTEAM + PROGRAM TOTAL
# =====================================================================

def compute_ev_by_subteam(df_raw: pd.DataFrame):
    df = normalize(df_raw)

    # Keep hours plug if present
    if "PLUG" in df.columns:
        df = df[df["PLUG"] == "HOURS"]

    # Required columns
    if "SUB_TEAM" not in df.columns:
        raise ValueError("Expected SUB_TEAM column in Cobra extract")

    if "DATE" not in df.columns:
        raise ValueError("Expected DATE column in Cobra extract")

    df["DATE"] = pd.to_datetime(df["DATE"])

    # Pivot by DATE & SUB_TEAM
    pivot = df.pivot_table(
        index=["DATE", "SUB_TEAM"],
        columns="COST_SET",
        values="HOURS",
        aggfunc="sum"
    ).fillna(0.0).reset_index()

    cost_cols = [c for c in pivot.columns if c not in ["DATE", "SUB_TEAM"]]
    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(cost_cols)

    # Rename standardized columns
    col_map = {
        bcws_col: "BCWS",
        bcwp_col: "BCWP",
        acwp_col: "ACWP",
    }
    if etc_col:
        col_map[etc_col] = "ETC"

    pivot = pivot.rename(columns=col_map)
    if "ETC" not in pivot.columns:
        pivot["ETC"] = 0.0

    pivot = pivot[["DATE", "SUB_TEAM", "BCWS", "BCWP", "ACWP", "ETC"]]
    pivot.sort_values(["SUB_TEAM", "DATE"], inplace=True)

    # Cumulative sums within each subteam
    for col in ["BCWS", "BCWP", "ACWP"]:
        pivot[f"{col}_CUM"] = pivot.groupby("SUB_TEAM")[col].cumsum()

    # Program-level series for trend chart
    prog_ts = pivot.groupby("DATE")[["BCWS", "BCWP", "ACWP"]].sum()
    prog_ts["BCWS_CUM"] = prog_ts["BCWS"].cumsum()
    prog_ts["BCWP_CUM"] = prog_ts["BCWP"].cumsum()
    prog_ts["ACWP_CUM"] = prog_ts["ACWP"].cumsum()

    # LSP (program-wide, then closest per subteam)
    lsp_cutoff = find_lsp_cutoff(pivot["DATE"])
    lsp_date = safe_lsp_date(pivot["DATE"], lsp_cutoff)

    # Summaries per subteam
    sub_ctd = []
    sub_lsp = []

    for st, grp in pivot.groupby("SUB_TEAM"):
        last = grp.iloc[-1]
        BCWS_CTD = last["BCWS_CUM"]
        BCWP_CTD = last["BCWP_CUM"]
        ACWP_CTD = last["ACWP_CUM"]
        ETC_last = last["ETC"]

        EAC = ACWP_CTD + ETC_last
        VAC = BCWS_CTD - EAC

        SPI_CTD = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan
        CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD else np.nan
        PCT_COMP = BCWP_CTD / BCWS_CTD if BCWS_CTD else np.nan

        # row at LSP or closest before it
        lsp_row = grp[grp["DATE"] <= lsp_date]
        if not lsp_row.empty:
            l_row = lsp_row.iloc[-1]
            SPI_LSP = l_row["BCWP"] / l_row["BCWS"] if l_row["BCWS"] else np.nan
            CPI_LSP = l_row["BCWP"] / l_row["ACWP"] if l_row["ACWP"] else np.nan
        else:
            SPI_LSP = CPI_LSP = np.nan

        sub_ctd.append({
            "SUB_TEAM": st,
            "BAC": BCWS_CTD,
            "ACWP_CTD": ACWP_CTD,
            "EAC": EAC,
            "VAC": VAC,
            "SPI_CTD": SPI_CTD,
            "CPI_CTD": CPI_CTD,
            "PCT_COMP": PCT_COMP,
        })
        sub_lsp.append({
            "SUB_TEAM": st,
            "SPI_LSP": SPI_LSP,
            "CPI_LSP": CPI_LSP,
        })

    sub_ctd_df = pd.DataFrame(sub_ctd).set_index("SUB_TEAM")
    sub_lsp_df = pd.DataFrame(sub_lsp).set_index("SUB_TEAM")

    # Program total row
    total = sub_ctd_df.sum(numeric_only=True)
    total.name = "TOTAL"
    sub_ctd_df = pd.concat([sub_ctd_df, total.to_frame().T])

    # For LSP, compute from prog_ts (safer)
    prog_lsp_series = prog_ts.loc[:lsp_date].iloc[-1]
    SPI_LSP_prog = prog_lsp_series["BCWP"] / prog_lsp_series["BCWS"] if prog_lsp_series["BCWS"] else np.nan
    CPI_LSP_prog = prog_lsp_series["BCWP"] / prog_lsp_series["ACWP"] if prog_lsp_series["ACWP"] else np.nan

    # Program-level CTD metrics
    last_prog = prog_ts.iloc[-1]
    BCWS_CTD_prog = last_prog["BCWS_CUM"]
    BCWP_CTD_prog = last_prog["BCWP_CUM"]
    ACWP_CTD_prog = last_prog["ACWP_CUM"]

    EAC_prog = ACWP_CTD_prog + 0.0  # no program-level ETC in this extract
    VAC_prog = BCWS_CTD_prog - EAC_prog
    SPI_CTD_prog = BCWP_CTD_prog / BCWS_CTD_prog if BCWS_CTD_prog else np.nan
    CPI_CTD_prog = BCWP_CTD_prog / ACWP_CTD_prog if ACWP_CTD_prog else np.nan
    PCT_COMP_prog = BCWP_CTD_prog / BCWS_CTD_prog if BCWS_CTD_prog else np.nan

    prog_summary = dict(
        LSP_DATE=lsp_date,
        BAC=BCWS_CTD_prog,
        EAC=EAC_prog,
        VAC=VAC_prog,
        ACWP_CTD=ACWP_CTD_prog,
        SPI_CTD=SPI_CTD_prog,
        CPI_CTD=CPI_CTD_prog,
        SPI_LSP=SPI_LSP_prog,
        CPI_LSP=CPI_LSP_prog,
        PCT_COMP=PCT_COMP_prog,
        prog_ts=prog_ts,
    )

    return pivot, sub_ctd_df, sub_lsp_df, prog_summary

# =====================================================================
# BEI FROM OPENPLAN (PROGRAM + SUBTEAM)
# =====================================================================

def compute_bei(openplan_norm: pd.DataFrame, program_key: str, lsp_date: pd.Timestamp):
    pname = PROGRAM_NAME_MAP[program_key]
    df = openplan_norm[openplan_norm["PROGRAM"] == pname].copy()
    if df.empty:
        return np.nan, np.nan, pd.DataFrame()

    # Normalize BEI-related cols
    base_col = "BASELINE_FINISH"
    act_col  = "ACTUAL_FINISH"

    if base_col not in df.columns:
        # best-effort fallback: look for BASELINE_FINISH-like
        base_col = next(c for c in df.columns if "BASELINE_FINISH" in c)
    if act_col not in df.columns:
        act_col = next((c for c in df.columns if "ACTUAL_FINISH" in c), base_col)

    df[base_col] = pd.to_datetime(df[base_col], errors="coerce")
    df[act_col]  = pd.to_datetime(df[act_col], errors="coerce")

    # Exclude milestones/LOE
    if "ACTIVITY_TYPE" in df.columns:
        df = df[~df["ACTIVITY_TYPE"].str.contains("M|LOE", na=False)]

    # Program-level BEI
    planned = df[df[base_col] <= lsp_date]
    completed = planned[planned[act_col].notna() & (planned[act_col] <= lsp_date)]
    bei_lsp_prog = len(completed) / len(planned) if len(planned) else np.nan
    bei_ctd_prog = bei_lsp_prog  # as of LSP, CTD = LSP for BEI

    # Subteam BEI
    sub_col = "SUBTEAM" if "SUBTEAM" in df.columns else "SUB_TEAM" if "SUB_TEAM" in df.columns else None
    bei_sub = []
    if sub_col:
        for st, grp in df.groupby(sub_col):
            p = grp[grp[base_col] <= lsp_date]
            c = p[p[act_col].notna() & (p[act_col] <= lsp_date)]
            bei_lsp = len(c) / len(p) if len(p) else np.nan
            bei_sub.append({"SUB_TEAM": st, "BEI_LSP": bei_lsp, "BEI_CTD": bei_lsp})
    bei_sub_df = pd.DataFrame(bei_sub).set_index("SUB_TEAM") if bei_sub else pd.DataFrame()

    return bei_ctd_prog, bei_lsp_prog, bei_sub_df

# =====================================================================
# MANPOWER (DEMAND/ACTUAL BY MONTH)
# =====================================================================

def compute_manpower(prog_ts: pd.DataFrame, lsp_date: pd.Timestamp):
    """Return Demand, Actual, Last, Next, %Var for program."""
    # Per-month BCWS and ACWP
    m_index = prog_ts.index.to_period("M")
    bcws_month = prog_ts["BCWS"].groupby(m_index).sum()
    acwp_month = prog_ts["ACWP"].groupby(m_index).sum()

    lsp_month = lsp_date.to_period("M")
    this_m = lsp_month
    last_m = this_m - 1
    next_m = this_m + 1

    def demand_for(period):
        if period not in bcws_month.index:
            return np.nan
        hrs = WORKING_HOURS.get(period.month, np.nan)
        return bcws_month.loc[period] / hrs if hrs else np.nan

    def actual_for(period):
        if period not in acwp_month.index:
            return np.nan
        hrs = WORKING_HOURS.get(period.month, np.nan)
        return acwp_month.loc[period] / hrs if hrs else np.nan

    last_demand = demand_for(last_m)
    this_demand = demand_for(this_m)
    next_demand = demand_for(next_m)
    this_actual = actual_for(this_m)

    pct_var = (this_actual - this_demand) / this_demand if this_demand else np.nan

    return last_demand, this_demand, next_demand, this_actual, pct_var

# =====================================================================
# EVMS TREND CHART
# =====================================================================

def make_evms_chart(program, prog_ts):
    df = prog_ts.copy()
    df["SPI_CUM"] = df["BCWP_CUM"] / df["BCWS_CUM"].replace(0, np.nan)
    df["CPI_CUM"] = df["BCWP_CUM"] / df["ACWP_CUM"].replace(0, np.nan)
    df["SPI_M"] = df["BCWP"] / df["BCWS"].replace(0, np.nan)
    df["CPI_M"] = df["BCWP"] / df["ACWP"].replace(0, np.nan)

    fig = go.Figure()

    # Bands
    fig.add_hrect(y0=0.90,  y1=0.945, fillcolor="red",    opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.945, y1=0.975, fillcolor="yellow", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.975, y1=1.02,  fillcolor="green",  opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.02,  y1=1.055, fillcolor="lightblue", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.055, y1=1.20,  fillcolor="rgb(200,220,255)", opacity=0.2, line_width=0)

    fig.add_trace(go.Scatter(x=df.index, y=df["SPI_CUM"], mode="lines", name="SPI (Cum)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["CPI_CUM"], mode="lines", name="CPI (Cum)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SPI_M"], mode="markers", name="SPI (M)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["CPI_M"], mode="markers", name="CPI (M)"))

    fig.update_layout(
        title=f"{program} EVMS Trend",
        yaxis=dict(range=[0.9, 1.2], title="EV Indices"),
        xaxis_title="Month",
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# =====================================================================
# MAIN
# =====================================================================

openplan_raw = pd.read_excel(openplan_path)
openplan_norm = normalize(openplan_raw)

for program, cobra_path in cobra_files.items():
    print(f"\nProcessing → {program} from {cobra_path} ...")

    cobra_raw = pd.read_excel(cobra_path)
    ts, sub_ctd_df, sub_lsp_df, prog_summary = compute_ev_by_subteam(cobra_raw)

    bei_ctd_prog, bei_lsp_prog, bei_sub_df = compute_bei(
        openplan_norm, program, prog_summary["LSP_DATE"]
    )

    # Merge LSP + BEI into subteam tables
    perf = sub_ctd_df.join(sub_lsp_df, how="left").join(bei_sub_df, how="left")

    # Program-level EVMS metrics table
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
                prog_summary["PCT_COMP"],  # same %complete at LSP
            ],
        },
        index=["SPI", "CPI", "BEI", "% Complete"]
    )

    # Labor Hours Performance (per subteam)
    labor_perf = perf[["PCT_COMP", "BAC", "EAC", "VAC"]].copy()
    labor_perf = labor_perf.sort_index()

    # Cost Performance (per subteam)
    cost_perf = perf[["CPI_LSP", "CPI_CTD"]].copy()

    # Schedule Performance (per subteam)
    sched_perf = perf[["SPI_LSP", "SPI_CTD", "BEI_LSP", "BEI_CTD"]].copy()

    # Program Manpower (single row, program level)
    last_d, this_d, next_d, actual_d, pct_var = compute_manpower(
        prog_summary["prog_ts"], prog_summary["LSP_DATE"]
    )

    manpower = pd.DataFrame(
        {
            "Demand": [this_d],
            "Actual": [actual_d],
            "Last Month": [last_d],
            "Next Month": [next_d],
            "%Var": [pct_var],
        },
        index=[PROGRAM_NAME_MAP[program]]
    )

    # -----------------------------------------------------------------
    # BUILD POWERPOINT
    # -----------------------------------------------------------------
    prs = Presentation()
    blank = prs.slide_layouts[6]

    # SLIDE 1: EVMS Metrics
    s1 = prs.slides.add_slide(blank)
    title = s1.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title.text_frame.text = f"{program} – EVMS Metrics"

    tbl1 = add_table(
        s1, rows=evms_metrics.shape[0] + 1, cols=3,
        left=Inches(0.3), top=Inches(0.8), width=Inches(7), height=Inches(1.5),
        headers=["Metric", "CTD", "LSP"]
    )
    for i, (metric, row) in enumerate(evms_metrics.iterrows(), start=1):
        tbl1.cell(i, 0).text = metric
        for j, col in enumerate(["CTD", "LSP"], start=1):
            val = row[col]
            tbl1.cell(i, j).text = "" if pd.isna(val) else f"{val:.2f}"
            if metric in ["SPI", "CPI", "BEI"]:
                set_bg(tbl1.cell(i, j), idx_color(val))

    # SLIDE 2: Labor Hours Performance (per subteam)
    s2 = prs.slides.add_slide(blank)
    title2 = s2.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title2.text_frame.text = f"{program} – Labor Hours Performance"

    tbl2 = add_table(
        s2, rows=labor_perf.shape[0] + 1, cols=6,
        left=Inches(0.3), top=Inches(0.7), width=Inches(9), height=Inches(3),
        headers=["Subteam", "%Comp", "BAC", "EAC", "VAC", "Comments"]
    )
    for i, (st, row) in enumerate(labor_perf.iterrows(), start=1):
        tbl2.cell(i, 0).text = str(st)
        tbl2.cell(i, 1).text = "" if pd.isna(row["PCT_COMP"]) else f"{row['PCT_COMP']*100:,.1f}%"
        tbl2.cell(i, 2).text = f"{row['BAC']:,.0f}"
        tbl2.cell(i, 3).text = f"{row['EAC']:,.0f}"
        tbl2.cell(i, 4).text = f"{row['VAC']:,.0f}"
        set_bg(tbl2.cell(i, 4), vac_color(row["VAC"] / row["BAC"] if row["BAC"] else np.nan))

    # SLIDE 3: Cost Performance
    s3 = prs.slides.add_slide(blank)
    title3 = s3.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title3.text_frame.text = f"{program} – Cost Performance"

    tbl3 = add_table(
        s3, rows=cost_perf.shape[0] + 1, cols=4,
        left=Inches(0.3), top=Inches(0.7), width=Inches(7), height=Inches(3),
        headers=["Subteam", "CPI LSP", "CPI CTD", "Comments"]
    )
    for i, (st, row) in enumerate(cost_perf.iterrows(), start=1):
        tbl3.cell(i, 0).text = str(st)
        for j, col in enumerate(["CPI_LSP", "CPI_CTD"], start=1):
            val = row[col]
            tbl3.cell(i, j).text = "" if pd.isna(val) else f"{val:.2f}"
            set_bg(tbl3.cell(i, j), idx_color(val))

    # SLIDE 4: Schedule Performance
    s4 = prs.slides.add_slide(blank)
    title4 = s4.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title4.text_frame.text = f"{program} – Schedule Performance"

    tbl4 = add_table(
        s4, rows=sched_perf.shape[0] + 1, cols=6,
        left=Inches(0.3), top=Inches(0.7), width=Inches(9), height=Inches(3),
        headers=["Subteam", "SPI LSP", "SPI CTD", "BEI LSP", "BEI CTD", "Comments"]
    )
    for i, (st, row) in enumerate(sched_perf.iterrows(), start=1):
        tbl4.cell(i, 0).text = str(st)
        for j, col in enumerate(["SPI_LSP", "SPI_CTD", "BEI_LSP", "BEI_CTD"], start=1):
            val = row.get(col, np.nan)
            tbl4.cell(i, j).text = "" if pd.isna(val) else f"{val:.2f}"
            if "SPI" in col or "BEI" in col:
                set_bg(tbl4.cell(i, j), idx_color(val))

    # SLIDE 5: Program Manpower
    s5 = prs.slides.add_slide(blank)
    title5 = s5.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title5.text_frame.text = f"{program} – Program Manpower"

    tbl5 = add_table(
        s5, rows=2, cols=7,
        left=Inches(0.3), top=Inches(0.7), width=Inches(9), height=Inches(1.5),
        headers=["Program", "Demand", "Actual", "Last Month", "Next Month", "%Var", "Comments"]
    )
    prog_row = manpower.iloc[0]
    tbl5.cell(1, 0).text = manpower.index[0]
    tbl5.cell(1, 1).text = "" if pd.isna(prog_row["Demand"]) else f"{prog_row['Demand']:.1f}"
    tbl5.cell(1, 2).text = "" if pd.isna(prog_row["Actual"]) else f"{prog_row['Actual']:.1f}"
    tbl5.cell(1, 3).text = "" if pd.isna(prog_row["Last Month"]) else f"{prog_row['Last Month']:.1f}"
    tbl5.cell(1, 4).text = "" if pd.isna(prog_row["Next Month"]) else f"{prog_row['Next Month']:.1f}"
    tbl5.cell(1, 5).text = "" if pd.isna(prog_row["%Var"]) else f"{prog_row['%Var']*100:.1f}%"
    set_bg(tbl5.cell(1, 5), manpower_color(1 + prog_row["%Var"] if not pd.isna(prog_row["%Var"]) else np.nan))

    # SLIDE 6: EVMS Trend Chart
    s6 = prs.slides.add_slide(blank)
    title6 = s6.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9), Inches(0.5))
    title6.text_frame.text = f"{program} – EVMS Trend"

    fig = make_evms_chart(program, prog_summary["prog_ts"])
    chart_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Trend.png")
    fig.write_image(chart_path, scale=3)
    s6.shapes.add_picture(chart_path, Inches(0.3), Inches(0.7), width=Inches(9))

    # SAVE PPT
    out_file = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_file)
    print(f"Saved → {out_file}")

print("\nALL EVMS DASHBOARDS COMPLETE ✅")