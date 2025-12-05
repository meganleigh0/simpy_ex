# =====================================================================
# EVMS DASHBOARD POWERPOINT GENERATOR (HOURS-BASED, CTD + LSP)
# - One PPT per program
# - 3 slides/program:
#     1) EVMS metrics table (CTD + Last Status Period) + comments box
#     2) Labor hours + manpower summary + comments box
#     3) EVMS trend chart (SPI/CPI, cumulative + monthly)
# =====================================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# -----------------------------------------------------------
# CONFIG: PROGRAMS + COBRA FILE PATHS (update to your paths)
# -----------------------------------------------------------
cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS":      "data/Cobra-Abrams STS.xlsx",
    "XM30":            "data/Cobra-XM30.xlsx"
}

OUTPUT_DIR = "EVMS_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you have a corporate template, set it here, else None for blank
THEME_PATH = None  # e.g. "data/ProgramDashboardTemplate.pptx"

# RGB palette approximating your GDLS threshold key
COLOR_BLUE   = RGBColor(31, 73, 125)   # top performance
COLOR_TEAL   = RGBColor(142, 180, 227) # good
COLOR_GREEN  = RGBColor(51, 153, 102)  # acceptable
COLOR_YELLOW = RGBColor(255, 255, 153) # caution
COLOR_RED    = RGBColor(192, 80, 77)   # concern
COLOR_WHITE  = RGBColor(255, 255, 255)
COLOR_BLACK  = RGBColor(0, 0, 0)

# -----------------------------------------------------------
# Helpers: basic normalization
# -----------------------------------------------------------
def normalize_columns(df):
    df = df.rename(columns=lambda c: c.strip())
    upper_map = {c: c.upper().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=upper_map)
    return df

# -----------------------------------------------------------
# Map Cobra COST_SET values to EV roles (BCWS/BCWP/ACWP/ETC)
# -----------------------------------------------------------
def map_cost_sets(cost_cols):
    """
    cost_cols: list of COST_SET values after pivot (column names)
    Returns (bcws_col, bcwp_col, acwp_col, etc_col_or_None)
    """
    cleaned = {col: str(col).replace("_", "").replace("-", "").upper() for col in cost_cols}
    bcws = bcwp = acwp = etc = None

    for orig, clean in cleaned.items():
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig
        elif ("BCWP" in clean) or ("PROGRESS" in clean) or ("EARNED" in clean):
            bcwp = orig
        elif "ETC" in clean:
            etc = orig

    return bcws, bcwp, acwp, etc

# -----------------------------------------------------------
# Compute EV metrics from a Cobra dataframe (Hours PLUG only)
# -----------------------------------------------------------
def compute_ev_from_cobra(df):
    """
    df: raw Cobra dataframe for a single program
    Returns:
        ev_df: DataFrame with DATE index and BCWS, BCWP, ACWP, ETC cols (hours)
        summary: dict with CTD + LSP metrics (SPI, CPI, %COMP, BAC, EAC, VAC, ETC, date info)
    """

    # Filter to PLUG = "Hours" if PLUG column exists
    if "PLUG" in df.columns:
        df = df[df["PLUG"].astype(str).str.lower().str.contains("hour")]

    required = ["COST_SET", "DATE", "HOURS"]
    # Try to locate columns by fuzzy match
    col_map = {}
    for need in required:
        for c in df.columns:
            if need in c:
                col_map[need] = c
                break
    if len(col_map) < 3:
        raise ValueError(f"Could not find COST_SET/DATE/HOURS in columns: {df.columns.tolist()}")

    df = df.rename(columns={col_map["COST_SET"]: "COST_SET",
                            col_map["DATE"]: "DATE",
                            col_map["HOURS"]: "HOURS"})

    # Ensure DATE is datetime
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Pivot by COST_SET
    pivot = df.pivot_table(index="DATE", columns="COST_SET", values="HOURS", aggfunc="sum").sort_index()
    cost_cols = [c for c in pivot.columns]

    bcws_col, bcwp_col, acwp_col, etc_col = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")
    if missing:
        raise ValueError(f"Missing required cost sets {missing}. Found {cost_cols}")

    ev = pd.DataFrame(index=pivot.index)
    ev["BCWS"] = pivot[bcws_col]
    ev["BCWP"] = pivot[bcwp_col]
    ev["ACWP"] = pivot[acwp_col]
    if etc_col is not None:
        ev["ETC"] = pivot[etc_col]
    else:
        ev["ETC"] = 0.0

    # Fill NaNs with 0 for accumulation
    ev = ev.fillna(0.0)

    # Cumulative sums
    ev["BCWS_CUM"] = ev["BCWS"].cumsum()
    ev["BCWP_CUM"] = ev["BCWP"].cumsum()
    ev["ACWP_CUM"] = ev["ACWP"].cumsum()

    # BAC = final cumulative BCWS
    BAC = ev["BCWS_CUM"].iloc[-1]
    ACWP_CTD = ev["ACWP_CUM"].iloc[-1]
    BCWP_CTD = ev["BCWP_CUM"].iloc[-1]

    # ETC total: use latest ETC row if ETC is a snapshot, else sum
    ETC_total = ev["ETC"].iloc[-1] if (ev["ETC"] != 0).any() else 0.0

    EAC = ACWP_CTD + ETC_total
    VAC = BAC - EAC

    # Contract-to-Date SPI/CPI/%COMP
    SPI_CTD = BCWP_CTD / BAC if BAC != 0 else np.nan
    CPI_CTD = BCWP_CTD / ACWP_CTD if ACWP_CTD != 0 else np.nan
    PCT_COMP_CTD = BCWP_CTD / BAC if BAC != 0 else np.nan

    # Last Status Period = last date in data
    lsp_date = ev.index.max()
    row_lsp = ev.loc[lsp_date]

    SPI_LSP = (row_lsp["BCWP"] / row_lsp["BCWS"]) if row_lsp["BCWS"] != 0 else np.nan
    CPI_LSP = (row_lsp["BCWP"] / row_lsp["ACWP"]) if row_lsp["ACWP"] != 0 else np.nan

    # %COMP at LSP = CTD % complete (same as CTD, but using all data through LSP)
    PCT_COMP_LSP = PCT_COMP_CTD

    summary = {
        "BAC": BAC,
        "EAC": EAC,
        "VAC": VAC,
        "ETC": ETC_total,
        "SPI_CTD": SPI_CTD,
        "SPI_LSP": SPI_LSP,
        "CPI_CTD": CPI_CTD,
        "CPI_LSP": CPI_LSP,
        "PCT_COMP_CTD": PCT_COMP_CTD,
        "PCT_COMP_LSP": PCT_COMP_LSP,
        "LSP_DATE": lsp_date
    }

    # Add monthly / cumulative SPI/CPI for plotting
    ev["SPI_M"] = ev["BCWP"] / ev["BCWS"].replace(0, np.nan)
    ev["CPI_M"] = ev["BCWP"] / ev["ACWP"].replace(0, np.nan)
    ev["SPI_CUM"] = ev["BCWP_CUM"] / ev["BCWS_CUM"].replace(0, np.nan)
    ev["CPI_CUM"] = ev["BCWP_CUM"] / ev["ACWP_CUM"].replace(0, np.nan)

    return ev, summary

# -----------------------------------------------------------
# Threshold coloring functions
# -----------------------------------------------------------
def idx_color(value):
    """
    Color for SPI/CPI/BEI style indices.
    Approx thresholds:
      >=1.05 -> blue
      1.05 > X >=1.02 -> teal
      1.02 > X >=0.98 -> green
      0.98 > X >=0.95 -> yellow
      <0.95 -> red
    """
    if value is None or np.isnan(value):
        return None
    if value >= 1.05:
        return COLOR_BLUE
    if value >= 1.02:
        return COLOR_TEAL
    if value >= 0.98:
        return COLOR_GREEN
    if value >= 0.95:
        return COLOR_YELLOW
    return COLOR_RED

def manpower_color(pct):
    """
    Program manpower thresholds (Actual / Demand).
    Rough mapping from your key:
      >=110% -> red
      110%>X>=105% -> yellow
      105%>X>=90%  -> green
      90%>X>=85%   -> yellow
      <85%         -> red
    """
    if pct is None or np.isnan(pct):
        return None
    if pct >= 1.10:
        return COLOR_RED
    if pct >= 1.05:
        return COLOR_YELLOW
    if pct >= 0.90:
        return COLOR_GREEN
    if pct >= 0.85:
        return COLOR_YELLOW
    return COLOR_RED

def vac_color(vac_ratio):
    """
    VAC/BAC thresholds, rough approximation:
      >= +0.05 -> blue
      +0.05 > X >= +0.02 -> green
      +0.02 > X >= -0.02 -> yellow
      -0.02 > X >= -0.05 -> teal
      < -0.05 -> red
    """
    if vac_ratio is None or np.isnan(vac_ratio):
        return None
    if vac_ratio >= 0.05:
        return COLOR_BLUE
    if vac_ratio >= 0.02:
        return COLOR_GREEN
    if vac_ratio >= -0.02:
        return COLOR_YELLOW
    if vac_ratio >= -0.05:
        return COLOR_TEAL
    return COLOR_RED

# -----------------------------------------------------------
# Plotting: EVMS trend (SPI/CPI) with colored bands
# -----------------------------------------------------------
def create_evms_figure(program, ev):
    fig = go.Figure()

    # Background EV bands (roughly matching your chart)
    fig.add_hrect(y0=0.90, y1=0.95, fillcolor="red", opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.95, y1=0.98, fillcolor="yellow", opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.98, y1=1.02, fillcolor="green", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.02, y1=1.05, fillcolor="lightblue", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.05, y1=1.20, fillcolor="rgb(200,220,255)", opacity=0.25, line_width=0)

    dates = ev.index

    # Monthly CPI (yellow diamonds)
    fig.add_trace(go.Scatter(
        x=dates, y=ev["CPI_M"],
        mode="markers",
        marker=dict(symbol="diamond", size=8, color="gold"),
        name="Monthly CPI"
    ))

    # Monthly SPI (black dots)
    fig.add_trace(go.Scatter(
        x=dates, y=ev["SPI_M"],
        mode="markers",
        marker=dict(size=7, color="black"),
        name="Monthly SPI"
    ))

    # Cumulative CPI (blue line)
    fig.add_trace(go.Scatter(
        x=dates, y=ev["CPI_CUM"],
        mode="lines",
        line=dict(color="blue", width=4),
        name="Cumulative CPI"
    ))

    # Cumulative SPI (grey line)
    fig.add_trace(go.Scatter(
        x=dates, y=ev["SPI_CUM"],
        mode="lines",
        line=dict(color="grey", width=4),
        name="Cumulative SPI"
    ))

    fig.update_layout(
        title=f"{program} EVMS Trend",
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.9, 1.2]),
        template="simple_white",
        height=500,
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig

# -----------------------------------------------------------
# PowerPoint helpers
# -----------------------------------------------------------
def new_presentation():
    if THEME_PATH and os.path.exists(THEME_PATH):
        return Presentation(THEME_PATH)
    return Presentation()

def add_comments_textbox(slide, left, top, width, height, title="Comments (RC / CA):"):
    tx_box = slide.shapes.add_textbox(left, top, width, height)
    tf = tx_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.bold = True
    p.font.size = Pt(12)
    tf.add_paragraph()  # blank line for user to type
    return tx_box

def make_table(slide, rows, cols, left, top, width, height, col_headers=None):
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    # Set column widths roughly proportional
    for i in range(cols):
        table.columns[i].width = int(width / cols)
    if col_headers:
        for j, hdr in enumerate(col_headers):
            cell = table.cell(0, j)
            cell.text = hdr
            cell.text_frame.paragraphs[0].font.bold = True
            cell.text_frame.paragraphs[0].font.size = Pt(12)
    return table

def set_cell_color(cell, rgb_color):
    if rgb_color is None:
        return
    fill = cell.fill
    fill.solid()
    fill.fore_color.rgb = rgb_color

# -----------------------------------------------------------
# MAIN LOOP: build PPT for each program
# -----------------------------------------------------------
for program, path in cobra_files.items():
    print(f"Processing {program} from {path} ...")

    # Load cobra
    df_raw = pd.read_excel(path)
    df_raw = normalize_columns(df_raw)

    # Compute EV
    ev, summary = compute_ev_from_cobra(df_raw)

    # ========== NEW PRESENTATION ==========
    prs = new_presentation()

    # =======================================================
    # SLIDE 1: EVMS METRICS (CTD + LSP) + COMMENTS
    # =======================================================
    slide_layout = prs.slide_layouts[5]  # blank-like
    slide1 = prs.slides.add_slide(slide_layout)
    title_shape = slide1.shapes.title if slide1.shapes.title else slide1.shapes.add_textbox(
        Inches(0.5), Inches(0.2), Inches(9), Inches(0.5)
    )
    title_tf = title_shape.text_frame
    title_tf.text = f"{program} – EVMS Metrics (Hours)"

    # Table position
    left = Inches(0.5)
    top = Inches(1.0)
    width = Inches(6.0)
    height = Inches(2.0)

    # Rows: header + SPI/CPI/BEI
    rows = 1 + 3
    cols = 3  # Metric, CTD, LSP
    table = make_table(slide1, rows, cols, left, top, width, height,
                       col_headers=["Metric", "Contract To Date", "Last Status Period"])

    # Row 1: SPI
    spi_ctd = summary["SPI_CTD"]
    spi_lsp = summary["SPI_LSP"]
    table.cell(1, 0).text = "SPI (Labor Hours)"
    table.cell(1, 1).text = f"{spi_ctd:.2f}" if not np.isnan(spi_ctd) else ""
    table.cell(1, 2).text = f"{spi_lsp:.2f}" if not np.isnan(spi_lsp) else ""
    set_cell_color(table.cell(1, 1), idx_color(spi_ctd))
    set_cell_color(table.cell(1, 2), idx_color(spi_lsp))

    # Row 2: CPI
    cpi_ctd = summary["CPI_CTD"]
    cpi_lsp = summary["CPI_LSP"]
    table.cell(2, 0).text = "CPI (Labor Hours)"
    table.cell(2, 1).text = f"{cpi_ctd:.2f}" if not np.isnan(cpi_ctd) else ""
    table.cell(2, 2).text = f"{cpi_lsp:.2f}" if not np.isnan(cpi_lsp) else ""
    set_cell_color(table.cell(2, 1), idx_color(cpi_ctd))
    set_cell_color(table.cell(2, 2), idx_color(cpi_lsp))

    # Row 3: BEI placeholder (you can wire in real BEI data later)
    table.cell(3, 0).text = "BEI / Hit Rate"
    table.cell(3, 1).text = ""
    table.cell(3, 2).text = ""

    # Comments textbox on right
    add_comments_textbox(
        slide1,
        left=Inches(6.7),
        top=Inches(1.0),
        width=Inches(3.0),
        height=Inches(3.0)
    )

    # Footer with LSP date
    footer_box = slide1.shapes.add_textbox(Inches(0.5), Inches(3.2), Inches(4.5), Inches(0.3))
    f_tf = footer_box.text_frame
    f_tf.text = f"Last Status Period: {summary['LSP_DATE'].strftime('%d-%b-%Y')}"

    # =======================================================
    # SLIDE 2: LABOR HOURS + MANPOWER SUMMARY + COMMENTS
    # =======================================================
    slide2 = prs.slides.add_slide(slide_layout)
    title2 = slide2.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.5))
    t2_tf = title2.text_frame
    t2_tf.text = f"{program} – Labor Hours & Manpower Summary"

    # ---- Labor summary table (BAC/EAC/VAC/ETC/%COMP) ----
    left = Inches(0.5)
    top = Inches(1.0)
    width = Inches(6.0)
    height = Inches(1.2)

    table2 = make_table(
        slide2,
        rows=2,
        cols=5,
        left=left,
        top=top,
        width=width,
        height=height,
        col_headers=["BAC (hrs)", "EAC (hrs)", "VAC (hrs)", "ETC (hrs)", "% Complete CTD"]
    )

    BAC = summary["BAC"]
    EAC = summary["EAC"]
    VAC = summary["VAC"]
    ETC_total = summary["ETC"]
    pct_comp = summary["PCT_COMP_CTD"]

    table2.cell(1, 0).text = f"{BAC:,.0f}"
    table2.cell(1, 1).text = f"{EAC:,.0f}"
    table2.cell(1, 2).text = f"{VAC:,.0f}"
    table2.cell(1, 3).text = f"{ETC_total:,.0f}"
    table2.cell(1, 4).text = f"{pct_comp*100:,.1f}%" if not np.isnan(pct_comp) else ""

    set_cell_color(table2.cell(1, 2), vac_color(VAC / BAC if BAC else np.nan))

    # ---- Manpower summary (simple, based on BAC vs EAC actual hours ratio) ----
    # Approximating Demand = BAC, Actual = ACWP_CTD
    demand = BAC
    actual = summary["EAC"] - summary["ETC"]  # approximate actual as ACWP_CTD
    pct_var = (actual / demand) if demand else np.nan
    next_month = np.nan  # placeholder; you can wire forecast in later

    top2 = Inches(2.4)
    table3 = make_table(
        slide2,
        rows=2,
        cols=4,
        left=left,
        top=top2,
        width=width,
        height=Inches(1.0),
        col_headers=["Demand (hrs)", "Actual (hrs)", "% Var", "Next (hrs)"]
    )

    table3.cell(1, 0).text = f"{demand:,.0f}"
    table3.cell(1, 1).text = f"{actual:,.0f}"
    table3.cell(1, 2).text = f"{pct_var*100:,.1f}%" if not np.isnan(pct_var) else ""
    table3.cell(1, 3).text = ""  # you can fill forecast later

    set_cell_color(table3.cell(1, 2), manpower_color(pct_var))

    # Comments textbox on right
    add_comments_textbox(
        slide2,
        left=Inches(6.7),
        top=Inches(1.0),
        width=Inches(3.0),
        height=Inches(3.0)
    )

    # =======================================================
    # SLIDE 3: EVMS TREND CHART
    # =======================================================
    slide3 = prs.slides.add_slide(slide_layout)
    title3 = slide3.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.5))
    title3.text_frame.text = f"{program} – EVMS Trend (SPI/CPI)"

    fig = create_evms_figure(program, ev)
    img_path = os.path.join(OUTPUT_DIR, f"{program}_EVMS.png")
    fig.write_image(img_path, scale=3)

    # Add chart image
    slide3.shapes.add_picture(
        img_path,
        left=Inches(0.5),
        top=Inches(0.8),
        width=Inches(9.0)
    )

    # =======================================================
    # SAVE PRESENTATION
    # =======================================================
    out_pptx = os.path.join(OUTPUT_DIR, f"{program}_EVMS_Dashboard.pptx")
    prs.save(out_pptx)
    print(f"Saved {out_pptx}")

print("ALL EVMS DASHBOARDS COMPLETE ✅")