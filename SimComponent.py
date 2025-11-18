import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 0) Load Cobra data
# -------------------------------------------------------------------
xl = pd.ExcelFile(DATA_PATH)
cobra = xl.parse(SHEET_NAME)

cobra["DATE"] = pd.to_datetime(cobra["DATE"], errors="coerce")
cobra = cobra[cobra["DATE"].notna()].copy()   # drop bad dates

# -------------------------------------------------------------------
# 1) PROGRAM MANPOWER TABLE  (SHC only, 9/80 schedule)
# -------------------------------------------------------------------

# 9/80 available hours table from OpPlan (2024–2028)
available_9_80 = {
    2024: [142, 160, 196, 156, 160, 191, 151, 160, 191, 160, 151, 155],
    2025: [124, 160, 200, 160, 160, 191, 152, 160, 191, 191, 160, 173],
    2026: [147, 160, 200, 160, 151, 195, 156, 160, 191, 160, 151, 151],
    2027: [147, 160, 192, 160, 151, 190, 151, 160, 191, 160, 151, 160],
    2028: [151, 160, 200, 160, 160, 191, 151, 160, 191, 160, 142, 160],
}

def get_available_hours(year, month):
    """Return 9/80 available hours for a given year/month; NaN if not in table."""
    try:
        return available_9_80[year][month - 1]
    except KeyError:
        return np.nan

# SHC only
shc = cobra[cobra[GROUP_COL] == "SHC"].copy()
shc["YEAR"] = shc["DATE"].dt.year
shc["MONTH"] = shc["DATE"].dt.month

if shc.empty:
    # No SHC data → return NaNs but do NOT crash
    program_manpower_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ["SHC"],
            "Demand": [np.nan],
            "Actual": [np.nan],
            "Next Month BCWS": [np.nan],
            "Next Month ETC": [np.nan],
        }
    )
else:
    # Use ANCHOR month if present in data; otherwise use last month from SHC data
    anchor_y, anchor_m = ANCHOR.year, ANCHOR.month
    has_anchor_month = ((shc["YEAR"] == anchor_y) & (shc["MONTH"] == anchor_m)).any()

    if has_anchor_month:
        cur_year, cur_month = anchor_y, anchor_m
    else:
        last_date = shc["DATE"].dropna().max()
        cur_year, cur_month = int(last_date.year), int(last_date.month)

    # Next month
    if cur_month == 12:
        next_year, next_month = cur_year + 1, 1
    else:
        next_year, next_month = cur_year, cur_month + 1

    # Sum SHC hours by COST-SET for current and next month
    cur_hours = (
        shc[(shc["YEAR"] == cur_year) & (shc["MONTH"] == cur_month)]
        .groupby("COST-SET")["HOURS"]
        .sum()
    )
    next_hours = (
        shc[(shc["YEAR"] == next_year) & (shc["MONTH"] == next_month)]
        .groupby("COST-SET")["HOURS"]
        .sum()
    )

    # Ensure required cost-sets exist
    cur_hours = cur_hours.reindex(["BCWS", "ACWP", "ETC"], fill_value=0.0)
    next_hours = next_hours.reindex(["BCWS", "ACWP", "ETC"], fill_value=0.0)

    # Available hours for current & next month
    cur_avail = get_available_hours(cur_year, cur_month)
    next_avail = get_available_hours(next_year, next_month)

    # Convert hours → FTE (headcount); guard against NaN / 0 available hours
    demand = cur_hours["BCWS"] / cur_avail if cur_avail and not np.isnan(cur_avail) else np.nan
    actual = cur_hours["ACWP"] / cur_avail if cur_avail and not np.isnan(cur_avail) else np.nan
    next_bcws_fte = (
        next_hours["BCWS"] / next_avail if next_avail and not np.isnan(next_avail) else np.nan
    )
    next_etc_fte = (
        next_hours["ETC"] / next_avail if next_avail and not np.isnan(next_avail) else np.nan
    )

    program_manpower_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ["SHC"],
            "Demand": [round(demand, 1)],
            "Actual": [round(actual, 1)],
            "Next Month BCWS": [round(next_bcws_fte, 1)],
            "Next Month ETC": [round(next_etc_fte, 1)],
        }
    )

# -------------------------------------------------------------------
# 2) EVMS TABLE (Monthly & Cumulative CPI / SPI)
# -------------------------------------------------------------------

# Use all teams, CTD through ANCHOR
cobra_ctd = cobra[cobra["DATE"] <= ANCHOR].copy()

ev = (
    cobra_ctd.groupby([pd.Grouper(key="DATE", freq="M"), "COST-SET"])["HOURS"]
    .sum()
    .unstack(fill_value=0.0)
    .sort_index()
)

# Make sure BCWP, BCWS, ACWP exist
for k in ["BCWP", "BCWS", "ACWP"]:
    if k not in ev.columns:
        ev[k] = 0.0

# Monthly indices
ev["CPI_month"] = np.where(ev["ACWP"] == 0, np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"] == 0, np.nan, ev["BCWP"] / ev["BCWS"])

# Cumulative indices
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

ev["CPI_cum"] = np.where(ev["ACWP_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

# Clean EVMS table for display
evms_tbl = ev[["CPI_month", "SPI_month", "CPI_cum", "SPI_cum"]].copy()
evms_tbl.index = evms_tbl.index.to_period("M").strftime("%b-%y")
evms_tbl = evms_tbl.reset_index().rename(columns={"index": "Month"})

# -------------------------------------------------------------------
# 3) EVMS PLOTLY CHART (like Jason’s slide)
# -------------------------------------------------------------------
fig = go.Figure()

# Colored performance bands (red / yellow / green / blue)
fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#ff4d4d", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=0.95, y1=1.00, fillcolor="#ffd633", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=1.00, y1=1.10, fillcolor="#66cc66", opacity=0.3, line_width=0, layer="below")
fig.add_hrect(y0=1.10, y1=1.20, fillcolor="#99ccff", opacity=0.3, line_width=0, layer="below")

x_vals = evms_tbl["Month"]

# Monthly CPI (diamonds)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl["CPI_month"],
        mode="markers",
        name="Monthly CPI",
        marker=dict(symbol="diamond", size=10, color="#f4b183"),
    )
)

# Monthly SPI (black dots)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl["SPI_month"],
        mode="markers",
        name="Monthly SPI",
        marker=dict(symbol="circle", size=10, color="#000000"),
    )
)

# Cumulative CPI (blue line)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl["CPI_cum"],
        mode="lines",
        name="Cumulative CPI",
        line=dict(color="#0050b3", width=3),
    )
)

# Cumulative SPI (gray line)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=evms_tbl["SPI_cum"],
        mode="lines",
        name="Cumulative SPI",
        line=dict(color="#7f7f7f", width=3),
    )
)

fig.update_layout(
    title="EV Indices",
    xaxis_title="Month",
    yaxis_title="Index",
    yaxis=dict(range=[0.9, 1.2]),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=40, r=20, t=60, b=80),
)

# -------------------------------------------------------------------
# 4) OUTPUTS
# -------------------------------------------------------------------
print("Program Manpower (SHC):")
display(program_manpower_tbl)

print("\nEVMS Indices Table:")
display(evms_tbl)

fig.show()